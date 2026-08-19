"""Build an augmentation pipeline from a config section, using the registry.

This replaces four hand-written `if` ladders -- one in `gpu/transforms.py`, two in
`gpu/transforms_list.py`, one in `cpu/transforms.py` -- that between them repeated
the same ~900 lines of dispatch and had already drifted apart. Each ladder decided
three things implicitly that are now explicit registry data: which class a key maps
to (`entry.cls`), where it sits in the pipeline (`entry.order`), and whether it runs
in sequence or inside a shuffled bucket (`entry.group` / `force_sequential`).

Validation is strict and cumulative: an unregistered name, an unaccepted parameter
or a missing required one are all collected and raised together, so a bad config is
fixed in one pass.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from smauglab import registry
from smauglab.registry import AugEntry, AugType, Backend, InvalidConfigError

#: Keys a config section may hold that are not augmentations.
NON_AUGMENTATION_KEYS = ("_",)


class PipelineMode(str, Enum):
    """How the GPU pipeline arranges what the registry gives it."""

    #: Everything in registry order. What `AugTransformsGPU` has always done.
    SEQUENTIAL = "sequential"
    #: Geometry in order, then TA and GE each shuffled inside a RandomChooseX.
    RANDOM_ORDER = "random_order"
    #: As above but the TA bucket keeps its order; GE is not bucketed separately.
    RANDOM_ORDER_TA = "random_order_ta"


def validate_section(section: dict, backend: Backend, *, source: str = "<config>") -> list[str]:
    """Return every problem in a config section. Empty means it will build."""
    problems: list[str] = []
    for name, params in section.items():
        if name.startswith(NON_AUGMENTATION_KEYS):
            continue
        try:
            entry = registry.get(name, backend)
        except registry.UnknownAugmentationError as exc:
            problems.append(str(exc).replace("\n", "\n    "))
            continue
        if not isinstance(params, dict):
            problems.append(f"{name}: expected a block of parameters, got {type(params).__name__}")
            continue

        accepted = registry.accepted_params(entry)
        problems.extend(registry.unknown_parameter_message(entry, key).replace("\n", "\n    ") for key in params if key not in accepted)
        missing = registry.required_params(entry) - set(params) - set(entry.context_params)
        if missing:
            problems.append(f"{name}: missing required parameter(s) {', '.join(sorted(missing))}")
    _ = source
    return problems


def _instantiate(entry: AugEntry, params: dict, context: dict[str, Any]) -> Any:
    """Build one transform, applying adapters and injecting runtime context."""
    kwargs = dict(params)

    # batchgeneratorsv2 ranges need wrapping in BGContrast, which samples differently
    # from the bare tuple -- see AugEntry.param_adapters.
    for name, adapter in entry.param_adapters.items():
        if name in kwargs:
            kwargs[name] = adapter(tuple(kwargs[name]))

    for name in entry.context_params:
        if name in context:
            kwargs[name] = context[name]

    if entry.backend is Backend.CPU and entry.wrap_random:
        # The probability lives on the wrapper, not the transform.
        from batchgeneratorsv2.transforms.utils.random import RandomTransform

        probability = kwargs.pop("p", 1.0)
        return RandomTransform(entry.cls(**kwargs), apply_probability=probability)

    return entry.cls(**kwargs)


def build_transforms(
    section: dict,
    backend: Backend,
    *,
    context: dict[str, Any] | None = None,
    source: str = "<config>",
) -> list[tuple[Any, AugEntry]]:
    """Instantiate every augmentation in the section, in registry order.

    Order comes from the registry rather than from the order of keys in the file:
    the two have never agreed, and honouring the file would silently reorder every
    pipeline the moment someone tidied a config.
    """
    problems = validate_section(section, backend, source=source)
    if problems:
        raise InvalidConfigError(source, problems)

    context = context or {}
    built = []
    for name, params in section.items():
        if name.startswith(NON_AUGMENTATION_KEYS):
            continue
        entry = registry.get(name, backend)
        built.append((_instantiate(entry, params, context), entry))
    return sorted(built, key=lambda pair: pair[1].order)


def _buckets(built: list[tuple[Any, AugEntry]], *, hoist_sequential: bool) -> tuple[list, list, list]:
    """Split into (sequential, transfer, general-enhancement).

    Geometry always leads, in order. `force_sequential` means "never inside the GE
    bucket" and so only bites when there *is* one: with both groups bucketed,
    `RandomLowResTransformGPU` is hoisted up front alongside the geometry, but when
    only the transfer group is bucketed it stays in its ordinary GE position. That
    is exactly what the two hand-written pipelines did, and the difference is
    invisible unless you line their two ladders up side by side.
    """
    sequential, transfer, enhancement = [], [], []
    for transform, entry in built:
        if entry.group is AugType.GEO or (hoist_sequential and entry.force_sequential):
            sequential.append(transform)
        elif entry.group is AugType.TA:
            transfer.append(transform)
        else:
            enhancement.append(transform)
    return sequential, transfer, enhancement


def build_gpu_pipeline(
    section: dict,
    *,
    mode: PipelineMode = PipelineMode.SEQUENTIAL,
    options: dict[str, Any] | None = None,
    source: str = "<config>",
) -> list[Any]:
    """The GPU transform list for a given pipeline mode."""
    built = build_transforms(section, Backend.GPU, source=source)
    if mode is PipelineMode.SEQUENTIAL:
        return [transform for transform, _ in built]

    from smauglab.transforms.gpu.transforms_list import RandomChooseXTransformsGPU

    options = options or {}
    sequential, transfer, enhancement = _buckets(built, hoist_sequential=mode is PipelineMode.RANDOM_ORDER)

    out = list(sequential)
    out.append(
        RandomChooseXTransformsGPU(
            transforms_list=transfer,
            num_transforms=len(transfer),
            p=options.get("ta_probability", 1.0),
            random_order=options.get("ta_random_order", True) if mode is PipelineMode.RANDOM_ORDER else False,
        )
    )
    if mode is PipelineMode.RANDOM_ORDER:
        out.append(
            RandomChooseXTransformsGPU(
                transforms_list=enhancement,
                num_transforms=len(enhancement),
                p=options.get("ge_probability", 1.0),
                random_order=options.get("ge_random_order", True),
            )
        )
    else:
        # RANDOM_ORDER_TA shuffles only the transfer augmentations; the general
        # enhancements stay in pipeline order, as they always have.
        out.extend(enhancement)
    return out


def build_cpu_pipeline(
    section: dict,
    *,
    do_dummy_2d_data_aug: bool,
    patch_size: Any,
    rotation: Any,
    source: str = "<config>",
) -> list[Any]:
    """The CPU transform list, with nnU-Net's runtime context injected.

    `SpatialTransform` is bracketed by the 2D/3D converters when dummy 2D
    augmentation is on. Those converters are not augmentations and carry no registry
    entry -- they are pipeline structure, emitted here around the slot.
    """
    patch_size_spatial = patch_size[1:] if do_dummy_2d_data_aug else patch_size
    context = {"patch_size": patch_size_spatial, "rotation": rotation}
    built = build_transforms(section, Backend.CPU, context=context, source=source)

    if not do_dummy_2d_data_aug:
        return [transform for transform, _ in built]

    from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform

    out: list[Any] = []
    for transform, entry in built:
        if entry.name == "SpatialTransform":
            out.extend([Convert3DTo2DTransform(), transform, Convert2DTo3DTransform()])
        else:
            out.append(transform)
    return out
