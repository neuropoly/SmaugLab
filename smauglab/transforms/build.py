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

from typing import Any

from smauglab import registry
from smauglab.config import NON_AUGMENTATION_KEYS, OrderSource, PipelineMode, validate_section
from smauglab.registry import AugEntry, AugType, Backend, InvalidConfigError


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
    order_source: OrderSource = OrderSource.REGISTRY,
) -> list[tuple[Any, AugEntry]]:
    """Instantiate every augmentation in the section, in pipeline order.

    Order comes from `registry.PIPELINE_ORDER` by default rather than from the order
    of keys in the file: the two have never agreed, and honouring the file silently
    reorders a pipeline the moment someone tidies a config. `OrderSource.CONFIG` asks
    for key order explicitly -- see `pipeline.order` in smauglab/config.py.
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
    if order_source is OrderSource.CONFIG:
        return built
    return sorted(built, key=lambda pair: registry.pipeline_position(pair[1]))


def _buckets(built: list[tuple[Any, AugEntry]], *, hoist_sequential: bool) -> tuple[list, list, list, list]:
    """Split into (sequential, transfer, general-enhancement, normalization).

    Geometry always leads, in order. `force_sequential` means "never inside the GE
    bucket" and so only bites when there *is* one: with both groups bucketed,
    `RandomLowResTransformGPU` is hoisted up front alongside the geometry, but when
    only the transfer group is bucketed it stays in its ordinary GE position. That
    is exactly what the two hand-written pipelines did, and the difference is
    invisible unless you line their two ladders up side by side.

    Normalization is its own list because it is the one group whose members are
    alternatives rather than additions -- see `AugType.NORM` and `_normalization_stage`.
    """
    sequential, transfer, enhancement, normalization = [], [], [], []
    for transform, entry in built:
        if entry.group is AugType.GEO or (hoist_sequential and entry.force_sequential):
            sequential.append(transform)
        elif entry.group is AugType.TA:
            transfer.append(transform)
        elif entry.group is AugType.NORM:
            normalization.append(transform)
        else:
            enhancement.append(transform)
    return sequential, transfer, enhancement, normalization


def _normalization_stage(normalization: list, *, options: dict[str, Any]) -> list:
    """Wrap the normalization group so that exactly one of its members runs.

    Chaining normalizations does not compose them -- a min-max rescale on top of a
    z-score simply discards the z-score -- so a config naming several of them is asking
    for a *choice*, not for a sequence. `RandomChooseXTransformsGPU` with
    `num_transforms=1` is that choice already: it draws one child uniformly and then
    spends that child's own `p`, which makes each scheme's `p` its weight
    (`P(scheme) = p / n`) and leaves the remainder as "no normalization at all".
    `random_order=True` in every mode, including the sequential one: with one transform
    to pick there is no order to shuffle, and `False` would pin the choice to whichever
    scheme sorts first.

    One draw per *batch*, not per sample, under the default `pipeline.same_on_batch`.
    That setting defaults to True and the container forces it onto every stage it holds,
    this bucket included, which is what makes a bucket take one decision for the whole
    batch -- the same is true of the `ta` and `ge` buckets today. A config wanting each
    sample to draw its own scheme sets `pipeline.same_on_batch: false`, which is also
    what makes every transform's `p` per-sample; see `SmaugConfig.same_on_batch`.

    Fewer than two members and nothing is wrapped. A single normalization keeps its
    pipeline position verbatim -- wrapping it would move it, which `pipeline.order:
    config` can notice, and would gate it behind a second probability for no gain.
    """
    if len(normalization) < 2:
        return normalization

    from smauglab.transforms.gpu.transforms_list import RandomChooseXTransformsGPU

    return [
        RandomChooseXTransformsGPU(
            transforms_list=normalization,
            num_transforms=1,
            p=options.get("norm_probability", 1.0),
            random_order=True,
            label="norm",
        )
    ]


def build_gpu_pipeline(
    section: dict,
    *,
    mode: PipelineMode = PipelineMode.SEQUENTIAL,
    options: dict[str, Any] | None = None,
    source: str = "<config>",
    order_source: OrderSource = OrderSource.REGISTRY,
) -> list[Any]:
    """The GPU transform list for a given pipeline mode."""
    built = build_transforms(section, Backend.GPU, source=source, order_source=order_source)
    options = options or {}
    if mode is PipelineMode.SEQUENTIAL:
        # Order is left exactly as `build_transforms` produced it, including under
        # `pipeline.order: config`. The only rearrangement is the normalization choice,
        # and only when there is a choice to make: a lone normalization is not touched.
        normalization = [transform for transform, entry in built if entry.group is AugType.NORM]
        if len(normalization) < 2:
            return [transform for transform, _ in built]
        kept = [transform for transform, entry in built if entry.group is not AugType.NORM]
        return kept + _normalization_stage(normalization, options=options)

    from smauglab.transforms.gpu.transforms_list import RandomChooseXTransformsGPU

    sequential, transfer, enhancement, normalization = _buckets(built, hoist_sequential=mode is PipelineMode.RANDOM_ORDER)

    out = list(sequential)
    out.append(
        RandomChooseXTransformsGPU(
            transforms_list=transfer,
            num_transforms=len(transfer),
            p=options.get("ta_probability", 1.0),
            random_order=options.get("ta_random_order", True) if mode is PipelineMode.RANDOM_ORDER else False,
            label="ta",
        )
    )
    if mode is PipelineMode.RANDOM_ORDER:
        out.append(
            RandomChooseXTransformsGPU(
                transforms_list=enhancement,
                num_transforms=len(enhancement),
                p=options.get("ge_probability", 1.0),
                random_order=options.get("ge_random_order", True),
                label="ge",
            )
        )
    else:
        # RANDOM_ORDER_TA shuffles only the transfer augmentations; the general
        # enhancements stay in pipeline order, as they always have.
        out.extend(enhancement)
    # Last in every mode: it standardizes whatever everything above it did.
    out.extend(_normalization_stage(normalization, options=options))
    return out


def build_cpu_pipeline(
    section: dict,
    *,
    do_dummy_2d_data_aug: bool,
    patch_size: Any,
    rotation: Any,
    source: str = "<config>",
    order_source: OrderSource = OrderSource.REGISTRY,
) -> list[Any]:
    """The CPU transform list, with nnU-Net's runtime context injected.

    `SpatialTransform` is bracketed by the 2D/3D converters when dummy 2D
    augmentation is on. Those converters are not augmentations and carry no registry
    entry -- they are pipeline structure, emitted here around the slot.
    """
    patch_size_spatial = patch_size[1:] if do_dummy_2d_data_aug else patch_size
    context = {"patch_size": patch_size_spatial, "rotation": rotation}
    built = build_transforms(section, Backend.CPU, context=context, source=source, order_source=order_source)

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
