"""Each GPU transform, exercised on its own.

The config-level tests in test_configs.py prove the pipelines people actually
use still work. These prove each transform works in isolation, so a failure
points at one class instead of a whole config.

Transforms are discovered by introspection rather than listed by hand, so a
newly added transform is covered the moment it lands.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch

from auglab.transforms.gpu.base import AugmentationSequentialCustom

TRANSFORM_MODULES = [
    "auglab.transforms.gpu.contrast",
    "auglab.transforms.gpu.spatial",
    "auglab.transforms.gpu.fromSeg",
    "auglab.transforms.gpu.domain_transfer",
]

# Not augmentations: helper modules that happen to be nn.Module subclasses.
NOT_A_TRANSFORM = {"DifferentiableHistogram3D"}


def _discover():
    """Collect transform classes that can be constructed without arguments."""
    found = []
    for module_name in TRANSFORM_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or obj.__module__ != module_name:
                continue
            if not issubclass(obj, torch.nn.Module) or name in NOT_A_TRANSFORM:
                continue
            signature = inspect.signature(obj.__init__)
            required = [
                param
                for param in list(signature.parameters.values())[1:]
                if param.default is inspect.Parameter.empty and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            ]
            if required:
                # Needs caller-supplied configuration; covered via test_configs.py.
                continue
            found.append((f"{module_name.rsplit('.', 1)[-1]}.{name}", obj, signature))
    return sorted(found, key=lambda item: item[0])


DISCOVERED = _discover()


def _build_kwargs(cls, signature) -> dict:
    """Construction arguments that make a transform actually do something.

    `p` is forced to 1.0 because most transforms default to a low probability
    and would otherwise pass through untouched most of the time.
    """
    kwargs = {"p": 1.0} if "p" in signature.parameters else {}
    if cls.__name__ == "RandomDomainTransferGPU":
        # Every parameter has a default, but the constructor still rejects a
        # missing source_label unless it is told to draw from every domain pair.
        kwargs["any_source"] = True
    return kwargs


def _skip_reason(cls) -> str | None:
    """Some transforms depend on assets that do not exist on a fresh checkout."""
    if cls.__name__ == "RandomDomainTransferGPU":
        from auglab.transforms.gpu.domain_transfer import DEFAULT_BANK_PATH

        if not Path(DEFAULT_BANK_PATH).is_file():
            return f"domain transfer bank not available at {DEFAULT_BANK_PATH}"
    return None


def test_discovery_found_transforms():
    assert len(DISCOVERED) >= 15, f"expected the bulk of the GPU transforms, found {len(DISCOVERED)}"


@pytest.mark.parametrize(
    ("cls", "signature"),
    [(cls, sig) for _, cls, sig in DISCOVERED],
    ids=[name for name, _, _ in DISCOVERED],
)
def test_transform_runs_on_a_tiny_volume(cls, signature, tiny_volume, tiny_seg):
    """Drive each transform the way AugTransformsGPU does and check the output."""
    reason = _skip_reason(cls)
    if reason:
        pytest.skip(reason)

    # Force the transform to actually fire; the default probability is often low.
    pipeline = AugmentationSequentialCustom(cls(**_build_kwargs(cls, signature)), data_keys=["input", "mask"], same_on_batch=True)

    result = pipeline(tiny_volume, tiny_seg)
    image = result[0] if isinstance(result, (list, tuple)) else result

    assert image.shape == tiny_volume.shape, f"{cls.__name__} changed the volume shape"
    assert image.dtype.is_floating_point
    assert torch.isfinite(image).all(), f"{cls.__name__} produced NaN or Inf"


@pytest.mark.parametrize(
    ("cls", "signature"),
    [(cls, sig) for _, cls, sig in DISCOVERED],
    ids=[name for name, _, _ in DISCOVERED],
)
def test_transform_leaves_the_mask_intact(cls, signature, tiny_volume, tiny_seg):
    """Image-only transforms must not silently alter the segmentation labels.

    Spatial transforms legitimately move the mask, so only the label *set* is
    checked -- values must stay in {0, 1}, never interpolated into something in
    between.
    """
    reason = _skip_reason(cls)
    if reason:
        pytest.skip(reason)

    pipeline = AugmentationSequentialCustom(cls(**_build_kwargs(cls, signature)), data_keys=["input", "mask"], same_on_batch=True)

    result = pipeline(tiny_volume, tiny_seg)
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        pytest.skip(f"{cls.__name__} does not return a mask")

    mask = result[1]
    assert torch.isfinite(mask).all(), f"{cls.__name__} produced a non-finite mask"
    unique = torch.unique(mask)
    assert unique.numel() <= 2, f"{cls.__name__} interpolated the mask into {unique.numel()} values"
