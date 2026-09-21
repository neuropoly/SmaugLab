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
import os
import unittest
from pathlib import Path

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from unit_tests.helpers import SmaugLabTestCase, first_output

TRANSFORM_MODULES = [
    "smauglab.transforms.gpu.contrast",
    "smauglab.transforms.gpu.spatial",
    "smauglab.transforms.gpu.fromSeg",
    "smauglab.transforms.gpu.domain_transfer",
    "smauglab.transforms.gpu.palette.transform",
]

# Not augmentations: helper modules that happen to be nn.Module subclasses.
NOT_A_TRANSFORM = {"DifferentiableHistogram3D"}


def discover_transforms():
    """Collect transform classes that can be constructed without arguments."""
    found = []
    for module_name in TRANSFORM_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or obj.__module__ != module_name:
                continue
            if name.startswith("_"):
                # Shared base classes (_RandomConvBaseGPU, _RandomNamedFunctionGPU, ...).
                # They are not augmentations; their concrete leaves are discovered instead.
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


DISCOVERED = discover_transforms()


def build_kwargs(cls, signature) -> dict:
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


def skip_reason(cls) -> str | None:
    """Some transforms depend on assets that do not exist on a fresh checkout.

    Which ones is registry data now (`external_asset` names the environment variable
    that points at the artefact), rather than a hardcoded class name here -- so a
    second such transform is covered the moment it is registered.
    """
    from smauglab import registry

    try:
        entry = registry.get(cls.__name__)
    except registry.UnknownAugmentationError:
        return None
    if not entry.external_asset:
        return None
    location = os.environ.get(entry.external_asset)
    if location and Path(location).is_file():
        return None
    return f"{cls.__name__} needs ${entry.external_asset} to point at its data"


class TestTransformDiscovery(unittest.TestCase):
    def test_discovery_found_transforms(self):
        self.assertGreaterEqual(
            len(DISCOVERED),
            15,
            f"expected the bulk of the GPU transforms, found {len(DISCOVERED)}",
        )


class TestTransformsRunStandalone(SmaugLabTestCase):
    def _pipeline(self, cls, signature):
        """Drive a single transform the way AugTransformsGPU does."""
        return AugmentationSequentialCustom(
            cls(**build_kwargs(cls, signature)),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )

    def test_transform_runs_on_a_tiny_volume(self):
        for label, cls, signature in DISCOVERED:
            with self.subTest(transform=label):
                reason = skip_reason(cls)
                if reason:
                    self.skipTest(reason)

                volume, seg = self.tiny_volume(), self.tiny_seg()
                image = first_output(self._pipeline(cls, signature)(volume, seg))

                self.assertIsImageLike(image, volume, cls.__name__)

    def test_transform_leaves_the_mask_intact(self):
        """Image-only transforms must not silently alter the segmentation labels.

        Spatial transforms legitimately move the mask, so only the label *set*
        is checked -- values must stay in {0, 1}, never interpolated into
        something in between.
        """
        for label, cls, signature in DISCOVERED:
            with self.subTest(transform=label):
                reason = skip_reason(cls)
                if reason:
                    self.skipTest(reason)

                result = self._pipeline(cls, signature)(self.tiny_volume(), self.tiny_seg())
                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    self.skipTest(f"{cls.__name__} does not return a mask")

                mask = result[1]
                self.assertTrue(bool(torch.isfinite(mask).all()), f"{cls.__name__} produced a non-finite mask")
                unique = torch.unique(mask)
                self.assertLessEqual(
                    unique.numel(),
                    2,
                    f"{cls.__name__} interpolated the mask into {unique.numel()} values",
                )


class TestTransformsOnADegeneratePatch(SmaugLabTestCase):
    """Every transform, on the inputs nnU-Net's dataloader actually produces.

    `tiny_volume()`/`tiny_seg()` are always well behaved -- random noise and a blob in
    the middle -- so this sweep never asked what happens when the patch has no variance
    or the mask has no foreground. Both are ordinary events in training, not edge cases:
    roughly 18% of the unconstrained samples on Dataset014 are all background, and a
    patch of air outside the body is *exactly* constant because preprocessing clips it to
    the 0.5th percentile. Four real defects lived behind that gap -- a silent [0, 1]
    rescale, an all-NaN volume, a CUDA-side crash and a fp-noise amplification -- and all
    four are caught by simply running the existing sweep on a degenerate pair.

    The bar here is deliberately low: finite, right shape, no exception. What a transform
    *should* do with such a patch is its own business and is pinned per transform
    elsewhere; what none of them may do is blow up or emit NaN.
    """

    def _cases(self):
        """The two degenerate shapes, and why each one is its own case.

        A constant image with a valid mask breaks range normalisation; a well-behaved
        image with an empty mask breaks anything that partitions by label. They fail in
        different places, so testing only their combination would miss half of it.
        """
        return [
            ("constant image, empty mask", self.constant_volume(), self.empty_seg()),
            ("constant image, normal mask", self.constant_volume(), self.tiny_seg()),
            ("normal image, empty mask", self.tiny_volume(), self.empty_seg()),
        ]

    def _pipeline(self, cls, signature):
        return AugmentationSequentialCustom(
            cls(**build_kwargs(cls, signature)),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )

    def test_no_transform_breaks_on_a_degenerate_patch(self):
        for label, cls, signature in DISCOVERED:
            for case, volume, seg in self._cases():
                with self.subTest(transform=label, case=case):
                    reason = skip_reason(cls)
                    if reason:
                        self.skipTest(reason)

                    image = first_output(self._pipeline(cls, signature)(volume.clone(), seg.clone()))

                    self.assertEqual(image.shape, volume.shape, f"{cls.__name__} changed the shape on {case}")
                    self.assertTrue(
                        bool(torch.isfinite(image).all()),
                        f"{cls.__name__} produced NaN or Inf on {case}",
                    )


if __name__ == "__main__":
    unittest.main()
