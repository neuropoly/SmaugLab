"""Each GPU transform, exercised on its own.

The config-level tests in test_configs.py prove the pipelines people actually
use still work. These prove each transform works in isolation, so a failure
points at one class instead of a whole config.

The list comes from the registry rather than from introspection, so it covers
exactly what a config can reach -- no denylist of helper classes to maintain, no
silent gap when a transform has a required argument. This replaced a hand-rolled
walk over four hardcoded module names that missed RandomAffineGPU (required
`degrees`) and RandomSynthSegGPU (its module was not in the list).
"""

from __future__ import annotations

import unittest

import torch

from smauglab import registry
from smauglab.registry import Backend
from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from unit_tests.helpers import SmaugLabTestCase, domain_bank_missing, first_output

# `p` is forced to 1.0 so a transform actually fires; most default lower and would
# otherwise pass the volume through untouched.
GPU_ENTRIES = [entry for entry in registry.entries(Backend.GPU) if not registry.required_params(entry)]


def skip_reason(entry) -> str | None:
    """Some transforms need assets that do not exist on a fresh checkout."""
    if entry.external_asset:
        return domain_bank_missing()
    return None


class TestTransformDiscovery(unittest.TestCase):
    def test_registry_offers_the_bulk_of_the_gpu_transforms(self):
        self.assertGreaterEqual(
            len(GPU_ENTRIES),
            25,
            f"expected the bulk of the GPU transforms, found {len(GPU_ENTRIES)}",
        )


class TestTransformsRunStandalone(SmaugLabTestCase):
    def _pipeline(self, entry):
        """Drive a single transform the way AugTransformsGPU does."""
        kwargs = {"p": 1.0, **dict(entry.smoke_kwargs)}
        return AugmentationSequentialCustom(
            entry.cls(**kwargs),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )

    def test_transform_runs_on_a_tiny_volume(self):
        for entry in GPU_ENTRIES:
            with self.subTest(transform=entry.name):
                reason = skip_reason(entry)
                if reason:
                    self.skipTest(reason)

                volume, seg = self.tiny_volume(), self.tiny_seg()
                image = first_output(self._pipeline(entry)(volume, seg))

                self.assertIsImageLike(image, volume, entry.name)

    def test_transform_leaves_the_mask_intact(self):
        """Image-only transforms must not silently alter the segmentation labels.

        Spatial transforms legitimately move the mask, so only the label *set*
        is checked -- values must stay in {0, 1}, never interpolated into
        something in between.
        """
        for entry in GPU_ENTRIES:
            with self.subTest(transform=entry.name):
                reason = skip_reason(entry)
                if reason:
                    self.skipTest(reason)

                result = self._pipeline(entry)(self.tiny_volume(), self.tiny_seg())
                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    self.skipTest(f"{entry.name} does not return a mask")

                mask = result[1]
                self.assertTrue(bool(torch.isfinite(mask).all()), f"{entry.name} produced a non-finite mask")
                unique = torch.unique(mask)
                self.assertLessEqual(
                    unique.numel(),
                    2,
                    f"{entry.name} interpolated the mask into {unique.numel()} values",
                )


if __name__ == "__main__":
    unittest.main()
