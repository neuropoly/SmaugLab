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
import unittest
from pathlib import Path

import torch

from auglab.transforms.gpu.base import AugmentationSequentialCustom
from unit_tests.helpers import AugLabTestCase, first_output

TRANSFORM_MODULES = [
    "auglab.transforms.gpu.contrast",
    "auglab.transforms.gpu.spatial",
    "auglab.transforms.gpu.fromSeg",
    "auglab.transforms.gpu.domain_transfer",
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
    """Some transforms depend on assets that do not exist on a fresh checkout."""
    if cls.__name__ == "RandomDomainTransferGPU":
        from auglab.transforms.gpu.domain_transfer import DEFAULT_BANK_PATH

        if not Path(DEFAULT_BANK_PATH).is_file():
            return f"domain transfer bank not available at {DEFAULT_BANK_PATH}"
    return None


class TestTransformDiscovery(unittest.TestCase):
    def test_discovery_found_transforms(self):
        self.assertGreaterEqual(
            len(DISCOVERED),
            15,
            f"expected the bulk of the GPU transforms, found {len(DISCOVERED)}",
        )


class TestTransformsRunStandalone(AugLabTestCase):
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


if __name__ == "__main__":
    unittest.main()
