"""The real registry, and the artifacts generated from it.

test_registry.py covers the mechanism against synthetic classes. This file covers
the actual augmentations: that every one is registered coherently, that the
generated matrix and template config are not stale, and that every registered
augmentation is genuinely constructible.

Together these are the answer to "which augmentations exist and which backends
have them" -- previously recoverable only by reading four `if` ladders side by side.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from smauglab import registry
from smauglab.registry import AugId, Backend

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "smauglab" / "configs" / "all_augmentations.json"


class TestRegistryIsPopulated(unittest.TestCase):
    def test_both_backends_have_augmentations(self):
        self.assertGreaterEqual(len(registry.names(Backend.GPU)), 25)
        self.assertGreaterEqual(len(registry.names(Backend.CPU)), 20)

    def test_no_monai_implementations_yet(self):
        """Tracked, not built. If this starts failing, the matrix gained a real cell."""
        self.assertEqual(registry.names(Backend.MONAI), [])

    def test_every_aug_id_is_used(self):
        """An unused AugId is a concept nothing implements -- almost always a typo."""
        used = {entry.aug_id for entry in registry.entries()}
        self.assertEqual(set(AugId) - used, set(), "AugId members with no implementation on any backend")

    def test_class_name_is_the_config_key(self):
        for entry in registry.entries():
            with self.subTest(entry=entry.name):
                self.assertEqual(entry.cls.__name__, entry.name)

    def test_every_registered_class_has_a_pipeline_position(self):
        """PIPELINE_ORDER and the decorators must describe the same set, in both
        directions: registration rejects an unlisted class, and this catches a listed
        name that nothing registers (a typo, or a class that was renamed)."""
        for backend in (Backend.GPU, Backend.CPU):
            with self.subTest(backend=backend.value):
                registered = {entry.name for entry in registry.entries(backend)}
                listed = set(registry.PIPELINE_ORDER[backend])
                self.assertEqual(listed, registered)

    def test_no_registered_class_hides_parameters_behind_kwargs(self):
        """**kwargs would make signature-derived validation accept anything."""
        for entry in registry.entries():
            with self.subTest(entry=entry.name):
                if registry._has_var_keyword(entry.cls):
                    self.assertIsNotNone(
                        entry.forwards_to,
                        f"{entry.name} takes **kwargs without declaring forwards_to",
                    )

    def test_gpu_transforms_expose_the_kornia_probability_parameters(self):
        for entry in registry.entries(Backend.GPU):
            accepted = registry.accepted_params(entry)
            with self.subTest(entry=entry.name):
                self.assertIn("p", accepted)
                self.assertIn("p_batch", accepted)

    def test_probability_is_never_a_parameter_name(self):
        """It was renamed to `p`; a survivor would mean a half-done migration."""
        for entry in registry.entries():
            with self.subTest(entry=entry.name):
                self.assertNotIn("probability", registry.accepted_params(entry))

    def test_legacy_config_keys_do_not_resolve(self):
        """The hard break, on the names that actually appear in the old configs."""
        for legacy in (
            "ScharrTransform",
            "UnsharpMaskTransform",
            "RandomConvTransform",
            "SynthSeg",
            "AffineTransform",
            "FlipTransform",
            "RandomPALETTETransform",
            "GammaTransform_invert",
            "ImageContrastGPUTransform",
            "PaletteSynthesisTransform",
        ):
            with self.subTest(legacy=legacy), self.assertRaises(registry.UnknownAugmentationError):
                registry.get(legacy, Backend.GPU)


class TestEveryEntryIsConstructible(unittest.TestCase):
    """The registry may not advertise an augmentation that cannot be built."""

    def test_constructible_with_declared_defaults(self):
        for entry in registry.entries():
            with self.subTest(entry=f"{entry.backend.value}.{entry.name}"):
                if entry.external_asset:
                    from unit_tests.helpers import domain_bank_missing

                    reason = domain_bank_missing(entry)
                    if reason:
                        self.skipTest(reason)
                required = registry.required_params(entry)
                if required:
                    # Legitimate: a few third-party CPU transforms take mandatory
                    # arguments that only a config or the trainer can supply.
                    self.assertTrue(
                        entry.backend is Backend.CPU or entry.context_params,
                        f"{entry.name} requires {sorted(required)} but nothing supplies them",
                    )
                    continue
                entry.cls(**dict(entry.smoke_kwargs))
