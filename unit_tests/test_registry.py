"""The registry machinery, exercised against synthetic transform classes.

Deliberately not against the real augmentations: those get registered in a later
stage, and these tests are about the mechanism -- registration invariants,
signature-derived parameter validation, and the coverage matrix. Keeping them
synthetic means they stay fast and cannot break for reasons unrelated to the
registry itself.

`registry.clear()` empties the global table, so every test here builds the world
it needs and tears it down again.
"""

from __future__ import annotations

import inspect
import unittest

from smauglab import registry
from smauglab.registry import (
    AugEntry,
    AugId,
    AugType,
    Backend,
    RegistryError,
    UnknownAugmentationError,
)


class RegistryTestCase(unittest.TestCase):
    """Isolates each test from the global registry."""

    def setUp(self) -> None:
        # Empties the registry so these tests see only what they register, and puts
        # the real augmentations back afterwards. A bare clear() cannot be undone:
        # load_all() re-imports an already-imported module, so the decorators never
        # run again and every later test in the process would see an empty registry.
        context = registry.isolated()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


def make_transform(name: str, **params):
    """A throwaway class whose __init__ has exactly the given parameters."""
    defaults = ", ".join(f"{k}={v!r}" for k, v in params.items())
    namespace: dict = {}
    exec(  # noqa: S102 -- building a signature is the point of this helper
        f"def __init__(self, {defaults}): pass" if defaults else "def __init__(self): pass",
        namespace,
    )
    return type(name, (), {"__init__": namespace["__init__"], "__doc__": f"{name} summary line.\n\nMore."})


class TestRegistration(RegistryTestCase):
    def test_decorator_registers_and_returns_the_class(self):
        cls = make_transform("RandomThingGPU", p=1.0)
        decorated = registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(cls)

        self.assertIs(decorated, cls, "the decorator must not replace the class")
        self.assertEqual(registry.get("RandomThingGPU", Backend.GPU).cls, cls)

    def test_summary_defaults_to_the_first_docstring_line(self):
        cls = make_transform("RandomThingGPU", p=1.0)
        registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(cls)
        self.assertEqual(registry.get("RandomThingGPU").summary, "RandomThingGPU summary line.")

    def test_name_must_match_the_class_name(self):
        """The config key IS the class name, so a mismatch has to be impossible."""
        with self.assertRaises(RegistryError) as caught:
            AugEntry(
                name="SomethingElse",
                cls=make_transform("RandomThingGPU"),
                backend=Backend.GPU,
                aug_id=AugId.SCHARR,
                group=AugType.TA,
            )
        self.assertIn("does not match class name", str(caught.exception))

    def test_duplicate_name_is_rejected(self):
        registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomThingGPU", p=1.0))
        with self.assertRaises(RegistryError) as caught:
            registry.register(aug_id=AugId.LAPLACE, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomThingGPU", p=1.0))
        self.assertIn("already registered", str(caught.exception))

    def test_registering_a_class_absent_from_pipeline_order_fails(self):
        """PIPELINE_ORDER is where a transform's pipeline position lives, so a class
        missing from it has nowhere to run. That is an error at import, not a silent
        append, which is what keeps the table from falling out of date."""
        order = {Backend.GPU: ("RandomListedGPU",)}
        with registry.isolated(order=order):
            registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomListedGPU", p=1.0))
            with self.assertRaises(registry.RegistryError) as caught:
                registry.register(aug_id=AugId.LAPLACE, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomUnlistedGPU", p=1.0))
        self.assertIn("PIPELINE_ORDER", str(caught.exception))

    def test_pipeline_order_positions_follow_the_table(self):
        order = {Backend.GPU: ("RandomSecondGPU", "RandomFirstGPU")}
        with registry.isolated(order=order):
            # Registered in the opposite order to the table, to prove the table wins.
            registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomFirstGPU", p=1.0))
            registry.register(aug_id=AugId.LAPLACE, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomSecondGPU", p=1.0))
            self.assertEqual(registry.names(Backend.GPU), ["RandomSecondGPU", "RandomFirstGPU"])

    def test_the_same_concept_can_exist_on_two_backends(self):
        registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomScharrGPU", p=1.0))
        registry.register(aug_id=AugId.SCHARR, backend=Backend.CPU, group=AugType.TA)(make_transform("ScharrConvTransform"))
        self.assertEqual(len(registry.entries()), 2)

    def test_var_keyword_without_forwards_to_is_rejected(self):
        """**kwargs would make signature-based validation accept anything."""

        class RandomSloppyGPU:
            def __init__(self, p: float = 1.0, **kwargs):
                pass

        with self.assertRaises(RegistryError) as caught:
            registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(RandomSloppyGPU)
        self.assertIn("**kwargs", str(caught.exception))

    def test_var_keyword_is_allowed_when_forwards_to_is_declared(self):
        target = make_transform("Generator", n_labels=3, blur=0.5)

        class RandomForwardingGPU:
            def __init__(self, p: float = 1.0, **kwargs):
                pass

        registry.register(aug_id=AugId.SYNTHSEG, backend=Backend.GPU, group=AugType.TA, forwards_to=target)(RandomForwardingGPU)
        self.assertIn("n_labels", registry.accepted_params(registry.get("RandomForwardingGPU")))


class TestLookup(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(
            make_transform("RandomScharrGPU", p=1.0, absolute=True)
        )
        registry.register(aug_id=AugId.FLIP, backend=Backend.GPU, group=AugType.GEO)(make_transform("RandomFlipTransformGPU", p=1.0))

    def test_entries_come_back_in_pipeline_order_not_registration_order(self):
        self.assertEqual(registry.names(Backend.GPU), ["RandomFlipTransformGPU", "RandomScharrGPU"])

    def test_filtering_by_group(self):
        self.assertEqual(registry.names(Backend.GPU, group=AugType.GEO), ["RandomFlipTransformGPU"])

    def test_unknown_name_suggests_a_close_match(self):
        with self.assertRaises(UnknownAugmentationError) as caught:
            registry.get("RandomScharGPU", Backend.GPU)
        self.assertIn("Did you mean: RandomScharrGPU?", str(caught.exception))

    def test_legacy_key_fails_but_points_at_its_replacement(self):
        """The hard break: old spellings must raise, not quietly work.

        difflib alone cannot bridge 'ScharrTransform' -> 'RandomScharrGPU'
        (too little shared prefix), so the stem fallback has to carry it.
        """
        with self.assertRaises(UnknownAugmentationError) as caught:
            registry.get("ScharrTransform", Backend.GPU)
        self.assertIn("Did you mean: RandomScharrGPU?", str(caught.exception))

    def test_renamed_hints_are_diagnostics_only(self):
        """Nothing in RENAMED_HINTS may be a working config key."""
        for stale in registry.RENAMED_HINTS:
            with self.subTest(key=stale), self.assertRaises(UnknownAugmentationError):
                registry.get(stale)


class TestAcceptedParams(RegistryTestCase):
    def test_params_come_from_the_constructor_signature(self):
        registry.register(aug_id=AugId.GAUSSIAN_NOISE, backend=Backend.GPU, group=AugType.GE)(
            make_transform("RandomGaussianNoiseGPU", p=1.0, mean=0.0, std=0.1)
        )
        entry = registry.get("RandomGaussianNoiseGPU")
        self.assertEqual(set(registry.accepted_params(entry)), {"p", "mean", "std"})

    def test_context_params_are_excluded(self):
        registry.register(
            aug_id=AugId.SPATIAL,
            backend=Backend.CPU,
            group=AugType.GEO,
            wrap_random=False,
            context_params=("rotation",),
        )(make_transform("SpatialTransform", rotation=None, p_rotation=0.2))
        entry = registry.get("SpatialTransform")
        self.assertNotIn("rotation", registry.accepted_params(entry))
        self.assertIn("p_rotation", registry.accepted_params(entry))

    def test_cpu_wrapped_transforms_accept_p_even_though_the_class_does_not(self):
        """batchgeneratorsv2 puts the probability on the RandomTransform wrapper."""
        registry.register(aug_id=AugId.SCHARR, backend=Backend.CPU, group=AugType.TA)(make_transform("ScharrConvTransform", absolute=True))
        entry = registry.get("ScharrConvTransform")
        self.assertIn("p", registry.accepted_params(entry))

    def test_unwrapped_cpu_transforms_do_not_get_a_synthetic_p(self):
        registry.register(aug_id=AugId.MIRROR, backend=Backend.CPU, group=AugType.GEO, wrap_random=False)(
            make_transform("MirrorTransform", allowed_axes=None)
        )
        self.assertNotIn("p", registry.accepted_params(registry.get("MirrorTransform")))

    def test_required_params_are_those_without_a_default(self):
        """RandomAffineGPU really does take a required `degrees` today."""

        class RandomAffineGPU:
            def __init__(self, degrees, p: float = 1.0):
                pass

        registry.register(aug_id=AugId.AFFINE, backend=Backend.GPU, group=AugType.GEO)(RandomAffineGPU)
        self.assertEqual(registry.required_params(registry.get("RandomAffineGPU")), {"degrees"})

    def test_unknown_parameter_message_suggests_p_for_probability(self):
        registry.register(aug_id=AugId.GAUSSIAN_NOISE, backend=Backend.GPU, group=AugType.GE)(
            make_transform("RandomGaussianNoiseGPU", p=1.0, std=0.1)
        )
        message = registry.unknown_parameter_message(registry.get("RandomGaussianNoiseGPU"), "probability")
        self.assertIn("'probability' -> p", message)
        self.assertIn("Accepted: p, std", message)

    def test_unknown_parameter_message_flags_context_params_specifically(self):
        registry.register(
            aug_id=AugId.SPATIAL,
            backend=Backend.CPU,
            group=AugType.GEO,
            wrap_random=False,
            context_params=("rotation",),
        )(make_transform("SpatialTransform", rotation=None))
        message = registry.unknown_parameter_message(registry.get("SpatialTransform"), "rotation")
        self.assertIn("supplied by the trainer", message)


class TestMatrix(RegistryTestCase):
    def test_every_aug_id_gets_a_row_even_with_no_implementations(self):
        self.assertEqual(set(registry.matrix()), set(AugId))

    def test_a_concept_joins_its_backends_into_one_row(self):
        registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomScharrGPU", p=1.0))
        registry.register(aug_id=AugId.SCHARR, backend=Backend.CPU, group=AugType.TA)(make_transform("ScharrConvTransform"))
        row = registry.matrix()[AugId.SCHARR]
        self.assertEqual(row[Backend.GPU].name, "RandomScharrGPU")
        self.assertEqual(row[Backend.CPU].name, "ScharrConvTransform")
        self.assertIsNone(row[Backend.MONAI], "no MONAI implementations exist yet")

    def test_markdown_render_marks_the_gap(self):
        registry.register(aug_id=AugId.SCHARR, backend=Backend.GPU, group=AugType.TA)(make_transform("RandomScharrGPU", p=1.0))
        rendered = registry.render_matrix("md")
        self.assertIn("| scharr | TA | `RandomScharrGPU` | — | — |", rendered)

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            registry.render_matrix("yaml")


class TestModuleHygiene(unittest.TestCase):
    def test_registry_does_not_import_torch(self):
        """`smauglab list` must be able to answer without paying for torch."""
        source = inspect.getsource(registry)
        for banned in ("import torch", "import kornia", "from smauglab.transforms"):
            with self.subTest(banned=banned):
                self.assertNotIn(f"\n{banned}", source, f"registry.py must not import {banned!r} at module scope")


if __name__ == "__main__":
    unittest.main()
