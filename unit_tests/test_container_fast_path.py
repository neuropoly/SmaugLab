"""The container short-circuits, checked against the kornia code they stand in for.

`smauglab.transforms.gpu.base` skips three things kornia does per module: the
leaf `forward` chain (`_apply_leaf`), the mask pass for transforms that cannot
touch a mask (`_leaves_masks_untouched`), and the per-module identity matrix.
Each is only valid because the step it replaces is an identity under stated
conditions -- which is a claim about *kornia's* code, and kornia is a moving
dependency (`tests.yml` runs a compat matrix over several versions).

So the tests here do not assert expected values. They run both routes and assert
the outputs match, which keeps the claim checked against whichever kornia is
installed.
"""

from __future__ import annotations

import inspect

import torch
from kornia.augmentation.container.ops import InputSequentialOps, MaskSequentialOps
from kornia.augmentation.container.params import ParamItem
from kornia.constants import DataKey

from smauglab import registry
from smauglab.registry import Backend
from smauglab.transforms.gpu.base import (
    ImageOnlyTransform,
    MaskSequentialOpsCustom,
    SegmentationRef,
    _apply_leaf,
    _can_apply_directly,
    _leaves_masks_untouched,
    autocast_active,
)
from unit_tests.helpers import SmaugLabTestCase, domain_bank_missing

SHAPE = (3, 1, 12, 14, 16)
EXTRA_KWARGS = {"RandomDomainTransferGPU": {"any_source": True}}


def build(entry):
    kwargs = dict(entry.smoke_kwargs)
    kwargs.update(EXTRA_KWARGS.get(entry.name, {}))
    if "p" in inspect.signature(entry.cls.__init__).parameters:
        kwargs["p"] = 1.0
    return entry.cls(**kwargs)


def fixtures():
    image = torch.rand(*SHAPE)
    seg = torch.zeros(*SHAPE)
    seg[:, :, 2:6, 2:6, 2:6] = 1
    seg[:, :, 6:10, 5:10, 5:10] = 2
    return image, seg


def params_for(module, image, seg, batch_prob):
    """A parameter dict as `AugmentationSequential` would have assembled it.

    `forward_parameters` draws the selection itself; here the selection is the
    thing under test, so the two halves of it are done by hand in the same order.
    Sizing `generate_parameters` to the selected rows is not cosmetic -- that is
    what kornia does, and the geometric transforms index their per-row parameters
    against exactly that count.
    """
    module.set_rng_device_and_dtype(image.device, torch.float32)
    selected = int((batch_prob > 0.5).sum())
    params = {}
    if getattr(module, "_param_generator", None) is not None:
        drawn = module.generate_parameters(torch.Size((selected, *image.shape[1:])))
        params = dict(drawn or {})
    params["batch_prob"] = batch_prob
    params["forward_input_shape"] = torch.tensor(image.shape, dtype=torch.long)
    params["seg"] = SegmentationRef(seg, batch_prob)
    return params


# Every row selected, none selected, and the partial case that exercises the
# `index_put` branch on both routes.
#: `p_batch != 1` is the case `_leaf_parameters` must hand back to kornia.
PROBABILITY_SETTINGS = (
    {"p": 1.0},
    {"p": 0.0},
    {"p": 0.35},
    {"p": 0.35, "same_on_batch": True},
    {"p": 0.5, "p_batch": 0.5},
    {"p": 0.5, "p_batch": 0.0},
)

SELECTIONS = {
    "all": torch.tensor([1.0, 1.0, 1.0]),
    "none": torch.tensor([0.0, 0.0, 0.0]),
    "partial": torch.tensor([1.0, 0.0, 1.0]),
}


class TestLeafFastPathMatchesKornia(SmaugLabTestCase):
    def test_every_registered_transform_agrees_on_both_routes(self):
        image, seg = fixtures()
        for entry in sorted(registry.entries(Backend.GPU), key=lambda e: e.name):
            reason = domain_bank_missing(entry)
            if reason:
                continue
            for label, batch_prob in SELECTIONS.items():
                with self.subTest(transform=entry.name, selection=label):
                    torch.manual_seed(4321)
                    direct_module = build(entry)
                    params = params_for(direct_module, image, seg, batch_prob)
                    self.assertTrue(_can_apply_directly(direct_module, image, None))
                    torch.manual_seed(99)
                    direct = _apply_leaf(direct_module, image.clone(), params)

                    torch.manual_seed(4321)
                    kornia_module = build(entry)
                    kornia_params = params_for(kornia_module, image, seg, batch_prob)
                    torch.manual_seed(99)
                    through_kornia = InputSequentialOps.transform(image.clone(), kornia_module, ParamItem(entry.name, kornia_params))

                    self.assertEqual(direct.shape, through_kornia.shape)
                    self.assertTrue(
                        torch.allclose(direct, through_kornia, atol=1e-5, equal_nan=True),
                        f"{entry.name} ({label}) differs between the fast path and kornia",
                    )

    def test_the_transform_matrix_is_the_one_kornia_would_have_stored(self):
        image, seg = fixtures()
        for entry in sorted(registry.entries(Backend.GPU), key=lambda e: e.name):
            if domain_bank_missing(entry):
                continue
            with self.subTest(transform=entry.name):
                torch.manual_seed(4321)
                module = build(entry)
                params = params_for(module, image, seg, SELECTIONS["partial"])
                torch.manual_seed(99)
                _apply_leaf(module, image.clone(), params)
                fast_matrix = module.transform_matrix

                torch.manual_seed(4321)
                other = build(entry)
                other_params = params_for(other, image, seg, SELECTIONS["partial"])
                torch.manual_seed(99)
                InputSequentialOps.transform(image.clone(), other, ParamItem(entry.name, other_params))

                self.assertTrue(torch.allclose(fast_matrix, other.transform_matrix, atol=1e-6))


class TestFastPathIsDeclined(SmaugLabTestCase):
    """The guard, on each condition it exists for."""

    def test_a_four_dimensional_volume_falls_through(self):
        module = registry.get("RandomBrightnessGPU", Backend.GPU).cls(p=1.0)
        self.assertFalse(_can_apply_directly(module, torch.rand(1, 8, 8, 8), None))

    def test_extra_args_fall_through(self):
        module = registry.get("RandomBrightnessGPU", Backend.GPU).cls(p=1.0)
        self.assertFalse(_can_apply_directly(module, torch.rand(*SHAPE), {"resample": "nearest"}))

    def test_a_foreign_module_falls_through(self):
        self.assertFalse(_can_apply_directly(torch.nn.Identity(), torch.rand(*SHAPE), None))

    def test_an_autocast_region_falls_through(self):
        module = registry.get("RandomBrightnessGPU", Backend.GPU).cls(p=1.0)
        self.assertFalse(autocast_active())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(autocast_active())
            self.assertFalse(_can_apply_directly(module, torch.rand(*SHAPE), None))


class TestMaskShortCircuit(SmaugLabTestCase):
    """Image-only transforms skip the mask pass; that must be what kornia returns."""

    def test_it_matches_kornia_for_every_image_only_transform(self):
        image, seg = fixtures()
        for entry in sorted(registry.entries(Backend.GPU), key=lambda e: e.name):
            if domain_bank_missing(entry):
                continue
            for label, batch_prob in SELECTIONS.items():
                torch.manual_seed(4321)
                module = build(entry)
                if not _leaves_masks_untouched(module):
                    continue
                with self.subTest(transform=entry.name, selection=label):
                    params = params_for(module, image, seg, batch_prob)
                    # The image pass is what sets `transform_matrix`, which the mask
                    # pass is handed; run it first, as the container does.
                    torch.manual_seed(99)
                    _apply_leaf(module, image.clone(), params)

                    item = ParamItem(entry.name, params)
                    ours = MaskSequentialOpsCustom.transform(seg.clone(), module, item)
                    flags = module.flags | {"data_keys": [DataKey.MASK]}
                    theirs = module.transform_masks(seg.clone(), params=params, flags=flags, transform=module.transform_matrix)
                    self.assertTrue(torch.equal(ours, theirs), f"{entry.name} ({label}) mask pass diverged")

    def test_a_geometric_transform_is_not_short_circuited(self):
        """The ones that do move a mask have to keep going through kornia."""
        for name in ("RandomFlipTransformGPU", "RandomAffineGPU", "RandomCropTransformGPU"):
            with self.subTest(transform=name):
                module = registry.get(name, Backend.GPU).cls()
                self.assertFalse(_leaves_masks_untouched(module))

    def test_an_override_disables_the_short_circuit(self):
        """A subclass that does implement a mask pass must not be skipped."""

        class MovesTheMask(ImageOnlyTransform):
            def apply_transform_mask(self, input, params, flags, transform=None):
                return input * 0

        self.assertFalse(_leaves_masks_untouched(MovesTheMask()))

    def test_the_short_circuit_is_the_identity_on_the_mask(self):
        """Not merely equal -- the same object, so nothing was allocated for it."""
        module = registry.get("RandomBrightnessGPU", Backend.GPU).cls(p=1.0)
        image, seg = fixtures()
        params = params_for(module, image, seg, SELECTIONS["all"])
        out = MaskSequentialOpsCustom.transform(seg, module, ParamItem("RandomBrightnessGPU", params))
        self.assertIs(out, seg)

    def test_the_custom_ops_still_extend_kornia(self):
        """Anything this package does not recognise must keep kornia's behaviour."""
        self.assertTrue(issubclass(MaskSequentialOpsCustom, MaskSequentialOps))


class TestLeafParameterDraw(SmaugLabTestCase):
    """`_leaf_parameters` against kornia's `forward_parameters`, draw for draw.

    The fast version skips an allocation, a sum and a broadcast multiply that
    `p_batch == 1` makes constant. None of those consume randomness, so under the
    same seed the two must produce the same dictionary -- not merely the same
    distribution.
    """

    def test_it_matches_kornia_for_every_probability_setting(self):
        from smauglab.transforms.gpu.base import _leaf_parameters

        shape = torch.Size(SHAPE)
        for name in ("RandomBrightnessGPU", "RandomAffineGPU", "RandomFlipTransformGPU", "RandomCropTransformGPU"):
            cls = registry.get(name, Backend.GPU).cls
            for settings in PROBABILITY_SETTINGS:
                for seed in (0, 7, 123):
                    with self.subTest(transform=name, settings=str(settings), seed=seed):
                        torch.manual_seed(seed)
                        ours = _leaf_parameters(cls(**settings), shape)
                        torch.manual_seed(seed)
                        theirs = cls(**settings).forward_parameters(shape)

                        self.assertEqual(set(ours), set(theirs))
                        for key in theirs:
                            self.assertTrue(
                                torch.equal(ours[key], theirs[key]),
                                f"{name} {settings} seed {seed}: `{key}` differs",
                            )

    def test_the_pipeline_draw_matches_kornia_module_for_module(self):
        """The container override, against the loop it replaces."""
        from kornia.augmentation.container.image import ImageSequential

        from smauglab.transforms.gpu.base import AugmentationSequentialCustom

        def build_pipeline():
            return AugmentationSequentialCustom(
                registry.get("RandomBrightnessGPU", Backend.GPU).cls(p=0.4),
                registry.get("RandomFlipTransformGPU", Backend.GPU).cls(p=0.6),
                registry.get("RandomGammaGPU", Backend.GPU).cls(p=1.0),
                data_keys=["input", "mask"],
            )

        for seed in (0, 11, 202):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                ours = build_pipeline().forward_parameters(torch.Size(SHAPE))
                torch.manual_seed(seed)
                theirs = ImageSequential.forward_parameters(build_pipeline(), torch.Size(SHAPE))

                self.assertEqual([p.name for p in ours], [p.name for p in theirs])
                for mine, kornias in zip(ours, theirs):
                    self.assertEqual(set(mine.data), set(kornias.data))
                    for key in kornias.data:
                        self.assertTrue(torch.equal(mine.data[key], kornias.data[key]), f"`{key}` differs")


class TestFallbacks(SmaugLabTestCase):
    """The escape hatches, exercised rather than assumed."""

    def test_the_pipeline_still_runs_without_kornias_private_helpers(self):
        """`FAST_PARAMETER_DRAW` off means kornia's own parameter loop, and it must work."""
        from smauglab.transforms.gpu import base

        def build():
            return base.AugmentationSequentialCustom(
                registry.get("RandomBrightnessGPU", Backend.GPU).cls(p=0.5),
                registry.get("RandomFlipTransformGPU", Backend.GPU).cls(p=0.5),
                data_keys=["input", "mask"],
            )

        image, seg = fixtures()
        torch.manual_seed(17)
        with_fast = build()(image.clone(), seg.clone())

        original = base.FAST_PARAMETER_DRAW
        base.FAST_PARAMETER_DRAW = False
        try:
            torch.manual_seed(17)
            without = build()(image.clone(), seg.clone())
        finally:
            base.FAST_PARAMETER_DRAW = original

        for mine, theirs in zip(with_fast, without):
            self.assertTrue(torch.equal(mine, theirs))

    def test_deepcopying_the_reference_shares_the_volume(self):
        """`transform_list` runs a real `copy.deepcopy`; it must not copy the mask."""
        import copy

        _, seg = fixtures()
        ref = SegmentationRef(seg, torch.tensor([1.0, 1.0, 1.0]))
        clone = copy.deepcopy(ref)

        self.assertIsNot(clone, ref)
        self.assertIs(clone.tensor(), ref.tensor())

    def test_an_unknown_payload_is_rejected(self):
        from smauglab.transforms.gpu.base import segmentation_from

        with self.assertRaises(TypeError):
            segmentation_from({"seg": [1, 2, 3]})
