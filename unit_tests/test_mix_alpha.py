"""`mix_alpha`: the range a transfer augmentation draws its blend weight from.

`mix_prob` decides *whether* a sample is blended back towards the image the transform
was handed; `mix_alpha` decides *how far*. Before this existed the weight came from a
hardcoded `U(0, 1)`, so the only expressible blends were "none" and "anywhere between
the identity and full strength" -- and eleven of the seventeen GPU transfer
augmentations had no blend at all.

alpha is the weight on the **original**: 0.0 is the transform's output, 1.0 is the input
untouched. Counter-intuitive in a config, and kept that way on purpose, because it is the
direction the three open-coded copies used and the default range therefore reproduces
their draw exactly. `test_mix_prob.py` is what holds that.

The cases that iterate over the registry are the ones that matter most: a transfer
augmentation added later cannot quietly arrive without the knob, or with the knob wired
into its constructor but never read.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from smauglab import registry
from smauglab.config import validate_section
from smauglab.registry import AugType, Backend
from smauglab.transforms.build import build_gpu_pipeline
from smauglab.transforms.gpu.mixing import normalise_blend_range
from unit_tests.helpers import SmaugLabTestCase, domain_bank_missing

BATCH = 6
SHAPE = (BATCH, 1, 24, 24, 24)

#: Their closing foreground z-score runs *after* the blend and is unconditional, so
#: alpha=1 hands the z-score the input rather than the synthesis -- which is not the
#: input. Everything else in the group either has no restore or restores the input's own
#: statistics, where that is a no-op.
NOT_IDENTITY_AT_ALPHA_ONE = {"RandomPaletteGPU", "PaletteSynthesisGPU"}


def transfer_augmentations():
    """Every registered GPU transfer augmentation."""
    return registry.entries(Backend.GPU, group=AugType.TA)


def build(entry, **kwargs):
    extra = dict(entry.smoke_kwargs)
    extra.update(kwargs)
    return entry.cls(p=1.0, **extra)


class TestEveryTransferAugmentationHonoursTheKnob(SmaugLabTestCase):
    """Registry-driven, so a new TA is covered the moment it is registered."""

    def volume(self):
        return torch.rand(*SHAPE, dtype=torch.float32)

    def seg(self):
        seg = torch.zeros(*SHAPE, dtype=torch.float32)
        seg[:, :, 6:18, 6:18, 6:18] = 1.0
        seg[:, :, 2:6, 2:6, 2:6] = 2.0
        return seg

    def run_with(self, entry, seed=3, **kwargs):
        torch.manual_seed(seed)
        transform = build(entry, **kwargs)
        torch.manual_seed(seed)
        return transform.apply_transform(self.volume().clone(), {"seg": self.seg()}, transform.flags)

    def test_the_constructor_accepts_both_parameters(self):
        """The guard against a TA that forgot to declare them."""
        for entry in transfer_augmentations():
            accepted = registry.accepted_params(entry)
            with self.subTest(entry=entry.name):
                self.assertIn("mix_prob", accepted)
                self.assertIn("mix_alpha", accepted)

    def test_the_defaults_leave_existing_configs_alone(self):
        """`mix_prob=0` is off, and `mix_alpha=(0, 1)` is the draw this replaced."""
        for entry in transfer_augmentations():
            accepted = registry.accepted_params(entry)
            with self.subTest(entry=entry.name):
                self.assertEqual(accepted["mix_prob"].default, 0.0)
                self.assertEqual(tuple(accepted["mix_alpha"].default), (0.0, 1.0))

    def test_a_mid_range_alpha_changes_the_output(self):
        """The guard against a TA whose constructor takes the knob but never reads it."""
        for entry in transfer_augmentations():
            with self.subTest(entry=entry.name):
                if entry.external_asset and domain_bank_missing(entry):
                    self.skipTest(domain_bank_missing(entry))
                plain = self.run_with(entry, mix_prob=0.0)
                mixed = self.run_with(entry, mix_prob=1.0, mix_alpha=0.5)
                self.assertFalse(
                    bool(torch.equal(plain, mixed)),
                    f"{entry.name} accepts mix_alpha but its output does not depend on it",
                )

    def test_alpha_one_is_the_identity(self):
        """All the weight on the input means the transform did nothing."""
        for entry in transfer_augmentations():
            with self.subTest(entry=entry.name):
                if entry.name in NOT_IDENTITY_AT_ALPHA_ONE:
                    self.skipTest("an unconditional z-score runs after the blend")
                if entry.external_asset and domain_bank_missing(entry):
                    self.skipTest(domain_bank_missing(entry))
                volume = self.volume()
                torch.manual_seed(3)
                transform = build(entry, mix_prob=1.0, mix_alpha=1.0)
                torch.manual_seed(3)
                out = transform.apply_transform(volume.clone(), {"seg": self.seg()}, transform.flags)
                # `allclose`, not `equal`: a `retain_stats` restore of the input's own
                # mean and std is only the identity up to its own 1e-8 epsilon.
                self.assertTrue(
                    bool(torch.allclose(out, volume, atol=1e-5)),
                    f"{entry.name} at mix_alpha=1.0 is not the identity (max deviation {float((out - volume).abs().max()):.3g})",
                )

    def test_alpha_zero_is_the_unmixed_transform(self):
        """All the weight on the output means the blend did nothing.

        Asserted on `RandomScharrGPU` rather than over the registry, because the
        comparison is only sound where the blend's two draws are the *only* randomness
        in `apply_transform`. `RandomRedistributeSegGPU` keeps drawing per sample after
        the blend, so switching `mix_prob` on necessarily shifts its later samples --
        which is why the draws are hoisted out of its loop rather than left in it, but
        hoisting cannot make two extra draws free. `test_alpha_one_is_the_identity`
        above is the contract that does hold for every one of them.
        """
        from smauglab.transforms.gpu.contrast import RandomScharrGPU

        torch.manual_seed(0)
        image = torch.rand(*SHAPE)
        mixed = RandomScharrGPU(p=1.0, mix_prob=1.0, mix_alpha=0.0, retain_stats=False)
        plain = RandomScharrGPU(p=1.0, mix_prob=0.0, retain_stats=False)
        self.assertTrue(
            bool(
                torch.equal(
                    mixed.apply_transform(image.clone(), {}, mixed.flags),
                    plain.apply_transform(image.clone(), {}, plain.flags),
                )
            )
        )


class TestTheRangeIsRespected(SmaugLabTestCase):
    """Recovering the drawn alpha, as `test_mix_prob.py` does.

    `RandomScharrGPU` is the probe because its kernel is fixed and, with no segmentation
    passed, `_choose_region_mode` never fires -- so the mix draws are the only randomness
    in `apply_transform` and an exact comparison against an unmixed run is sound.
    """

    def recovered_alphas(self, **kwargs):
        from smauglab.transforms.gpu.contrast import RandomScharrGPU

        torch.manual_seed(0)
        image = torch.rand(*SHAPE)

        mixed = RandomScharrGPU(p=1.0, mix_prob=1.0, retain_stats=False, **kwargs)
        out = mixed.apply_transform(image.clone(), {}, mixed.flags)
        plain = RandomScharrGPU(p=1.0, mix_prob=0.0, retain_stats=False)
        reference = plain.apply_transform(image.clone(), {}, plain.flags)

        alphas = []
        for b in range(BATCH):
            denominator = image[b] - reference[b]
            usable = denominator.abs() > 1e-3
            alphas.append(float(((out[b] - reference[b])[usable] / denominator[usable]).median()))
        return alphas

    def test_a_narrow_range_bounds_every_draw(self):
        alphas = self.recovered_alphas(mix_alpha=[0.8, 0.9])
        for alpha in alphas:
            self.assertGreaterEqual(alpha, 0.8 - 1e-3, alphas)
            self.assertLessEqual(alpha, 0.9 + 1e-3, alphas)

    def test_two_disjoint_ranges_stay_disjoint(self):
        low = self.recovered_alphas(mix_alpha=[0.0, 0.1])
        high = self.recovered_alphas(mix_alpha=[0.9, 1.0])
        self.assertLess(max(low), min(high), f"{low} vs {high}")

    def test_a_scalar_gives_every_sample_the_same_alpha(self):
        alphas = [round(a, 4) for a in self.recovered_alphas(mix_alpha=0.35)]
        self.assertEqual(set(alphas), {0.35}, alphas)

    def test_the_default_range_still_spreads(self):
        """The replacement for the hardcoded U(0, 1) is not accidentally degenerate."""
        alphas = [round(a, 4) for a in self.recovered_alphas()]
        self.assertGreater(len(set(alphas)), 1, alphas)


class TestMixProbZeroSpendsNothing(SmaugLabTestCase):
    """Turning mixing off has to leave the RNG stream where it found it.

    Otherwise a `mix_alpha` sweep could not be compared against its own baseline at a
    fixed seed: the transforms downstream would all see a different stream. The two
    per-sample copies this replaced got that wrong -- they drew a gate per sample even at
    `mix_prob=0` -- and 15 shipped configs changed output when it was fixed.
    """

    def scharr_then_a_draw(self, **kwargs):
        from smauglab.transforms.gpu.contrast import RandomScharrGPU

        torch.manual_seed(5)
        image = torch.rand(*SHAPE)
        transform = RandomScharrGPU(p=1.0, **kwargs)
        torch.manual_seed(11)
        transform.apply_transform(image.clone(), {}, transform.flags)
        return torch.rand(4)

    def test_the_range_does_not_shift_the_stream_when_mixing_is_off(self):
        self.assertTrue(
            bool(torch.equal(self.scharr_then_a_draw(mix_prob=0.0, mix_alpha=[0.3, 0.3]), self.scharr_then_a_draw(mix_prob=0.0))),
            "mix_alpha moved the RNG stream with mix_prob at zero",
        )

    def test_the_range_does_not_shift_the_stream_when_mixing_is_on(self):
        """Two draws either way, so an A/B over ranges stays comparable."""
        self.assertTrue(
            bool(
                torch.equal(
                    self.scharr_then_a_draw(mix_prob=1.0, mix_alpha=[0.2, 0.2]),
                    self.scharr_then_a_draw(mix_prob=1.0, mix_alpha=[0.4, 0.9]),
                )
            ),
            "a degenerate range skipped the alpha draw",
        )


class TestItSurvivesTheConfigPath(SmaugLabTestCase):
    """A range reaches the transform from JSON as a `list`, not a tuple.

    The constructors are the config schema -- `registry.accepted_params` reads
    `inspect.signature` -- so a parameter is only really settable once it has been
    through `validate_section` and `build_gpu_pipeline`. Asserted on a section rather
    than a shipped config file so no new file has to be kept in step.
    """

    SECTION: ClassVar[dict] = {
        "RandomScharrGPU": {"p": 1.0, "mix_prob": 0.5, "mix_alpha": [0.2, 0.8]},
        "RandomSynthSegGPU": {"p": 0.0, "mix_alpha": 0.6},
        "RandomLog1pGPU": {"p": 1.0, "mix_prob": 1.0, "mix_alpha": [0.0, 0.25]},
    }

    def test_a_section_validates(self):
        self.assertEqual(validate_section(self.SECTION, Backend.GPU), [])

    def test_the_built_transforms_carry_the_parsed_range(self):
        built = build_gpu_pipeline(self.SECTION)
        by_name = {type(t).__name__: t for t in built}
        self.assertEqual(by_name["RandomScharrGPU"].mix_alpha, (0.2, 0.8))
        self.assertEqual(by_name["RandomSynthSegGPU"].mix_alpha, (0.6, 0.6))
        self.assertEqual(by_name["RandomLog1pGPU"].mix_alpha, (0.0, 0.25))

    def test_a_bad_range_in_a_section_fails_the_build(self):
        """Not at validation -- that only checks keys -- but before the first batch."""
        section = {"RandomScharrGPU": {"mix_alpha": [0.9, 0.1]}}
        self.assertEqual(validate_section(section, Backend.GPU), [])
        with self.assertRaises(ValueError):
            build_gpu_pipeline(section)


class TestTheRangeIsValidated(SmaugLabTestCase):
    """At construction, which is where every value check in this package happens."""

    def test_a_descending_range_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            normalise_blend_range([0.6, 0.4])
        self.assertIn("non-decreasing", str(caught.exception))

    def test_a_range_outside_the_unit_interval_is_rejected(self):
        for bad in ([-0.1, 0.5], [0.5, 1.5], 1.5, -0.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as caught:
                normalise_blend_range(bad)
            self.assertIn("[0, 1]", str(caught.exception))

    def test_a_malformed_range_is_rejected(self):
        # `True` is an int subtype, so without an explicit check it would quietly mean 1.0
        # -- a transform configured to do nothing at all.
        for bad in ([0.1, 0.2, 0.3], [0.1], "half", [None, 1.0], True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalise_blend_range(bad)

    def test_the_message_names_the_parameter_and_the_class(self):
        from smauglab.transforms.gpu.contrast import RandomScharrGPU

        with self.assertRaises(ValueError) as caught:
            RandomScharrGPU(mix_alpha=[0.6, 0.4])
        self.assertIn("mix_alpha", str(caught.exception))
        self.assertIn("RandomScharrGPU", str(caught.exception))

    def test_a_scalar_and_a_range_both_round_trip(self):
        self.assertEqual(normalise_blend_range(0.25), (0.25, 0.25))
        self.assertEqual(normalise_blend_range([0.25, 0.75]), (0.25, 0.75))
        self.assertEqual(normalise_blend_range((0, 1)), (0.0, 1.0))


if __name__ == "__main__":
    import unittest

    unittest.main()
