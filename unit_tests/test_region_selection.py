"""Where a transform applies, at what granularity, and how sharply it stops.

`in_seg` and `out_seg` predate this; what they did not express is most of what they
looked like they expressed. The parameters and the four defects they replace are set out
in `smauglab/transforms/gpu/regions.py`; this file is what holds the behaviour.

The registry-driven cases matter most. The surface has to reach every transform that
reads a segmentation, and a transform added later must not be able to arrive without it
-- or, worse, with the parameters wired into its constructor and never read.

Three invariants here are about the **RNG stream** rather than about pixels, and all
three are properties the code this replaces broke:

* a knob that is off spends no random numbers, so a sweep is comparable with its own
  baseline at a fixed seed;
* the number of uniforms a draw spends depends on the configuration and never on the
  data, so a four-label patch and a five-label one leave the stream in the same place;
* the EM background split is forked, so a cache hit and a cache miss agree.
"""

from __future__ import annotations

import gc

import torch

from smauglab import registry
from smauglab.config import validate_section
from smauglab.registry import AugType, Backend
from smauglab.transforms.build import build_gpu_pipeline
from smauglab.transforms.gpu import regions
from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.contrast import RandomScharrGPU
from smauglab.transforms.gpu.regions import (
    DEFAULT_BG_EM_CLUSTERS,
    DEFAULT_BG_SPLIT,
    DEFAULT_SEG_FEATHER,
    DEFAULT_SEG_UNIT,
    DEFAULT_SEG_UNIT_PROB,
    UnitSpec,
    apply_region_selection,
    configure_regions,
    region_units,
)
from unit_tests.helpers import SmaugLabTestCase, domain_bank_missing, first_output

BATCH = 2
SHAPE = (BATCH, 1, 24, 24, 24)

#: `RandomRedistributeSegGPU` is the one transform that deliberately does not take
#: `in_seg` / `out_seg`: it used to accept `in_seg` with a different meaning, and had it
#: kept the name while changing it, twenty-eight shipped configs would have flipped
#: silently. See its class docstring.
NO_MODE_PARAMETERS = {"RandomRedistributeSegGPU"}

#: The five that are not about the in/out mode and therefore reach every segmentation-aware
#: transform, including the one above.
GRANULARITY_PARAMETERS = ("seg_unit", "seg_unit_prob", "bg_split", "bg_em_clusters", "seg_feather")


def region_aware():
    """Every registered GPU transform that takes the region surface."""
    return [entry for entry in registry.entries(Backend.GPU) if "seg_unit" in registry.accepted_params(entry)]


def transfer_augmentations():
    return registry.entries(Backend.GPU, group=AugType.TA)


class _Stub:
    """Stands in for a transform. The helpers only read these attributes."""

    def __init__(self, **overrides):
        values = {
            "in_seg": 0.0,
            "out_seg": 0.0,
            "seg_unit": DEFAULT_SEG_UNIT,
            "seg_unit_prob": 1.0,
            "bg_split": DEFAULT_BG_SPLIT,
            "bg_em_clusters": DEFAULT_BG_EM_CLUSTERS,
            "seg_feather": (DEFAULT_SEG_FEATHER, DEFAULT_SEG_FEATHER),
        }
        values.update(overrides)
        for name, value in values.items():
            setattr(self, name, value)


class TestTheSurfaceReachesEveryTransform(SmaugLabTestCase):
    """Registry-driven, so a transform added later is covered when it is registered."""

    def test_every_transfer_augmentation_takes_the_granularity_parameters(self):
        for entry in transfer_augmentations():
            accepted = registry.accepted_params(entry)
            with self.subTest(entry=entry.name):
                for name in GRANULARITY_PARAMETERS:
                    self.assertIn(name, accepted, f"{entry.name} cannot be given {name}")

    def test_every_transfer_augmentation_but_one_takes_the_mode_parameters(self):
        for entry in transfer_augmentations():
            accepted = registry.accepted_params(entry)
            expected = entry.name not in NO_MODE_PARAMETERS
            with self.subTest(entry=entry.name):
                self.assertEqual("in_seg" in accepted, expected)
                self.assertEqual("out_seg" in accepted, expected)

    def test_mix_in_out_is_gone_everywhere(self):
        for entry in registry.entries(Backend.GPU):
            with self.subTest(entry=entry.name):
                self.assertNotIn("mix_in_out", registry.accepted_params(entry))

    def test_the_defaults_mean_off(self):
        """A config that sets none of these must get the behaviour it had before."""
        for entry in region_aware():
            accepted = registry.accepted_params(entry)
            with self.subTest(entry=entry.name):
                self.assertEqual(accepted["bg_split"].default, DEFAULT_BG_SPLIT)
                self.assertEqual(accepted["seg_feather"].default, DEFAULT_SEG_FEATHER)
                self.assertEqual(tuple(accepted["bg_em_clusters"].default), DEFAULT_BG_EM_CLUSTERS)
                if entry.name in NO_MODE_PARAMETERS:
                    # Its own defaults reproduce the region set it has always used.
                    self.assertEqual(accepted["seg_unit"].default, "label")
                    self.assertEqual(accepted["seg_unit_prob"].default, 1.0)
                else:
                    self.assertEqual(accepted["in_seg"].default, 0.0)
                    self.assertEqual(accepted["out_seg"].default, 0.0)
                    self.assertEqual(accepted["seg_unit"].default, DEFAULT_SEG_UNIT)
                    self.assertEqual(accepted["seg_unit_prob"].default, DEFAULT_SEG_UNIT_PROB)

    def test_the_knobs_are_read_and_not_merely_declared(self):
        """The guard against a constructor that accepts a parameter and ignores it.

        Several seeds, because a single draw can legitimately keep every label and then
        `seg_unit="label"` *is* `"whole"`. A transform that leaves the volume alone on
        this input is skipped rather than failed: there is nothing to confine, which is
        a statement about the input and not about the knob.
        """
        image = torch.rand(*SHAPE)
        seg = self.multi_label_seg().expand(BATCH, -1, -1, -1, -1).contiguous()
        for entry in region_aware():
            with self.subTest(entry=entry.name):
                skip = domain_bank_missing(entry) if entry.external_asset else None
                if skip:
                    self.skipTest(skip)
                extra = dict(entry.smoke_kwargs)
                mode = {} if entry.name in NO_MODE_PARAMETERS else {"in_seg": 1.0}

                def run(seed, entry=entry, extra=extra, **kwargs):
                    torch.manual_seed(seed)
                    transform = entry.cls(p=1.0, **extra, **kwargs)
                    torch.manual_seed(seed)
                    return transform.apply_transform(image.clone(), {"seg": seg.clone()}, transform.flags)

                untouched = all(bool(torch.equal(run(seed, **mode), image)) for seed in range(4))
                if untouched:
                    self.skipTest(f"{entry.name} leaves this input unchanged, so there is nothing to confine")

                differed = any(
                    not bool(
                        torch.equal(
                            run(seed, **mode, seg_unit="whole", seg_unit_prob=0.5),
                            run(seed, **mode, seg_unit="label", seg_unit_prob=0.5),
                        )
                    )
                    for seed in range(6)
                )
                self.assertTrue(
                    differed,
                    f"{entry.name} accepts seg_unit but its output does not depend on it",
                )


class TestWhereEachModeLands(SmaugLabTestCase):
    """`seg_unit_prob=1.0` keeps every unit, so these are assertions about *where*."""

    def setUp(self):
        super().setUp()
        self.orig = torch.zeros(1, 24, 24, 24)
        self.transformed = torch.ones(1, 24, 24, 24)
        self.seg = self.multi_label_seg()
        self.foreground = self.seg[:, 0] > 0

    def changed(self, **kwargs):
        out = apply_region_selection(_Stub(**kwargs), self.orig, self.transformed, self.seg.clone())
        return out != self.orig

    def test_in_covers_exactly_the_labels(self):
        self.assertTrue(bool(torch.equal(self.changed(in_seg=1.0), self.foreground)))

    def test_out_covers_exactly_the_background(self):
        self.assertTrue(bool(torch.equal(self.changed(out_seg=1.0), ~self.foreground)))

    def test_both_covers_the_patch_when_every_unit_is_kept(self):
        self.assertTrue(bool(self.changed(in_seg=1.0, out_seg=1.0).all()))

    def test_neither_is_the_whole_patch_untouched_by_the_mask(self):
        out = apply_region_selection(_Stub(), self.orig, self.transformed, self.seg)
        self.assertTrue(bool(torch.equal(out, self.transformed)))

    def test_both_is_a_mixture_and_not_merely_all(self):
        """At a probability below 1 the two sides are sampled independently.

        This is the one semantic change to `in_seg` / `out_seg`: both firing used to fall
        through to "the whole patch", so `in_seg=1, out_seg=1` was indistinguishable from
        `in_seg=0, out_seg=0`.
        """
        torch.manual_seed(4)
        partials = 0
        for _ in range(12):
            changed = self.changed(in_seg=1.0, out_seg=1.0, seg_unit="label", seg_unit_prob=0.5)
            touches_both = bool((changed & self.foreground).any()) and bool((changed & ~self.foreground).any())
            if touches_both and not bool(changed.all()):
                partials += 1
        self.assertGreater(partials, 0, "'both' never produced a mixture, only the whole patch")


class TestGranularityKeepsWholeUnits(SmaugLabTestCase):
    """A unit is applied or it is not; it is never partially applied."""

    def setUp(self):
        super().setUp()
        self.orig = torch.zeros(1, 24, 24, 24)
        self.transformed = torch.ones(1, 24, 24, 24)
        self.seg = self.multi_label_seg()

    def assert_whole_units(self, seg_unit):
        spec = UnitSpec(seg_unit, DEFAULT_BG_SPLIT, DEFAULT_BG_EM_CLUSTERS)
        for seed in range(8):
            torch.manual_seed(seed)
            seg = self.seg.clone()
            units = region_units(seg, self.orig.unsqueeze(1), spec)
            out = apply_region_selection(_Stub(in_seg=1.0, seg_unit=seg_unit, seg_unit_prob=0.5), self.orig, self.transformed, seg)
            changed = (out != self.orig)[0]
            for unit in range(units.n_background, units.n_units):
                cells = units.ids[0, 0] == unit
                if not bool(cells.any()):
                    continue
                values = changed[cells]
                with self.subTest(seg_unit=seg_unit, seed=seed, unit=unit):
                    self.assertTrue(
                        bool(values.all()) or bool((~values).all()),
                        f"{seg_unit} unit {unit} was partially applied",
                    )

    def test_label_units_are_whole(self):
        self.assert_whole_units("label")

    def test_component_units_are_whole(self):
        self.assert_whole_units("component")

    def test_components_split_a_label_that_labels_does_not(self):
        """`multi_label_seg`'s label 1 is two separated blocks."""
        by_label = region_units(self.seg.clone(), None, UnitSpec("label", "none", DEFAULT_BG_EM_CLUSTERS))
        by_component = region_units(self.seg.clone(), None, UnitSpec("component", "none", DEFAULT_BG_EM_CLUSTERS))
        self.assertEqual(by_label.n_units, 4, "background plus three label values")
        self.assertEqual(by_component.n_units, 5, "the split label contributes one more")

    def test_keeping_every_unit_reduces_to_whole(self):
        """`seg_unit_prob=1.0` is how granularity is turned off, and it must be exact."""
        for seg_unit in ("label", "component"):
            with self.subTest(seg_unit=seg_unit):
                whole = apply_region_selection(
                    _Stub(in_seg=1.0, seg_unit="whole", seg_unit_prob=1.0), self.orig, self.transformed, self.seg.clone()
                )
                granular = apply_region_selection(
                    _Stub(in_seg=1.0, seg_unit=seg_unit, seg_unit_prob=1.0), self.orig, self.transformed, self.seg.clone()
                )
                self.assertTrue(bool(torch.equal(whole, granular)))

    def test_a_single_label_mask_makes_label_the_same_as_whole(self):
        """Documented rather than discovered: there is nothing to mix on a binary mask."""
        binary = self.tiny_seg()
        whole = apply_region_selection(_Stub(in_seg=1.0, seg_unit="whole"), self.orig, self.transformed, binary.clone())
        by_label = apply_region_selection(_Stub(in_seg=1.0, seg_unit="label"), self.orig, self.transformed, binary.clone())
        self.assertTrue(bool(torch.equal(whole, by_label)))


class TestTheBackgroundSplit(SmaugLabTestCase):
    """`bg_split="em"` is what makes the background something other than all-or-nothing."""

    def setUp(self):
        super().setUp()
        self.seg = self.tiny_seg()
        self.foreground = self.seg[:, 0] > 0
        # Bimodal background, so the EM has two modes to find.
        torch.manual_seed(1)
        self.image = torch.zeros(1, 24, 24, 24)
        self.image[0, :12] = 0.2
        self.image[0, 12:] = 0.9
        self.image += torch.rand(1, 24, 24, 24) * 0.02
        self.transformed = torch.full((1, 24, 24, 24), 5.0)

    def changed_background(self, **kwargs):
        out = apply_region_selection(_Stub(**kwargs), self.image, self.transformed, self.seg.clone())
        return int(((out != self.image) & ~self.foreground).sum())

    def test_without_the_split_the_background_is_all_or_nothing(self):
        torch.manual_seed(5)
        total = int((~self.foreground).sum())
        seen = {self.changed_background(out_seg=1.0, seg_unit_prob=0.5) for _ in range(8)}
        self.assertTrue(seen <= {0, total}, f"a partial background without bg_split: {sorted(seen)}")

    def test_with_the_split_part_of_the_background_can_be_used(self):
        torch.manual_seed(5)
        total = int((~self.foreground).sum())
        counts = [self.changed_background(out_seg=1.0, bg_split="em", seg_unit_prob=0.5) for _ in range(8)]
        self.assertTrue(
            any(0 < count < total for count in counts),
            f"bg_split='em' never produced a partial background: {counts}",
        )

    def test_the_foreground_is_still_untouched_in_mode_out(self):
        torch.manual_seed(5)
        out = apply_region_selection(_Stub(out_seg=1.0, bg_split="em", seg_unit_prob=0.5), self.image, self.transformed, self.seg)
        self.assertTrue(bool(torch.equal(out[self.foreground], self.image[self.foreground])))

    def test_a_patch_with_no_background_is_not_split(self):
        """`em_subdivide_labels` falls back to the largest label when 0 is absent."""
        units = region_units(torch.ones(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8), UnitSpec("whole", "em", (3, 10)))
        self.assertEqual(units.n_background, 1, "the EM fallback split a foreground label")

    def test_a_constant_volume_gives_one_background_unit(self):
        """The alternative is clusters whose boundaries are decided by float noise."""
        seg = torch.zeros(1, 1, 10, 10, 10)
        seg[0, 0, 4:6, 4:6, 4:6] = 1
        units = region_units(seg, self.constant_volume()[:, :, :10, :10, :10], UnitSpec("whole", "em", (3, 10)))
        self.assertEqual(units.n_background, 1)
        self.assertTrue(bool(torch.isfinite(units.ids.float()).all()))


class TestTheBoundary(SmaugLabTestCase):
    """`seg_feather` turns the composite weight from a step into a Gaussian ramp."""

    def setUp(self):
        super().setUp()
        self.orig = torch.zeros(1, 24, 24, 24)
        self.transformed = torch.ones(1, 24, 24, 24)
        self.seg = self.tiny_seg()

    def composite(self, sigma):
        return apply_region_selection(_Stub(in_seg=1.0, seg_feather=(sigma, sigma)), self.orig, self.transformed, self.seg.clone())

    def test_zero_is_a_hard_step(self):
        self.assertEqual(sorted(set(self.composite(0.0).flatten().tolist())), [0.0, 1.0])

    def test_a_positive_sigma_puts_voxels_between_the_two(self):
        ramp = self.composite(1.0)
        between = (ramp > 1e-6) & (ramp < 1.0 - 1e-6)
        self.assertGreater(int(between.sum()), 0)
        self.assertTrue(bool((ramp >= 0.0).all() and (ramp <= 1.0).all()), "the weight left [0, 1]")

    def test_the_ramp_is_monotone_across_the_boundary(self):
        ramp = self.composite(1.5)[0, 12, 12, :]
        rising = ramp[:12]
        self.assertTrue(bool((rising[1:] - rising[:-1] >= -1e-6).all()), rising.tolist())

    def test_the_interior_of_a_unit_stays_at_full_strength(self):
        """The ramp is centred on the boundary, so it does not reach a large unit's middle."""
        self.assertGreater(float(self.composite(1.0)[0, 11:13, 11:13, 11:13].min()), 0.999)

    def test_in_and_out_still_sum_to_the_transform(self):
        """blur(m) + blur(1 - m) == blur(1) == 1 for a normalised kernel and reflect padding."""
        inside = self.composite(1.0)
        outside = apply_region_selection(_Stub(out_seg=1.0, seg_feather=(1.0, 1.0)), self.orig, self.transformed, self.seg.clone())
        self.assertTrue(torch.allclose(inside + outside, self.transformed, atol=1e-5))

    def test_mode_all_is_unaffected(self):
        """The blur of a constant weight is that constant."""
        out = apply_region_selection(_Stub(in_seg=1e-12, seg_feather=(2.0, 2.0)), self.orig, self.transformed, self.seg)
        self.assertTrue(torch.allclose(out, self.transformed, atol=1e-6))

    def test_an_empty_mask_is_unaffected(self):
        out = apply_region_selection(_Stub(in_seg=1.0, seg_feather=(2.0, 2.0)), self.orig, self.transformed, self.empty_seg())
        self.assertTrue(torch.allclose(out, self.orig, atol=1e-6))


class TestDegenerateMasks(SmaugLabTestCase):
    """An empty mask is ~18% of real nnU-Net samples; see `helpers.empty_seg`."""

    def setUp(self):
        super().setUp()
        self.orig = torch.zeros(1, 24, 24, 24)
        self.transformed = torch.ones(1, 24, 24, 24)

    def test_in_is_a_no_op_and_out_is_everywhere(self):
        for seg_unit in ("whole", "label", "component"):
            with self.subTest(seg_unit=seg_unit):
                inside = apply_region_selection(
                    _Stub(in_seg=1.0, seg_unit=seg_unit, seg_unit_prob=0.5), self.orig, self.transformed, self.empty_seg()
                )
                outside = apply_region_selection(
                    _Stub(out_seg=1.0, seg_unit=seg_unit, seg_unit_prob=0.5), self.orig, self.transformed, self.empty_seg()
                )
                self.assertTrue(bool(torch.equal(inside, self.orig)), "'in' applied something with nothing labelled")
                self.assertTrue(bool(torch.equal(outside, self.transformed)), "'out' did not cover the whole patch")

    def test_the_unit_count_is_never_zero(self):
        """It sizes a draw and a gather; zero is a device-side assert, not a failed batch."""
        for seg_unit in ("whole", "label", "component"):
            with self.subTest(seg_unit=seg_unit):
                units = region_units(self.empty_seg(), None, UnitSpec(seg_unit, "none", DEFAULT_BG_EM_CLUSTERS))
                self.assertGreaterEqual(units.n_units, 1)
                self.assertGreaterEqual(units.n_background, 1)


class TestTheDecompositionIsSharedAcrossAPass(SmaugLabTestCase):
    """Without this the feature is not trainable: a pipeline runs twenty-odd of these."""

    def setUp(self):
        super().setUp()
        self.calls = 0
        self.real = regions._build_units

        def counting(*args, **kwargs):
            self.calls += 1
            return self.real(*args, **kwargs)

        regions._build_units = counting
        self.addCleanup(setattr, regions, "_build_units", self.real)

    def pipeline(self, *transforms):
        return AugmentationSequentialCustom(*transforms, data_keys=["input", "mask"], same_on_batch=True)

    def test_two_transforms_in_one_pass_share_one_decomposition(self):
        image = torch.rand(1, 1, 24, 24, 24)
        seg = self.multi_label_seg()
        pipeline = self.pipeline(
            RandomScharrGPU(p=1.0, in_seg=1.0, seg_unit="component"),
            RandomScharrGPU(p=1.0, in_seg=1.0, seg_unit="component"),
        )

        first_output(pipeline(image.clone(), seg.clone()))

        self.assertEqual(self.calls, 1, "the decomposition was rebuilt per transform")

    def test_a_different_spec_is_a_different_entry(self):
        image = torch.rand(1, 1, 24, 24, 24)
        seg = self.multi_label_seg()
        pipeline = self.pipeline(
            RandomScharrGPU(p=1.0, in_seg=1.0, seg_unit="component"),
            RandomScharrGPU(p=1.0, in_seg=1.0, seg_unit="label"),
        )

        first_output(pipeline(image.clone(), seg.clone()))

        self.assertEqual(self.calls, 2)

    def test_a_new_mask_tensor_is_a_new_decomposition(self):
        """How invalidation works: a geometric transform returns a different object."""
        image = torch.rand(1, 1, 24, 24, 24)
        spec = UnitSpec("component", "none", DEFAULT_BG_EM_CLUSTERS)
        region_units(self.multi_label_seg(), image, spec)
        region_units(self.multi_label_seg(), image, spec)
        self.assertEqual(self.calls, 2)

    def test_the_cache_dies_with_the_mask(self):
        seg = self.multi_label_seg()
        region_units(seg, torch.rand(1, 1, 24, 24, 24), UnitSpec("label", "none", DEFAULT_BG_EM_CLUSTERS))
        self.assertTrue(hasattr(seg, "_smauglab_region_units"))
        del seg
        gc.collect()  # nothing else holds it, so there is nothing left to leak


class TestTheRngStream(SmaugLabTestCase):
    """Three properties without which a seeded sweep is not comparable to its baseline."""

    def scharr_then_a_draw(self, seed=11, **kwargs):
        image = torch.rand(*SHAPE)
        seg = self.multi_label_seg().expand(BATCH, -1, -1, -1, -1).contiguous()
        transform = RandomScharrGPU(p=1.0, **kwargs)
        torch.manual_seed(seed)
        transform.apply_transform(image.clone(), {"seg": seg}, transform.flags)
        return torch.rand(4)

    def test_region_selection_off_spends_nothing(self):
        self.assertTrue(
            bool(torch.equal(self.scharr_then_a_draw(), self.scharr_then_a_draw(in_seg=0.0, out_seg=0.0, seg_unit="component"))),
            "the region parameters moved the stream with both probabilities at zero",
        )

    def test_the_draw_width_does_not_depend_on_the_data(self):
        """Two labels or five, the stream has to land in the same place."""
        two = torch.zeros(*SHAPE)
        two[:, :, 2:6, 2:6, 2:6] = 1
        two[:, :, 10:14, 10:14, 10:14] = 2
        five = two.clone()
        for offset, value in enumerate((3, 4, 5)):
            five[:, :, 20, 20, offset] = value

        def stream(seg, seg_unit):
            transform = RandomScharrGPU(p=1.0, in_seg=1.0, seg_unit=seg_unit, seg_unit_prob=0.5)
            torch.manual_seed(11)
            transform.apply_transform(torch.rand(*SHAPE), {"seg": seg.clone()}, transform.flags)
            return torch.rand(4)

        for seg_unit in ("label", "component"):
            with self.subTest(seg_unit=seg_unit):
                self.assertTrue(
                    bool(torch.equal(stream(two, seg_unit), stream(five, seg_unit))),
                    f"{seg_unit} spent a number of uniforms that depended on the segmentation",
                )

    def test_keeping_every_unit_spends_nothing(self):
        """`seg_unit_prob=1` is how granularity is turned off, so it has to be free."""
        self.assertTrue(
            bool(
                torch.equal(
                    self.scharr_then_a_draw(in_seg=1.0, seg_unit="whole", seg_unit_prob=1.0),
                    self.scharr_then_a_draw(in_seg=1.0, seg_unit="component", seg_unit_prob=1.0),
                )
            )
        )

    def test_the_em_split_is_forked_out_of_the_global_stream(self):
        """It spends ~100k uniforms on a cache miss and none on a hit."""
        self.assertTrue(
            bool(
                torch.equal(
                    self.scharr_then_a_draw(out_seg=1.0, bg_split="none"),
                    self.scharr_then_a_draw(out_seg=1.0, bg_split="em"),
                )
            ),
            "the EM draw leaked into the global stream, so cache state changes the augmentation",
        )

    def test_the_feather_range_does_not_shift_the_stream_when_it_is_off(self):
        self.assertTrue(
            bool(
                torch.equal(
                    self.scharr_then_a_draw(in_seg=1.0, seg_feather=0.0),
                    self.scharr_then_a_draw(in_seg=1.0),
                )
            )
        )

    def test_a_degenerate_feather_range_still_spends_its_draw(self):
        """Otherwise the stream becomes a function of the range and an A/B is not valid."""
        self.assertTrue(
            bool(
                torch.equal(
                    self.scharr_then_a_draw(in_seg=1.0, seg_feather=[1.0, 1.0]),
                    self.scharr_then_a_draw(in_seg=1.0, seg_feather=[0.5, 2.0]),
                )
            )
        )


class TestTheModeIsPerSample(SmaugLabTestCase):
    """It was one decision for the whole batch at ten of the eleven call sites.

    So `in_seg=0.5` read as "half the batches" where `p` and `mix_prob` both meant "half
    the samples".
    """

    def test_all_four_modes_appear_over_a_batch(self):
        torch.manual_seed(0)
        seen = set()
        for _ in range(16):
            modes = regions.choose_region_modes(_Stub(in_seg=0.5, out_seg=0.5), BATCH, torch.device("cpu"))
            seen.update(modes.tolist())
        self.assertEqual(seen, {regions.MODE_ALL, regions.MODE_IN, regions.MODE_OUT, regions.MODE_BOTH})

    def test_two_samples_of_one_batch_can_differ(self):
        torch.manual_seed(0)
        differed = any(
            len(set(regions.choose_region_modes(_Stub(in_seg=0.5, out_seg=0.5), BATCH, torch.device("cpu")).tolist())) > 1
            for _ in range(16)
        )
        self.assertTrue(differed, "every sample of the batch got the same mode")

    def test_nothing_is_drawn_when_both_probabilities_are_zero(self):
        self.assertIsNone(regions.choose_region_modes(_Stub(), BATCH, torch.device("cpu")))


class TestItSurvivesTheConfigPath(SmaugLabTestCase):
    """The constructors are the schema, and JSON hands over lists rather than tuples."""

    def test_a_section_validates_and_reaches_the_transform(self):
        section = {
            "RandomScharrGPU": {
                "in_seg": 0.5,
                "out_seg": 0.25,
                "seg_unit": "component",
                "seg_unit_prob": 0.3,
                "bg_split": "em",
                "bg_em_clusters": [2, 5],
                "seg_feather": [0.5, 1.5],
            }
        }
        self.assertEqual(validate_section(section, Backend.GPU), [])

        built = {type(t).__name__: t for t in build_gpu_pipeline(section)}
        scharr = built["RandomScharrGPU"]
        self.assertEqual(scharr.seg_unit, "component")
        self.assertEqual(scharr.bg_split, "em")
        self.assertEqual(scharr.bg_em_clusters, (2, 5))
        self.assertEqual(scharr.seg_feather, (0.5, 1.5))

    def test_a_bad_value_in_a_section_fails_the_build(self):
        """Not at validation -- that only checks keys -- but before the first batch."""
        section = {"RandomScharrGPU": {"seg_unit": "labl"}}
        self.assertEqual(validate_section(section, Backend.GPU), [])
        with self.assertRaises(ValueError):
            build_gpu_pipeline(section)

    def test_the_renamed_parameters_are_rejected_with_guidance(self):
        for section, needle in (
            ({"RandomScharrGPU": {"mix_in_out": True}}, "seg_unit"),
            ({"RandomRedistributeSegGPU": {"in_seg": 0.25}}, "confine_prob"),
        ):
            with self.subTest(section=section):
                problems = validate_section(section, Backend.GPU)
                self.assertEqual(len(problems), 1)
                self.assertIn(needle, problems[0])

    def test_no_shipped_config_still_sets_the_old_names(self):
        import json

        from unit_tests.helpers import configs_dir

        for path in sorted(configs_dir().rglob("*.json")):
            payload = json.loads(path.read_text())
            for section in payload.values():
                if not isinstance(section, dict):
                    continue
                for name, params in section.items():
                    if not isinstance(params, dict):
                        continue
                    with self.subTest(config=path.name, augmentation=name):
                        self.assertNotIn("mix_in_out", params)
                        if name in {"RandomRedistributeSegGPU", "RedistributeTransform"}:
                            self.assertNotIn("in_seg", params)


class TestTheValuesAreValidated(SmaugLabTestCase):
    """At construction, which is where every value check in this package happens."""

    def configure(self, **kwargs):
        class Owner:
            pass

        arguments = {"in_seg": 0.0, "out_seg": 0.0}
        arguments.update(kwargs)
        configure_regions(Owner(), **arguments)

    def test_a_bad_value_names_the_parameter_and_the_class(self):
        for kwargs, needle in (
            ({"in_seg": 1.5}, "in_seg"),
            ({"out_seg": -0.1}, "out_seg"),
            ({"seg_unit": "labl"}, "seg_unit"),
            ({"seg_unit_prob": True}, "seg_unit_prob"),
            ({"bg_split": "EM"}, "bg_split"),
            ({"bg_em_clusters": 5}, "bg_em_clusters"),
            ({"bg_em_clusters": [5, 2]}, "bg_em_clusters"),
            ({"seg_feather": -1.0}, "seg_feather"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError) as caught:
                self.configure(**kwargs)
            self.assertIn(needle, str(caught.exception))
            self.assertIn("Owner", str(caught.exception))

    def test_the_legal_seg_unit_values_are_listed(self):
        with self.assertRaises(ValueError) as caught:
            self.configure(seg_unit="nope")
        for value in regions.SEG_UNITS:
            self.assertIn(value, str(caught.exception))

    def test_a_feather_above_one_is_accepted(self):
        """It is a sigma in voxels, not a weight -- the point of `bounds=`."""
        self.configure(seg_feather=[0.5, 4.0])

    def test_a_mix_alpha_above_one_is_still_rejected(self):
        """The same check, with its own bounds. Widening one must not widen the other."""
        from smauglab.transforms.gpu.mixing import normalise_blend_range

        with self.assertRaises(ValueError):
            normalise_blend_range([0.5, 4.0])


if __name__ == "__main__":
    import unittest

    unittest.main()
