"""The GPU normalization schemes, and the choice the builder makes between them.

`test_transforms_gpu.py` already runs every registered transform over a tiny volume and
checks it stays finite and leaves the mask alone. That is not enough here: a normalization
whose arithmetic is subtly wrong still produces a finite volume of the right shape, and
three of these schemes claim to reproduce an nnU-Net preprocessing scheme exactly. So this
file asserts the maths, the parity, and the two guards -- the constant patch and the
negligible divisor -- that decide when a scheme declines to run.

The bucket tests are here rather than in `test_builder.py` because the thing being tested
is semantic, not structural: with `num_transforms=1` a scheme's `p` stops being "how often
do I fire" and becomes "what share of the choice is mine", and nothing else in the
repository works that way.
"""

from __future__ import annotations

import unittest
from typing import ClassVar

import torch

from smauglab.config import OrderSource, PipelineMode
from smauglab.registry import AugType, Backend, names
from smauglab.transforms.build import build_gpu_pipeline
from smauglab.transforms.gpu.contrast import ZscoreNormalizationGPU
from smauglab.transforms.gpu.normalization import (
    MaxScaleNormalizationGPU,
    MeanScaleNormalizationGPU,
    MinMaxNormalizationGPU,
    PercentileNormalizationGPU,
    RobustZscoreNormalizationGPU,
)
from smauglab.transforms.gpu.transforms_list import RandomChooseXTransformsGPU
from unit_tests.helpers import SmaugLabTestCase

#: Every scheme in the group, at defaults. Built from the registry rather than hand-listed
#: so a sixth one cannot be added without the guard tests below seeing it.
SCHEMES = (
    MinMaxNormalizationGPU,
    PercentileNormalizationGPU,
    RobustZscoreNormalizationGPU,
    MaxScaleNormalizationGPU,
    MeanScaleNormalizationGPU,
)


def apply(transform, volume, seg=None):
    """Call a transform the way its pipeline would, without the kornia container."""
    params = {} if seg is None else {"seg": seg}
    return transform.apply_transform(volume.clone(), params, transform.flags)


class TestTheGroupIsComplete(unittest.TestCase):
    def test_every_registered_normalization_is_covered(self):
        """A new scheme joins SCHEMES, or this says so."""
        registered = set(names(Backend.GPU, group=AugType.NORM))
        covered = {cls.__name__ for cls in SCHEMES} | {ZscoreNormalizationGPU.__name__}
        self.assertEqual(registered, covered)


class TestEachSchemeDoesWhatItsNameSays(SmaugLabTestCase):
    """One assertion per scheme on the property the class name claims."""

    def volume(self):
        # Offset and scaled away from zero mean and unit variance, so a scheme that
        # silently did nothing would not pass by accident.
        return torch.randn(2, 1, 12, 12, 12) * 3.0 + 5.0

    def test_minmax_lands_on_its_output_range(self):
        for out_range in ((0.0, 1.0), (-1.0, 1.0)):
            with self.subTest(out_range=out_range):
                out = apply(MinMaxNormalizationGPU(out_range=out_range, p=1.0), self.volume())
                self.assertAlmostEqual(float(out.amin()), out_range[0], places=5)
                self.assertAlmostEqual(float(out.amax()), out_range[1], places=5)

    def test_percentile_standardizes_what_it_clipped(self):
        out = apply(PercentileNormalizationGPU(p=1.0), self.volume())
        self.assertAlmostEqual(float(out.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out.std(unbiased=False)), 1.0, places=3)

    def test_percentile_clips_to_the_window_it_was_given(self):
        """The clip is the scheme, not a detail: a wider window is a weaker transform."""
        volume = self.volume()
        narrow = apply(PercentileNormalizationGPU(clip_range=[0.0, 6.0], mean=3.0, std=1.0, p=1.0), volume)
        self.assertAlmostEqual(float(narrow.amin()), -3.0, places=5)
        self.assertAlmostEqual(float(narrow.amax()), 3.0, places=5)

    def test_robust_zscore_centres_the_median(self):
        for spread in RobustZscoreNormalizationGPU.SPREADS:
            with self.subTest(spread=spread):
                out = apply(RobustZscoreNormalizationGPU(spread=spread, p=1.0), self.volume())
                # Per sample: each one is centred on its own median, and the median of two
                # pooled distributions is not the median of either. Compared against
                # `torch.quantile`, not `torch.median`: on an even number of voxels the
                # transform interpolates between the two middle values, as quantile does,
                # while `median` returns the lower of them.
                for sample in out:
                    self.assertAlmostEqual(float(torch.quantile(sample.flatten(), 0.5)), 0.0, places=5)
                    # Gaussian input, so the unit_variance rescaling should land near 1.
                    self.assertAlmostEqual(float(sample.std(unbiased=False)), 1.0, places=1)

    def test_robust_zscore_ignores_an_outlier_a_mean_would_chase(self):
        """What "robust" buys, stated as the comparison that fails without it.

        One 1e4 voxel takes the sample's standard deviation from ~3 to ~240, so an ordinary
        z-score divides the anatomy by eighty times too much and flattens it. The IQR does
        not move, so the robust scheme still returns a unit-variance volume.
        """
        volume = self.volume()[:1]
        volume[0, 0, 0, 0, 0] = 1e4
        interior = (slice(None), slice(None), slice(1, None))

        robust = apply(RobustZscoreNormalizationGPU(p=1.0), volume)[interior]
        plain = apply(ZscoreNormalizationGPU(p=1.0), volume)[interior]
        self.assertAlmostEqual(float(robust.std(unbiased=False)), 1.0, places=1)
        self.assertLess(float(plain.std(unbiased=False)), 0.1)

    def test_max_scale_puts_the_maximum_at_one_and_leaves_zero_alone(self):
        volume = self.volume()
        volume[:, :, :2] = 0.0
        out = apply(MaxScaleNormalizationGPU(p=1.0), volume)
        self.assertAlmostEqual(float(out.abs().amax()), 1.0, places=5)
        self.assertEqual(float(out[:, :, :2].abs().amax()), 0.0)

    def test_max_scale_percentile_leaves_the_hot_voxels_above_one(self):
        out = apply(MaxScaleNormalizationGPU(percentile=0.99, p=1.0), self.volume())
        self.assertGreater(float(out.abs().amax()), 1.0)

    def test_mean_scale_puts_the_mean_at_one(self):
        out = apply(MeanScaleNormalizationGPU(stats_from="all", p=1.0), self.volume())
        self.assertAlmostEqual(float(out.mean()), 1.0, places=4)


class TestParityWithNnunetPreprocessing(SmaugLabTestCase):
    """Three schemes claim an nnU-Net equivalent; this is what makes the claim checkable.

    Exact equality wherever the arithmetic is fully determined -- the schemes were written
    against nnU-Net's own, including its floored denominators
    (`np.clip(image.max(), 1e-8)`, `max(std, 1e-8)`) and its population standard
    deviation. The one case with a tolerance is the per-sample mean and standard
    deviation, where numpy and torch sum 1000 float32 values in a different order; the
    gap there is one ulp and is not something a test should pin to zero.

    Not asserted: the masked variants. nnU-Net masks on `seg >= 0`, i.e. everything inside
    the body including unlabelled background, while `stats_from="seg"` here means the
    labelled anatomy only. The two are different populations on purpose.
    """

    #: Intensity properties of the shape nnU-Net's CT scheme reads from its dataset
    #: fingerprint. A transform is never handed these at training time, which is why
    #: `PercentileNormalizationGPU` takes them as parameters.
    PROPERTIES: ClassVar[dict] = {"mean": 38.5, "std": 115.0, "percentile_00_5": -150.0, "percentile_99_5": 260.0}

    def setUp(self):
        super().setUp()
        try:
            from nnunetv2.preprocessing.normalization import default_normalization_schemes
        except ImportError as missing:  # pragma: no cover -- an nnU-Net that moved the module
            self.skipTest(f"nnU-Net normalization schemes not importable: {missing}")
        self.schemes = default_normalization_schemes
        # Hounsfield-ish, so the CT scheme's percentile window actually bites.
        self.volume = torch.randn(1, 1, 10, 10, 10) * 120.0 + 40.0
        self.image = self.volume[0, 0].numpy().copy()

    def _compare(self, reference, transform, atol=0.0):
        got = apply(transform, self.volume)[0, 0].numpy()
        self.assertTrue(
            torch.allclose(torch.from_numpy(reference), torch.from_numpy(got), rtol=0.0, atol=atol),
            f"{type(transform).__name__} diverged from nnU-Net by up to {abs(reference - got).max()}",
        )

    def test_minmax_matches_rescale_to_01(self):
        reference = self.schemes.RescaleTo01Normalization(use_mask_for_norm=False, intensityproperties={}).run(self.image.copy())
        self._compare(reference, MinMaxNormalizationGPU(p=1.0))

    def test_percentile_matches_ct_normalization(self):
        reference = self.schemes.CTNormalization(use_mask_for_norm=False, intensityproperties=self.PROPERTIES).run(self.image.copy())
        self._compare(
            reference,
            PercentileNormalizationGPU(
                clip_range=[self.PROPERTIES["percentile_00_5"], self.PROPERTIES["percentile_99_5"]],
                mean=self.PROPERTIES["mean"],
                std=self.PROPERTIES["std"],
                p=1.0,
            ),
        )

    def test_percentile_without_a_window_matches_zscore(self):
        """The degenerate case of the CT scheme: clip to the full range, measure per sample."""
        reference = self.schemes.ZScoreNormalization(use_mask_for_norm=False, intensityproperties={}).run(self.image.copy())
        # Both reductions, so float32 accumulation order shows up; see the class docstring.
        self._compare(reference, PercentileNormalizationGPU(percentiles=[0.0, 1.0], p=1.0), atol=1e-6)


class TestTheGuardsDecline(SmaugLabTestCase):
    """Both guards, and the paired assertion that neither disables the scheme outright."""

    def test_a_constant_channel_is_returned_unchanged(self):
        """No scheme has anything to say about a volume with no variance.

        `ZscoreNormalizationGPU` used to return a constant -1.0 for a constant -2.709 CT-air
        patch, an O(1) value decided entirely by rounding; the same trap is waiting in every
        divisor here.
        """
        for cls in SCHEMES:
            for value in (-2.709, 0.0, 1.0):
                with self.subTest(scheme=cls.__name__, value=value):
                    volume = self.constant_volume(value)
                    out = apply(cls(p=1.0), volume)
                    self.assertTrue(bool(torch.equal(out, volume)), f"a constant {value} volume was rewritten")

    def test_a_normal_volume_is_still_normalized(self):
        """The guard against passing the test above by doing nothing at all."""
        volume = torch.randn(1, 1, 12, 12, 12) * 2.0 + 4.0
        for cls in SCHEMES:
            with self.subTest(scheme=cls.__name__):
                out = apply(cls(p=1.0), volume)
                self.assertFalse(bool(torch.equal(out, volume)), "the scheme left a perfectly normal volume alone")
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_mean_scaling_declines_on_already_zscored_data(self):
        """The reason `min_scale` exists.

        The incoming patch has been z-scored by preprocessing, so its mean over every voxel
        is ~0 and `x / mean` is rounding noise times a hundred million. The result is finite,
        so nothing downstream rejects it.
        """
        volume = torch.randn(1, 1, 12, 12, 12)
        volume = volume - volume.mean()
        out = apply(MeanScaleNormalizationGPU(stats_from="all", p=1.0), volume)
        self.assertTrue(bool(torch.equal(out, volume)), "divided by a mean of ~0")

    def test_mean_scaling_still_runs_where_the_mean_is_real(self):
        volume = torch.rand(1, 1, 12, 12, 12) + 1.0
        out = apply(MeanScaleNormalizationGPU(stats_from="all", p=1.0), volume)
        self.assertAlmostEqual(float(out.mean()), 1.0, places=4)

    def test_an_unknown_stats_source_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MinMaxNormalizationGPU(stats_from="foreground")

    def test_an_empty_out_range_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            MinMaxNormalizationGPU(out_range=(1.0, 1.0))


class TestMaskedStatistics(SmaugLabTestCase):
    """`stats_from` changes which voxels are measured, never which ones are written."""

    def _bordered(self):
        """Interior in [1, 2], border exactly zero -- what masked preprocessing leaves."""
        volume = torch.zeros(1, 1, 12, 12, 12)
        volume[:, :, 2:10, 2:10, 2:10] = torch.rand(1, 1, 8, 8, 8) + 1.0
        return volume

    def test_nonzero_statistics_ignore_the_zeroed_border(self):
        volume = self._bordered()
        interior = (slice(None), slice(None), slice(2, 10), slice(2, 10), slice(2, 10))

        out = apply(MinMaxNormalizationGPU(stats_from="nonzero", p=1.0), volume)
        self.assertAlmostEqual(float(out[interior].amin()), 0.0, places=5)
        self.assertAlmostEqual(float(out[interior].amax()), 1.0, places=5)
        # The border was still written, it just was not measured: 0 sits below the
        # interior minimum, so it maps outside the output range.
        self.assertLess(float(out.amin()), 0.0)

    def test_the_unmasked_scheme_measures_the_border_too(self):
        """The comparison that makes the test above about masking rather than luck."""
        out = apply(MinMaxNormalizationGPU(stats_from="all", p=1.0), self._bordered())
        self.assertAlmostEqual(float(out.amin()), 0.0, places=5)

    def test_seg_statistics_come_from_the_labelled_region(self):
        volume = torch.rand(1, 1, 12, 12, 12)
        volume[:, :, 4:8, 4:8, 4:8] += 4.0
        labelled = (slice(None), slice(None), slice(4, 8), slice(4, 8), slice(4, 8))

        out = apply(MinMaxNormalizationGPU(stats_from="seg", p=1.0), volume, self.tiny_seg_like(volume))
        self.assertAlmostEqual(float(out[labelled].amin()), 0.0, places=5)
        self.assertAlmostEqual(float(out[labelled].amax()), 1.0, places=5)

    def test_an_empty_mask_falls_back_to_the_whole_volume(self):
        """An unlabelled patch is not an error, and must not print a non-finite warning."""
        volume = torch.rand(1, 1, 12, 12, 12)
        out = apply(MinMaxNormalizationGPU(stats_from="seg", p=1.0), volume, torch.zeros_like(volume))
        self.assertAlmostEqual(float(out.amin()), 0.0, places=5)
        self.assertAlmostEqual(float(out.amax()), 1.0, places=5)

    def tiny_seg_like(self, volume):
        seg = torch.zeros_like(volume)
        seg[:, :, 4:8, 4:8, 4:8] = 1.0
        return seg


class TestTheBuilderMakesAChoiceNotASequence(SmaugLabTestCase):
    """Where the normalization stage lands, and when it is a bucket at all."""

    ONE: ClassVar[dict] = {"RandomFlipTransformGPU": {"p": 1.0}, "ZscoreNormalizationGPU": {"p": 1.0}}
    MANY: ClassVar[dict] = {
        "RandomFlipTransformGPU": {"p": 1.0},
        "ZscoreNormalizationGPU": {"p": 1.0},
        "MinMaxNormalizationGPU": {"p": 1.0},
        "MaxScaleNormalizationGPU": {"p": 1.0},
    }

    def test_a_single_normalization_is_not_wrapped(self):
        """Wrapping it would move it, and gate it behind a second probability."""
        for mode in PipelineMode:
            with self.subTest(mode=mode.value):
                built = build_gpu_pipeline(self.ONE, mode=mode)
                self.assertIsInstance(built[-1], ZscoreNormalizationGPU)

    def test_a_single_normalization_keeps_its_configured_position(self):
        """`pipeline.order: config` put it first; the normalization stage must not hoist it."""
        built = build_gpu_pipeline(self.ONE, order_source=OrderSource.CONFIG)
        self.assertEqual([type(t).__name__ for t in built], ["RandomFlipTransformGPU", "ZscoreNormalizationGPU"])

    def test_several_normalizations_become_one_trailing_choice(self):
        for mode in PipelineMode:
            with self.subTest(mode=mode.value):
                built = build_gpu_pipeline(self.MANY, mode=mode)
                bucket = built[-1]
                self.assertIsInstance(bucket, RandomChooseXTransformsGPU)
                self.assertEqual(bucket.provenance_label, "norm")
                self.assertEqual(bucket.num_transforms, 1)
                self.assertEqual(len(bucket.transforms_list), 3)
                self.assertTrue(bucket.random_order, "a fixed order would pin the choice to one scheme")
                others = [t for t in built[:-1] if isinstance(t, RandomChooseXTransformsGPU)]
                self.assertNotIn("norm", [t.provenance_label for t in others], "only one normalization stage")

    def test_no_normalization_emits_no_stage(self):
        built = build_gpu_pipeline({"RandomFlipTransformGPU": {"p": 1.0}})
        self.assertEqual([type(t).__name__ for t in built], ["RandomFlipTransformGPU"])

    def test_the_bucket_probability_comes_from_the_pipeline_options(self):
        built = build_gpu_pipeline(self.MANY, options={"norm_probability": 0.25})
        self.assertAlmostEqual(built[-1].p, 0.25)


class TestOneDrawPerBatchOrPerSample(SmaugLabTestCase):
    """Which `same_on_batch` decides, because it is not the bucket's own argument.

    `_normalization_stage` builds the bucket with `same_on_batch=False`, and then the
    container overwrites it: `pipeline.same_on_batch` defaults to True and kornia forces a
    non-None value onto every stage it holds. So the shipped default is one scheme for the
    whole batch, as it already is for the `ta` and `ge` buckets. This was written the wrong
    way round in a docstring first; the assertion is here so the next reader does not have
    to run a batch to find out.
    """

    SECTION: ClassVar[dict] = {
        "MinMaxNormalizationGPU": {"p": 1.0},
        "MaxScaleNormalizationGPU": {"p": 1.0},
    }

    def _bucket(self, *, same_on_batch):
        from smauglab.transforms.gpu.base import AugmentationSequentialCustom

        stages = build_gpu_pipeline(self.SECTION)
        container = AugmentationSequentialCustom(
            *stages,
            data_keys=["input", "mask"],
            same_on_batch=True if same_on_batch else None,
        )
        return container[-1]

    def test_the_bucket_is_built_per_sample(self):
        self.assertFalse(RandomChooseXTransformsGPU(transforms_list=[], num_transforms=1).same_on_batch)

    def test_the_container_default_makes_the_draw_batch_wide(self):
        self.assertTrue(self._bucket(same_on_batch=True).same_on_batch)

    def test_opting_out_leaves_the_draw_per_sample(self):
        self.assertFalse(self._bucket(same_on_batch=False).same_on_batch)

    def test_a_per_sample_bucket_can_give_two_samples_different_schemes(self):
        """The property the per-sample setting buys, on identical inputs."""
        torch.manual_seed(0)
        bucket = self._bucket(same_on_batch=False)
        volume = torch.rand(1, 1, 8, 8, 8) + 2.0
        batch = torch.cat([volume, volume], dim=0)

        differed = sum(not torch.allclose(*apply(bucket, batch), atol=1e-6) for _ in range(40))
        self.assertGreater(differed, 5, "the two samples always drew the same scheme")


class TestProbabilityIsTheSchemesWeight(SmaugLabTestCase):
    """Inside a `num_transforms=1` bucket, `p` stops meaning "how often" and starts
    meaning "what share".

    `RandomChooseXTransformsGPU` draws one child uniformly and then spends that child's own
    `p` (`transforms_list.py`), so `P(scheme) = p / n` and the leftover is no normalization
    at all. That is the identity arm of the random choice, and it exists without a
    registered no-op transform -- which is the whole reason there isn't one.
    """

    DRAWS = 400

    def test_a_scheme_at_half_probability_takes_half_its_share(self):
        torch.manual_seed(0)
        # Two schemes, one of them switched off: the bucket should normalize on ~half the
        # draws (minmax chosen, p=1.0) and leave the volume alone on the rest (the p=0
        # scheme chosen, nothing applied).
        bucket = RandomChooseXTransformsGPU(
            transforms_list=[MinMaxNormalizationGPU(p=1.0), MaxScaleNormalizationGPU(p=0.0)],
            num_transforms=1,
            p=1.0,
        )
        volume = torch.rand(1, 1, 8, 8, 8) + 2.0

        normalized = sum(not torch.equal(apply(bucket, volume), volume) for _ in range(self.DRAWS))
        expected = 0.5 * self.DRAWS
        self.assertGreater(normalized, 0.65 * expected, "the switched-off scheme is swallowing draws")
        self.assertLess(normalized, 1.35 * expected, "the switched-off scheme is not taking its share")

    def test_every_scheme_in_the_bucket_gets_chosen(self):
        """A uniform draw over the members, not a favourite.

        The three schemes are told apart by what they do to a [2, 3] volume: min-max puts
        the maximum at 1 and the minimum at 0, max-scale puts the maximum at 1 and leaves
        the minimum positive, and the identity leaves the maximum above 1.
        """
        torch.manual_seed(1)
        bucket = RandomChooseXTransformsGPU(
            transforms_list=[MinMaxNormalizationGPU(p=1.0), MaxScaleNormalizationGPU(p=1.0), MinMaxNormalizationGPU(p=0.0)],
            num_transforms=1,
            p=1.0,
        )
        volume = torch.rand(1, 1, 8, 8, 8) + 2.0

        seen = {"minmax": 0, "maxscale": 0, "identity": 0}
        for _ in range(self.DRAWS):
            out = apply(bucket, volume)
            if float(out.amax()) > 1.5:
                seen["identity"] += 1
            elif float(out.amin()) < 1e-6:
                seen["minmax"] += 1
            else:
                seen["maxscale"] += 1

        expected = self.DRAWS / 3
        for scheme, count in seen.items():
            with self.subTest(scheme=scheme):
                self.assertGreater(count, 0.65 * expected, f"{scheme} is under-represented: {seen}")
                self.assertLess(count, 1.35 * expected, f"{scheme} is over-represented: {seen}")


if __name__ == "__main__":
    unittest.main()
