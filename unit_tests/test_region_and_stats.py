"""Region selection and per-sample statistics.

Three defects, each of which made a transform depend on something it should not:

* `_apply_region_mode` reduced the mask's class axis with `torch.argmax(mask, dim) > 0`.
  For the ordinary single-channel mask that is always False, so `in_seg` applied the
  transform nowhere and `out_seg` applied it everywhere -- both knobs silently
  inoperative. For a one-hot mask it dropped the first foreground class, because this
  repository encodes channel `c` as label `c + 1`.
* The elementwise function transforms normalised with a slab-wide `x.min()`/`x.max()`,
  so a volume's augmentation depended on which other volumes shared its batch.
* `RandomHistogramEqualizationGPU` wrote through an `input[:, c]` view, so its
  non-finite guard skipped over values already in the batch.
"""

import numpy as np
import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.contrast import (
    RandomHistogramEqualizationGPU,
    RandomScharrGPU,
    RandomSqrtGPU,
    _apply_region_mode,
    _foreground,
)
from unit_tests.helpers import SmaugLabTestCase, first_output


class TestForegroundReduction(SmaugLabTestCase):
    def test_a_single_channel_mask_is_not_collapsed_to_nothing(self):
        """argmax over a length-1 axis is always 0, so `> 0` was always False."""
        mask = torch.zeros(1, 1, 4, 4, 4)
        mask[0, 0, 1:3, 1:3, 1:3] = 1.0

        found = _foreground(mask, dim=1)

        self.assertEqual(int(found.sum()), 8, "the labelled block was not recognised as foreground")
        self.assertTrue(bool(found[0, 1, 1, 1]))
        self.assertFalse(bool(found[0, 0, 0, 0]))

    def test_the_old_argmax_reduction_really_did_collapse_it(self):
        """Control: what the previous expression produced on the same input."""
        mask = torch.zeros(1, 1, 4, 4, 4)
        mask[0, 0, 1:3, 1:3, 1:3] = 1.0
        self.assertEqual(int((torch.argmax(mask, dim=1) > 0).sum()), 0)

    def test_the_first_one_hot_channel_counts_as_foreground(self):
        """Channel 0 encodes label 1 -- see collapse_onehot_to_index in gpu/fromSeg.py."""
        mask = torch.zeros(1, 3, 4, 4, 4)
        mask[0, 0, 0, 0, 0] = 1.0  # only the first class is present here
        mask[0, 2, 3, 3, 3] = 1.0

        found = _foreground(mask, dim=1)

        self.assertTrue(bool(found[0, 0, 0, 0]), "the first one-hot class was dropped")
        self.assertTrue(bool(found[0, 3, 3, 3]))
        self.assertEqual(int(found.sum()), 2)

    def test_background_stays_background(self):
        self.assertEqual(int(_foreground(torch.zeros(1, 4, 5, 5, 5), dim=1).sum()), 0)


class TestApplyRegionMode(SmaugLabTestCase):
    """The behaviour a config actually asks for when it sets in_seg or out_seg."""

    def setUp(self):
        super().setUp()
        # orig/transformed are one channel of the batch: [N, D, H, W].
        self.orig = torch.zeros(1, 4, 4, 4)
        self.transformed = torch.ones(1, 4, 4, 4)
        # A single-channel mask covering one corner: [N, 1, D, H, W].
        self.mask = torch.zeros(1, 1, 4, 4, 4)
        self.mask[0, 0, :2, :2, :2] = 1.0
        self.inside = (slice(None), slice(0, 2), slice(0, 2), slice(0, 2))

    def test_mode_in_changes_only_the_masked_voxels(self):
        out = _apply_region_mode(self.orig, self.transformed, self.mask, "in")

        self.assertTrue(bool((out[self.inside] == 1.0).all()), "'in' did not apply the transform inside the mask")
        self.assertEqual(int(out.sum()), 8, "'in' leaked outside the mask")

    def test_mode_out_changes_only_the_unmasked_voxels(self):
        out = _apply_region_mode(self.orig, self.transformed, self.mask, "out")

        self.assertTrue(bool((out[self.inside] == 0.0).all()), "'out' applied the transform inside the mask")
        self.assertEqual(int(out.sum()), 64 - 8)

    def test_in_and_out_partition_the_volume(self):
        inside = _apply_region_mode(self.orig, self.transformed, self.mask, "in")
        outside = _apply_region_mode(self.orig, self.transformed, self.mask, "out")
        self.assertTrue(torch.equal(inside + outside, self.transformed))

    def test_mode_all_ignores_the_mask(self):
        out = _apply_region_mode(self.orig, self.transformed, self.mask, "all")
        self.assertTrue(torch.equal(out, self.transformed))

    def test_a_missing_mask_means_apply_everywhere(self):
        out = _apply_region_mode(self.orig, self.transformed, None, "in")
        self.assertTrue(torch.equal(out, self.transformed))

    def test_the_unbatched_3d_path_behaves_the_same(self):
        orig = torch.zeros(4, 4, 4)
        transformed = torch.ones(4, 4, 4)
        mask = torch.zeros(2, 4, 4, 4)
        mask[0, :2, :2, :2] = 1.0

        out = _apply_region_mode(orig, transformed, mask, "in")
        self.assertEqual(int(out.sum()), 8)

    def test_an_unsupported_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            _apply_region_mode(torch.rand(2, 2), torch.rand(2, 2), torch.rand(1, 2, 2), "in")


class TestRegionModeReachesTheRealTransforms(SmaugLabTestCase):
    """End to end: a GPU transform with in_seg=1.0 must respect the mask."""

    def test_in_seg_confines_a_scharr_transform_to_the_mask(self):
        image = self.tiny_volume()
        seg = self.tiny_seg()  # single channel, a centred cube

        # Driven through the container, as AugTransformsGPU does: that is what routes
        # the mask into params["seg"], which is where _apply_region_mode reads it.
        pipeline = AugmentationSequentialCustom(
            RandomScharrGPU(p=1.0, in_seg=1.0, out_seg=0.0, mix_prob=0.0),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )
        out = first_output(pipeline(image.clone(), seg.clone()))

        outside = ~(seg[0, 0] > 0)
        self.assertTrue(
            torch.allclose(out[0, 0][outside], image[0, 0][outside], atol=1e-5),
            "in_seg=1.0 changed voxels outside the segmentation",
        )
        self.assertFalse(
            torch.allclose(out[0, 0], image[0, 0], atol=1e-5),
            "in_seg=1.0 changed nothing at all -- this is what the argmax bug did",
        )


class TestFunctionTransformIsPerSample(SmaugLabTestCase):
    """The normalisation used a slab-wide min/max, coupling every volume in the batch."""

    def _run(self, volume: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)
        transform = RandomSqrtGPU(p=1.0)
        return transform.apply_transform(volume.clone(), {}, {}, transform=None)

    def test_a_volume_is_augmented_the_same_alone_and_in_a_batch(self):
        subject = torch.rand(1, 1, 8, 8, 8)
        # A neighbour with a much wider range: under a slab-wide min/max it drags the
        # subject's normalisation with it.
        neighbour = torch.rand(1, 1, 8, 8, 8) * 100.0
        batch = torch.cat([subject, neighbour], dim=0)

        alone = self._run(subject)
        together = self._run(batch)

        self.assertTrue(
            torch.allclose(alone[0, 0], together[0, 0], atol=1e-5),
            "the same volume was augmented differently depending on its batch neighbours",
        )

    def test_each_sample_is_normalised_onto_its_own_range(self):
        batch = torch.cat([torch.rand(1, 1, 6, 6, 6), torch.rand(1, 1, 6, 6, 6) * 50.0], dim=0)

        out = self._run(batch)

        for b in range(2):
            with self.subTest(sample=b):
                # sqrt of a [0, 1]-normalised sample still spans close to the full range.
                self.assertAlmostEqual(float(out[b, 0].max()), 1.0, places=3)


class TestNonFiniteGuard(SmaugLabTestCase):
    """Guard tests for `RandomHistogramEqualizationGPU`, not regression tests.

    The `.clone()` this pins is unobservable from outside today, in the same way as
    the `resample` change in the previous PR. `channel_data` was `input[:, c]`, a view,
    and the loop assigns into `channel_data[b]` -- so by the time the non-finite guard
    at the bottom `continue`d, every value it meant to withhold was already in the
    batch, and the closing `input[:, c] = channel_data` was a no-op. With the clone the
    guard does what it says. Reaching it needs the equalisation to *produce* a
    non-finite value from finite input, which no input constructed here manages: a NaN
    supplied by the caller raises out of `torch.histc` first. So these tests pin the
    behaviour the clone must not disturb, and pass either way.
    """

    def test_a_degenerate_constant_channel_stays_finite(self):
        transform = RandomHistogramEqualizationGPU(p=1.0, mix_prob=0.0)
        # A constant channel makes img_max == img_min, the degenerate histogram case.
        volume = torch.ones(1, 1, 8, 8, 8)

        out = transform.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertTrue(bool(torch.isfinite(out).all()), "a non-finite result reached the batch")

    def test_a_non_finite_input_is_still_rejected_loudly(self):
        """Documents why the guard cannot be exercised: histc rejects the range first."""
        transform = RandomHistogramEqualizationGPU(p=1.0, mix_prob=0.0)
        volume = torch.rand(1, 1, 8, 8, 8)
        volume[0, 0, 0, 0, 0] = float("nan")

        with self.assertRaises(RuntimeError):
            transform.apply_transform(volume, {}, {}, transform=None)

    def test_equalisation_still_changes_the_image(self):
        transform = RandomHistogramEqualizationGPU(p=1.0, mix_prob=0.0)
        volume = torch.rand(1, 1, 8, 8, 8)

        out = transform.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertFalse(torch.allclose(out, volume, atol=1e-6))


class TestDilationRankFollowsTheData(SmaugLabTestCase):
    """scipy requires the structuring element's rank to match the input's.

    `aug_redistribute_seg` is called per channel (`img[c], seg[c]`), so it sees a bare
    spatial array -- 3-D for a volume, 2-D for an image -- and returns (img, seg).
    """

    def test_a_2d_image_does_not_raise(self):
        from smauglab.transforms.cpu.fromSeg import aug_redistribute_seg

        image = torch.rand(12, 12)
        seg = torch.zeros(12, 12)
        seg[3:8, 3:8] = 1.0

        out, _ = aug_redistribute_seg(image.clone(), seg, in_seg=1.0)

        self.assertEqual(tuple(out.shape), tuple(image.shape))
        self.assertTrue(bool(np.isfinite(out.numpy()).all()))

    def test_the_3d_path_is_unchanged(self):
        from smauglab.transforms.cpu.fromSeg import aug_redistribute_seg

        image = torch.rand(10, 10, 10)
        seg = torch.zeros(10, 10, 10)
        seg[2:6, 2:6, 2:6] = 1.0

        out, _ = aug_redistribute_seg(image.clone(), seg, in_seg=1.0)

        self.assertEqual(tuple(out.shape), tuple(image.shape))


class _Stub:
    """Stands in for a transform: _select_and_check only reads these three attributes."""

    def __init__(self, in_seg: float = 0.0, out_seg: float = 0.0, mix_in_out: bool = False):
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out


class TestExtractedTailHelpers(SmaugLabTestCase):
    """The three helpers that replaced the tail repeated in nine apply_transform methods."""

    def test_channel_stats_are_per_sample(self):
        from smauglab.transforms.gpu.contrast import _channel_stats

        x = torch.stack([torch.full((4, 4, 4), 2.0), torch.full((4, 4, 4), 9.0)])

        means, stds = _channel_stats(x)

        self.assertEqual(tuple(means.shape), (2,))
        self.assertAlmostEqual(float(means[0]), 2.0, places=5)
        self.assertAlmostEqual(float(means[1]), 9.0, places=5)
        self.assertTrue(bool((stds == 0).all()))

    def test_restore_stats_puts_mean_and_std_back(self):
        from smauglab.transforms.gpu.contrast import _channel_stats, _restore_stats

        x = torch.rand(3, 5, 5, 5) * 4.0 + 1.0
        stats = _channel_stats(x)

        restored = _restore_stats(x * 100.0 - 7.0, stats)

        means, stds = _channel_stats(restored)
        for b in range(3):
            with self.subTest(sample=b):
                self.assertAlmostEqual(float(means[b]), float(stats[0][b]), places=4)
                self.assertAlmostEqual(float(stds[b]), float(stats[1][b]), places=4)

    def test_select_and_check_rejects_a_non_finite_result(self):
        from smauglab.transforms.gpu.contrast import _select_and_check

        orig = torch.zeros(1, 4, 4, 4)
        broken = torch.full((1, 4, 4, 4), float("nan"))

        self.assertIsNone(_select_and_check(_Stub(), orig, broken, None))

    def test_select_and_check_passes_a_finite_result_through(self):
        from smauglab.transforms.gpu.contrast import _select_and_check

        orig = torch.zeros(1, 4, 4, 4)
        fine = torch.ones(1, 4, 4, 4)

        self.assertTrue(torch.equal(_select_and_check(_Stub(), orig, fine, None), fine))

    def test_select_and_check_applies_the_region_mode(self):
        from smauglab.transforms.gpu.contrast import _select_and_check

        orig = torch.zeros(1, 4, 4, 4)
        transformed = torch.ones(1, 4, 4, 4)
        mask = torch.zeros(1, 1, 4, 4, 4)
        mask[0, 0, :2, :2, :2] = 1.0

        out = _select_and_check(_Stub(in_seg=1.0), orig, transformed, mask)

        self.assertEqual(int(out.sum()), 8)
