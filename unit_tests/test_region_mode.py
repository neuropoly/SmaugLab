"""`in_seg` / `out_seg`: applying a GPU contrast transform only inside or outside the mask.

The bug these pin down: `_apply_region_mode` reduced the class axis with
`torch.argmax(mask, dim) > 0`. For the ordinary single-channel mask that is always
False, so `in_seg` applied the transform nowhere and `out_seg` applied it everywhere --
both knobs silently inoperative. For a one-hot mask it dropped the first foreground
class, because this repository encodes channel `c` as label `c + 1`.
"""

import torch

from smauglab.transforms.gpu.contrast import _apply_region_mode, _foreground
from unit_tests.helpers import SmaugLabTestCase


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


class TestNaNGuard(SmaugLabTestCase):
    def test_histogram_equalisation_does_not_write_through_a_view(self):
        """It used to take `input[:, c]` as a view and assign into it per batch element,
        so the NaN guard's `continue` skipped over values already written to the batch."""
        from smauglab.transforms.gpu.contrast import RandomHistogramEqualizationGPU

        transform = RandomHistogramEqualizationGPU(p=1.0, mix_prob=0.0)
        # A constant channel makes img_max == img_min, the degenerate histogram case.
        volume = torch.ones(1, 1, 8, 8, 8)
        out = transform.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertTrue(bool(torch.isfinite(out).all()), "a non-finite result reached the batch despite the guard")


class TestRegionModeReachesTheRealTransforms(SmaugLabTestCase):
    """End to end: a registered GPU transform with in_seg=1.0 must respect the mask."""

    def test_in_seg_confines_a_scharr_transform_to_the_mask(self):
        from smauglab.transforms.gpu.base import AugmentationSequentialCustom
        from smauglab.transforms.gpu.contrast import RandomScharrGPU

        image = self.tiny_volume()
        seg = self.tiny_seg()  # single channel, a centred cube

        # Driven through the container, as AugTransformsGPU does: that is what routes
        # the mask into params["seg"], which is where _apply_region_mode reads it.
        pipeline = AugmentationSequentialCustom(
            RandomScharrGPU(p=1.0, in_seg=1.0, out_seg=0.0, mix_prob=0.0),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )
        out = pipeline(image.clone(), seg.clone())
        out = out[0] if isinstance(out, (list, tuple)) else out

        outside = ~(seg[0, 0] > 0)
        self.assertTrue(
            torch.allclose(out[0, 0][outside], image[0, 0][outside], atol=1e-5),
            "in_seg=1.0 changed voxels outside the segmentation",
        )
        self.assertFalse(
            torch.allclose(out[0, 0], image[0, 0], atol=1e-5),
            "in_seg=1.0 changed nothing at all -- this is what the argmax bug did",
        )
