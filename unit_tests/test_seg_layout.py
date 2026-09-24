"""A label map and its one-hot equivalent must augment identically.

SmaugLab's transforms are handed the segmentation in whichever layout the caller
uses, and two of them read the region count off the channel axis:

* `RandomRedistributeSegGPU` did `masks = seg_b.bool(); R = masks.shape[0]`, so every
  per-region statistic was computed per channel.
* `RandomDomainTransferGPU` did `w = blur(seg[:, :NC].float())` and normalised across
  the class axis, so with one channel `w` normalised to 1 everywhere and one global
  LUT was applied.

`nnUNetTrainerDAExtGPU.train_step` passes `batch["target"]`, a single-channel int16
label map, straight into the pipeline -- nothing one-hots it. So in training both
transforms saw exactly one region covering the whole foreground, for every run, while
the sample-generation scripts (which one-hot via `prep_data.load_sample`) saw one
region per label and looked correct. `RandomRedistributeSegGPU` was the
highest-probability transform in Dataset014's trained config at p=0.4.

`seg_region_masks` makes the region set come from the *labels* rather than the
storage layout, so the two agree.
"""

import torch

from smauglab.transforms.gpu.fromSeg import RandomRedistributeSegGPU, seg_region_masks
from unit_tests.helpers import SmaugLabTestCase, seed_everything

SHAPE = (1, 1, 24, 24, 24)


def labelled_seg() -> tuple[torch.Tensor, torch.Tensor]:
    """Four labelled blocks plus background, as a label map and as its one-hot."""
    labels = torch.zeros(*SHAPE, dtype=torch.float32)
    labels[..., :12, :12, :] = 1.0
    labels[..., :12, 12:, :] = 2.0
    labels[..., 12:, :12, :] = 3.0
    labels[..., 12:, 12:, :12] = 4.0
    onehot = torch.cat([(labels == v) for v in range(5)], dim=1).float()
    return labels, onehot


class TestSegRegionMasks(SmaugLabTestCase):
    def test_a_label_map_expands_to_one_channel_per_label(self):
        labels, onehot = labelled_seg()
        self.assertTrue(torch.equal(seg_region_masks(labels), onehot.bool()))

    def test_one_hot_is_returned_unchanged(self):
        """One-hot callers must keep byte-identical behaviour."""
        _, onehot = labelled_seg()
        self.assertTrue(torch.equal(seg_region_masks(onehot), onehot.bool()))

    def test_max_regions_truncates_like_the_old_channel_slice(self):
        labels, _ = labelled_seg()
        self.assertEqual(seg_region_masks(labels, max_regions=3).shape[1], 3)

    def test_a_warped_float_mask_does_not_invent_regions(self):
        """Geometric transforms hand the mask back as float. Taking `unique` of
        unrounded floats would make one region per interpolated value."""
        labels, _ = labelled_seg()
        self.assertEqual(seg_region_masks(labels + 1e-4).shape[1], 5)

    def test_region_order_is_shared_across_a_batch(self):
        """Channel c must mean the same label for every sample, or a batched
        transform mixes regions between samples."""
        labels, _ = labelled_seg()
        batch = torch.cat([labels, torch.where(labels > 2, labels, torch.zeros_like(labels))])
        masks = seg_region_masks(batch)
        self.assertEqual(masks.shape[0], 2)
        # sample 1 has no label 1 or 2, but still carries their (empty) channels
        self.assertEqual(int(masks[1, 1].sum()), 0)
        self.assertGreater(int(masks[1, 3].sum()), 0)

    def test_the_old_channel_axis_read_saw_one_region(self):
        """Control: what `seg.bool()` gave on the same label map, and why it was wrong."""
        labels, _ = labelled_seg()
        self.assertEqual(labels.bool().shape[1], 1, "four labels used to collapse to one region")


def _apply(build, seg, image):
    seed_everything()
    return build().apply_transform(image.clone(), {"seg": seg}, {})


class TestRedistributeSegAgreesBetweenLayouts(SmaugLabTestCase):
    def test_label_map_and_one_hot_give_the_same_image(self):
        labels, onehot = labelled_seg()
        image = torch.rand(*SHAPE)

        def build():
            return RandomRedistributeSegGPU(p=1.0, in_seg=1.0)

        torch.testing.assert_close(_apply(build, labels, image), _apply(build, onehot, image))

    def test_the_labels_actually_drive_the_result(self):
        """Guard against the test passing because nothing happens: a different
        labelling of the same voxels must give a different image."""
        labels, _ = labelled_seg()
        merged = (labels > 0).float()  # every label fused into one region
        image = torch.rand(*SHAPE)

        def build():
            return RandomRedistributeSegGPU(p=1.0, in_seg=1.0)

        per_label = _apply(build, labels, image)
        one_blob = _apply(build, merged, image)
        self.assertFalse(
            torch.allclose(per_label, one_blob),
            "per-label and single-blob segmentations produced the same image, so the region set is still being ignored",
        )


class TestDomainTransferAgreesBetweenLayouts(SmaugLabTestCase):
    """The LUT bank is built offline and not shipped, so this skips without it."""

    def setUp(self) -> None:
        super().setUp()
        from smauglab import registry
        from smauglab.registry import Backend
        from unit_tests.helpers import requires_external_asset  # noqa: F401  (import check)

        registry.load_all()
        from unit_tests.helpers import domain_bank_missing

        reason = domain_bank_missing(registry.get("RandomDomainTransferGPU", Backend.GPU))
        if reason:
            self.skipTest(reason)

    def _build(self):
        from smauglab.transforms.gpu.domain_transfer import RandomDomainTransferGPU

        return RandomDomainTransferGPU(p=1.0, any_source=True, include_self=False)

    def test_label_map_and_one_hot_give_the_same_image(self):
        labels, onehot = labelled_seg()
        image = torch.rand(*SHAPE)
        torch.testing.assert_close(_apply(self._build, labels, image), _apply(self._build, onehot, image))

    def test_the_labels_actually_drive_the_result(self):
        labels, _ = labelled_seg()
        merged = (labels > 0).float()
        image = torch.rand(*SHAPE)
        self.assertFalse(
            torch.allclose(_apply(self._build, labels, image), _apply(self._build, merged, image)),
            "per-label and single-blob segmentations produced the same image, so the class "
            "weights are still being read off the channel axis",
        )
