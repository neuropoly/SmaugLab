"""`_foreground` must give the same answer for every segmentation layout.

The package carries two one-hot conventions:

* `collapse_onehot_to_index` documents background as *implicit* -- channel c is
  label c + 1, and a background voxel is all-zero.
* `seg_region_masks` emits one channel per distinct value for a single-channel
  label map, background *included*, and that is what `test_seg_layout.py` builds
  too.

`_foreground` was written for the first and returns True everywhere for the
second: the background channel covers exactly the voxels the foreground channels
do not, so `amax > 0` is True at every voxel. `in_seg` then applied the
transform to the whole patch and `out_seg` to none of it -- the precise failure
`_foreground`'s docstring says it exists to fix, in the layout it was not
checked against.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.contrast import _foreground
from smauglab.transforms.gpu.fromSeg import seg_region_masks
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (1, 1, 8, 8, 8)


def label_map() -> torch.Tensor:
    """Half background, half foreground, in two distinct labels."""
    labels = torch.zeros(*SHAPE)
    labels[:, :, :2] = 1.0
    labels[:, :, 2:4] = 2.0
    return labels


class TestForegroundAgreesAcrossLayouts(SmaugLabTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.labels = label_map()
        self.expected = float((self.labels > 0).float().mean())

    def _fraction(self, mask) -> float:
        return float(_foreground(mask, 1).float().mean())

    def test_a_single_channel_label_map(self):
        self.assertAlmostEqual(self._fraction(self.labels), self.expected, places=6)

    def test_a_background_implicit_one_hot(self):
        """`collapse_onehot_to_index`'s documented convention."""
        onehot = torch.cat([(self.labels == v) for v in (1, 2)], dim=1).float()

        self.assertAlmostEqual(self._fraction(onehot), self.expected, places=6)

    def test_a_background_inclusive_one_hot(self):
        """What `seg_region_masks` emits, and what used to read as all-foreground."""
        onehot = torch.cat([(self.labels == v) for v in (0, 1, 2)], dim=1).float()

        self.assertAlmostEqual(self._fraction(onehot), self.expected, places=6)

    def test_seg_region_masks_output_is_read_correctly(self):
        """The two helpers have to agree, since one feeds the other's layout."""
        masks = seg_region_masks(self.labels).float()

        self.assertAlmostEqual(self._fraction(masks), self.expected, places=6)

    def test_an_empty_mask_has_no_foreground(self):
        empty = torch.zeros(*SHAPE)

        self.assertEqual(self._fraction(empty), 0.0)

    def test_a_fully_labelled_single_channel_map_is_all_foreground(self):
        """Single-channel input must not be second-guessed."""
        full = torch.ones(*SHAPE)

        self.assertEqual(self._fraction(full), 1.0)
