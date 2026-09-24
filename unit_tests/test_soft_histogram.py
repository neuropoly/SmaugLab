"""`DifferentiableHistogram3D` counts voxels, and carries no dead state.

`__init__` registered a `bin_centers` buffer that `forward` never read -- it
computes bin indices from `min_value` and `bin_width` arithmetically. The buffer
still appeared in `state_dict()` and was moved by every `.to(device)`.

The class had no test at all, so this pins what it computes as well as removing
the buffer.
"""

import torch

from smauglab.transforms.gpu.fromSeg import DifferentiableHistogram3D
from unit_tests.helpers import SmaugLabTestCase


class TestSoftHistogram(SmaugLabTestCase):
    def test_it_carries_no_unused_state(self):
        histogram = DifferentiableHistogram3D(num_bins=8)

        self.assertEqual(dict(histogram.named_buffers()), {})

    def test_the_counts_sum_to_the_voxel_count(self):
        """Soft binning splits each voxel between two bins; the total is still 1 each."""
        volume = torch.rand(2, 1, 4, 4, 4)

        hist = DifferentiableHistogram3D(num_bins=16)(volume)

        self.assertEqual(hist.shape, (2, 1, 16))
        torch.testing.assert_close(hist.sum(dim=2), torch.full((2, 1), 64.0))

    def test_a_mask_restricts_the_count(self):
        volume = torch.rand(1, 1, 4, 4, 4)
        mask = torch.zeros_like(volume)
        mask[..., :2, :, :] = 1.0

        hist = DifferentiableHistogram3D(num_bins=16)(volume, mask)

        torch.testing.assert_close(hist.sum(dim=2), torch.full((1, 1), 32.0))

    def test_a_constant_volume_lands_in_one_bin(self):
        volume = torch.full((1, 1, 2, 2, 2), 0.0)

        hist = DifferentiableHistogram3D(num_bins=8)(volume)

        self.assertAlmostEqual(float(hist[0, 0, 0]), 8.0, places=4)
        self.assertAlmostEqual(float(hist[0, 0, 1:].sum()), 0.0, places=4)

    def test_it_rejects_a_non_5d_input(self):
        with self.assertRaises(ValueError):
            DifferentiableHistogram3D()(torch.rand(1, 4, 4, 4))
