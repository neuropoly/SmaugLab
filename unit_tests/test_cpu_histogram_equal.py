"""The CPU histogram equalisation must put a voxel in the bin it belongs to.

`torch.searchsorted(bin_edges[:-1], v)` returns the number of edges *strictly
below* v, which is `bin + 1` for every value above the image minimum, and 256 at
the maximum -- one past the end of a 256-entry CDF, hidden by a `clamp(0, 255)`.

The GPU sibling (`gpu/contrast.py`, `RandomHistogramEqualizationGPU`) computes
the floor index and is the reference here. Nothing covered the CPU class.
"""

import torch

from smauglab.transforms.cpu.contrast import HistogramEqualTransform
from unit_tests.helpers import SmaugLabTestCase


def reference_equalisation(channel: torch.Tensor) -> torch.Tensor:
    """Textbook equalisation: CDF of the histogram, indexed by the value's own bin."""
    flat = channel.flatten().to(torch.float32)
    lo, hi = flat.min(), flat.max()
    hist, edges = torch.histogram(flat, bins=256)
    cdf = hist.cumsum(dim=0)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    cdf = cdf * (hi - lo) + lo
    return cdf[torch.bucketize(flat, edges[1:-1])].reshape(channel.shape)


class TestHistogramEqualisationBinning(SmaugLabTestCase):
    def _apply(self, volume):
        transform = HistogramEqualTransform()
        return transform(image=volume.clone())["image"]

    def test_it_matches_a_textbook_equalisation(self):
        volume = self.tiny_volume()[0]

        out = self._apply(volume)

        torch.testing.assert_close(out[0], reference_equalisation(volume[0]))

    def test_a_ramp_is_not_shifted_by_one_bin(self):
        """The clearest form of the defect: a linear ramp equalises to itself.

        With 256 bins over 256 distinct values, every bin holds one voxel, so the
        CDF is the identity and equalisation must return the ramp unchanged. An
        index that is one too high returns the *next* value everywhere.
        """
        ramp = torch.linspace(0.0, 1.0, 256).reshape(1, 4, 8, 8)

        out = self._apply(ramp)

        torch.testing.assert_close(out, ramp, atol=1e-5, rtol=0)

    def test_the_maximum_stays_in_range(self):
        """The top voxel used to index 256 of a 256-entry CDF."""
        volume = self.tiny_volume()[0]

        out = self._apply(volume)

        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertLessEqual(float(out.max()), float(volume.max()) + 1e-5)
