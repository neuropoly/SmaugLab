"""The 2-D and 3-D Scharr tables must agree on which way a gradient points.

`SCHARR_2D`'s x-kernel has middle row `[-10, 0, 10]` -- the usual Scharr/Sobel
convention, a `[-1, 0, +1]` derivative with the positive lobe on the far side.
`SCHARR_3D` was the exact negation of that, so the same image ran through a 2-D
and a 3-D Scharr produced opposite-signed gradients.

`test_kernel_correctness.py` checks that each kernel sums to zero and is
antisymmetric; both properties are sign-agnostic, which is why this survived.

Every shipped config sets `absolute: true`, and `|(-k) * x| == |k * x|`, so no
shipped pipeline changes -- that invariance is asserted below rather than
asserted in prose. It matters for `absolute=False`, which is what
`RandomScharrGPU` defaults to.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from smauglab.transforms.kernels import scharr_kernels
from unit_tests.helpers import SmaugLabTestCase


def ramp(dims: int, size: int = 5) -> torch.Tensor:
    """A volume increasing along the last axis, so the gradient has a known sign."""
    shape = [1, 1] + [size] * dims
    values = torch.arange(size, dtype=torch.float32)
    view = [1, 1] + [1] * dims
    view[-1] = size
    return values.view(view).expand(shape).clone()


class TestScharrSignConvention(SmaugLabTestCase):
    def test_both_tables_put_the_positive_lobe_on_the_far_side(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                first = scharr_kernels(dims)[0].flatten()

                self.assertLess(float(first[0]), 0.0)
                self.assertGreater(float(first[-1]), 0.0)

    def test_an_increasing_ramp_gives_a_positive_gradient_in_both(self):
        """The property that actually matters: the same image, the same sign."""
        conv = {2: F.conv2d, 3: F.conv3d}
        for dims in (2, 3):
            with self.subTest(dims=dims):
                kernel = scharr_kernels(dims)[0]
                weight = kernel.view(1, 1, *kernel.shape)

                response = conv[dims](ramp(dims), weight)

                self.assertGreater(float(response.mean()), 0.0, "an increasing ramp must give a positive derivative")

    def test_every_kernel_still_sums_to_zero(self):
        for dims in (2, 3):
            for index, kernel in enumerate(scharr_kernels(dims)):
                with self.subTest(dims=dims, kernel=index):
                    self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)

    def test_the_absolute_response_is_unchanged_by_the_sign(self):
        """Why no shipped config moves: they all set absolute=true."""
        kernel = scharr_kernels(3)[0]
        weight = kernel.view(1, 1, *kernel.shape)
        volume = torch.rand(1, 1, 6, 6, 6)

        positive = F.conv3d(volume, weight).abs()
        negated = F.conv3d(volume, -weight).abs()

        torch.testing.assert_close(positive, negated)
