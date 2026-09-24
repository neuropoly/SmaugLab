"""RandConv must not change the image scale with its kernel size.

`_RandomConvBaseGPU.get_kernel` built its random kernel with
`std = 1.0 / sqrt(k * k)` -- the 2-D normalisation -- for a `k**3`-tap 3-D
kernel. Output variance is then `k**3 * (1/k**2) * var(x) = k * var(x)`, so the
augmentation's strength grew as `sqrt(k)` with a kernel size that is itself drawn
at random from `[1, 3, 5, 7]`.

Nothing corrected it downstream: `RandomRandConvGPU` defaults
`retain_stats=False`. `test_bucket_and_rng.py` covers RandConv reproducibility,
not its gain.
"""

import math

import torch

from smauglab.transforms.gpu.contrast import RandomRandConvGPU
from unit_tests.helpers import SmaugLabTestCase

KERNEL_SIZES = (1, 3, 5, 7)
DRAWS = 40


class TestRandConvPreservesScale(SmaugLabTestCase):
    def _kernels(self, k):
        transform = RandomRandConvGPU(p=1.0, kernel_sizes=[k])
        return [transform.get_kernel(torch.device("cpu")) for _ in range(DRAWS)]

    def test_the_expected_kernel_energy_is_one_for_every_size(self):
        """A k**3-tap kernel of i.i.d. taps has E[sum of squares] = k**3 * var(tap).

        Energy, not its square root: the square root of a chi-squared draw is
        biased low for few taps (a single tap averages E|z| = 0.798), so the mean
        gain is not 1 even when the normalisation is right. The expected *energy*
        is 1 for every k when it is.
        """
        for k in KERNEL_SIZES:
            with self.subTest(kernel_size=k):
                torch.manual_seed(0)
                energies = [float(kernel.pow(2).sum()) for kernel in self._kernels(k)]
                mean_energy = sum(energies) / len(energies)

                self.assertAlmostEqual(
                    mean_energy,
                    1.0,
                    delta=0.4,
                    msg=f"k={k} has mean energy {mean_energy:.3f}; k={k} would give {float(k):.1f} under the 2-D formula",
                )

    def test_the_output_scale_does_not_track_the_kernel_size(self):
        """The symptom: augmentation strength decided by the drawn kernel size.

        Compared in RMS over draws, not in mean. A single RandConv kernel is random
        by design, and for k=1 the transform is just a multiply by one normal draw,
        whose *mean* magnitude is E|z| = 0.798 however it is normalised. The
        root-mean-square is 1 for every k when the normalisation is right.
        """
        image = torch.randn(1, 1, 32, 32, 32)
        stds = {}
        for k in KERNEL_SIZES:
            torch.manual_seed(0)
            transform = RandomRandConvGPU(p=1.0, kernel_sizes=[k], mix_prob=0.0)
            draws = [float(transform.apply_transform(image.clone(), {}, transform.flags).std()) for _ in range(DRAWS)]
            stds[k] = math.sqrt(sum(d * d for d in draws) / len(draws))

        spread = max(stds.values()) / min(stds.values())
        self.assertLess(spread, 1.25, f"output scale still tracks the kernel size: {stds}")
