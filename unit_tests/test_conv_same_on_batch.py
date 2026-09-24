"""`same_on_batch` has to mean something on the convolution transforms.

It was inert on two of them and inverted on the third:

* `get_kernel` is called once, before the channel loop, so the GaussianBlur and
  UnsharpMask sigma was shared across the batch whatever the flag said --
  `same_on_batch=False` did nothing. The unsharp amount was a single draw too.
* RandConv drew a fresh kernel per batch element *unconditionally*, so
  `same_on_batch=True` did nothing -- and True is what
  `AugmentationSequentialCustom` forces onto every child, so on a shipped config
  RandConv was the one transform not sharing its draws.

The test feeds a batch of identical samples: shared draws must then give
identical outputs, and per-sample draws must not. That separates "the flag is
honoured" from "the transform happens to be deterministic", which comparing
different samples cannot.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.contrast import (
    RandomGaussianBlurGPU,
    RandomLaplaceGPU,
    RandomRandConvGPU,
    RandomUnsharpMaskGPU,
)
from unit_tests.helpers import SmaugLabTestCase

BATCH = 6
RANDOMISED = (RandomGaussianBlurGPU, RandomUnsharpMaskGPU, RandomRandConvGPU)


def identical_batch() -> torch.Tensor:
    """Every sample the same, so any difference in the output is a difference in draws."""
    torch.manual_seed(0)
    return torch.rand(1, 1, 12, 12, 12).repeat(BATCH, 1, 1, 1, 1)


def run(cls, same_on_batch, image):
    torch.manual_seed(1)
    transform = cls(p=1.0, same_on_batch=same_on_batch, mix_prob=0.0, retain_stats=False)
    return transform.apply_transform(image.clone(), {}, transform.flags)


class TestSameOnBatchIsHonoured(SmaugLabTestCase):
    def test_true_shares_the_draws_across_the_batch(self):
        image = identical_batch()
        for cls in RANDOMISED:
            with self.subTest(transform=cls.__name__):
                out = run(cls, True, image)

                for b in range(1, BATCH):
                    self.assertTrue(
                        bool(torch.allclose(out[0], out[b])),
                        f"{cls.__name__} used a different draw for sample {b} despite same_on_batch=True",
                    )

    def test_false_draws_per_sample(self):
        image = identical_batch()
        for cls in RANDOMISED:
            with self.subTest(transform=cls.__name__):
                out = run(cls, False, image)

                self.assertFalse(
                    all(bool(torch.allclose(out[0], out[b])) for b in range(1, BATCH)),
                    f"{cls.__name__} shared its draw despite same_on_batch=False",
                )

    def test_a_deterministic_kernel_is_unaffected(self):
        """Laplace has nothing random to share; the flag must not change it."""
        image = identical_batch()

        torch.testing.assert_close(run(RandomLaplaceGPU, True, image), run(RandomLaplaceGPU, False, image))

    def test_the_output_stays_well_formed(self):
        image = identical_batch()
        for cls in RANDOMISED:
            for same_on_batch in (True, False):
                with self.subTest(transform=cls.__name__, same_on_batch=same_on_batch):
                    out = run(cls, same_on_batch, image)

                    self.assertEqual(out.shape, image.shape)
                    self.assertTrue(bool(torch.isfinite(out).all()))
