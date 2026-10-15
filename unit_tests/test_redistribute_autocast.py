"""`RandomRedistributeSegGPU` must still redistribute under `autocast`.

`_region_moments` computes each region's voxel count, mean and variance as one
`[R, S] x [S, 3]` matmul. `autocast` demotes matmuls to fp16, whose largest finite
value is 65504, so the `1` column -- a plain voxel count -- saturates to `inf` for
any region above about 40^3 voxels. At the 128^3 patch size nnU-Net trains on, the
background region always does.

Nothing then raises. `counts=inf` makes `means=NaN`, every per-region Gaussian
contribution is NaN, `tmax - tmin > 1e-8` is False on NaN so the `torch.where`
returns the input, and `retain_stats` maps that back to the original mean and std --
an exact identity. The transform's own `torch.isfinite(x).all()` guard passes,
because the NaN only ever lived in `to_add`.

So the augmentation reported itself as applied and did nothing, for every real
training run between `20360d8` (which introduced the matmul; the reduction it
replaced was fp32 under autocast and did not overflow) and this test. It sits at
`p=0.4` in the paper's "Ours" config, the highest probability of any transfer
augmentation there.

fp16 autocast on CPU demotes the same op, so these do not need a GPU.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.fromSeg import RandomRedistributeSegGPU, _region_moments
from unit_tests.helpers import SmaugLabTestCase

#: Big enough to reproduce the identity, small enough to stay a unit test. Both the
#: count column and the `sum(x)` column have to overflow: with only the count at `inf`
#: the mean comes out 0 rather than NaN, which is still wrong but still moves the patch.
#: `x` is the transform's [0, 1] working space, so `sum(x)` needs roughly 2 * 65504
#: voxels. 48^3 is not enough; 64^3 is, and matches the real failure at 128^3.
SHAPE = (1, 1, 64, 64, 64)


def pair() -> tuple[torch.Tensor, torch.Tensor]:
    """A z-scored patch and the single-channel integer label map the trainer passes."""
    seg = torch.zeros(*SHAPE)
    seg[:, :, 16:32, 16:32, 16:32] = 1.0
    seg[:, :, 36:48, 36:48, 36:48] = 2.0
    return torch.randn(*SHAPE), seg


class TestRegionCountsSurviveAutocast(SmaugLabTestCase):
    def test_the_count_does_not_overflow_fp16(self):
        values = torch.rand(100_000)
        masks = torch.ones(2, 100_000, dtype=torch.bool)
        design = torch.stack([torch.ones_like(values), values, values * values], dim=1)

        with torch.autocast("cpu", torch.float16):
            counts, means, stds = _region_moments(masks, design)

        self.assertTrue(torch.isfinite(counts).all(), f"counts overflowed: {counts}")
        self.assertEqual(float(counts[0]), 100_000.0)
        self.assertAlmostEqual(float(means[0]), float(values.mean()), places=4)
        self.assertAlmostEqual(float(stds[0]), float(values.std(unbiased=False)), places=4)

    def test_it_agrees_with_the_same_call_outside_autocast(self):
        values = torch.rand(100_000)
        masks = torch.rand(3, 100_000) > 0.5
        design = torch.stack([torch.ones_like(values), values, values * values], dim=1)

        plain = _region_moments(masks, design)
        with torch.autocast("cpu", torch.float16):
            amp = _region_moments(masks, design)

        for got, want in zip(amp, plain):
            self.assertTrue(torch.equal(got, want))


class TestRedistributeIsNotAnIdentityUnderAutocast(SmaugLabTestCase):
    def test_it_moves_the_patch(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                img, seg = pair()
                transform = RandomRedistributeSegGPU(confine_prob=1.0, p=1.0, p_batch=1.0)

                with torch.autocast("cpu", torch.float16):
                    out = transform.apply_transform(img.clone(), {"seg": seg}, {})

                moved = (out.float() - img).abs().max().item()
                self.assertTrue(torch.isfinite(out).all())
                # The perturbation is up to 2.0 in the transform's [0, 1] working space,
                # so anything near zero means the identity branch fired.
                self.assertGreater(moved, 1e-2, "redistribute was an identity under autocast")

    def test_autocast_does_not_change_the_result(self):
        """The moments are the only thing autocast could touch here, and they are pinned."""
        img, seg = pair()
        transform = RandomRedistributeSegGPU(confine_prob=1.0, p=1.0, p_batch=1.0)

        torch.manual_seed(0)
        plain = transform.apply_transform(img.clone(), {"seg": seg}, {})
        torch.manual_seed(0)
        with torch.autocast("cpu", torch.float16):
            amp = transform.apply_transform(img.clone(), {"seg": seg}, {})

        self.assertTrue(torch.allclose(plain, amp.float(), atol=1e-5))
