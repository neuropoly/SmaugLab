"""`in_seg` must confine the redistribution to the segmentation.

`aug_redistribute_seg` builds a `to_add` field -- zero outside the labels on the
`in_seg` branch -- and then normalised it with

    img += 2 * (to_add - to_add.min()) / (to_add.max() - to_add.min() + 1e-6)

A min-max rescale moves zero. A voxel where nothing was redistributed picked up
`-2 * min / range`, which is non-zero whenever `min < 0` -- and it always is,
because the per-label scale is `2 * rand - 1` on [-1, 1]. So `in_seg` restricted
where the redistribution was *computed* but not where it landed: the whole patch
got a uniform DC offset.

Scaling by the largest magnitude keeps the same peak amplitude and maps zero to
zero.
"""

from __future__ import annotations

import torch

from smauglab.transforms.cpu.fromSeg import aug_redistribute_seg
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (1, 16, 16, 16)


def pair() -> tuple[torch.Tensor, torch.Tensor]:
    seg = torch.zeros(*SHAPE)
    seg[:, 4:12, 4:12, 4:12] = 1.0
    return torch.rand(*SHAPE), seg


def normalised(img: torch.Tensor) -> torch.Tensor:
    """What the function does to the image before adding anything."""
    return (img - img.min()) / (img.max() - img.min()).clamp_min(1e-6)


class TestInSegStaysInsideTheSegmentation(SmaugLabTestCase):
    def test_nothing_changes_outside_the_labels(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                img, seg = pair()

                out, _ = aug_redistribute_seg(img.clone(), seg, in_seg=1.0)

                delta = out - normalised(img)
                self.assertAlmostEqual(
                    float(delta[seg == 0].abs().max()),
                    0.0,
                    places=6,
                    msg="in_seg applied a DC offset to the background",
                )

    def test_something_still_changes_inside_the_labels(self):
        """The guard against fixing the leak by disabling the transform."""
        torch.manual_seed(0)
        img, seg = pair()

        out, _ = aug_redistribute_seg(img.clone(), seg, in_seg=1.0)

        delta = out - normalised(img)
        self.assertGreater(float(delta[seg == 1].abs().max()), 0.1)

    def test_the_peak_amplitude_is_unchanged(self):
        """The rescale still tops out at 2, as the min-max form did."""
        torch.manual_seed(0)
        img, seg = pair()

        out, _ = aug_redistribute_seg(img.clone(), seg, in_seg=1.0)

        delta = (out - normalised(img)).abs().max()
        self.assertLessEqual(float(delta), 2.0 + 1e-4)

    def test_the_whole_image_branch_still_redistributes(self):
        """`in_seg=0` takes the other branch, which must keep working."""
        torch.manual_seed(0)
        img, seg = pair()

        out, _ = aug_redistribute_seg(img.clone(), seg, in_seg=0.0)

        delta = out - normalised(img)
        self.assertGreater(float(delta.abs().max()), 0.1)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_a_constant_patch_is_still_finite(self):
        """The degenerate case the clamp above exists for."""
        torch.manual_seed(0)
        _, seg = pair()

        out, _ = aug_redistribute_seg(torch.full(SHAPE, -2.709), seg, in_seg=1.0)

        self.assertTrue(bool(torch.isfinite(out).all()))
