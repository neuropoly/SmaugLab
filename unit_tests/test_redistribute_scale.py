"""`RandomRedistributeSegGPU` must not throw away the input's intensity scale.

The transform works in a per-sample [0, 1] min-max space and adds a perturbation
of up to 2.0 *in that space*. With `retain_stats=False` nothing maps the result
back, so the output does not depend on the input's scale at all -- a z-scored
patch and the same patch multiplied by 100 come out identical. For a network fed
z-scored patches that is finite, silent and badly out of distribution, which is
the same failure the no-foreground branch of this method carries a comment
about.

Mapping back is not a fix on its own: the perturbation is defined in normalised
units, so rescaling it by the input range makes it much larger (measured, an
output mean of 18.4 rather than 2.2). Bounding the amplitude in input units
would be a redesign of an augmentation inherited from totalspineseg. So the
default changed and the limitation is pinned here rather than papered over.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.fromSeg import RandomRedistributeSegGPU
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (1, 1, 16, 16, 16)


def seg() -> torch.Tensor:
    mask = torch.zeros(*SHAPE)
    mask[:, :, 4:12, 4:12, 4:12] = 1.0
    return mask


def run(image, **kwargs):
    torch.manual_seed(3)
    transform = RandomRedistributeSegGPU(p=1.0, **kwargs)
    return transform.apply_transform(image.clone(), {"seg": seg()}, transform.flags)


class TestScaleIsPreservedByDefault(SmaugLabTestCase):
    def test_the_default_keeps_the_input_statistics(self):
        torch.manual_seed(0)
        image = torch.randn(*SHAPE) * 1.5

        out = run(image)

        self.assertAlmostEqual(float(out.mean()), float(image.mean()), places=4)
        self.assertAlmostEqual(float(out.std()), float(image.std()), places=3)

    def test_the_default_still_changes_the_image(self):
        torch.manual_seed(0)
        image = torch.randn(*SHAPE) * 1.5

        out = run(image)

        self.assertFalse(bool(torch.equal(out, image)))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_the_default_tracks_the_input_scale(self):
        """Two inputs differing only in scale must not produce the same output."""
        torch.manual_seed(0)
        image = torch.randn(*SHAPE)

        small = run(image)
        large = run(image * 100.0)

        self.assertGreater(float(large.std() / small.std()), 50.0)


class TestRetainStatsFalseIsStillScaleBlind(SmaugLabTestCase):
    """The known limitation, pinned so it cannot be forgotten or silently change."""

    def test_the_output_is_independent_of_the_input_scale(self):
        torch.manual_seed(0)
        image = torch.randn(*SHAPE)

        outputs = [run(image * scale, retain_stats=False) for scale in (1.0, 1.5, 100.0)]

        for other in outputs[1:]:
            torch.testing.assert_close(outputs[0], other)
