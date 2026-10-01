"""The elementwise function transforms normalise one channel at a time.

`_FunctionBaseTransform._apply_to_image` works on channel 0 only, but its
normalisation step reduced over the whole slab:

    img[c] = (img[c] - img.min()) / (img.max() - img.min() + 0.00001)

so a second channel decided channel 0's range. That is not hypothetical: the
two-channel layout `torchio_ops.apply_tio` produces carries a 0/1 label map in
channel 1, which pins `img.max()` at 1 whatever the image holds.

The GPU sibling documents this as already fixed ("This used to be a bare
`x.min()` / `x.max()`, which reduces over the whole [N, ...] slab").
"""

import torch

from smauglab.transforms.cpu.contrast import SqrtTransform
from unit_tests.helpers import SmaugLabTestCase


class TestFunctionTransformIsPerChannel(SmaugLabTestCase):
    def _apply(self, image):
        return SqrtTransform()(image=image.clone())["image"]

    def test_a_second_channel_does_not_change_the_first(self):
        """The contract: channel 0's result must not depend on what sits beside it."""
        channel0 = torch.rand(1, 8, 8, 8)
        alone = self._apply(channel0)

        for label, companion in (
            ("a 0/1 label map", (torch.rand(1, 8, 8, 8) > 0.5).float()),
            ("a second modality", torch.rand(1, 8, 8, 8) * 100.0),
        ):
            with self.subTest(companion=label):
                paired = self._apply(torch.cat([channel0, companion], dim=0))

                torch.testing.assert_close(paired[0], alone[0])

    def test_the_second_channel_is_left_alone(self):
        """The loop is `range(1)`; only channel 0 is transformed."""
        image = torch.cat([torch.rand(1, 8, 8, 8), torch.rand(1, 8, 8, 8)], dim=0)

        out = self._apply(image)

        torch.testing.assert_close(out[1], image[1])

    def test_the_channel_is_normalised_to_its_own_range(self):
        """sqrt of a [0, 1]-normalised channel: the channel minimum maps to 0."""
        channel0 = torch.rand(1, 8, 8, 8) * 5.0 + 10.0

        out = self._apply(channel0)

        self.assertAlmostEqual(float(out.min()), 0.0, places=4)
        self.assertAlmostEqual(float(out.max()), 1.0, places=4)
