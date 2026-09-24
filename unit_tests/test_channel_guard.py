"""An `apply_to_channel` the input does not have must fail, not be ignored.

`RandomBiasFieldGPU` and `ZscoreNormalizationGPU` guarded the index with
`continue  # skip invalid channel index`. Every other transform indexes
`input[:, c]` directly and raises `IndexError`, so a config typo -- most easily
`apply_to_channel: [1]` on single-channel data -- made those two silently do
nothing while the rest of the pipeline failed loudly.

Silent is the worse half of that pair: a transform the config asks for and that
never runs is invisible for the length of a training run.
"""

import torch

from smauglab.transforms.gpu.contrast import RandomBiasFieldGPU, RandomBrightnessGPU, ZscoreNormalizationGPU
from unit_tests.helpers import SmaugLabTestCase

GUARDED = (RandomBiasFieldGPU, ZscoreNormalizationGPU)


class TestOutOfRangeChannelRaises(SmaugLabTestCase):
    def _apply(self, cls, channels, apply_to_channel):
        transform = cls(p=1.0, apply_to_channel=apply_to_channel)
        image = torch.rand(1, channels, 8, 8, 8)
        return transform.apply_transform(image, {}, transform.flags)

    def test_a_channel_beyond_the_input_is_rejected(self):
        for cls in GUARDED:
            with self.subTest(transform=cls.__name__):
                with self.assertRaises(IndexError) as caught:
                    self._apply(cls, channels=1, apply_to_channel=[1])

                self.assertIn("apply_to_channel", str(caught.exception))

    def test_a_negative_channel_is_rejected(self):
        for cls in GUARDED:
            with self.subTest(transform=cls.__name__), self.assertRaises(IndexError):
                self._apply(cls, channels=2, apply_to_channel=[-1])

    def test_a_valid_channel_still_works(self):
        for cls in GUARDED:
            with self.subTest(transform=cls.__name__):
                out = self._apply(cls, channels=2, apply_to_channel=[1])

                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_the_unguarded_transforms_already_raised(self):
        """The behaviour being matched, not invented."""
        with self.assertRaises(IndexError):
            self._apply(RandomBrightnessGPU, channels=1, apply_to_channel=[1])
