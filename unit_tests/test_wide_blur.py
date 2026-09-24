"""A Gaussian wider than the patch must blur it, not crash.

`gaussian_blur3d` pads with `mode="reflect"`, which `F.pad` only accepts while
the padding is smaller than the axis it pads. The kernel is sized
`2 * ceil(3 * sigma) + 1`, so sigma 6 asks for 18 voxels of padding -- more than
a 16-voxel axis has.

That is not an exotic setting: SynthSeg's `blurring_sigma_for_downsampling`
returns sigmas in that range for the anisotropic resolutions the generator
samples, so the failure showed up as an intermittent crash on a small patch
rather than something reproducible.

`test_kernels.py` blurs 11- to 15-cube volumes with sigma <= 2, which never gets
near the limit.
"""

import torch

from smauglab.transforms.kernels import gaussian_blur3d
from unit_tests.helpers import SmaugLabTestCase


class TestBlurWiderThanTheVolume(SmaugLabTestCase):
    def test_a_sigma_wider_than_the_patch_still_blurs(self):
        for shape, sigma in (
            ((1, 1, 16, 16, 16), 6.0),  # the SynthSeg case: pad 18 against 16
            ((1, 1, 8, 8, 8), 10.0),
            ((1, 1, 4, 4, 4), 3.0),
        ):
            with self.subTest(shape=shape, sigma=sigma):
                volume = torch.rand(*shape)

                out = gaussian_blur3d(volume, sigma)

                self.assertEqual(out.shape, volume.shape)
                self.assertTrue(bool(torch.isfinite(out).all()))
                self.assertFalse(bool(torch.equal(out, volume)), "nothing was blurred")

    def test_a_singleton_axis_is_handled(self):
        """Reflect is undefined for an axis of length 1; replicate takes over."""
        volume = torch.rand(1, 1, 1, 8, 8)

        out = gaussian_blur3d(volume, 4.0)

        self.assertEqual(out.shape, volume.shape)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_a_narrow_sigma_is_unchanged_by_the_fix(self):
        """One reflect pass was already legal here, and must stay byte-identical."""
        volume = torch.rand(1, 2, 24, 24, 24)

        out = gaussian_blur3d(volume, 1.5)

        self.assertEqual(out.shape, volume.shape)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_the_synthseg_generator_survives_a_small_patch(self):
        """The path this was found on: ~2% of calls used to raise."""
        from smauglab.transforms.synthseg.generator import SynthSegGenerator

        labels = torch.zeros(1, 1, 16, 16, 16)
        labels[:, :, 4:12, 4:12, 4:12] = 1.0
        generator = SynthSegGenerator(generation_labels=[0, 1], output_labels=[0, 1])

        for seed in range(60):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                image, _ = generator(labels.clone())

                self.assertTrue(bool(torch.isfinite(image).all()))
