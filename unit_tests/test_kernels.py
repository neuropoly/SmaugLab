"""The shared kernel module, exercised as the consolidation it is.

Four Gaussian blurs, three bias fields and two copies of the Laplace/Scharr tables
lived in five modules. `test_kernel_correctness.py` covers the two that were wrong;
this file covers the properties the merged implementation has to preserve for the
call sites it replaced, and the equivalences that make it one implementation rather
than a sixth copy.
"""

import torch

from smauglab.transforms.kernels import gaussian_blur3d, gaussian_kernel1d, random_bias_field3d
from unit_tests.helpers import SmaugLabTestCase


class TestGaussianBlur(SmaugLabTestCase):
    def test_a_constant_volume_survives_the_blur(self):
        """Normalised kernel plus reflect padding means no darkening at the border.

        The copy in gpu/fromSeg.py zero-padded, which pulled the border towards 0.
        """
        volume = torch.full((1, 1, 12, 12, 12), 3.0)
        blurred = gaussian_blur3d(volume, 1.0)
        self.assertTrue(torch.allclose(blurred, volume, atol=1e-5))

    def test_zero_sigma_is_a_no_op(self):
        volume = self.tiny_volume()
        self.assertTrue(torch.equal(gaussian_blur3d(volume, 0.0), volume))

    def test_anisotropic_sigma_blurs_only_the_named_axis(self):
        volume = torch.zeros(1, 1, 15, 15, 15)
        volume[0, 0, 7, 7, 7] = 1.0

        blurred = gaussian_blur3d(volume, torch.tensor([2.0, 0.0, 0.0]))

        # Axis 0 spread out; the other two still hold a single non-zero plane.
        self.assertGreater(int((blurred[0, 0, :, 7, 7] > 1e-6).sum()), 1)
        self.assertEqual(int((blurred[0, 0, 7, :, 7] > 1e-6).sum()), 1)
        self.assertEqual(int((blurred[0, 0, 7, 7, :] > 1e-6).sum()), 1)

    def test_multichannel_volumes_are_blurred_per_channel(self):
        volume = torch.zeros(2, 3, 11, 11, 11)
        volume[:, :, 5, 5, 5] = 1.0
        blurred = gaussian_blur3d(volume, 1.0)
        self.assertEqual(blurred.shape, volume.shape)
        for c in range(3):
            self.assertAlmostEqual(float(blurred[0, c].sum()), 1.0, places=4)

    def test_a_non_5d_input_is_rejected(self):
        with self.assertRaises(ValueError):
            gaussian_blur3d(torch.rand(4, 4, 4), 1.0)

    def test_kernel1d_is_normalised_and_odd_length(self):
        for sigma in (0.5, 1.0, 2.5):
            kernel = gaussian_kernel1d(sigma, torch.device("cpu"))
            self.assertEqual(kernel.numel() % 2, 1)
            self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)
            self.assertEqual(int(kernel.argmax()), kernel.numel() // 2)

    def test_a_non_positive_sigma_gives_the_identity_kernel(self):
        self.assertTrue(torch.equal(gaussian_kernel1d(0.0, torch.device("cpu")), torch.tensor([1.0])))


class TestRandomBiasField(SmaugLabTestCase):
    def test_the_field_is_strictly_positive(self):
        field = random_bias_field3d((8, 8, 8), std=0.7, scale=0.025, device=torch.device("cpu"))
        self.assertTrue(bool((field > 0).all()), "a log-space Gaussian field must exponentiate to positive values")

    def test_the_shape_follows_batch_and_channels(self):
        field = random_bias_field3d((6, 7, 8), std=0.5, scale=0.1, device=torch.device("cpu"), batch=3, channels=2)
        self.assertEqual(tuple(field.shape), (3, 2, 6, 7, 8))

    def test_a_zero_std_disables_the_field(self):
        field = random_bias_field3d((5, 5, 5), std=0.0, scale=0.025, device=torch.device("cpu"))
        self.assertTrue(torch.equal(field, torch.ones_like(field)))

    def test_the_field_is_smooth(self):
        """Coarse grid plus trilinear upsampling: neighbours must be close."""
        torch.manual_seed(0)
        field = random_bias_field3d((32, 32, 32), std=0.7, scale=0.025, device=torch.device("cpu"))[0, 0]
        largest_step = max(float(field.diff(dim=d).abs().max()) for d in range(3))
        self.assertLess(largest_step, 0.5)

    def test_synthseg_and_domain_transfer_now_share_one_implementation(self):
        from smauglab.transforms.gpu.domain_transfer import _random_bias_field3d
        from smauglab.transforms.synthseg.functional import bias_field

        torch.manual_seed(7)
        local = _random_bias_field3d((6, 6, 6), 0.7, 0.025, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(local.shape), (6, 6, 6))

        torch.manual_seed(7)
        applied = bias_field(torch.ones(1, 1, 6, 6, 6), bias_field_std=0.7, bias_scale=0.025)
        self.assertTrue(torch.allclose(applied[0, 0], local, atol=1e-6))
