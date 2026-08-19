"""The shared convolution kernels and smooth random fields.

Four Gaussian blurs, three bias fields and two copies of the Laplace/Scharr tables
used to live in five modules. Two of those copies were wrong, and stayed wrong because
nothing compared them against each other. These tests are that comparison.
"""

import torch

from smauglab.transforms.kernels import (
    gaussian_blur3d,
    gaussian_kernel1d,
    gaussian_kernel3d,
    laplace_kernel,
    random_bias_field3d,
    scharr_kernels,
)
from unit_tests.helpers import SmaugLabTestCase


class TestGaussianKernelIsCentred(SmaugLabTestCase):
    """The bug: the 1-D kernel was sampled at arange(k), so its peak sat at index 0.

    A blur built from that kernel translates the image by about a voxel as well as
    blurring it -- while the segmentation mask, which is not convolved, stays put.
    """

    def test_the_1d_kernel_peaks_in_the_middle(self):
        for kernel_size in (3, 5, 7):
            with self.subTest(kernel_size=kernel_size):
                kernel = gaussian_kernel3d(kernel_size, 1.0, torch.float32, torch.device("cpu"))
                flat_argmax = int(kernel.argmax())
                centre = kernel_size // 2
                expected = (centre * kernel_size + centre) * kernel_size + centre
                self.assertEqual(flat_argmax, expected, "the 3D Gaussian's maximum is not the centre voxel")

    def test_the_kernel_is_symmetric(self):
        kernel = gaussian_kernel3d(5, 1.3, torch.float32, torch.device("cpu"))
        self.assertTrue(torch.allclose(kernel, kernel.flip(0), atol=1e-6))
        self.assertTrue(torch.allclose(kernel, kernel.flip(1), atol=1e-6))
        self.assertTrue(torch.allclose(kernel, kernel.flip(2), atol=1e-6))

    def test_the_kernel_sums_to_one(self):
        kernel = gaussian_kernel3d(3, torch.tensor([0.5, 1.0, 2.0]), torch.float32, torch.device("cpu"))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_the_old_uncentred_formula_really_was_off_centre(self):
        """Pin down what was wrong, so nobody reintroduces it as a 'simplification'."""
        kernel_size, sigma = 3, 1.0
        x = torch.arange(kernel_size, dtype=torch.float32)  # the old sample points
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        old = pdf / pdf.sum()
        self.assertEqual(int(old.argmax()), 0, "control: the old kernel peaked at index 0")

        half = (kernel_size - 1) / 2.0
        x_centred = torch.linspace(-half, half, kernel_size)
        pdf = torch.exp(-0.5 * (x_centred / sigma).pow(2))
        new = pdf / pdf.sum()
        self.assertEqual(int(new.argmax()), kernel_size // 2)

    def test_blurring_an_impulse_leaves_its_centre_of_mass_in_place(self):
        volume = torch.zeros(1, 1, 15, 15, 15)
        volume[0, 0, 7, 7, 7] = 1.0

        blurred = gaussian_blur3d(volume, 1.5)

        grid = torch.arange(15, dtype=torch.float32)
        for axis in (2, 3, 4):
            marginal = blurred.sum(dim=[d for d in (2, 3, 4) if d != axis]).flatten()
            centre_of_mass = float((marginal * grid).sum() / marginal.sum())
            self.assertAlmostEqual(centre_of_mass, 7.0, places=3, msg=f"blur shifted the impulse along axis {axis}")


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


class TestDerivativeKernels(SmaugLabTestCase):
    """The bug: the 2-D Scharr x-kernel had [-10, 0, -10] as its middle row on the CPU
    side, so it summed to -20 and was not a gradient operator at all."""

    def test_every_scharr_kernel_sums_to_zero(self):
        for dims in (2, 3):
            for axis, kernel in enumerate(scharr_kernels(dims)):
                with self.subTest(dims=dims, axis=axis):
                    self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)

    def test_every_laplace_kernel_sums_to_zero(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                self.assertAlmostEqual(float(laplace_kernel(dims).sum()), 0.0, places=5)

    def test_the_2d_scharr_x_kernel_is_antisymmetric(self):
        """A gradient operator negates when its differencing axis is flipped."""
        kernel_x, kernel_y = scharr_kernels(2)
        self.assertTrue(torch.allclose(kernel_x, -kernel_x.flip(1), atol=1e-6))
        self.assertTrue(torch.allclose(kernel_y, -kernel_y.flip(0), atol=1e-6))

    def test_the_3d_scharr_kernels_are_antisymmetric(self):
        for axis, kernel in enumerate(scharr_kernels(3)):
            with self.subTest(axis=axis):
                # kernel index 0 differences along the last axis, 1 the middle, 2 the first
                flip_dim = (2, 1, 0)[axis]
                self.assertTrue(torch.allclose(kernel, -kernel.flip(flip_dim), atol=1e-6))

    def test_a_scharr_kernel_gives_zero_response_on_a_constant_image(self):
        import torch.nn.functional as F

        constant = torch.full((1, 1, 7, 7), 4.0)
        for kernel in scharr_kernels(2):
            response = F.conv2d(constant, kernel.view(1, 1, 3, 3))
            self.assertTrue(torch.allclose(response, torch.zeros_like(response), atol=1e-4))

    def test_the_kernels_are_shared_by_both_backends(self):
        """The CPU and GPU tables were separate copies; that is how one drifted."""
        from smauglab.transforms.cpu.contrast import _ConvBaseTransform

        cpu_kernels = _ConvBaseTransform(kernel_type="Scharr").get_parameters(image=torch.rand(1, 8, 8, 8))["kernel"]
        for shared, from_cpu in zip(scharr_kernels(3), cpu_kernels):
            self.assertTrue(torch.equal(shared, from_cpu))

    def test_the_cpu_transform_hands_out_a_valid_2d_scharr(self):
        """The 2-D CPU path is where the sign typo lived, so exercise it directly."""
        from smauglab.transforms.cpu.contrast import _ConvBaseTransform

        kernels = _ConvBaseTransform(kernel_type="Scharr").get_parameters(image=torch.rand(1, 8, 8))["kernel"]
        self.assertEqual(len(kernels), 2)
        for axis, kernel in enumerate(kernels):
            with self.subTest(axis=axis):
                self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)

    def test_the_cpu_transform_hands_out_a_valid_2d_laplace(self):
        from smauglab.transforms.cpu.contrast import _ConvBaseTransform

        kernel = _ConvBaseTransform(kernel_type="Laplace").get_parameters(image=torch.rand(1, 8, 8))["kernel"]
        self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)

    def test_unsupported_dimensionality_is_rejected(self):
        with self.assertRaises(ValueError):
            scharr_kernels(4)
        with self.assertRaises(ValueError):
            laplace_kernel(1)


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
