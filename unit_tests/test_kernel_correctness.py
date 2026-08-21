"""Convolution kernels that were not what their name says.

* The 1-D Gaussian was sampled at `arange(k)`, so its peak sat at index 0 and the 3-D
  kernel built from it had its maximum at corner [0, 0, 0]. A blur through that kernel
  translates the image by about a voxel as well as blurring it -- while the
  segmentation mask, which is never convolved, stays put.
* The 2-D CPU Scharr x-kernel had `[-10, 0, -10]` as its middle row, so it summed to
  -20 instead of 0 and was not a gradient operator.
"""

import torch
import torch.nn.functional as F

from smauglab.transforms.cpu.contrast import LaplaceConvTransform, ScharrConvTransform
from smauglab.transforms.kernels import gaussian_kernel1d, gaussian_kernel3d
from unit_tests.helpers import SmaugLabTestCase

CPU = torch.device("cpu")


class TestGaussianKernelIsCentred(SmaugLabTestCase):
    def test_the_1d_kernel_peaks_in_the_middle(self):
        for sigma in (0.5, 1.0, 2.5):
            with self.subTest(sigma=sigma):
                kernel = gaussian_kernel1d(sigma, CPU)
                self.assertEqual(kernel.numel() % 2, 1, "an even-length kernel has no centre tap")
                self.assertEqual(int(kernel.argmax()), kernel.numel() // 2, "the 1D Gaussian's peak is not the centre tap")

    def test_the_1d_kernel_is_symmetric_and_normalised(self):
        for sigma in (0.5, 1.3, 2.5):
            with self.subTest(sigma=sigma):
                kernel = gaussian_kernel1d(sigma, CPU)
                self.assertTrue(torch.allclose(kernel, kernel.flip(0), atol=1e-6))
                self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_a_non_positive_sigma_gives_the_identity_kernel(self):
        self.assertTrue(torch.equal(gaussian_kernel1d(0.0, CPU), torch.tensor([1.0])))

    def test_the_3d_kernel_peaks_at_the_centre_voxel(self):
        for kernel_size in (3, 5):
            with self.subTest(kernel_size=kernel_size):
                kernel = gaussian_kernel3d(kernel_size, 1.0, torch.float32, CPU)
                centre = kernel_size // 2
                expected = (centre * kernel_size + centre) * kernel_size + centre
                self.assertEqual(int(kernel.argmax()), expected, "the 3D Gaussian's maximum is not the centre voxel")

    def test_the_3d_kernel_is_symmetric_on_every_axis(self):
        kernel = gaussian_kernel3d(5, 1.3, torch.float32, CPU)
        for axis in (0, 1, 2):
            with self.subTest(axis=axis):
                self.assertTrue(torch.allclose(kernel, kernel.flip(axis), atol=1e-6))

    def test_the_3d_kernel_sums_to_one(self):
        kernel = gaussian_kernel3d(3, torch.tensor([0.5, 1.0, 2.0]), torch.float32, CPU)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_blurring_an_impulse_leaves_its_centre_of_mass_in_place(self):
        """The translation is the part that actually hurt: the mask does not move with it."""
        volume = torch.zeros(1, 1, 15, 15, 15)
        volume[0, 0, 7, 7, 7] = 1.0
        kernel = gaussian_kernel3d(7, 1.5, torch.float32, CPU)

        blurred = F.conv3d(volume, kernel.view(1, 1, 7, 7, 7), padding=3)

        grid = torch.arange(15, dtype=torch.float32)
        for axis in (2, 3, 4):
            with self.subTest(axis=axis):
                marginal = blurred.sum(dim=[d for d in (2, 3, 4) if d != axis]).flatten()
                centre_of_mass = float((marginal * grid).sum() / marginal.sum())
                self.assertAlmostEqual(centre_of_mass, 7.0, places=3, msg=f"blur shifted the impulse along axis {axis}")

    def test_the_old_uncentred_formula_really_was_off_centre(self):
        """Pin down what was wrong, so nobody reintroduces it as a 'simplification'."""
        kernel_size, sigma = 3, 1.0
        x = torch.arange(kernel_size, dtype=torch.float32)  # the old sample points
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        old = pdf / pdf.sum()

        self.assertEqual(int(old.argmax()), 0, "control: the old kernel peaked at index 0")
        self.assertEqual(int(gaussian_kernel3d(kernel_size, sigma, torch.float32, CPU)[:, 1, 1].argmax()), 1)


class TestScharrIsAGradientOperator(SmaugLabTestCase):
    def _kernels(self, spatial_dims: int):
        image = torch.rand(1, *([8] * spatial_dims))
        return ScharrConvTransform().get_parameters(image=image)["kernel"]

    def test_every_2d_scharr_kernel_sums_to_zero(self):
        for axis, kernel in enumerate(self._kernels(2)):
            with self.subTest(axis=axis):
                self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)

    def test_every_3d_scharr_kernel_sums_to_zero(self):
        for axis, kernel in enumerate(self._kernels(3)):
            with self.subTest(axis=axis):
                self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)

    def test_the_2d_kernels_are_antisymmetric_about_their_axis(self):
        """A gradient operator negates when its differencing axis is flipped."""
        kernel_x, kernel_y = self._kernels(2)
        self.assertTrue(torch.allclose(kernel_x, -kernel_x.flip(1), atol=1e-6))
        self.assertTrue(torch.allclose(kernel_y, -kernel_y.flip(0), atol=1e-6))

    def test_a_constant_image_gives_no_gradient(self):
        constant = torch.full((1, 1, 7, 7), 4.0)
        for axis, kernel in enumerate(self._kernels(2)):
            with self.subTest(axis=axis):
                response = F.conv2d(constant, kernel.view(1, 1, 3, 3))
                self.assertTrue(torch.allclose(response, torch.zeros_like(response), atol=1e-4))

    def test_the_2d_x_kernel_responds_to_a_horizontal_edge(self):
        """Sanity: it must still detect the thing it is for."""
        image = torch.zeros(1, 1, 7, 7)
        image[0, 0, :, 4:] = 1.0
        kernel_x = self._kernels(2)[0]

        response = F.conv2d(image, kernel_x.view(1, 1, 3, 3))

        self.assertGreater(float(response.abs().max()), 1.0)

    def test_the_laplace_kernels_still_sum_to_zero(self):
        """Untouched by this change, but the neighbouring branch of the same method."""
        for spatial_dims in (2, 3):
            with self.subTest(spatial_dims=spatial_dims):
                image = torch.rand(1, *([8] * spatial_dims))
                kernel = LaplaceConvTransform().get_parameters(image=image)["kernel"]
                self.assertAlmostEqual(float(kernel.sum()), 0.0, places=5)
