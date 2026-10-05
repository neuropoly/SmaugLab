"""Each optimised primitive, against the straightforward code it replaced.

The speed work in this area replaced several inner loops with batched or
reformulated equivalents. Every one of them is supposed to compute the same thing
as the obvious spelling, so every one of them is checked against the obvious
spelling here rather than against a stored expected value -- that way the test
says what the optimisation claims, and keeps saying it if the fast path is ever
rewritten again.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from smauglab.transforms.gpu.contrast import (
    RandomBiasFieldGPU,
    _axis_power_tables,
    _batched_quantiles,
    apply_convolution_per_sample,
)
from smauglab.transforms.gpu.fromSeg import _kmeans_1d, _region_moments, foreground_classes, segment_sum, voxel_coordinates
from smauglab.transforms.kernels import LAPLACE_2D, LAPLACE_3D, depthwise_conv3d, laplace_kernel, laplacian_response
from smauglab.transforms.synthseg.functional import _assign_gmm, _em_gmm_1d, distinct_values
from unit_tests.helpers import SmaugLabTestCase

CPU = torch.device("cpu")


class TestDepthwiseConv3d(SmaugLabTestCase):
    """`depthwise_conv3d` folds the batch into `groups`; the result must not move."""

    @staticmethod
    def _reference(padded: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """One `F.conv3d` per (sample, channel) plane -- what the loops used to do."""
        planes = padded.shape[0] * padded.shape[1]
        flat = padded.reshape(planes, 1, *padded.shape[2:])
        return torch.stack(
            [F.conv3d(flat[p : p + 1], kernel[min(p, kernel.shape[0] - 1) : min(p, kernel.shape[0] - 1) + 1])[0] for p in range(planes)]
        )

    def test_a_shared_kernel_matches_a_per_plane_loop(self):
        for batch, channels in ((1, 1), (3, 1), (2, 2)):
            with self.subTest(batch=batch, channels=channels):
                volume = torch.randn(batch, channels, 7, 8, 9)
                kernel = torch.randn(1, 1, 3, 3, 3)
                padded = F.pad(volume, [1] * 6, mode="reflect")
                got = depthwise_conv3d(padded, kernel)
                self.assertTrue(torch.allclose(got, self._reference(padded, kernel), atol=1e-5))

    def test_the_single_plane_duplication_does_not_change_the_answer(self):
        """A one-sample, one-channel call is the case that needs `groups == 2` faked."""
        volume = torch.randn(1, 1, 6, 7, 8)
        kernel = torch.randn(1, 1, 3, 3, 3)
        padded = F.pad(volume, [1] * 6, mode="reflect")

        got = depthwise_conv3d(padded, kernel)

        self.assertEqual(got.shape, (1, 1, 6, 7, 8))
        self.assertTrue(torch.allclose(got, F.conv3d(padded, kernel), atol=1e-5))

    def test_a_per_plane_kernel_bank_is_applied_plane_by_plane(self):
        volume = torch.randn(3, 1, 6, 6, 6)
        kernels = torch.randn(3, 1, 3, 3, 3)
        padded = F.pad(volume, [1] * 6, mode="reflect")

        got = depthwise_conv3d(padded, kernels)

        for plane in range(3):
            expected = F.conv3d(padded[plane : plane + 1], kernels[plane : plane + 1])
            self.assertTrue(torch.allclose(got[plane : plane + 1], expected, atol=1e-5))

    def test_several_filters_per_plane_land_on_the_channel_axis(self):
        volume = torch.randn(2, 1, 6, 6, 6)
        bank = torch.randn(4, 1, 3, 3, 3)
        padded = F.pad(volume, [1] * 6, mode="reflect")

        got = depthwise_conv3d(padded, bank, filters_per_plane=4)

        self.assertEqual(got.shape, (2, 4, 6, 6, 6))
        for plane in range(2):
            expected = F.conv3d(padded[plane : plane + 1], bank)
            self.assertTrue(torch.allclose(got[plane : plane + 1], expected, atol=1e-5))


class TestPerSampleConvolution(SmaugLabTestCase):
    def test_it_matches_convolving_each_sample_separately(self):
        volume = torch.randn(3, 6, 7, 8)
        kernels = [torch.randn(3, 3, 3) for _ in range(3)]

        got = apply_convolution_per_sample(volume, kernels)

        for b, kernel in enumerate(kernels):
            padded = F.pad(volume[b : b + 1].unsqueeze(1), [1] * 6, mode="reflect")
            expected = F.conv3d(padded, kernel.view(1, 1, 3, 3, 3))[0, 0]
            self.assertTrue(torch.allclose(got[b], expected, atol=1e-5))

    def test_kernels_of_different_sizes_are_zero_padded_not_resized(self):
        """A 1^3 kernel padded out to 7^3 must still be a 1^3 convolution."""
        volume = torch.randn(2, 10, 10, 10)
        small = torch.full((1, 1, 1), 2.0)
        large = torch.randn(7, 7, 7)

        got = apply_convolution_per_sample(volume, [small, large])

        self.assertTrue(torch.allclose(got[0], volume[0] * 2.0, atol=1e-5))
        padded = F.pad(volume[1:2].unsqueeze(1), [3] * 6, mode="reflect")
        self.assertTrue(torch.allclose(got[1], F.conv3d(padded, large.view(1, 1, 7, 7, 7))[0, 0], atol=1e-4))


class TestLaplacianIdentity(SmaugLabTestCase):
    """`laplacian_response` uses `3**d * x - box(x)` instead of the dense kernel."""

    def test_it_matches_the_dense_kernel_in_three_dimensions(self):
        volume = torch.randn(1, 1, 9, 10, 11)
        dense = F.conv3d(volume, laplace_kernel(3).view(1, 1, 3, 3, 3), padding="same")
        self.assertTrue(torch.allclose(laplacian_response(volume, 3), dense, atol=1e-4))

    def test_it_matches_the_dense_kernel_in_two_dimensions(self):
        image = torch.randn(1, 1, 11, 12)
        dense = F.conv2d(image, laplace_kernel(2).view(1, 1, 3, 3), padding="same")
        self.assertTrue(torch.allclose(laplacian_response(image, 2), dense, atol=1e-4))

    def test_the_tables_really_are_a_scaled_delta_minus_a_box(self):
        """The identity the reformulation rests on, stated directly."""
        for dims, table in ((2, LAPLACE_2D), (3, LAPLACE_3D)):
            with self.subTest(dims=dims):
                kernel = torch.tensor(table, dtype=torch.float32)
                box = torch.ones_like(kernel)
                delta = torch.zeros_like(kernel)
                delta[(1,) * dims] = float(3**dims)
                self.assertTrue(torch.equal(kernel, delta - box))


class TestCachedKernels(SmaugLabTestCase):
    def test_repeated_calls_return_the_same_tensor(self):
        self.assertIs(laplace_kernel(3, CPU), laplace_kernel(3, CPU))

    def test_the_cached_kernel_still_holds_the_table(self):
        self.assertTrue(torch.equal(laplace_kernel(3, CPU), torch.tensor(LAPLACE_3D, dtype=torch.float32)))


class TestSegmentSum(SmaugLabTestCase):
    def test_it_matches_scatter_add(self):
        ids = torch.randint(0, 6, (5000,))
        values = torch.rand(5000)
        expected = torch.zeros(6).scatter_add_(0, ids, values)
        self.assertTrue(torch.allclose(segment_sum(ids, values, 6), expected, atol=1e-4))

    def test_a_trailing_empty_segment_is_still_returned(self):
        """`bincount` sizes itself from the largest id present; the pad restores it."""
        ids = torch.zeros(10, dtype=torch.long)
        got = segment_sum(ids, torch.ones(10), 4)
        self.assertEqual(got.shape, (4,))
        self.assertTrue(torch.equal(got, torch.tensor([10.0, 0.0, 0.0, 0.0])))


class TestDistinctValues(SmaugLabTestCase):
    def test_foreground_classes_matches_unique(self):
        labels = torch.randint(0, 5, (2, 1, 6, 6, 6))
        expected = labels.unique()
        self.assertTrue(torch.equal(foreground_classes(labels), expected[expected > 0]))

    def test_distinct_values_matches_unique(self):
        volume = torch.randint(0, 9, (2, 1, 5, 5, 5))
        self.assertTrue(torch.equal(distinct_values(volume), volume.unique()))


class TestBatchedQuantiles(SmaugLabTestCase):
    def test_it_matches_torch_quantile_per_sample(self):
        data = torch.randn(4, 1, 7, 8, 9)
        lower = torch.tensor([0.0, 0.05, 0.2, 0.33])
        upper = torch.tensor([1.0, 0.95, 0.8, 0.67])

        got_lo, got_hi = _batched_quantiles(data, lower, upper)

        for b in range(4):
            flat = data[b].flatten()
            self.assertAlmostEqual(float(got_lo[b]), float(torch.quantile(flat, lower[b])), places=5)
            self.assertAlmostEqual(float(got_hi[b]), float(torch.quantile(flat, upper[b])), places=5)

    def test_the_no_op_shortcut_returns_the_actual_range(self):
        data = torch.randn(3, 1, 5, 5, 5)
        lo, hi = _batched_quantiles(data, torch.zeros(3), torch.ones(3))
        self.assertTrue(torch.equal(lo, data.reshape(3, -1).amin(dim=1)))
        self.assertTrue(torch.equal(hi, data.reshape(3, -1).amax(dim=1)))


class TestRegionMoments(SmaugLabTestCase):
    def test_it_matches_the_explicit_masked_statistics(self):
        values = torch.rand(400)
        masks = torch.rand(5, 400) > 0.6
        design = torch.stack([torch.ones_like(values), values, values * values], dim=1)

        counts, means, stds = _region_moments(masks, design)

        for r in range(5):
            selected = values[masks[r]]
            self.assertAlmostEqual(float(counts[r]), max(float(masks[r].sum()), 1.0), places=4)
            self.assertAlmostEqual(float(means[r]), float(selected.mean()), places=4)
            self.assertAlmostEqual(float(stds[r]), float(selected.std(unbiased=False)), places=4)


class TestKMeans1D(SmaugLabTestCase):
    def test_it_separates_two_well_separated_clusters(self):
        values = torch.cat([torch.randn(500) * 0.02, torch.randn(500) * 0.02 + 1.0])
        centroids = _kmeans_1d(values, 2)
        low, high = sorted(float(c) for c in centroids)
        self.assertAlmostEqual(low, 0.0, places=1)
        self.assertAlmostEqual(high, 1.0, places=1)

    def test_the_centroids_stay_sorted(self):
        """The `searchsorted` assignment is only valid while they are."""
        centroids = _kmeans_1d(torch.rand(2000), 5)
        self.assertTrue(bool((centroids[1:] >= centroids[:-1]).all()))

    def test_a_constant_input_does_not_produce_nan(self):
        centroids = _kmeans_1d(torch.full((100,), 0.5), 4)
        self.assertTrue(bool(torch.isfinite(centroids).all()))


class TestBiasFieldContraction(SmaugLabTestCase):
    """The polynomial is evaluated as a separable contraction, not a term loop."""

    def test_it_matches_the_explicit_monomial_sum(self):
        for order in (0, 1, 3):
            with self.subTest(order=order):
                transform = RandomBiasFieldGPU(order=order, p=1.0)
                spatial = (5, 6, 7)
                coefficients = torch.randn(transform._num_coeffs(3), 2)

                cube = transform._coefficient_cube(coefficients, 3)
                x_pow, y_pow, z_pow = _axis_power_tables(order, spatial, CPU, torch.float32)
                partial = torch.einsum("bxyz,zd->bxyd", cube, z_pow)
                partial = torch.einsum("bxyd,yh->bxdh", partial, y_pow)
                got = torch.einsum("bxdh,xw->bdhw", partial, x_pow)

                grids = transform._make_grids(spatial, CPU, torch.float32)
                expected = torch.zeros((2, *spatial))
                index = 0
                for xo in range(order + 1):
                    for yo in range(order + 1 - xo):
                        for zo in range(order + 1 - (xo + yo)):
                            term = grids[0].pow(xo) * grids[1].pow(yo) * grids[2].pow(zo)
                            expected += coefficients[index].view(-1, 1, 1, 1) * term
                            index += 1
                self.assertTrue(torch.allclose(got, expected, atol=1e-4))


class TestEmGmm(SmaugLabTestCase):
    """The EM steps are matrix multiplies now; the fit itself must still be a fit."""

    def test_it_recovers_two_separated_components(self):
        data = torch.cat([torch.randn(800) * 0.1 - 2.0, torch.randn(800) * 0.1 + 3.0])
        means, var, weights = _em_gmm_1d(data, 2, 40, 1e-6)
        low, high = sorted(float(m) for m in means)
        self.assertAlmostEqual(low, -2.0, places=1)
        self.assertAlmostEqual(high, 3.0, places=1)
        self.assertTrue(bool((var > 0).all()))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=4)

    def test_the_hard_assignment_splits_them_the_obvious_way(self):
        data = torch.cat([torch.randn(400) * 0.1 - 2.0, torch.randn(400) * 0.1 + 3.0])
        fit = _em_gmm_1d(data, 2, 40, 1e-6)
        assignment = _assign_gmm(data, *fit, eps=1e-6)
        self.assertEqual(len(assignment.unique()), 2)
        self.assertTrue(bool((assignment[:400] == assignment[0]).all()))
        self.assertTrue(bool((assignment[400:] == assignment[-1]).all()))
        self.assertNotEqual(int(assignment[0]), int(assignment[-1]))

    def test_a_constant_region_stays_finite(self):
        means, var, weights = _em_gmm_1d(torch.full((200,), 1.5), 3, 20, 1e-6)
        for name, tensor in (("means", means), ("var", var), ("weights", weights)):
            self.assertTrue(bool(torch.isfinite(tensor).all()), f"{name} went non-finite")

    def test_the_quadratic_form_is_the_gaussian_log_density(self):
        """`a x^2 + b x + c` has to be `log w + log N(x | m, v)`, or the E-step is wrong."""
        from smauglab.transforms.synthseg.functional import _quadratic_log_likelihood_terms

        means = torch.tensor([0.5, -1.25])
        var = torch.tensor([0.3, 2.0])
        weights = torch.tensor([0.4, 0.6])
        x = torch.linspace(-3, 3, 11).view(-1, 1)

        terms = _quadratic_log_likelihood_terms(means, var, weights, 1e-6)
        got = torch.cat([x * x, x, torch.ones_like(x)], dim=1) @ terms

        expected = (
            torch.log(weights).view(1, 2)
            - 0.5 * (math.log(2.0 * math.pi) + torch.log(var).view(1, 2))
            - 0.5 * (x - means.view(1, 2)) ** 2 / var.view(1, 2)
        )
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))


class TestVoxelCoordinateCache(SmaugLabTestCase):
    def test_the_grid_is_the_meshgrid_it_replaced(self):
        coords = voxel_coordinates((2, 3, 4), CPU)
        expected = torch.stack(torch.meshgrid(torch.arange(2.0), torch.arange(3.0), torch.arange(4.0), indexing="ij"), dim=-1).reshape(
            24, 3
        )
        self.assertTrue(torch.equal(coords, expected))

    def test_the_same_shape_is_not_rebuilt(self):
        self.assertIs(voxel_coordinates((4, 4, 4), CPU), voxel_coordinates((4, 4, 4), CPU))
