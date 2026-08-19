"""Variant leaves fix their variant and hide it from the config surface.

Four convolution kernels used to share one class behind a `kernel_type=` argument,
gamma and inverted-gamma shared one behind `invert_image=`, and five elementwise
functions shared one behind an un-serialisable `func=` callable. That made the
config key ambiguous -- the same augmentation could be spelled several ways, and
`FunctionTransform` could not name a single function at all.

Each variant is now its own class. These tests pin both halves of that: the leaf
really does apply its own variant, and the variant argument is gone, so a config
cannot contradict the class it just named.
"""

from __future__ import annotations

import inspect
import unittest

import torch

from smauglab.transforms.cpu.contrast import (
    ExpTransform,
    InvertedGammaTransform,
    LaplaceConvTransform,
    Log1pTransform,
    ScharrConvTransform,
    SigmoidTransform,
    SinTransform,
    SqrtTransform,
)
from smauglab.transforms.gpu.contrast import (
    RandomExpGPU,
    RandomGammaGPU,
    RandomGaussianBlurGPU,
    RandomInvGammaGPU,
    RandomLaplaceGPU,
    RandomLog1pGPU,
    RandomRandConvGPU,
    RandomScharrGPU,
    RandomSigmoidGPU,
    RandomSinGPU,
    RandomSqrtGPU,
    RandomUnsharpMaskGPU,
)

GPU_KERNELS = {
    RandomLaplaceGPU: "Laplace",
    RandomScharrGPU: "Scharr",
    RandomGaussianBlurGPU: "GaussianBlur",
    RandomUnsharpMaskGPU: "UnsharpMask",
    RandomRandConvGPU: "RandConv",
}
GPU_FUNCTIONS = [RandomLog1pGPU, RandomSqrtGPU, RandomSinGPU, RandomExpGPU, RandomSigmoidGPU]
CPU_FUNCTIONS = [Log1pTransform, SqrtTransform, SinTransform, ExpTransform, SigmoidTransform]


def params(cls) -> set[str]:
    return set(inspect.signature(cls).parameters)


class TestConvLeaves(unittest.TestCase):
    def test_each_leaf_fixes_its_kernel(self):
        for cls, kernel in GPU_KERNELS.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls().kernel_type, kernel)

    def test_kernel_type_is_not_configurable(self):
        for cls in GPU_KERNELS:
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("kernel_type", params(cls))

    def test_scharr_keeps_the_defaults_the_old_ladder_passed(self):
        """The ladder passed absolute=True and retain_stats=True only for Scharr."""
        scharr = RandomScharrGPU()
        self.assertTrue(scharr.absolute)
        self.assertTrue(scharr.retain_stats)
        self.assertFalse(RandomLaplaceGPU().absolute)

    def test_unsharp_keeps_its_ladder_amount(self):
        self.assertEqual(RandomUnsharpMaskGPU().unsharp_amount, 1.5)

    def test_cpu_leaves_mirror_the_gpu_split(self):
        self.assertEqual(LaplaceConvTransform().kernel_type, "Laplace")
        self.assertEqual(ScharrConvTransform().kernel_type, "Scharr")
        for cls in (LaplaceConvTransform, ScharrConvTransform):
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("kernel_type", params(cls))


class TestGammaLeaves(unittest.TestCase):
    def test_leaves_fix_opposite_inversions(self):
        self.assertFalse(RandomGammaGPU().invert_image)
        self.assertTrue(RandomInvGammaGPU().invert_image)

    def test_invert_image_is_not_configurable(self):
        for cls in (RandomGammaGPU, RandomInvGammaGPU):
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("invert_image", params(cls))

    def test_cpu_inverted_gamma_fixes_the_flag(self):
        self.assertNotIn("p_invert_image", params(InvertedGammaTransform))
        self.assertEqual(InvertedGammaTransform().p_invert_image, 1)


class TestFunctionLeaves(unittest.TestCase):
    def test_func_is_not_configurable(self):
        for cls in GPU_FUNCTIONS + CPU_FUNCTIONS:
            with self.subTest(cls=cls.__name__):
                self.assertNotIn("func", params(cls))
                self.assertNotIn("function", params(cls))

    def test_each_leaf_applies_a_distinct_function(self):
        x = torch.linspace(0.1, 2.0, 8)
        results = {cls.__name__: cls().func(x) for cls in GPU_FUNCTIONS}
        for name, value in results.items():
            with self.subTest(cls=name):
                self.assertTrue(torch.isfinite(value).all())
        flat = list(results.values())
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                self.assertFalse(torch.allclose(flat[i], flat[j]), "two function leaves compute the same thing")

    def test_arithmetic_matches_the_original_lambdas_bit_for_bit(self):
        """torch.log1p / torch.sigmoid differ in the last ulp and would move the
        seeded determinism hashes that published experiments depend on."""
        x = torch.linspace(0.1, 2.0, 64)
        self.assertTrue(torch.equal(RandomLog1pGPU().func(x), torch.log(1 + x)))
        self.assertTrue(torch.equal(RandomSigmoidGPU().func(x), 1 / (1 + torch.exp(-x))))

    def test_gpu_and_cpu_leaves_compute_the_same_functions(self):
        x = torch.linspace(0.1, 2.0, 16)
        for gpu_cls, cpu_cls in zip(GPU_FUNCTIONS, CPU_FUNCTIONS):
            with self.subTest(pair=f"{gpu_cls.__name__}/{cpu_cls.__name__}"):
                self.assertTrue(torch.equal(gpu_cls().func(x), cpu_cls().function(x)))


class TestAcqIsSingleAxis(unittest.TestCase):
    def test_one_dim_is_not_configurable(self):
        """RandomAcqTransformGPU *is* the single-axis case; the isotropic one is
        RandomLowResTransformGPU. Exposing the flag let either key do either job."""
        from smauglab.transforms.gpu.spatial import RandomAcqTransformGPU

        self.assertNotIn("one_dim", params(RandomAcqTransformGPU))
        self.assertTrue(RandomAcqTransformGPU()._param_generator.one_dim)


if __name__ == "__main__":
    unittest.main()
