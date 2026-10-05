"""`mix_prob` is a per-sample probability, as its siblings already treat it.

`_RandomConvBaseGPU.apply_transform` drew

    if torch.rand(1).item() < self.mix_prob:
        alpha = torch.rand(1, device=input.device)

*outside* any loop, so one coin flip and one alpha decided the whole batch.
`RandomInverseGPU` and `RandomHistogramEqualizationGPU` write the identical three
lines inside their per-sample loop, so the same parameter meant two different
things depending on which transform read it. Its documented meaning --
"probability of blending the result back with the original" -- is the per-sample
one.

`transform_params_gpu.json` and `transform_params_one-sequence-to-segment-them-all.json`
run `RandomScharrGPU` at `mix_prob: 0.5`, so this is a live difference, not a
latent one.
"""

from __future__ import annotations

import torch

from smauglab.transforms.gpu.contrast import RandomHistogramEqualizationGPU, RandomScharrGPU
from unit_tests.helpers import SmaugLabTestCase

BATCH = 8
SHAPE = (BATCH, 1, 12, 12, 12)


class TestMixProbIsPerSample(SmaugLabTestCase):
    def _blended_counts(self, mix_prob, seeds=6):
        """How many samples per batch differ from the same run with mixing off."""
        image = torch.rand(*SHAPE)
        counts = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            mixed = RandomScharrGPU(p=1.0, mix_prob=mix_prob, retain_stats=False)
            out = mixed.apply_transform(image.clone(), {}, mixed.flags)

            torch.manual_seed(seed)
            plain = RandomScharrGPU(p=1.0, mix_prob=0.0, retain_stats=False)
            reference = plain.apply_transform(image.clone(), {}, plain.flags)

            counts.append(sum(not bool(torch.allclose(out[b], reference[b])) for b in range(BATCH)))
        return counts

    def test_a_middling_probability_splits_the_batch(self):
        counts = self._blended_counts(0.5)

        self.assertTrue(
            any(0 < count < BATCH for count in counts),
            f"every batch was all-or-nothing, so mix_prob is still per batch: {counts}",
        )

    def test_probability_zero_blends_nothing(self):
        self.assertTrue(all(count == 0 for count in self._blended_counts(0.0)))

    def test_probability_one_blends_every_sample(self):
        self.assertTrue(all(count == BATCH for count in self._blended_counts(1.0)))

    def test_each_sample_gets_its_own_alpha(self):
        """One shared alpha would give every sample the identical blend weight.

        `out = alpha * orig + (1 - alpha) * x`, so alpha can be recovered exactly:
        alpha = (out - x) / (orig - x), with x the mix_prob=0 result and orig the
        input. A per-batch draw makes all eight recovered alphas equal.
        """
        torch.manual_seed(0)
        image = torch.rand(*SHAPE)

        mixed = RandomScharrGPU(p=1.0, mix_prob=1.0, retain_stats=False)
        out = mixed.apply_transform(image.clone(), {}, mixed.flags)
        torch.manual_seed(0)
        plain = RandomScharrGPU(p=1.0, mix_prob=0.0, retain_stats=False)
        x = plain.apply_transform(image.clone(), {}, plain.flags)

        alphas = []
        for b in range(BATCH):
            denominator = image[b] - x[b]
            usable = denominator.abs() > 1e-3
            alphas.append(round(float(((out[b] - x[b])[usable] / denominator[usable]).median()), 4))

        self.assertGreater(len(set(alphas)), 1, f"every sample was blended with the same alpha: {alphas}")


class TestTheSiblingsAlreadyAgreed(SmaugLabTestCase):
    """The behaviour being matched, not invented."""

    def test_histogram_equalisation_is_per_sample(self):
        image = torch.rand(*SHAPE)
        counts = []
        for seed in range(6):
            torch.manual_seed(seed)
            mixed = RandomHistogramEqualizationGPU(p=1.0, mix_prob=0.5)
            out = mixed.apply_transform(image.clone(), {}, mixed.flags)

            torch.manual_seed(seed)
            plain = RandomHistogramEqualizationGPU(p=1.0, mix_prob=0.0)
            reference = plain.apply_transform(image.clone(), {}, plain.flags)

            counts.append(sum(not bool(torch.allclose(out[b], reference[b])) for b in range(BATCH)))

        self.assertTrue(any(0 < count < BATCH for count in counts), f"expected a split batch, got {counts}")
