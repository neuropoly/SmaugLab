"""`SynthSegTransformsGPU.probability` applies per sample, not per batch.

`forward` drew a single `torch.rand(())` and returned the batch untouched if it
lost, so `probability=0.5` meant "half of all *batches* are entirely synthetic"
rather than "half of all samples are". The network then saw all-synthetic and
all-real batches instead of a mix of both in each -- a different training signal,
and out of step with the per-sample convention every other transform here
follows.

This class is the standalone driver; it carries no registry entry, so no shipped
config reaches it. The registered `RandomSynthSegGPU` gets its per-sample
behaviour from kornia and is unaffected.
"""

from __future__ import annotations

import torch

from smauglab.transforms.synthseg.transforms import SynthSegTransformsGPU
from unit_tests.helpers import SmaugLabTestCase

BATCH = 8
SHAPE = (BATCH, 1, 16, 16, 16)


def driver(probability: float) -> SynthSegTransformsGPU:
    return SynthSegTransformsGPU(params={"generation_labels": [0, 1], "n_channels": 1, "probability": probability})


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros(*SHAPE)
    labels[:, :, 4:12, 4:12, 4:12] = 1
    return torch.randn(*SHAPE), labels


class TestSynthSegProbabilityIsPerSample(SmaugLabTestCase):
    def _synthesised(self, probability, seeds=6):
        transform = driver(probability)
        image, labels = inputs()
        counts = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            out_image, _ = transform(image.clone(), labels.clone())
            counts.append(sum(not bool(torch.equal(out_image[b], image[b])) for b in range(BATCH)))
        return counts

    def test_a_middling_probability_splits_the_batch(self):
        counts = self._synthesised(0.5)

        self.assertTrue(
            any(0 < count < BATCH for count in counts),
            f"every batch was all-or-nothing, so the probability is still per batch: {counts}",
        )

    def test_probability_one_synthesises_everything(self):
        self.assertTrue(all(count == BATCH for count in self._synthesised(1.0)))

    def test_probability_zero_leaves_the_batch_alone(self):
        self.assertTrue(all(count == 0 for count in self._synthesised(0.0)))

    def test_the_untouched_rows_are_bit_identical(self):
        """A spliced batch must not perturb the samples that were not selected."""
        transform = driver(0.5)
        image, labels = inputs()

        torch.manual_seed(0)
        out_image, out_labels = transform(image.clone(), labels.clone())

        for b in range(BATCH):
            if bool(torch.equal(out_image[b], image[b])):
                with self.subTest(sample=b):
                    self.assertTrue(bool(torch.equal(out_labels[b], labels[b])), "an unselected sample's labels moved")

    def test_the_output_keeps_its_shape_and_stays_finite(self):
        transform = driver(0.5)
        image, labels = inputs()

        torch.manual_seed(0)
        out_image, out_labels = transform(image.clone(), labels.clone())

        self.assertEqual(out_image.shape, image.shape)
        self.assertEqual(out_labels.shape, labels.shape)
        self.assertTrue(bool(torch.isfinite(out_image).all()))
