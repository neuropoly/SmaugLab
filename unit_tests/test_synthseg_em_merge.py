"""`em_merge_prob`: EM subregions sometimes share a generation Gaussian.

`em_subdivide_labels` splits every label into intensity-coherent subregions and
gives each one its own Gaussian, so an EM split is *always* visible in the
synthetic image as an intensity boundary the segmentation target does not have.
The network can then learn that such a boundary means something. Merging siblings
back at generation time makes each boundary show up in only a fraction of the
samples instead.

The paper gives every subregion its own Gaussian, so the faithful setting is 0,
which the shipped paper config pins (and which must not consume random numbers,
or the run it reproduces would shift).
"""

from __future__ import annotations

import importlib.resources
import json
from unittest import mock

import torch

from smauglab import configs
from smauglab.transforms.synthseg import functional as FN
from smauglab.transforms.synthseg.generator import SynthSegGenerator
from unit_tests.helpers import SmaugLabTestCase

# Three background subregions and two foreground ones, as `em_subdivide_labels`
# reports them: the parent label of each generation label, in generation order.
PARENTS = [0, 0, 0, 1, 1]
SHAPE = (24, 24, 24)


def sparse_pair(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """A one-structure label map and a real image whose background is bimodal."""
    depth, height, width = SHAPE
    labels = torch.zeros(batch, 1, depth, height, width, dtype=torch.long)
    labels[..., 6:18, 6:18, 6:18] = 1
    image = torch.rand(batch, 1, depth, height, width)
    image[..., : depth // 2, :, :] += 3.0
    return labels, image


def generator(merge_prob: float) -> SynthSegGenerator:
    """EM completion on, every corruption off, zero GMM noise.

    Each Gaussian then paints one constant value, so counting the distinct
    intensities in the output counts the Gaussians that were actually used.
    """
    return SynthSegGenerator(
        em_label_completion=True,
        em_merge_prob=merge_prob,
        prior_stds=[0.0, 0.0],
        apply_affine=False,
        apply_nonlinear=False,
        apply_bias_field=False,
        apply_intensity_augmentation=False,
        apply_resolution=False,
        flipping=False,
    )


class TestRandomMergeClasses(SmaugLabTestCase):
    def test_zero_leaves_every_subregion_independent(self):
        classes = FN.random_merge_classes(PARENTS, 0.0, batch=4)
        for row in classes.tolist():
            self.assertEqual(row, list(range(len(PARENTS))))

    def test_one_collapses_each_parent_to_a_single_gaussian(self):
        classes = FN.random_merge_classes(PARENTS, 1.0, batch=4)
        for row in classes.tolist():
            self.assertEqual(len(set(row)), len(set(PARENTS)))
            self.assertEqual(len(set(row[:3])), 1)
            self.assertEqual(len(set(row[3:])), 1)

    def test_merging_never_crosses_parent_labels(self):
        """A foreground subregion on the background's Gaussian would hide the structure."""
        for _ in range(20):
            for row in FN.random_merge_classes(PARENTS, 0.9, batch=4).tolist():
                self.assertEqual(set(row[:3]) & set(row[3:]), set())

    def test_class_ids_are_contiguous_from_zero(self):
        """`sample_gmm_parameters` draws `max(class) + 1` Gaussians, so gaps waste draws."""
        for row in FN.random_merge_classes(PARENTS, 0.5, batch=8).tolist():
            self.assertEqual(sorted(set(row)), list(range(len(set(row)))))

    def test_the_pattern_is_drawn_per_sample(self):
        """One pattern for the whole batch would correlate the samples in it."""
        rows = {tuple(row) for row in FN.random_merge_classes(PARENTS, 0.5, batch=16).tolist()}
        self.assertGreater(len(rows), 1)


class TestPerSampleClassesReachTheGmm(SmaugLabTestCase):
    def test_tied_labels_share_their_gaussian_within_a_sample(self):
        classes = torch.tensor([[0, 0, 0, 1, 1]] * 4)
        means, stds = FN.sample_gmm_parameters(5, 1, 4, torch.device("cpu"), generation_classes=classes, background_label_index=None)

        for tied in ((0, 1), (0, 2), (3, 4)):
            self.assertTrue(torch.equal(means[:, tied[0]], means[:, tied[1]]))
            self.assertTrue(torch.equal(stds[:, tied[0]], stds[:, tied[1]]))

    def test_each_sample_gets_its_own_draw(self):
        """Indexing instead of gathering would broadcast one row over the batch."""
        classes = torch.tensor([[0, 0, 0, 1, 1]] * 4)
        means, _ = FN.sample_gmm_parameters(5, 1, 4, torch.device("cpu"), generation_classes=classes, background_label_index=None)

        self.assertFalse(torch.equal(means[0], means[1]))

    def test_a_row_per_batch_element_is_required(self):
        with self.assertRaises(ValueError):
            FN.sample_gmm_parameters(5, 1, 4, torch.device("cpu"), generation_classes=torch.zeros(3, 5, dtype=torch.long))


class TestGeneratorHonoursTheProbability(SmaugLabTestCase):
    def _levels(self, merge_prob: float, seed: int = 5) -> list[int]:
        labels, image = sparse_pair()
        torch.manual_seed(seed)
        synth, _ = generator(merge_prob)(labels, image=image)
        return [int(torch.unique(synth[b]).numel()) for b in range(synth.shape[0])]

    def test_full_merging_leaves_one_intensity_per_parent_label(self):
        self.assertEqual(self._levels(1.0), [2, 2])

    def test_no_merging_keeps_one_intensity_per_subregion(self):
        self.assertTrue(all(level > 2 for level in self._levels(0.0)), "nothing was subdivided, so the test proves nothing")

    def test_merging_is_what_removes_the_internal_boundaries(self):
        self.assertLess(sum(self._levels(1.0)), sum(self._levels(0.0)))

    def test_zero_never_reaches_the_merge_draws(self):
        """Consuming a random number at 0 would shift the stream of the paper run."""
        labels, image = sparse_pair()
        calls = []
        original = FN.random_merge_classes

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        with mock.patch.object(FN, "random_merge_classes", spy):
            generator(0.0)(labels, image=image)
            self.assertEqual(calls, [], "the faithful path drew merge patterns")

            generator(0.5)(labels, image=image)
            self.assertEqual(len(calls), 1, "the merge is not wired into the generator at all")

    def _classes_reaching_the_gmm(self, same_on_batch: bool) -> set[tuple[int, ...]]:
        labels, image = sparse_pair(batch=4)
        captured = []
        original = FN.sample_gmm_parameters

        def spy(*args, **kwargs):
            captured.append(kwargs["generation_classes"])
            return original(*args, **kwargs)

        gen = generator(0.5)
        gen.em_same_on_batch = same_on_batch
        with mock.patch.object(FN, "sample_gmm_parameters", spy):
            gen(labels, image=image)

        return {tuple(row) for row in captured[0].tolist()}

    def test_same_on_batch_shares_one_merge_pattern(self):
        """It shares the EM cluster count; sharing only half of the EM draws would surprise."""
        self.assertEqual(len(self._classes_reaching_the_gmm(same_on_batch=True)), 1)

    def test_otherwise_each_sample_merges_differently(self):
        self.assertGreater(len(self._classes_reaching_the_gmm(same_on_batch=False)), 1)


class TestShippedConfigs(SmaugLabTestCase):
    def test_the_paper_config_pins_the_faithful_zero(self):
        path = importlib.resources.files(configs) / "transform_params_paper-synthseg.json"
        synthseg = json.loads(path.read_text())["GPU"]["RandomSynthSegGPU"]

        self.assertEqual(synthseg["em_merge_prob"], 0.0)
        self.assertTrue(synthseg["em_label_completion"], "the knob is only read when EM completion is on")
