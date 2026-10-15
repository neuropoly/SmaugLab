"""`connected_components_3d` against scipy, which is the only reference that matters.

This is the one new algorithm in the region-selection feature, so it is checked against
`scipy.ndimage.label` rather than against its own output. scipy is an unconditional
runtime dependency (see pyproject.toml), so there is no skip guard.

Two things the tests pin that are easy to get wrong and silent when wrong:

* **The connectivity is 6, not 26.** The obvious GPU implementation is one
  `max_pool3d(kernel=3)`, which is a 3x3x3 neighbourhood and therefore merges blobs that
  touch only at a corner. `scipy.ndimage.label`'s default structuring element is
  `generate_binary_structure(3, 1)`, the six-neighbour cross, and that is also the
  conventional reading of "connected components of a label".
* **Components do not cross label boundaries**, and that holds in a single propagation
  pass rather than one pass per label value. A spine patch carries five to twenty-five
  labels, so the difference is not academic.

Partitions are compared, not ids: the numbering is an implementation detail and only the
grouping is the contract.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from smauglab.transforms.gpu.regions import connected_components_3d
from unit_tests.helpers import SmaugLabTestCase

#: scipy's default structuring element, spelled out because the default is the whole point.
SIX_CONNECTED = ndimage.generate_binary_structure(3, 1)


def as_batch(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A `[D, H, W]` bool mask as the `(labels, foreground)` pair the function takes."""
    labels = mask.long().view(1, 1, *mask.shape)
    return labels, labels > 0


def blobs(size: int = 20, seed: int = 0) -> torch.Tensor:
    """A smoothed random volume thresholded at its mean: many components, odd shapes."""
    torch.manual_seed(seed)
    smoothed = F.avg_pool3d(F.pad(torch.rand(1, 1, size, size, size), [3] * 6, mode="reflect"), 7, 1, 0)
    return smoothed[0, 0] > smoothed.mean()


def same_partition(ours: np.ndarray, theirs: np.ndarray, mask: np.ndarray) -> bool:
    """Whether two id maps group the masked voxels identically.

    True when the (ours, theirs) pairs present form a bijection: every one of our
    components maps to exactly one of theirs and vice versa. Comparing ids directly would
    only test that two numberings happen to agree.
    """
    pairs = set(zip(ours[mask].tolist(), theirs[mask].tolist()))
    ours_seen = {a for a, _ in pairs}
    theirs_seen = {b for _, b in pairs}
    return len(pairs) == len(ours_seen) == len(theirs_seen)


class TestAgreesWithScipy(SmaugLabTestCase):
    def test_random_blobs_partition_identically(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                mask = blobs(seed=seed)
                labels, foreground = as_batch(mask)

                ids, count = connected_components_3d(labels, foreground)
                reference, expected = ndimage.label(mask.numpy(), structure=SIX_CONNECTED)

                self.assertEqual(count, expected, "a different number of components than scipy found")
                self.assertTrue(
                    same_partition(ids[0, 0].numpy(), reference, mask.numpy()),
                    "the same count but a different grouping",
                )

    def test_a_multi_label_volume_matches_scipy_run_per_label(self):
        """One propagation pass must equal the sum of scipy's per-label runs."""
        for seed in range(4):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                smoothed = F.avg_pool3d(F.pad(torch.rand(1, 1, 20, 20, 20), [3] * 6, mode="reflect"), 7, 1, 0)[0, 0]
                labels = torch.zeros(20, 20, 20, dtype=torch.long)
                labels[smoothed > smoothed.mean()] = 1
                labels[smoothed > smoothed.mean() + 0.6 * smoothed.std()] = 2
                batched = labels.view(1, 1, 20, 20, 20)

                _ids, count = connected_components_3d(batched, batched > 0)
                expected = sum(ndimage.label((labels == value).numpy(), structure=SIX_CONNECTED)[1] for value in (1, 2))

                self.assertEqual(count, expected)


class TestTheConnectivityIsSix(SmaugLabTestCase):
    def test_two_voxels_touching_at_a_corner_are_two_components(self):
        """The test that pins 6- over 26-connectivity. A single max_pool3d merges these."""
        labels = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
        labels[0, 0, 1, 1, 1] = 1
        labels[0, 0, 2, 2, 2] = 1

        _ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 2, "corner-touching voxels were merged, so the neighbourhood is 26 not 6")
        self.assertEqual(ndimage.label((labels[0, 0] > 0).numpy(), structure=SIX_CONNECTED)[1], 2)

    def test_two_voxels_touching_at_a_face_are_one_component(self):
        labels = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
        labels[0, 0, 1, 1, 1] = 1
        labels[0, 0, 1, 1, 2] = 1

        _ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 1)


class TestLabelBoundariesAreRespected(SmaugLabTestCase):
    def test_two_adjacent_slabs_of_different_labels_stay_apart(self):
        """They touch across a face, so only the label gate keeps them separate."""
        labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        labels[0, 0, :, :, :4] = 1
        labels[0, 0, :, :, 4:] = 2

        ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 2, "the propagation crossed a label boundary")
        self.assertEqual(ids[0, 0, :, :, :4].unique().numel(), 1)
        self.assertEqual(ids[0, 0, :, :, 4:].unique().numel(), 1)
        self.assertNotEqual(int(ids[0, 0, 0, 0, 0]), int(ids[0, 0, 0, 0, 7]))


class TestLongThinStructuresConverge(SmaugLabTestCase):
    """The iteration-count test: pointer jumping is what makes these logarithmic.

    A serpentine ribbon has a geodesic length far beyond its Euclidean extent, so plain
    neighbour propagation would need one round per voxel along it.
    """

    def test_a_serpentine_ribbon_is_one_component(self):
        size = 24
        labels = torch.zeros(1, 1, size, size, size, dtype=torch.long)
        row = 0
        for plane in range(0, size, 2):
            labels[0, 0, plane, row, :] = 1
            labels[0, 0, plane : min(plane + 2, size), row, size - 1 if plane % 4 == 0 else 0] = 1
            row = min(row + 2, size - 1)

        ids, count = connected_components_3d(labels, labels > 0)
        reference, expected = ndimage.label((labels[0, 0] > 0).numpy(), structure=SIX_CONNECTED)

        self.assertEqual(count, expected)
        self.assertTrue(same_partition(ids[0, 0].numpy(), reference, (labels[0, 0] > 0).numpy()))


class TestDegenerateMasks(SmaugLabTestCase):
    def test_an_empty_mask_has_no_components_and_does_not_raise(self):
        labels = torch.zeros(1, 1, 5, 5, 5, dtype=torch.long)

        ids, count = connected_components_3d(labels, labels > 0)

        self.assertTrue(bool((ids == -1).all()), "background must be -1, not a unit id")
        self.assertGreaterEqual(count, 1, "the count sizes a draw and must never be zero")

    def test_a_full_volume_is_one_component(self):
        labels = torch.ones(1, 1, 5, 5, 5, dtype=torch.long)

        ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 1)
        self.assertEqual(ids.unique().tolist(), [0])

    def test_a_single_voxel_is_one_component(self):
        labels = torch.zeros(1, 1, 5, 5, 5, dtype=torch.long)
        labels[0, 0, 2, 2, 2] = 1

        _ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 1)


class TestPerSampleNumbering(SmaugLabTestCase):
    """Ids are 0-based within each sample, because the keep draw is per sample."""

    def test_each_sample_is_numbered_from_zero(self):
        labels = torch.zeros(2, 1, 6, 6, 6, dtype=torch.long)
        labels[0, 0, 0, 0, 0] = 1
        labels[1, 0, 0, 0, 0] = 1
        labels[1, 0, 2, 2, 2] = 1
        labels[1, 0, 4, 4, 4] = 1

        ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 3, "the count is the largest per-sample count, which sizes the draw")
        self.assertEqual(ids[0, 0][labels[0, 0] > 0].tolist(), [0])
        self.assertEqual(sorted(ids[1, 0][labels[1, 0] > 0].tolist()), [0, 1, 2])

    def test_two_samples_rooted_at_the_same_index_are_not_merged(self):
        """Seeds are globally unique for exactly this reason."""
        labels = torch.zeros(2, 1, 4, 4, 4, dtype=torch.long)
        labels[:, 0, 0, 0, 0] = 1

        ids, count = connected_components_3d(labels, labels > 0)

        self.assertEqual(count, 1)
        self.assertEqual(int(ids[0, 0, 0, 0, 0]), 0)
        self.assertEqual(int(ids[1, 0, 0, 0, 0]), 0)


class TestIdsAreUsableAsIndices(SmaugLabTestCase):
    """`region_alpha` gathers a per-unit draw with these, so they must be dense."""

    def test_ids_are_contiguous_from_zero(self):
        mask = blobs(seed=3)
        labels, foreground = as_batch(mask)

        ids, count = connected_components_3d(labels, foreground)

        present = ids[foreground].unique().tolist()
        self.assertEqual(present, list(range(count)))


if __name__ == "__main__":
    import unittest

    unittest.main()
