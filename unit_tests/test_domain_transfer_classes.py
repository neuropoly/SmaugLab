"""A label must reach the same LUT class whatever else is in the patch.

`RandomDomainTransferGPU` built its per-class weights with

    class_masks = seg_region_masks(seg, max_regions=NC)

which, for a single-channel label map, orders channels by the values that happen
to be *present*. `_accumulate` then indexes the bank's class axis by that
position. So a patch holding labels {0, 3, 7} sent label 3 to LUT class 1, while
a patch holding {0, 1, 3, 7} sent it to class 2 -- the bank's class axis is a
fixed taxonomy, so a class's transfer curve landed on whatever anatomy sorted
into its slot, and which anatomy that was changed from batch to batch.

The real LUT bank is hundreds of megabytes, built offline and deliberately not
shipped (`SMAUGLAB_DOMAIN_BANK`), so these tests build a synthetic one whose
class curves are trivially distinguishable: class c maps every intensity to the
constant c / NC. Then "which class did this voxel get" is readable straight off
the output, which is the only property under test here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unit_tests.helpers import SmaugLabTestCase

L = 32
NC = 4


def write_bank(directory) -> str:
    """A tiny .npz in the real bank's format, with trivially readable curves.

    Class c maps every intensity to the constant c / NC, so "which class did this
    voxel get" reads straight off the output -- which is the only property these
    tests are about. The real bank is hundreds of megabytes, built offline and
    deliberately not shipped (SMAUGLAB_DOMAIN_BANK), so it cannot be used here.
    """
    labels = ["a", "b"]
    curves = np.zeros((NC, L), dtype=np.float32)
    for c in range(NC):
        curves[c, :] = c / NC

    payload = {"labels": np.array(labels), "L": np.array(L), "num_classes": np.array(NC)}
    for x in labels:
        for y in labels:
            payload[f"{x}__{y}"] = curves.copy()

    path = str(Path(directory) / "bank.npz")
    np.savez(path, **payload)
    return path


def build(directory, **kwargs):
    from smauglab.transforms.gpu.domain_transfer import RandomDomainTransferGPU

    return RandomDomainTransferGPU(p=1.0, any_source=True, bank_path=write_bank(directory), **kwargs)


def patch_with(labels: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """A volume whose label `labels[i]` occupies slab i."""
    image = torch.rand(1, 1, 8, 8, 8)
    seg = torch.zeros(1, 1, 8, 8, 8)
    width = 8 // max(len(labels), 1)
    for i, value in enumerate(labels):
        seg[:, :, i * width : (i + 1) * width] = float(value)
    return image, seg


class TestLabelToClassMappingIsStable(SmaugLabTestCase):
    """The property that must hold whatever the bank's taxonomy is."""

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = directory.name

    def _class_field(self, transform, image, seg):
        out = transform.apply_transform(image.clone(), {"seg": seg}, transform.flags)
        # The synthetic curves are constants c / NC, so the output reads back as the
        # class index that dominated each voxel.
        return (out * NC).round()

    def test_a_label_lands_on_the_same_class_whatever_else_is_present(self):
        transform = build(self.tmp, sigma=0.0)

        torch.manual_seed(0)
        image_a, seg_a = patch_with([0, 3])
        torch.manual_seed(0)
        field_a = self._class_field(transform, image_a, seg_a)

        torch.manual_seed(0)
        image_b, seg_b = patch_with([0, 1, 3, 2])
        torch.manual_seed(0)
        field_b = self._class_field(transform, image_b, seg_b)

        # patch_with splits the volume evenly: with two labels the slabs are four
        # voxels deep, with four labels two. So label 3 is z[4:8] in A and z[4:6]
        # in B -- a different channel position in each, which is the whole point.
        label3_in_a = field_a[:, :, 4:8].flatten()
        label3_in_b = field_b[:, :, 4:6].flatten()

        self.assertAlmostEqual(
            float(label3_in_a.median()),
            float(label3_in_b.median()),
            places=4,
            msg="label 3 was mapped to a different LUT class depending on which other labels were present",
        )

    def test_each_label_maps_to_its_own_class(self):
        transform = build(self.tmp, sigma=0.0)
        torch.manual_seed(0)
        image, seg = patch_with([0, 1, 2, 3])

        torch.manual_seed(0)
        field = self._class_field(transform, image, seg)

        for index, label in enumerate([0, 1, 2, 3]):
            with self.subTest(label=label):
                slab = field[:, :, index * 2 : (index + 1) * 2]
                self.assertAlmostEqual(float(slab.median()), float(label), places=4)

    def test_the_output_stays_finite_and_shaped(self):
        transform = build(self.tmp)
        torch.manual_seed(0)
        image, seg = patch_with([0, 2])

        out = transform.apply_transform(image.clone(), {"seg": seg}, transform.flags)

        self.assertEqual(out.shape, image.shape)
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main()
