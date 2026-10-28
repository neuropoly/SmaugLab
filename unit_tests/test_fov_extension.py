"""`RandomFOVExtensionGPU`: shift the scan inside the patch, fill what lies outside it.

The fixture is a small sagittal-slab stand-in: tissue in [1, 2], a slab of exact zeros at
both ends of the first axis (nnU-Net's padding), and a labelled block at intensity 1000.
The sentinel makes two properties checkable voxel by voxel: the image and the mask moved
together (1000 sits exactly under the label), and no foreground intensity reached the
synthesised background (nothing else comes near 1000).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.context import RandomFOVExtensionGPU, _dilate, _em_regions, _shift
from unit_tests.helpers import SmaugLabTestCase

SHAPE = (3, 1, 28, 24, 20)  # non-cubic on purpose
SENTINEL = 1000.0


def slab(shape=SHAPE):
    image = torch.rand(shape) + 1.0
    image[:, :, :6] = 0.0
    image[:, :, -6:] = 0.0
    seg = torch.zeros(shape)
    seg[:, :, 10:18, 6:16, 7:13] = 1.0
    image[seg.bool()] = SENTINEL
    return image, seg


def run(transform, image, seg, same_on_batch=True):
    pipeline = AugmentationSequentialCustom(transform, data_keys=["input", "mask"], same_on_batch=same_on_batch)
    return pipeline(image.clone(), seg.clone())


class TestShiftAndFill(SmaugLabTestCase):
    def test_image_and_mask_move_together(self):
        image, seg = slab()
        for seed in range(6):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                out, out_seg = run(RandomFOVExtensionGPU(shift=1.0, p=1.0), image, seg)
                self.assertTrue(bool(torch.equal(out > SENTINEL / 2, out_seg > 0)))

    def test_the_shift_moves_something(self):
        """The guard against passing the alignment test by never shifting."""
        image, seg = slab()
        torch.manual_seed(0)
        _, out_seg = run(RandomFOVExtensionGPU(shift=1.0, p=1.0), image, seg)
        self.assertFalse(bool(torch.equal(out_seg, seg)))

    def test_outside_the_fill_is_the_shifted_input(self):
        image, seg = slab()
        transform = RandomFOVExtensionGPU(shift=0.5, p=1.0)
        torch.manual_seed(3)
        params = transform.generate_parameters(image.shape)
        params["seg"] = seg
        out = transform.apply_transform(image, params, transform.flags)
        offsets = transform._offsets(params, image.shape[2:])
        shifted = _shift(image, offsets, 0.0)
        scan = _shift(torch.ones_like(image, dtype=torch.bool), offsets, False) & (shifted != 0.0)
        self.assertTrue(bool(torch.equal(out[scan], shifted[scan])))

    def test_no_shift_touches_only_the_padding(self):
        image, seg = slab()
        out, out_seg = run(RandomFOVExtensionGPU(shift=0.0, p=1.0), image, seg)
        scan = image != 0.0
        self.assertTrue(bool(torch.equal(out_seg, seg)))
        self.assertTrue(bool(torch.equal(out[scan], image[scan])))
        self.assertEqual(int((out[~scan] == 0.0).sum()), 0, "padding left unfilled")

    def test_no_foreground_intensity_reaches_the_fill(self):
        image, seg = slab()
        for margin in (0, 2):
            with self.subTest(margin=margin):
                torch.manual_seed(1)
                out, out_seg = run(RandomFOVExtensionGPU(shift=1.0, margin=margin, p=1.0), image, seg)
                background = out[out_seg == 0]
                self.assertLess(float(background.max()), 10.0)
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_unselected_rows_are_untouched(self):
        image, seg = slab()
        torch.manual_seed(0)
        out, out_seg = run(RandomFOVExtensionGPU(p=0.0), image, seg)
        self.assertTrue(bool(torch.equal(out, image)))
        self.assertTrue(bool(torch.equal(out_seg, seg)))

    def test_no_background_still_shifts_consistently(self):
        """Nothing to fit: the fill stays at pad_value, and the mask still follows the image."""
        image, seg = slab()
        seg[:, :, 6:-6] = 1.0
        image[seg.bool()] = SENTINEL
        torch.manual_seed(2)
        out, out_seg = run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg)
        self.assertTrue(bool(torch.equal(out > SENTINEL / 2, out_seg > 0)))
        self.assertTrue(bool(((out == 0.0) | (out == SENTINEL)).all()))

    def test_without_a_segmentation_every_scan_voxel_is_a_source(self):
        image, _ = slab()
        transform = RandomFOVExtensionGPU(shift=0.0, p=1.0)
        params = transform.generate_parameters(image.shape)
        out = transform.apply_transform(image, params, transform.flags)
        self.assertEqual(int((out == 0.0).sum()), 0)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_padding_means_every_channel(self):
        image, seg = slab((2, 2, 28, 24, 20))
        image[:, 1, :6] = 5.0  # channel 1 is not padding there, so neither is the voxel
        out, _ = run(RandomFOVExtensionGPU(shift=0.0, p=1.0), image, seg)
        self.assertTrue(bool(torch.equal(out[:, :, :6], image[:, :, :6])))
        self.assertEqual(int((out[:, :, -6:] == 0.0).all(dim=1).sum()), 0)


class TestSelectionAndDeterminism(SmaugLabTestCase):
    def test_rows_are_selected_per_sample_under_same_on_batch(self):
        """The paper config hands every child same_on_batch=True; this transform ignores it."""
        image, seg = slab((8, 1, 28, 24, 20))
        counts = []
        for seed in range(8):
            torch.manual_seed(seed)
            transform = RandomFOVExtensionGPU(shift=0.0, p=0.5)
            out, _ = run(transform, image, seg, same_on_batch=True)
            self.assertFalse(transform.same_on_batch)
            counts.append(int((out != image).flatten(1).any(dim=1).sum()))
        self.assertTrue(any(0 < c < 8 for c in counts), counts)

    def test_seed_determinism(self):
        image, seg = slab()
        results = []
        for _ in range(2):
            torch.manual_seed(11)
            results.append(run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg))
        self.assertTrue(bool(torch.equal(results[0][0], results[1][0])))
        self.assertTrue(bool(torch.equal(results[0][1], results[1][1])))

    def test_autocast_matches_fp32(self):
        image, seg = slab()
        torch.manual_seed(5)
        ref, ref_seg = run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg)
        torch.manual_seed(5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out, out_seg = run(RandomFOVExtensionGPU(shift=0.5, p=1.0), image, seg)
        self.assertTrue(bool(torch.equal(out_seg, ref_seg)))
        self.assertTrue(bool(torch.allclose(out.float(), ref, atol=1e-5)))

    def test_rejects_bad_arguments(self):
        for kwargs in ({"shift": 1.5}, {"shift": (0.5, 0.5)}, {"regions": (0, 3)}, {"regions": (5, 3)}, {"layers": 0}, {"margin": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                RandomFOVExtensionGPU(**kwargs)


class TestHelpers(SmaugLabTestCase):
    def test_em_separates_two_populations(self):
        x = torch.cat([torch.randn(1, 4000) * 0.1, torch.randn(1, 4000) * 0.1 + 5.0], dim=1)
        assign = _em_regions(x, torch.ones_like(x), torch.tensor([2]), k_max=4)
        first, second = assign[0, :4000], assign[0, 4000:]
        self.assertGreater(float((first == first.mode().values).float().mean()), 0.99)
        self.assertGreater(float((second == second.mode().values).float().mean()), 0.99)
        self.assertNotEqual(int(first.mode().values), int(second.mode().values))
        self.assertLess(int(assign.max()), 2, "a masked-out component was assigned")

    def test_em_fits_only_weighted_voxels(self):
        """With weight on one population only, both components land inside it."""
        x = torch.cat([torch.randn(1, 4000) * 0.1, torch.randn(1, 4000) * 0.1 + 5.0], dim=1)
        weight = torch.cat([torch.ones(1, 4000), torch.zeros(1, 4000)], dim=1)
        assign = _em_regions(x, weight, torch.tensor([2]), k_max=2)
        self.assertEqual(int(assign[0, 4000:].unique().numel()), 1)

    def test_dilate_matches_a_dense_pool(self):
        mask = torch.rand(2, 1, 12, 10, 9) > 0.95
        for radius in (0, 1, 3):
            with self.subTest(radius=radius):
                k = 2 * radius + 1
                dense = F.max_pool3d(mask.float(), k, stride=1, padding=radius) > 0
                self.assertTrue(bool(torch.equal(_dilate(mask, radius), dense)))

    def test_shift_beyond_the_patch_empties_the_row(self):
        volume = torch.ones(1, 1, 4, 4, 4)
        self.assertTrue(bool((_shift(volume, [[0, 4, 0]], 0.0) == 0).all()))
        self.assertTrue(bool(torch.equal(_shift(volume, [[0, 0, 0]], 0.0), volume)))
