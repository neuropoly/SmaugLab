"""Resolution simulation must not touch the segmentation.

`RandomLowResTransformGPU` down- and up-samples the image to model a thicker
slice. It inherits `RigidAffineAugmentationBase3D`, so `MaskSequentialOpsCustom`
routes the mask through `apply_transform_mask` -- and that override used to
forward straight to `apply_transform`, nearest-resampling the label map too.

Anatomy does not move when the acquisition gets coarser, so this corrupted the
training target. `RandomAcqTransformGPU`, the single-axis form of the same
operation, is an `ImageOnlyTransform` and always got this right, which is the
control case below.

The existing sweep could not catch it: `test_transforms_gpu.py::
test_transform_leaves_the_mask_intact` only asserts `unique(mask).numel() <= 2`,
and a nearest-resampled binary mask is still binary.
"""

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.spatial import RandomAcqTransformGPU, RandomLowResTransformGPU
from unit_tests.helpers import SmaugLabTestCase


class TestResolutionSimulationLeavesTheMaskAlone(SmaugLabTestCase):
    def _run(self, cls, seed):
        torch.manual_seed(seed)
        pipeline = AugmentationSequentialCustom(cls(p=1.0), data_keys=["input", "mask"], same_on_batch=True)
        image, seg = self.tiny_volume(), self.tiny_seg()
        out_image, out_seg = pipeline(image.clone(), seg.clone())
        return image, seg, out_image, out_seg

    def test_the_mask_is_returned_bit_identical(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                _, seg, _, out_seg = self._run(RandomLowResTransformGPU, seed)

                self.assertTrue(bool(torch.equal(out_seg, seg)), "the segmentation was resampled along with the image")

    def test_the_foreground_count_is_unchanged(self):
        """The symptom that made this visible: the label grew by about 7%."""
        _, seg, _, out_seg = self._run(RandomLowResTransformGPU, 3)

        self.assertEqual(int(out_seg.sum()), int(seg.sum()))

    def test_the_image_is_still_degraded(self):
        """The guard against fixing the mask by disabling the transform."""
        image, _, out_image, _ = self._run(RandomLowResTransformGPU, 3)

        self.assertFalse(bool(torch.equal(out_image, image)), "the transform stopped doing anything to the image")

    def test_the_single_axis_sibling_agrees(self):
        """`RandomAcqTransformGPU` is the same operation and always left the mask alone."""
        _, seg, out_image, out_seg = self._run(RandomAcqTransformGPU, 3)

        self.assertTrue(bool(torch.equal(out_seg, seg)))
