"""What the transforms do with a patch that carries no information.

Two shapes, both of which nnU-Net hands the network as a matter of course:

* **no foreground** -- the mask is entirely background. `oversample_foreground_percent`
  forces foreground for only *some* samples of a batch; the rest are placed at random,
  and on a whole-body CT roughly 18% of those land outside every label. A few cases are
  also effectively unlabelled, so every patch drawn from them is this.
* **no variance** -- the image is exactly constant. Preprocessing clips CT to the 0.5th
  percentile, so air outside the body is not merely dark, it is all one value.

Neither is exotic and neither used to be tested: `helpers.tiny_volume()` is random noise
and `helpers.tiny_seg()` always has a blob in the middle, so every sweep in this suite
was asking the easy question. The sweep in test_transforms_gpu.py now runs on the
degenerate pair too, but its bar is only "finite, and does not raise". These are the
behavioural tests -- what each transform is *supposed* to do -- because the two worst
defects here were perfectly finite and perfectly silent.
"""

import torch

from smauglab.transforms.cpu.fromSeg import aug_redistribute_seg
from smauglab.transforms.gpu.contrast import ZscoreNormalizationGPU
from smauglab.transforms.gpu.fromSeg import RandomRedistributeSegGPU
from smauglab.transforms.gpu.palette.transform import PaletteSynthesisGPU
from unit_tests.helpers import SmaugLabTestCase


class TestRedistributeSkipsWithoutForeground(SmaugLabTestCase):
    """`RandomRedistributeSegGPU` has nothing to redistribute without regions.

    It says so itself -- "Quick skip if no foreground" -- but the skip used to write back
    `x_batch[b]`, the [0, 1] min-max normalised image it had built for the real path, and
    `continue` past the `retain_stats` restore that would have undone it. A z-scored air
    patch at about [-2.74, -2.67] came out as [0.0, 1.0]: the mean moved by three units
    and the standard deviation by a factor of thirteen, on 40% of empty patches, with a
    perfectly finite result that no NaN guard could see.

    A skip has to be a no-op. That is the whole of these tests.
    """

    def _transform(self, **kwargs):
        return RandomRedistributeSegGPU(p=1.0, **kwargs)

    def _apply(self, volume, seg, **kwargs):
        transform = self._transform(**kwargs)
        return transform.apply_transform(volume.clone(), {"seg": seg}, transform.flags)

    def test_a_normal_image_with_an_empty_mask_is_untouched(self):
        """The case that matters most: real anatomy whose mask happens to be empty."""
        volume = self.tiny_volume()

        out = self._apply(volume, self.empty_seg())

        self.assertTrue(bool(torch.equal(out, volume)), "an empty mask must leave the image alone")

    def test_a_constant_image_with_an_empty_mask_is_untouched(self):
        volume = self.constant_volume()

        out = self._apply(volume, self.empty_seg())

        self.assertTrue(bool(torch.equal(out, volume)))

    def test_the_skip_holds_whatever_retain_stats_says(self):
        """`retain_stats` restores the statistics *after* the real path runs.

        The skip returns before it, so the flag cannot be what rescues the patch -- and
        when the input has zero variance there is no scale for it to restore anyway.
        """
        volume = self.tiny_volume()
        for retain_stats in (True, False):
            with self.subTest(retain_stats=retain_stats):
                out = self._apply(volume, self.empty_seg(), retain_stats=retain_stats)

                self.assertTrue(bool(torch.equal(out, volume)))

    def test_a_mask_with_foreground_still_redistributes(self):
        """The guard against fixing the skip by disabling the transform."""
        volume = self.tiny_volume()

        out = self._apply(volume, self.tiny_seg())

        self.assertFalse(bool(torch.equal(out, volume)), "the transform did nothing on a normal patch")
        self.assertTrue(bool(torch.isfinite(out).all()))


class TestCpuRedistributeSurvivesAConstantPatch(SmaugLabTestCase):
    """The CPU twin normalised by a range it never checked.

    `(img - img_min) / (img_max - img_min)` is 0/0 for a constant patch, so the whole
    volume came back NaN -- and unlike the GPU path, nothing downstream inspects it, so
    the NaN reached the loss. `GradScaler` then skips the step in silence, which is how a
    training run stops learning without a single line in the log.
    """

    def test_a_constant_patch_does_not_become_nan(self):
        volume, seg = self.constant_volume()[0], self.tiny_seg()[0]

        out, _ = aug_redistribute_seg(volume.clone(), seg)

        self.assertTrue(bool(torch.isfinite(out).all()), "a constant patch produced NaN")

    def test_a_constant_patch_with_an_empty_mask_does_not_become_nan(self):
        volume, seg = self.constant_volume()[0], self.empty_seg()[0]

        out, _ = aug_redistribute_seg(volume.clone(), seg)

        self.assertTrue(bool(torch.isfinite(out).all()))


class TestPaletteSynthesisSurvivesAConstantPatch(SmaugLabTestCase):
    """A constant image gives PALETTE an empty foreground and no regions to remap.

    `images_01` collapses to zero, so the `> dark_threshold` foreground test keeps
    nothing, so the Voronoi refinement never allocates a region and reports zero of them
    -- and `scatter_add_` into a zero-length tensor raises. On CUDA that is a device-side
    assert, which does not merely fail the batch: it poisons the context and takes the
    whole training run with it.

    Note the segmentation is irrelevant. A perfectly good mask crashed too, because it is
    the image that has no contrast.
    """

    def test_it_does_not_raise_with_an_empty_mask(self):
        transform = PaletteSynthesisGPU(p=1.0)
        volume = self.constant_volume()

        out = transform.apply_transform(volume.clone(), {"seg": self.empty_seg()}, transform.flags)

        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_it_does_not_raise_with_a_normal_mask(self):
        transform = PaletteSynthesisGPU(p=1.0)
        volume = self.constant_volume()

        out = transform.apply_transform(volume.clone(), {"seg": self.tiny_seg()}, transform.flags)

        self.assertTrue(bool(torch.isfinite(out).all()))


class TestZscoreLeavesAConstantChannelAlone(SmaugLabTestCase):
    """z-scoring a constant volume is meaningless; it must not be inventive about it.

    The `.clamp_min(1e-8)` on the standard deviation stops the NaN, but the mean is still
    *computed*, so what is left after subtracting it is floating-point residue -- and
    dividing that by 1e-8 multiplies it by a hundred million. A constant -2.709 patch
    came back as a constant -1.0, an O(1) value decided entirely by rounding.
    """

    def _apply(self, volume):
        transform = ZscoreNormalizationGPU(p=1.0)
        return transform.apply_transform(volume.clone(), {}, transform.flags)

    def test_a_constant_channel_is_returned_unchanged(self):
        for value in (-2.709, 0.0, 1.0, 100.0):
            with self.subTest(value=value):
                volume = self.constant_volume(value)

                out = self._apply(volume)

                self.assertTrue(bool(torch.equal(out, volume)), f"a constant {value} volume was rewritten")

    def test_a_normal_volume_is_still_z_scored(self):
        """The guard against fixing the constant case by disabling the transform."""
        out = self._apply(self.tiny_volume())

        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(out.std(unbiased=False)), 1.0, places=4)
