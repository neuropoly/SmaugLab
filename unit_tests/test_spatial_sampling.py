"""Spatial transforms that were advertised as random but were not.

The first two groups fail against the implementation that preceded them:

* `RandomFlipTransformGPU` never read the flip flags its own generator sampled, so it
  flipped every configured axis, identically, on every call.
* The single-axis generators drew their axis in `make_samplers`, which kornia calls
  once and caches -- so "degrade a random axis" degraded the same axis for a whole
  training run, and `CropGenerator3D` neutralised the crop *position* to the far edge
  instead of the centre.

`TestMaskResampleRestore` is a guard rather than a regression test; see its docstring.
"""

import torch
from kornia.constants import Resample

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.spatial import CropGenerator3D, RandomAffine3DCustom, RandomFlipTransformGPU, ScaleGenerator3D
from unit_tests.helpers import SmaugLabTestCase, first_output


class TestFlipIsActuallyRandom(SmaugLabTestCase):
    def _pipeline(self, **kwargs):
        return AugmentationSequentialCustom(
            RandomFlipTransformGPU(p=1.0, **kwargs),
            data_keys=["input", "mask"],
            same_on_batch=False,
        )

    def test_two_seeds_give_two_different_flips(self):
        volume, seg = self.tiny_volume(), self.tiny_seg()
        pipeline = self._pipeline(flip_axis=(0, 1, 2))

        outputs = []
        for seed in range(12):
            torch.manual_seed(seed)
            outputs.append(first_output(pipeline(volume.clone(), seg.clone())).clone())

        distinct = {tuple(out.flatten()[:64].tolist()) for out in outputs}
        self.assertGreater(len(distinct), 1, "every seed produced the same flip -- params['flip'] is being ignored")

    def test_batch_elements_flip_independently(self):
        """The generator samples [B, 3]; the loop used to discard it and flip all of them."""
        torch.manual_seed(0)
        volume = torch.rand(8, 1, 8, 8, 8)
        seg = torch.zeros(8, 1, 8, 8, 8)
        pipeline = self._pipeline(flip_axis=(0, 1, 2))

        out = first_output(pipeline(volume.clone(), seg.clone()))
        flipped = [bool(torch.allclose(out[b], torch.flip(volume[b], dims=(1, 2, 3)))) for b in range(8)]

        self.assertIn(False, flipped, "every batch element got the identical all-axis flip")

    def test_the_mask_is_flipped_the_same_way_as_the_image(self):
        """Image and mask read the same params['flip'], so they cannot disagree."""
        torch.manual_seed(3)
        volume = torch.rand(4, 1, 8, 8, 8)
        seg = (volume > 0.5).float()

        result = self._pipeline(flip_axis=(0, 1, 2))(volume.clone(), seg.clone())
        image, mask = result[0], result[1]

        self.assertTrue(torch.equal((image > 0.5).float(), mask), "the mask was flipped differently from the image")

    def test_only_configured_axes_are_ever_flipped(self):
        torch.manual_seed(1)
        volume = torch.rand(6, 1, 8, 8, 8)
        seg = torch.zeros(6, 1, 8, 8, 8)

        out = first_output(self._pipeline(flip_axis=(0,))(volume.clone(), seg.clone()))
        for b in range(6):
            unchanged = torch.allclose(out[b], volume[b])
            flipped_axis0 = torch.allclose(out[b], torch.flip(volume[b], dims=(1,)))
            self.assertTrue(unchanged or flipped_axis0, f"batch element {b} was flipped along an axis that was not configured")

    def test_apply_transform_without_params_still_flips_every_configured_axis(self):
        """The fallback for callers that reach past the generator into apply_transform."""
        transform = RandomFlipTransformGPU(p=1.0, flip_axis=(0, 1, 2))
        volume = torch.rand(2, 1, 8, 8, 8)

        out = transform.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertTrue(torch.equal(out, torch.flip(volume, dims=(2, 3, 4))))


class TestSingleAxisIsRedrawnEveryCall(SmaugLabTestCase):
    """The axis used to be chosen in make_samplers, which kornia calls once."""

    def test_the_scale_generator_does_not_pin_one_axis_forever(self):
        torch.manual_seed(0)
        generator = ScaleGenerator3D(scale=(0.3, 1.0), one_dim=True)
        generator.make_samplers(torch.device("cpu"), torch.float32)

        degraded_axes = set()
        for _ in range(20):
            scale = generator((4,), same_on_batch=False)["scale"]
            # Exactly one axis per row is scaled; the rest sit at the neutral 1.0.
            for row in scale:
                self.assertEqual(int((row != 1.0).sum()), 1, "more than one axis was degraded")
                degraded_axes.add(int((row != 1.0).nonzero().item()))

        self.assertGreater(len(degraded_axes), 1, "the same axis was degraded on every call")

    def test_same_on_batch_degrades_one_shared_axis(self):
        torch.manual_seed(0)
        generator = ScaleGenerator3D(scale=(0.3, 1.0), one_dim=True)
        generator.make_samplers(torch.device("cpu"), torch.float32)

        scale = generator((5,), same_on_batch=True)["scale"]
        chosen = {int((row != 1.0).nonzero().item()) for row in scale}
        self.assertEqual(len(chosen), 1, "same_on_batch should pick one axis for the whole batch")

    def test_isotropic_scaling_is_left_alone(self):
        """one_dim=False must keep every axis independent, as before."""
        torch.manual_seed(0)
        generator = ScaleGenerator3D(scale=(0.3, 0.9), one_dim=False)
        generator.make_samplers(torch.device("cpu"), torch.float32)

        scale = generator((4,), same_on_batch=False)["scale"]
        self.assertTrue(bool((scale != 1.0).all()), "one_dim=False should not neutralise any axis")

    def test_the_crop_generator_uses_one_axis_for_crop_and_position(self):
        """make_samplers drew a separate axis for each, so they could disagree."""
        torch.manual_seed(0)
        generator = CropGenerator3D(crop=(0.5, 0.9), pos=(0.2, 0.8), one_dim=True)
        generator.make_samplers(torch.device("cpu"), torch.float32)

        params = generator((4,), same_on_batch=False)
        crop, pos = params["crop"], params["pos"]

        for b in range(4):
            cropped = (crop[b] != 1.0).nonzero().flatten().tolist()
            positioned = (pos[b] != 0.5).nonzero().flatten().tolist()
            self.assertEqual(len(cropped), 1)
            self.assertEqual(cropped, positioned, "the crop and its position were placed on different axes")

    def test_the_crop_generator_neutralises_position_at_the_centre(self):
        """The old code copied the crop's neutral 1.0 onto `pos`, i.e. the far edge."""
        torch.manual_seed(0)
        generator = CropGenerator3D(crop=(0.5, 0.9), pos=(0.2, 0.8), one_dim=True)
        generator.make_samplers(torch.device("cpu"), torch.float32)

        params = generator((4,), same_on_batch=False)
        crop, pos = params["crop"], params["pos"]

        for b in range(4):
            kept = (crop[b] != 1.0).nonzero().flatten().tolist()
            untouched = [axis for axis in range(3) if axis != kept[0]]
            for axis in untouched:
                self.assertAlmostEqual(float(pos[b, axis]), 0.5, places=5, msg="a non-cropped axis was not centred")


class TestMaskResampleRestore(SmaugLabTestCase):
    """Guard tests for `RandomAffine3DCustom.apply_transform_mask`.

    Unlike the rest of this file these pass before the change too: `resample_method`
    was annotated but assigned only inside the `if`, so the restore below it could
    read an unbound local -- except that `apply_transform` indexes `flags["resample"]`
    unguarded and raises `KeyError` first, which makes the unbound local unreachable
    rather than harmless. `flags.get(...)` removes the hazard; these tests pin the
    behaviour that has to survive it.
    """

    def _transform(self):
        return RandomAffine3DCustom(p=1.0, degrees=5, align_corners=True)

    def _flags(self, resample: str = "bilinear"):
        transform = self._transform()
        flags = dict(transform.flags)
        flags["resample"] = Resample.get(resample)
        return transform, flags

    def test_a_mask_is_resampled_and_keeps_its_shape(self):
        transform, flags = self._flags()
        seg = self.tiny_seg()
        params = transform.forward_parameters(seg.shape)
        matrix = transform.compute_transformation(seg, params, flags)

        out = transform.apply_transform_mask(seg.clone(), params, flags, transform=matrix)

        self.assertEqual(out.shape, seg.shape)

    def test_the_callers_resample_mode_is_restored(self):
        """The method flips the flag to "nearest" for the mask and must put it back."""
        transform, flags = self._flags()
        seg = self.tiny_seg()
        params = transform.forward_parameters(seg.shape)
        matrix = transform.compute_transformation(seg, params, flags)

        transform.apply_transform_mask(seg.clone(), params, flags, transform=matrix)

        self.assertEqual(flags["resample"], Resample.get("bilinear"))
