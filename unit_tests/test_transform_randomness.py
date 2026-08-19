"""Transforms that were advertised as random but were not.

Each test here fails against the implementation that preceded it:

* `RandomFlipTransformGPU` never read the flip flags its generator sampled, so it
  flipped every configured axis, identically, on every call.
* The single-axis generators drew their axis in `make_samplers`, which kornia calls
  once and caches -- so "degrade a random axis" degraded the same axis for the whole
  run.
* `RandomLowResTransformGPU` read `flags["data_keys"]` unguarded, which only the mask
  path injects.
* `RandomChooseXTransformsGPU` wrote into the caller's batch.
* Several transforms drew from Python's `random`, which `torch.manual_seed` does not
  reach.
"""

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.spatial import CropGenerator3D, RandomFlipTransformGPU, RandomLowResTransformGPU, ScaleGenerator3D
from smauglab.transforms.gpu.transforms_list import RandomChooseXTransformsGPU
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

    def test_the_crop_generator_neutralises_position_at_the_centre(self):
        """The old code copied the crop's neutral 1.0 onto `pos`, i.e. the far edge."""
        torch.manual_seed(0)
        generator = CropGenerator3D(crop=(0.5, 0.9), pos=(0.2, 0.8), one_dim=True)
        generator.make_samplers(torch.device("cpu"), torch.float32)

        params = generator((4,), same_on_batch=False)
        crop, pos = params["crop"], params["pos"]

        for b in range(4):
            kept = (crop[b] != 1.0).nonzero().flatten().tolist()
            self.assertEqual(len(kept), 1)
            untouched = [axis for axis in range(3) if axis != kept[0]]
            for axis in untouched:
                self.assertAlmostEqual(float(pos[b, axis]), 0.5, places=5, msg="a non-cropped axis was not centred")


class TestLowResRunsOutsideTheMaskPath(SmaugLabTestCase):
    def test_it_runs_standalone(self):
        """flags['data_keys'] is only injected by MaskSequentialOpsCustom."""
        torch.manual_seed(0)
        transform = RandomLowResTransformGPU(p=1.0)
        out = transform(self.tiny_volume())
        self.assertIsImageLike(first_output(out), self.tiny_volume(), "RandomLowResTransformGPU")

    def test_it_runs_inside_a_random_choose_bucket(self):
        """The bucket calls apply_transform with the transform's own flags, which
        carry no data_keys either."""
        torch.manual_seed(0)
        bucket = RandomChooseXTransformsGPU(
            transforms_list=[RandomLowResTransformGPU(p=1.0)],
            num_transforms=1,
            p=1.0,
        )
        volume = self.tiny_volume()
        out = bucket.apply_transform(volume.clone(), {}, {}, transform=None)
        self.assertIsImageLike(out, volume, "RandomLowResTransformGPU in a bucket")


class TestRandomChooseDoesNotMutateItsInput(SmaugLabTestCase):
    def test_the_callers_tensor_is_left_alone(self):
        torch.manual_seed(0)
        bucket = RandomChooseXTransformsGPU(
            transforms_list=[RandomLowResTransformGPU(p=1.0)],
            num_transforms=1,
            p=1.0,
            same_on_batch=False,
        )
        volume = torch.rand(3, 1, 12, 12, 12)
        before = volume.clone()

        bucket.apply_transform(volume, {}, {}, transform=None)

        self.assertTrue(torch.equal(volume, before), "RandomChooseXTransformsGPU wrote into the caller's batch")

    def test_an_empty_bucket_is_a_no_op(self):
        bucket = RandomChooseXTransformsGPU(transforms_list=[], num_transforms=0, p=1.0)
        volume = self.tiny_volume()
        self.assertTrue(torch.equal(bucket.apply_transform(volume.clone(), {}, {}, transform=None), volume))


class TestTorchSeedReachesEveryDraw(SmaugLabTestCase):
    """`torch.manual_seed` alone must be enough; Python's `random` is seeded separately."""

    def _run_twice(self, build):
        outputs = []
        for _ in range(2):
            torch.manual_seed(1234)
            transform = build()
            outputs.append(first_output(transform(self.tiny_volume())).clone())
        return outputs

    def test_randconv_is_reproducible_under_torch_seed_alone(self):
        from smauglab.transforms.gpu.contrast import RandomRandConvGPU

        first, second = self._run_twice(lambda: RandomRandConvGPU(p=1.0, kernel_sizes=(1, 3, 5, 7)))
        self.assertTrue(torch.equal(first, second), "RandomRandConvGPU drew its kernel size from an unseeded generator")

    def test_shared_choice_covers_the_whole_sequence(self):
        from smauglab.transforms.rng import shared_choice

        torch.manual_seed(0)
        options = (1, 3, 5, 7)
        seen = {shared_choice(options) for _ in range(200)}
        self.assertEqual(seen, set(options))

    def test_shared_choice_rejects_an_empty_sequence(self):
        from smauglab.transforms.rng import shared_choice

        with self.assertRaises(ValueError):
            shared_choice([])
