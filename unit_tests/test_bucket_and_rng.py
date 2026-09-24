"""`RandomChooseXTransformsGPU`, and the draws that `torch.manual_seed` did not reach.

* The bucket wrote into the caller's batch, and could not run any transform with a
  kornia parameter generator: calling `apply_transform` directly skips the
  `forward_parameters` step that fills `params`, so those raised
  "params must contain 'scale'".
* `RandomLowResTransformGPU` read `flags["data_keys"]` unguarded, which only the mask
  path injects -- so it raised `KeyError` standalone and inside a bucket.
* Blur sigmas and kernel sizes were drawn with Python's `random`, which
  `torch.manual_seed` does not reach and which diverges across DDP ranks.
* The bucket spent a generator transform's probability twice -- once on its own gate,
  then again inside `forward_parameters` -- so below `p=1.0` it both crashed on an
  empty parameter draw and, when it survived, applied the transform at `p**2`. Every
  test above builds its transforms at `p=1.0`, where the second draw always selects,
  which is exactly why it went unnoticed.
"""

import torch

from smauglab.config import write_temp_config
from smauglab.transforms.gpu.base import record_applications
from smauglab.transforms.gpu.contrast import RandomLaplaceGPU, RandomRandConvGPU
from smauglab.transforms.gpu.spatial import RandomAcqTransformGPU, RandomLowResTransformGPU
from smauglab.transforms.gpu.transforms_list import AugTransformsGPURandomOrder, RandomChooseXTransformsGPU
from smauglab.transforms.rng import shared_choice, shared_rand
from unit_tests.helpers import SmaugLabTestCase, first_output


class TestLowResRunsOutsideTheMaskPath(SmaugLabTestCase):
    def test_it_runs_standalone(self):
        """flags['data_keys'] is only injected by MaskSequentialOpsCustom."""
        torch.manual_seed(0)
        transform = RandomLowResTransformGPU(p=1.0)
        volume = self.tiny_volume()

        out = first_output(transform(volume))

        self.assertIsImageLike(out, volume, "RandomLowResTransformGPU")

    def test_it_runs_inside_a_random_choose_bucket(self):
        """The bucket calls apply_transform with the transform's own flags, which
        carry no data_keys either."""
        torch.manual_seed(0)
        bucket = RandomChooseXTransformsGPU(transforms_list=[RandomLowResTransformGPU(p=1.0)], num_transforms=1, p=1.0)
        volume = self.tiny_volume()

        out = bucket.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertIsImageLike(out, volume, "RandomLowResTransformGPU in a bucket")

    def test_an_explicit_mask_key_still_selects_nearest(self):
        """The branch that does exist must keep working."""
        from kornia.constants import DataKey

        torch.manual_seed(0)
        transform = RandomLowResTransformGPU(p=1.0)
        seg = self.tiny_seg()
        params = transform.forward_parameters(seg.shape)

        out = transform.apply_transform(seg.clone(), params, {"data_keys": [DataKey.MASK]}, transform=None)

        self.assertEqual(out.shape, seg.shape)
        self.assertTrue(bool(torch.isin(out, torch.tensor([0.0, 1.0])).all()), "a mask was resampled with interpolation")


class TestBucketDoesNotMutateItsInput(SmaugLabTestCase):
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

    def test_it_still_returns_something_transformed(self):
        """Cloning must not turn the bucket into a no-op."""
        torch.manual_seed(0)
        bucket = RandomChooseXTransformsGPU(
            transforms_list=[RandomLaplaceGPU(p=1.0)],
            num_transforms=1,
            p=1.0,
            same_on_batch=False,
        )
        volume = torch.rand(2, 1, 10, 10, 10)

        out = bucket.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertFalse(torch.allclose(out, volume, atol=1e-6))

    def test_an_empty_bucket_is_a_no_op(self):
        bucket = RandomChooseXTransformsGPU(transforms_list=[], num_transforms=0, p=1.0)
        volume = self.tiny_volume()
        self.assertTrue(torch.equal(bucket.apply_transform(volume.clone(), {}, {}, transform=None), volume))

    def test_the_same_on_batch_path_also_runs_a_generator_transform(self):
        torch.manual_seed(0)
        bucket = RandomChooseXTransformsGPU(
            transforms_list=[RandomLowResTransformGPU(p=1.0)],
            num_transforms=1,
            p=1.0,
            same_on_batch=True,
        )
        volume = self.tiny_volume()

        out = bucket.apply_transform(volume.clone(), {}, {}, transform=None)

        self.assertIsImageLike(out, volume, "bucket with same_on_batch")


class TestTorchSeedReachesEveryDraw(SmaugLabTestCase):
    """`torch.manual_seed` alone must be enough.

    The suite's own `seed_everything` seeds torch, numpy *and* Python's `random`, which
    is exactly why this went unnoticed -- training does not call it. These tests seed
    only torch.
    """

    def test_randconv_is_reproducible_under_torch_seed_alone(self):
        outputs = []
        for _ in range(2):
            torch.manual_seed(1234)
            transform = RandomRandConvGPU(p=1.0, kernel_sizes=[1, 3, 5, 7])
            outputs.append(transform.apply_transform(self.tiny_volume(), {}, {}, transform=None).clone())

        self.assertTrue(torch.equal(outputs[0], outputs[1]), "RandomRandConvGPU drew its kernel size from an unseeded generator")

    def test_shared_choice_covers_the_whole_sequence(self):
        torch.manual_seed(0)
        options = (1, 3, 5, 7)

        seen = {shared_choice(options) for _ in range(200)}

        self.assertEqual(seen, set(options))

    def test_shared_choice_is_reproducible(self):
        def draw():
            torch.manual_seed(7)
            return [shared_choice((1, 3, 5, 7)) for _ in range(20)]

        self.assertEqual(draw(), draw())

    def test_shared_choice_rejects_an_empty_sequence(self):
        with self.assertRaises(ValueError):
            shared_choice([])

    def test_shared_rand_is_plain_torch_rand_outside_ddp(self):
        """No process group initialised, so it must stay on the caller's generator."""
        torch.manual_seed(11)
        expected = torch.rand((4,))
        torch.manual_seed(11)

        self.assertTrue(torch.equal(shared_rand((4,), torch.device("cpu")), expected))


class TestProbabilityIsSpentOnce(SmaugLabTestCase):
    """A generator transform below p=1.0, which is the configuration that broke.

    `RandomAcqTransformGPU` is the only registered augmentation that reaches this path
    in a real pipeline -- the only one with a `_param_generator` that is neither GEO nor
    force_sequential, so the only one the builder puts inside a bucket.
    """

    P = 0.5
    DRAWS = 400

    def _bucket(self):
        return RandomChooseXTransformsGPU(transforms_list=[RandomAcqTransformGPU(p=self.P, scale=[0.3, 1.0])], num_transforms=1, p=1.0)

    def test_a_low_probability_generator_transform_never_raises(self):
        """It used to raise IndexError on the draws where the second gate said no."""
        torch.manual_seed(0)
        bucket = self._bucket()
        volume = self.tiny_volume()

        for draw in range(self.DRAWS):
            with self.subTest(draw=draw):
                out = bucket.apply_transform(volume.clone(), {}, {}, transform=None)
                self.assertIsImageLike(out, volume, "RandomAcqTransformGPU in a bucket")

    def test_the_transform_applies_at_p_not_p_squared(self):
        """The crash was half the bug; the surviving draws were also too rare.

        Bounds are wide because the rate is binomial and a scale that rounds back to
        the original size is a legitimate no-op, but p=0.5 and p**2=0.25 are far enough
        apart that the band separates them.
        """
        torch.manual_seed(0)
        bucket = self._bucket()
        volume = self.tiny_volume()

        applied = sum(not torch.allclose(bucket.apply_transform(volume.clone(), {}, {}, transform=None), volume) for _ in range(self.DRAWS))

        expected = self.P * self.DRAWS
        self.assertGreater(applied, 0.65 * expected, f"applied {applied}/{self.DRAWS}, near p**2 -- probability spent twice")
        self.assertLess(applied, 1.35 * expected, f"applied {applied}/{self.DRAWS}, above p -- gate not applied")

    def test_sampling_leaves_the_transforms_own_probability_alone(self):
        """The fix forces p during the draw; a leak would make the transform p=1.0."""
        torch.manual_seed(0)
        transform = RandomAcqTransformGPU(p=self.P, scale=[0.3, 1.0])

        RandomChooseXTransformsGPU._sample_params(transform, self.tiny_volume().shape)

        self.assertEqual(transform.p, self.P)
        self.assertEqual(transform.p_batch, 1.0)

    def test_sampling_restores_probability_even_when_the_draw_raises(self):
        """`finally`, not a bare restore after the call."""
        transform = RandomAcqTransformGPU(p=self.P, scale=[0.3, 1.0])

        with self.assertRaises(TypeError):
            RandomChooseXTransformsGPU._sample_params(transform, "not a shape")

        self.assertEqual(transform.p, self.P)


class TestProvenanceNamesWhatItRecorded(SmaugLabTestCase):
    """`record_applications` reports one entry per (transform, data key).

    Two things in its output read as bugs and are not, and both cost an afternoon
    before they were labelled. A 3D geometric transform appears twice because
    `AugmentationSequential` runs each module over the image and then the mask, and
    those transforms implement `apply_transform_mask` by calling `self.apply_transform`
    -- which is the recorder's own instance-attribute patch. And `random_order` emits
    two `RandomChooseXTransformsGPU`, the transfer bucket and the general-enhancement
    one, which the class name alone cannot tell apart.
    """

    def _pipeline(self):
        payload = {
            "GPU": {
                "RandomCropTransformGPU": {"p": 1.0, "crop": [0.5, 0.6]},
                "RandomGammaGPU": {"p": 1.0},
                "RandomGaussianNoiseGPU": {"p": 1.0},
            },
            "pipeline": {"mode": "random_order"},
        }
        return AugTransformsGPURandomOrder(str(write_temp_config(payload)))

    def _fired(self):
        pipeline = self._pipeline()
        with record_applications(pipeline) as fired:
            pipeline(self.tiny_volume(), self.tiny_seg())
        return fired

    def test_a_geometric_transform_is_recorded_once_per_data_key(self):
        entries = [e for e in self._fired() if e["transform"] == "RandomCropTransformGPU"]

        self.assertEqual([e["target"] for e in entries], ["image", "mask"])
        self.assertEqual(entries[0]["params"], entries[1]["params"], "image and mask moved on different draws")

    def test_an_intensity_transform_is_recorded_once(self):
        entries = [e for e in self._fired() if e["transform"] == "RandomGammaGPU"]

        self.assertEqual([e["target"] for e in entries], ["image"])

    def test_the_two_buckets_are_distinguishable(self):
        entries = [e for e in self._fired() if e["transform"] == "RandomChooseXTransformsGPU"]

        self.assertEqual([e["label"] for e in entries], ["ta", "ge"])

    def test_the_old_flag_name_still_reads(self):
        """`changed_image` predates `target` and analyses still load it."""
        for entry in self._fired():
            self.assertEqual(entry["changed_image"], entry["changed"])
