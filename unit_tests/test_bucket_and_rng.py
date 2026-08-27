"""`RandomChooseXTransformsGPU`, and the draws that `torch.manual_seed` did not reach.

* The bucket wrote into the caller's batch, and could not run any transform with a
  kornia parameter generator: calling `apply_transform` directly skips the
  `forward_parameters` step that fills `params`, so those raised
  "params must contain 'scale'".
* `RandomLowResTransformGPU` read `flags["data_keys"]` unguarded, which only the mask
  path injects -- so it raised `KeyError` standalone and inside a bucket.
* Blur sigmas and kernel sizes were drawn with Python's `random`, which
  `torch.manual_seed` does not reach and which diverges across DDP ranks.
"""

import torch

from smauglab.transforms.gpu.contrast import RandomLaplaceGPU, RandomRandConvGPU
from smauglab.transforms.gpu.spatial import RandomLowResTransformGPU
from smauglab.transforms.gpu.transforms_list import RandomChooseXTransformsGPU
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
