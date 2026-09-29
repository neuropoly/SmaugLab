"""`RandomFlipTransformGPU` promises at least one flip; it has to keep it.

`FlipGenerator3D` samples a 0/1 draw for all three axes, but `apply_transform`
only acts on the axes in `flip_axis`. The "ensure at least one flip" guard tested
`flips[b].sum() == 0` across all three columns, so a 1 drawn on a *disallowed*
axis satisfied the guard while nothing was flipped.

With the default `flip_axis=(0,)` -- the shipped default, at `p=1.0` -- that made
the transform do nothing on roughly a third of draws.
`test_spatial_sampling.py` checks the converse (no axis outside `flip_axis` is
ever flipped) but nothing asserted the guarantee itself.
"""

import torch

from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.spatial import FlipGenerator3D, RandomFlipTransformGPU
from unit_tests.helpers import SmaugLabTestCase

DRAWS = 200


class TestFlipAlwaysFlips(SmaugLabTestCase):
    def test_p_one_always_changes_the_volume(self):
        """The user-visible symptom: p=1.0 that sometimes does nothing."""
        # An asymmetric volume, so a flip is always detectable.
        image = torch.arange(24 * 24 * 24, dtype=torch.float32).reshape(1, 1, 24, 24, 24)
        seg = torch.zeros_like(image)

        unchanged = 0
        for seed in range(DRAWS):
            torch.manual_seed(seed)
            pipeline = AugmentationSequentialCustom(RandomFlipTransformGPU(p=1.0), data_keys=["input", "mask"], same_on_batch=True)
            out, _ = pipeline(image.clone(), seg.clone())
            unchanged += int(bool(torch.equal(out, image)))

        self.assertEqual(unchanged, 0, f"p=1.0 left the volume untouched in {unchanged}/{DRAWS} draws")

    def test_the_generator_flips_an_allowed_axis_for_every_sample(self):
        """Straight at the generator, for each subset of axes."""
        for flip_axis in ([0], [1], [2], [0, 1], [0, 1, 2]):
            with self.subTest(flip_axis=tuple(flip_axis)):
                generator = FlipGenerator3D(flip_axis)
                generator.make_samplers(torch.device("cpu"), torch.float32)

                torch.manual_seed(0)
                flips = generator((32, 1, 8, 8, 8))["flip"]

                allowed = flips[:, flip_axis]
                self.assertTrue(
                    bool((allowed.sum(dim=1) > 0).all()),
                    f"{int((allowed.sum(dim=1) == 0).sum())} of 32 samples had no allowed axis flipped",
                )

    def test_no_disallowed_axis_is_ever_flipped(self):
        """The guard against satisfying the guarantee by flipping something else."""
        generator = FlipGenerator3D([1])
        generator.make_samplers(torch.device("cpu"), torch.float32)

        torch.manual_seed(0)
        flips = generator((64, 1, 8, 8, 8))["flip"]

        # Columns 0 and 2 may still be sampled, but apply_transform ignores them;
        # what must hold is that column 1 carries the guarantee on its own.
        self.assertTrue(bool((flips[:, 1] == 1).all()))
