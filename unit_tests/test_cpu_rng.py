"""The CPU transforms must draw from the RNG a training run actually seeds.

`smauglab/transforms/rng.py` exists because `torch.manual_seed(...)` does not
seed Python's `random`, so a transform reaching for `random.choice` made a
"seeded" run unreproducible -- and under DDP each rank drew something different.
That was fixed on the GPU side and left in place on the CPU side:

* `cpu/spatial.py` picked its crop shape with `random.randint`
* `cpu/torchio_ops.py` picked which artifact to keep with `random.choice`

while the batchgeneratorsv2 `RandomTransform` wrapping both of them gates on
`torch.rand`. One pipeline, two RNG streams.

CI could not see it: `helpers.seed_everything` seeds torch, numpy *and* random,
which training does not. These tests seed **only** torch, which is what
`nnUNetTrainer` does.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from smauglab.transforms.cpu.spatial import ShapeTransform
from smauglab.transforms.cpu.torchio_ops import select

PACKAGE = Path(__file__).resolve().parent.parent / "smauglab"


class TestOnlyTorchSeedingIsNeeded(unittest.TestCase):
    """Seed torch alone -- as a training run does -- and expect reproducibility."""

    def _crop_shapes(self, draws=6):
        shapes = []
        torch.manual_seed(99)
        transform = ShapeTransform(shape_min=4)
        for _ in range(draws):
            image = torch.rand(1, 12, 12, 12)
            seg = torch.zeros(1, 12, 12, 12)
            out = transform(image=image, segmentation=seg)
            shapes.append(tuple(out["image"].shape))
        return shapes

    def test_the_crop_shape_is_reproducible_under_torch_seeding_alone(self):
        first = self._crop_shapes()
        second = self._crop_shapes()

        self.assertEqual(first, second)

    def test_the_crop_shape_actually_varies(self):
        """Otherwise the test above would pass on a transform that does nothing."""
        self.assertGreater(len(set(self._crop_shapes(12))), 1)

    def test_select_is_reproducible_under_torch_seeding_alone(self):
        flags = {"motion": True, "ghosting": True, "spike": True, "bias": True}

        def picks(n=8):
            torch.manual_seed(5)
            return [tuple(sorted(k for k, v in select(flags, random_pick=True).items() if v)) for _ in range(n)]

        self.assertEqual(picks(), picks())

    def test_select_actually_varies(self):
        flags = {"motion": True, "ghosting": True, "spike": True, "bias": True}
        torch.manual_seed(5)
        picks = {tuple(sorted(k for k, v in select(flags, random_pick=True).items() if v)) for _ in range(30)}

        self.assertGreater(len(picks), 1)

    def test_select_keeps_exactly_one(self):
        flags = {"motion": True, "ghosting": True, "spike": False}
        torch.manual_seed(5)

        chosen = select(flags, random_pick=True)

        self.assertEqual(sum(chosen.values()), 1)
        self.assertFalse(chosen["spike"], "a disabled entry must not be picked")


class TestNoStdlibRandomLeftInTheTransforms(unittest.TestCase):
    """A grep would catch a comment; this only counts real imports."""

    def test_no_transform_module_imports_random(self):
        for path in sorted((PACKAGE / "transforms").rglob("*.py")):
            tree = ast.parse(path.read_text())
            imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | {
                node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
            }

            with self.subTest(module=str(path.relative_to(PACKAGE.parent))):
                self.assertNotIn("random", imported, "use smauglab.transforms.rng instead of Python's random")


class TestCropHandlesAnOversizedMinimum(unittest.TestCase):
    def test_a_shape_min_larger_than_the_axis_does_not_raise(self):
        """`random.randint(shape_min, s)` raised when shape_min > s."""
        torch.manual_seed(0)
        transform = ShapeTransform(shape_min=64)

        out = transform(image=torch.rand(1, 8, 8, 8), segmentation=torch.zeros(1, 8, 8, 8))

        self.assertEqual(tuple(out["image"].shape), (1, 8, 8, 8))
