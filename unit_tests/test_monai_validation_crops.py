"""Validation must score the same patches every epoch.

`val_transforms` is a verbatim copy of `train_transforms`, including

    RandCropByPosNegLabeld(..., pos=3, neg=1, num_samples=3)

so each epoch validated on three freshly drawn, foreground-biased crops per
volume, with no fixed state. `if val_dsc > val_dsc_best: torch.save(...)` then
selected the epoch that drew the easiest crops as much as the best model -- the
model-selection signal carried a per-epoch crop lottery on top of it.

This is asserted on the source rather than by running the loop: `main()` needs
monai, wandb, a dataset on disk and a GPU, none of which are available to the
suite, and `RandCropByPosNegLabeld` is monai's. What can be checked cheaply is
the property the fix depends on -- that the reset happens, that it happens
*before* the validation pass, and that it uses the run's seed -- plus the
underlying principle, which is not monai-specific.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "scripts" / "train_monai.py"


def main_body() -> list[ast.stmt]:
    tree = ast.parse(SOURCE.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main").body


class TestValidationCropsAreFixed(unittest.TestCase):
    def _lines(self, needle: str) -> list[int]:
        return [i + 1 for i, line in enumerate(SOURCE.read_text().splitlines()) if needle in line]

    def test_the_validation_transforms_are_reset_each_epoch(self):
        resets = self._lines("val_transforms.set_random_state(")

        self.assertEqual(len(resets), 1, "the reset must happen exactly once, in the epoch loop")

    def test_the_reset_precedes_the_validation_pass(self):
        reset = self._lines("val_transforms.set_random_state(")[0]
        validate_call = self._lines("val_loss, val_dsc = validate(")[0]

        self.assertLess(reset, validate_call, "resetting after the pass would not fix anything")

    def test_the_reset_uses_the_run_seed(self):
        line = SOURCE.read_text().splitlines()[self._lines("val_transforms.set_random_state(")[0] - 1]

        self.assertIn("seed=seed", line, "a literal here would drift from the run's seed")

    def test_the_reset_is_inside_the_epoch_loop(self):
        """Once at construction would not do: each pass advances the RNG."""
        loops = [node for node in main_body() if isinstance(node, ast.For)]
        epoch_loop = next(loop for loop in loops if isinstance(loop.target, ast.Name) and loop.target.id == "epoch")

        calls = [
            node
            for node in ast.walk(epoch_loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_random_state"
        ]

        self.assertEqual(len(calls), 1)


class TestResettingRngMakesPassesAgree(unittest.TestCase):
    """The principle the fix relies on, without monai in the way."""

    class Sampler:
        def __init__(self):
            import random

            self._rng = random.Random()

        def set_random_state(self, seed):
            self._rng.seed(seed)

        def draw(self, n):
            return [self._rng.random() for _ in range(n)]

    def test_two_passes_differ_without_a_reset(self):
        sampler = self.Sampler()
        sampler.set_random_state(seed=42)

        self.assertNotEqual(sampler.draw(5), sampler.draw(5))

    def test_two_passes_agree_when_reset_first(self):
        sampler = self.Sampler()

        sampler.set_random_state(seed=42)
        first = sampler.draw(5)
        sampler.set_random_state(seed=42)
        second = sampler.draw(5)

        self.assertEqual(first, second)
