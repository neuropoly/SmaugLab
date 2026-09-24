"""What `train_monai`'s seeding does and does not cover.

Two claims worth pinning, because they point in opposite directions and both are
easy to get wrong:

* `os.environ["PYTHONHASHSEED"] = str(seed)` inside `main()` does nothing. The
  interpreter reads that variable at startup, so by the time `main` runs the
  hash seed is already fixed. The line reads as though it made hashing
  deterministic.
* the dataloader workers, by contrast, need no seeding hook here. torch's
  `_worker_loop` derives a distinct seed per worker from the loader's generator
  and applies it to `random`, torch *and* numpy, so MONAI's random transforms
  already differ between workers and are already reproducible. Adding a
  `worker_init_fn` would be redundant, and this test says so rather than leaving
  the next reader to re-derive it.
"""

from __future__ import annotations

import random
import subprocess
import sys
import unittest

import numpy as np
import torch


class DrawsFromNumpy(torch.utils.data.Dataset):
    """Each item is a numpy and a stdlib draw, so shared worker state is visible."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor([np.random.rand(), random.random()])


def draws() -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    loader = torch.utils.data.DataLoader(DrawsFromNumpy(), batch_size=1, num_workers=2, generator=generator)
    return torch.cat(list(loader))


class TestPythonHashSeedIsNotSettableAtRuntime(unittest.TestCase):
    def test_setting_it_in_process_does_not_change_hashing(self):
        probe = "import os, sys; os.environ['PYTHONHASHSEED'] = '42'; print(os.environ['PYTHONHASHSEED'], sys.flags.hash_randomization)"

        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        value, randomization = result.stdout.split()
        self.assertEqual(value, "42", "the variable is set")
        self.assertEqual(randomization, "1", "and hash randomisation is still on, so the assignment changed nothing")


class TestWorkersAreAlreadySeeded(unittest.TestCase):
    def test_workers_draw_different_numbers(self):
        rows = draws()

        unique = {tuple(round(float(v), 12) for v in row) for row in rows}
        self.assertEqual(len(unique), len(rows), f"workers repeated draws: {rows}")

    def test_the_draws_are_reproducible(self):
        torch.testing.assert_close(draws(), draws())
