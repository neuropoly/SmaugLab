"""The reported DSC must not carry a phantom zero.

Both epoch loops seeded their accumulator with `dsc_list = [0]` and then
appended only scores above zero, so `np.mean(dsc_list)` always averaged in one
extra zero: ten batches at 0.8 report 0.727. The bias shrinks as the epoch
lengthens, which is why it reads as "our DSC is oddly low" rather than as a bug.

`val_dsc` is also what gates checkpoint saving (`if val_dsc > val_dsc_best`), so
the bias sat directly on model selection.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# See test_monai_epoch_metrics.py: monai and wandb are optional extras imported at
# module level, and train() needs neither.
for _optional in ("wandb", "monai", "monai.data", "monai.losses", "monai.networks.nets", "monai.transforms"):
    sys.modules.setdefault(_optional, MagicMock())


class PerfectModel(torch.nn.Module):
    """Predicts the target, so every batch scores the same known DSC."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.weight


class ZeroLoss(torch.nn.Module):
    def forward(self, prediction, target):
        return prediction.sum() * 0.0


def constant_batches(n, value=1.0, shape=(1, 1, 4, 4, 4)):
    """Identical batches, so every per-batch DSC is identical too."""
    target = torch.full(shape, value)
    return [{"image": target.clone(), "segmentation": target.clone()} for _ in range(n)]


class TestDscHasNoPhantomZero(unittest.TestCase):
    def _train(self, batches):
        import train_monai

        model = PerfectModel()
        return train_monai.train(
            batches,
            gpu_transforms=lambda x, y: (x, y),
            model=model,
            loss_func=ZeroLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            device=torch.device("cpu"),
        )

    def test_identical_batches_report_their_shared_score(self):
        """Every batch scores the same, so the epoch mean must equal that score."""
        from _common import compute_dsc

        batches = constant_batches(10)
        target = batches[0]["segmentation"].numpy()
        expected = compute_dsc(target, target, sigmoid=True)

        _, epoch_dsc = self._train(batches)

        self.assertAlmostEqual(epoch_dsc, expected, places=5)

    def test_the_phantom_zero_is_gone(self):
        """What the old accumulator would have reported, spelled out."""
        batches = constant_batches(10)

        _, epoch_dsc = self._train(batches)

        biased = float(np.mean([0] + [epoch_dsc] * 10))
        self.assertNotAlmostEqual(epoch_dsc, biased, places=5)

    def test_an_epoch_with_no_score_reports_nan(self):
        """0.0 would be indistinguishable from a real score of zero.

        Asserted on the helper rather than through `train`, because an empty
        loader also trips the separate epoch-loss defect (`loss` is the loop
        variable, so the return raises UnboundLocalError). That is fixed in its
        own change; this one is only about the accumulator.
        """
        import train_monai

        self.assertTrue(np.isnan(train_monai._dsc_mean([])))
        self.assertAlmostEqual(train_monai._dsc_mean([0.8, 0.8]), 0.8, places=6)
