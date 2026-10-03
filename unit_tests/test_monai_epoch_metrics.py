"""`train_monai`'s epoch loss has to be the epoch's loss.

Both loops ended with

    return loss.mean().item(), np.mean(dsc_list)

where `loss` is the loop variable -- so the value logged to wandb as
`Loss_train/epoch` / `Loss_val/epoch` was the **last mini-batch's** loss, while
the DSC returned beside it was accumulated over the epoch. On an empty loader it
raised UnboundLocalError instead.
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

# `train_monai` imports monai and wandb at module level; both are optional extras
# and neither is installed for the test run. `train()` itself uses only torch,
# numpy, tqdm and `_common.compute_dsc` (pure numpy), so stubbing the two heavy
# imports lets the real function be driven rather than skipped.
for _optional in ("wandb", "monai", "monai.data", "monai.losses", "monai.networks.nets", "monai.transforms"):
    sys.modules.setdefault(_optional, MagicMock())


class ConstantLoss(torch.nn.Module):
    """A known loss per batch, so the epoch mean is something we can assert."""

    def __init__(self, values):
        super().__init__()
        self.values = list(values)
        self.calls = 0

    def forward(self, prediction, target):
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        # Keep it attached to the graph so scaler.scale(loss).backward() works.
        return prediction.sum() * 0.0 + value


class Identity3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.weight


def batches(n, shape=(1, 1, 4, 4, 4)):
    for _ in range(n):
        yield {"image": torch.rand(*shape), "segmentation": (torch.rand(*shape) > 0.5).float()}


class TestEpochLossIsTheEpochMean(unittest.TestCase):
    def _train(self, losses, batch_count):
        import train_monai

        model = Identity3D()
        loss_func = ConstantLoss(losses)
        return train_monai.train(
            list(batches(batch_count)),
            gpu_transforms=lambda x, y: (x, y),
            model=model,
            loss_func=loss_func,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            device=torch.device("cpu"),
        )

    def test_the_returned_loss_is_the_mean_not_the_last_batch(self):
        losses = [1.0, 2.0, 9.0]

        epoch_loss, _ = self._train(losses, batch_count=3)

        self.assertAlmostEqual(epoch_loss, float(np.mean(losses)), places=5)
        self.assertNotAlmostEqual(epoch_loss, losses[-1], places=5)

    def test_an_empty_loader_does_not_raise(self):
        import train_monai

        model = Identity3D()
        epoch_loss, _ = train_monai.train(
            [],
            gpu_transforms=lambda x, y: (x, y),
            model=model,
            loss_func=ConstantLoss([1.0]),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            device=torch.device("cpu"),
        )

        self.assertTrue(np.isnan(epoch_loss), "an empty epoch should report nan, not raise")
