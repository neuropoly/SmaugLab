"""The shared seed has to survive the backend it is broadcast on.

`next_shared_seed` built its tensor with `torch.tensor([seed], dtype=torch.long)`
-- on the CPU, unconditionally -- and handed it to `dist.broadcast`. NCCL, the
default backend for GPU DDP and the exact case this module exists for, only
accepts CUDA tensors and raises on a CPU one. So the DDP branch failed on the
setup it was written for, while the gloo path (CPU tensors are fine there) kept
working.

The gloo cases below run a real single-process group, so the broadcast is really
executed. The NCCL device choice is asserted on `broadcast_device` with the
backend stubbed, because there is no CUDA and no second rank here -- and that is
said out loud rather than dressed up.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist

from smauglab.transforms import rng


@unittest.skipUnless(dist.is_available(), "torch.distributed is not available")
class TestSeedBroadcastOverGloo(unittest.TestCase):
    """A real process group, so `dist.broadcast` actually runs."""

    def setUp(self) -> None:
        if not dist.is_gloo_available():
            self.skipTest("the gloo backend is not available")
        # A path that does not exist yet: gloo's file:// rendezvous creates it.
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        store = Path(self._dir.name) / "rendezvous"
        dist.init_process_group(backend="gloo", init_method=f"file://{store}", world_size=1, rank=0)
        self.addCleanup(self._teardown)

    def _teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_a_seed_can_be_broadcast(self):
        torch.manual_seed(1234)

        seed = rng.next_shared_seed()

        self.assertIsInstance(seed, int)

    def test_successive_seeds_differ(self):
        torch.manual_seed(1234)

        self.assertNotEqual(rng.next_shared_seed(), rng.next_shared_seed())

    def test_shared_rand_goes_through_the_ddp_branch(self):
        torch.manual_seed(1234)

        drawn = rng.shared_rand((4,), torch.device("cpu"))

        self.assertEqual(drawn.shape, (4,))
        self.assertTrue(bool(((drawn >= 0) & (drawn < 1)).all()))

    def test_gloo_still_broadcasts_on_the_cpu(self):
        self.assertEqual(rng.broadcast_device().type, "cpu")


class TestBroadcastDeviceChoice(unittest.TestCase):
    """The NCCL half, which needs neither CUDA nor a second rank to pin."""

    def test_nccl_asks_for_a_cuda_tensor(self):
        with (
            mock.patch.object(dist, "get_backend", return_value=dist.Backend.NCCL),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
        ):
            device = rng.broadcast_device()

        self.assertEqual(device.type, "cuda", "NCCL rejects a CPU tensor, which is the whole bug")
        self.assertEqual(device.index, 0)

    def test_gloo_asks_for_a_cpu_tensor(self):
        with mock.patch.object(dist, "get_backend", return_value=dist.Backend.GLOO):
            self.assertEqual(rng.broadcast_device().type, "cpu")

    def test_nccl_without_cuda_falls_back_rather_than_raising(self):
        with (
            mock.patch.object(dist, "get_backend", return_value=dist.Backend.NCCL),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            self.assertEqual(rng.broadcast_device().type, "cpu")
