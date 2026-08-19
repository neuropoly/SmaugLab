"""Shared helpers and the base TestCase for the SmaugLab test suite.

Everything here runs on CPU with tiny volumes, so the whole suite stays fast
enough to gate every pull request. No GPU and no image data on disk required.
"""

from __future__ import annotations

import importlib.resources
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch

# Small enough to be fast, large enough that the spatial transforms (crop,
# low-res simulation, flips) still have something to work with.
VOLUME_SHAPE = (1, 1, 24, 24, 24)
SEED = 1234


def seed_everything(seed: int = SEED) -> None:
    """Pin every RNG the transforms reach for.

    Some transforms use the stdlib and numpy generators, not just torch's, so
    all three have to be set or results are not reproducible between runs.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class SmaugLabTestCase(unittest.TestCase):
    """Base class that seeds the RNGs and hands out the standard test volumes."""

    def setUp(self) -> None:
        super().setUp()
        seed_everything()

    def tiny_volume(self) -> torch.Tensor:
        """A [N, C, D, H, W] float image in roughly the range the transforms expect."""
        return torch.rand(*VOLUME_SHAPE, dtype=torch.float32)

    def tiny_seg(self) -> torch.Tensor:
        """A binary segmentation mask matching `tiny_volume`."""
        seg = torch.zeros(*VOLUME_SHAPE, dtype=torch.float32)
        seg[:, :, 6:18, 6:18, 6:18] = 1.0
        return seg

    def assertIsImageLike(self, image: torch.Tensor, reference: torch.Tensor, label: str) -> None:
        """Assert `image` is a finite float tensor shaped like `reference`."""
        self.assertEqual(image.shape, reference.shape, f"{label} changed the volume shape")
        self.assertTrue(image.dtype.is_floating_point, f"{label} returned a non-float tensor")
        self.assertTrue(bool(torch.isfinite(image).all()), f"{label} produced NaN or Inf")


def first_output(result):
    """Pipelines return either a tensor or a (image, mask) sequence."""
    return result[0] if isinstance(result, (list, tuple)) else result


def configs_dir() -> Path:
    """Locate the packaged config directory.

    Uses importlib.resources rather than a path relative to this file, which is
    how smauglab.add_trainer resolves its own package data -- so the tests
    exercise the same lookup that ships to users.
    """
    from smauglab import configs

    return Path(str(importlib.resources.files(configs)))


def all_config_paths() -> list[Path]:
    """Every JSON config shipped in smauglab/configs (excluding the data/ examples)."""
    return sorted(p for p in configs_dir().glob("*.json"))


def gpu_config_paths() -> list[Path]:
    """Configs that drive the GPU augmentation pipeline."""
    paths = [p for p in all_config_paths() if p.name.startswith("transform_params_gpu")]
    extra = configs_dir() / "transform_params_one-sequence-to-segment-them-all.json"
    if extra.is_file():
        paths.append(extra)
    return sorted(paths)


def domain_bank_missing() -> str | None:
    """Return a skip reason if the domain-transfer LUT bank is not on this machine.

    The bank is a large offline-built artefact that ships with neither the wheel nor
    the repo, so it is only present where someone has exported SMAUGLAB_DOMAIN_BANK.
    Rather than fail CI, skip and say why.
    """
    from smauglab.transforms.gpu.domain_transfer import BANK_PATH_ENV_VAR, resolve_bank_path

    try:
        resolve_bank_path()
    except FileNotFoundError:
        return f"domain transfer bank not available (set {BANK_PATH_ENV_VAR})"
    return None


def requires_external_asset(config_path: Path) -> str | None:
    """Return a skip reason if a config needs an asset that is not on this machine."""
    params = json.loads(config_path.read_text())
    params = params.get("GPU", params)
    if not isinstance(params, dict):
        return None
    uses_transfer = params.get("RandomDomainTransferGPU") or params.get("DomainTransferTransform")
    if not uses_transfer:
        return None
    # A config may name its own bank; only fall back to the env var when it does not.
    if isinstance(uses_transfer, dict) and uses_transfer.get("bank_path"):
        return None if Path(uses_transfer["bank_path"]).is_file() else f"domain transfer bank not available at {uses_transfer['bank_path']}"
    return domain_bank_missing()
