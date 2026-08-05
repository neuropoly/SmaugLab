"""Shared fixtures for the AugLab test suite.

Everything here runs on CPU with tiny volumes, so the whole suite stays fast
enough to gate every pull request. No GPU and no image data on disk required.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import pytest
import torch

# Small enough to be fast, large enough that the spatial transforms (crop,
# low-res simulation, flips) still have something to work with.
VOLUME_SHAPE = (1, 1, 24, 24, 24)
SEED = 1234


@pytest.fixture(autouse=True)
def _seeded():
    """Seed every RNG the transforms reach for, so failures are reproducible."""
    import random

    import numpy as np

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def tiny_volume() -> torch.Tensor:
    """A [N, C, D, H, W] float image in roughly the range the transforms expect."""
    return torch.rand(*VOLUME_SHAPE, dtype=torch.float32)


@pytest.fixture
def tiny_seg() -> torch.Tensor:
    """A binary segmentation mask matching `tiny_volume`."""
    seg = torch.zeros(*VOLUME_SHAPE, dtype=torch.float32)
    seg[:, :, 6:18, 6:18, 6:18] = 1.0
    return seg


def configs_dir() -> Path:
    """Locate the packaged config directory.

    Uses importlib.resources rather than a path relative to this file, which is
    how auglab.add_trainer resolves its own package data -- so the tests
    exercise the same lookup that ships to users.
    """
    from auglab import configs

    return Path(str(importlib.resources.files(configs)))


def all_config_paths() -> list[Path]:
    """Every JSON config shipped in auglab/configs (excluding the data/ examples)."""
    return sorted(p for p in configs_dir().glob("*.json"))


def gpu_config_paths() -> list[Path]:
    """Configs that drive the GPU augmentation pipeline."""
    paths = [p for p in all_config_paths() if p.name.startswith("transform_params_gpu")]
    extra = configs_dir() / "transform_params_one-sequence-to-segment-them-all.json"
    if extra.is_file():
        paths.append(extra)
    return sorted(paths)


def requires_external_asset(config_path: Path) -> str | None:
    """Return a skip reason if a config needs an asset that is not on this machine.

    RandomDomainTransferGPU loads a precomputed histogram bank from an absolute
    path baked into the module, which only exists on the authors' machines.
    Rather than fail CI, skip those configs and say why.
    """
    from auglab.transforms.gpu.domain_transfer import DEFAULT_BANK_PATH

    params = json.loads(config_path.read_text())
    params = params.get("GPU", params)
    if not isinstance(params, dict):
        return None
    uses_transfer = params.get("RandomDomainTransferGPU") or params.get("DomainTransferTransform")
    if uses_transfer and not Path(DEFAULT_BANK_PATH).is_file():
        return f"domain transfer bank not available at {DEFAULT_BANK_PATH}"
    return None
