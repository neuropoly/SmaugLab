"""The shipped JSON configs must parse and actually drive the pipeline.

Every `transform_params_gpu*.json` is built into an `AugTransformsGPU` and run
over a tiny CPU volume. This is what catches a config that references a
transform the code no longer provides, or a parameter that was renamed.
"""

from __future__ import annotations

import json

import pytest
import torch

from unit_tests.conftest import all_config_paths, gpu_config_paths, requires_external_asset

ALL_CONFIGS = all_config_paths()
GPU_CONFIGS = gpu_config_paths()


def _ids(paths):
    return [p.name for p in paths]


def test_configs_are_shipped():
    assert ALL_CONFIGS, "no config JSONs found -- package data is missing"
    assert GPU_CONFIGS, "no transform_params_gpu*.json found"


@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=_ids(ALL_CONFIGS))
def test_config_is_valid_json(config_path):
    payload = json.loads(config_path.read_text())
    assert isinstance(payload, dict), f"{config_path.name} should hold a JSON object"


@pytest.mark.parametrize("config_path", GPU_CONFIGS, ids=_ids(GPU_CONFIGS))
def test_gpu_config_builds_and_runs(config_path, tiny_volume, tiny_seg):
    """Build the pipeline from the config and push one volume through it."""
    skip_reason = requires_external_asset(config_path)
    if skip_reason:
        pytest.skip(skip_reason)

    from auglab.transforms.gpu.transforms import AugTransformsGPU

    pipeline = AugTransformsGPU(json_path=str(config_path))
    result = pipeline(tiny_volume, tiny_seg)

    image = result[0] if isinstance(result, (list, tuple)) else result
    assert image.shape == tiny_volume.shape, f"{config_path.name} changed the volume shape"
    assert image.dtype.is_floating_point
    assert torch.isfinite(image).all(), f"{config_path.name} produced NaN or Inf"


@pytest.mark.parametrize("config_path", GPU_CONFIGS, ids=_ids(GPU_CONFIGS))
def test_gpu_config_is_deterministic_under_a_seed(config_path, tiny_volume, tiny_seg):
    """Same seed, same output -- otherwise published experiments are not reproducible."""
    skip_reason = requires_external_asset(config_path)
    if skip_reason:
        pytest.skip(skip_reason)

    from auglab.transforms.gpu.transforms import AugTransformsGPU

    def run_once():
        # Some transforms reach for the stdlib/numpy RNGs, not just torch's,
        # so all three have to be pinned for the comparison to mean anything.
        import random

        import numpy as np

        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        pipeline = AugTransformsGPU(json_path=str(config_path))
        out = pipeline(tiny_volume.clone(), tiny_seg.clone())
        return out[0] if isinstance(out, (list, tuple)) else out

    first, second = run_once(), run_once()
    assert torch.equal(first, second), f"{config_path.name} is not reproducible under a fixed seed"
