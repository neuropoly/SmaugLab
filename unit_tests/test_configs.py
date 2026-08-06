"""The shipped JSON configs must parse and actually drive the pipeline.

Every `transform_params_gpu*.json` is built into an `AugTransformsGPU` and run
over a tiny CPU volume. This is what catches a config that references a
transform the code no longer provides, or a parameter that was renamed.
"""

from __future__ import annotations

import json
import unittest

import torch

from unit_tests.helpers import (
    AugLabTestCase,
    all_config_paths,
    first_output,
    gpu_config_paths,
    requires_external_asset,
    seed_everything,
)

ALL_CONFIGS = all_config_paths()
GPU_CONFIGS = gpu_config_paths()


class TestConfigsArePresent(unittest.TestCase):
    def test_configs_are_shipped(self):
        self.assertTrue(ALL_CONFIGS, "no config JSONs found -- package data is missing")
        self.assertTrue(GPU_CONFIGS, "no transform_params_gpu*.json found")

    def test_every_config_is_valid_json(self):
        for config_path in ALL_CONFIGS:
            with self.subTest(config=config_path.name):
                payload = json.loads(config_path.read_text())
                self.assertIsInstance(payload, dict, f"{config_path.name} should hold a JSON object")


class TestGpuConfigs(AugLabTestCase):
    def test_gpu_config_builds_and_runs(self):
        """Build the pipeline from each config and push one volume through it."""
        from auglab.transforms.gpu.transforms import AugTransformsGPU

        for config_path in GPU_CONFIGS:
            with self.subTest(config=config_path.name):
                skip_reason = requires_external_asset(config_path)
                if skip_reason:
                    self.skipTest(skip_reason)

                volume, seg = self.tiny_volume(), self.tiny_seg()
                pipeline = AugTransformsGPU(json_path=str(config_path))
                image = first_output(pipeline(volume, seg))

                self.assertIsImageLike(image, volume, config_path.name)

    def test_gpu_config_is_deterministic_under_a_seed(self):
        """Same seed, same output -- otherwise published experiments are not reproducible."""
        from auglab.transforms.gpu.transforms import AugTransformsGPU

        def run_once(config_path):
            # Some transforms reach for the stdlib/numpy RNGs, not just torch's,
            # so all three have to be pinned for the comparison to mean anything.
            seed_everything(7)
            pipeline = AugTransformsGPU(json_path=str(config_path))
            return first_output(pipeline(self.tiny_volume(), self.tiny_seg()))

        for config_path in GPU_CONFIGS:
            with self.subTest(config=config_path.name):
                skip_reason = requires_external_asset(config_path)
                if skip_reason:
                    self.skipTest(skip_reason)

                first, second = run_once(config_path), run_once(config_path)
                self.assertTrue(
                    torch.equal(first, second),
                    f"{config_path.name} is not reproducible under a fixed seed",
                )


if __name__ == "__main__":
    unittest.main()
