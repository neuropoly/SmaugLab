"""The trainer has to honour `pipeline.order` like every other entry point.

`AugTransformsGPU` and `AugTransforms` both pass `order_source=config.order_source()`
into the builders. `nnUNetTrainerDAExtGPU` built both pipelines itself and did not,
so `build_gpu_pipeline` / `build_cpu_pipeline` fell back to their
`OrderSource.REGISTRY` default.

A config asking for `"order": "config"` was therefore honoured in the demo script,
in the standalone pipelines and in `test_builder.py`, and silently ignored in the
one place the order actually changes what a model is trained on.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("nnunetv2") is None:
    raise unittest.SkipTest("the trainer needs the nnunetv2 extra")

from unit_tests.test_trainers import training_transforms  # noqa: E402

#: Deliberately in the opposite order to registry.PIPELINE_ORDER[CPU], where
#: GaussianNoiseTransform comes well before MirrorTransform.
SECTION = {
    "MirrorTransform": {"allowed_axes": [0, 1, 2]},
    "GaussianNoiseTransform": {"p": 1.0},
}


class TestTrainerHonoursPipelineOrder(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get("SMAUGLAB_PARAMS_JSON")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._saved is None:
            os.environ.pop("SMAUGLAB_PARAMS_JSON", None)
        else:
            os.environ["SMAUGLAB_PARAMS_JSON"] = self._saved

    def _built(self, order):
        payload: dict = {"CPU": dict(SECTION)}
        if order is not None:
            payload["pipeline"] = {"order": order}
        # A distinct filename per case: load_config is lru_cached by path.
        path = Path(self._tmp.name) / f"transform_params_order_{order}.json"
        path.write_text(json.dumps(payload))
        return training_transforms(str(path))

    def test_the_default_is_still_registry_order(self):
        built = self._built(None)

        self.assertLess(
            built.index("GaussianNoiseTransform"),
            built.index("MirrorTransform"),
            f"registry order puts noise before mirror, got {built}",
        )

    def test_config_order_is_honoured(self):
        built = self._built("config")

        self.assertLess(
            built.index("MirrorTransform"),
            built.index("GaussianNoiseTransform"),
            f"the trainer ignored pipeline.order, got {built}",
        )
