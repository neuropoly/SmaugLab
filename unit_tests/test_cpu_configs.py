"""Every shipped config's CPU section must build, not just parse.

`test_configs.py` builds only the `transform_params_gpu*` configs, so nothing
ever instantiated a CPU pipeline from a shipped file. That gap hid two coupled
defects in `all_augmentations.json`, the template CI checks for staleness:

* `render_template` wrote `null` for every parameter that has no default, and
* `validate_section` counted a key present with value `null` as supplied.

So the template validated clean and then died at build time with
`TypeError: 'NoneType' object is not iterable`. The whole point of the file --
per `render_template`'s docstring, "an augmentation exists but no config can
reach it" is a test failure -- did not hold for the CPU half.
"""

from __future__ import annotations

import json

from smauglab.config import validate_file
from smauglab.registry import Backend
from smauglab.transforms.build import build_cpu_pipeline
from unit_tests.helpers import SmaugLabTestCase, all_config_paths

# What nnU-Net supplies at runtime; the builder injects it as context.
PATCH_SIZE = (24, 24, 24)
ROTATION = (-0.1, 0.1)


class TestCpuSectionsBuild(SmaugLabTestCase):
    def _cpu_configs(self):
        for path in all_config_paths():
            payload = json.loads(path.read_text())
            if payload.get(Backend.CPU.value):
                yield path, payload

    def test_there_is_something_to_check(self):
        self.assertTrue(list(self._cpu_configs()), "no shipped config has a CPU section")

    def test_every_cpu_section_builds(self):
        for path, payload in self._cpu_configs():
            with self.subTest(config=path.name):
                transforms = build_cpu_pipeline(
                    payload[Backend.CPU.value],
                    do_dummy_2d_data_aug=False,
                    patch_size=PATCH_SIZE,
                    rotation=ROTATION,
                    source=path.name,
                )

                self.assertTrue(transforms, f"{path.name} built an empty CPU pipeline")

    def test_every_cpu_section_validates(self):
        for path, _ in self._cpu_configs():
            with self.subTest(config=path.name):
                self.assertEqual(validate_file(path), [])


class TestNullIsNotASuppliedValue(SmaugLabTestCase):
    """A required parameter written as `null` is a hole, not a value."""

    def _problems(self, tmp, payload):
        path = tmp / "transform_params_null.json"
        path.write_text(json.dumps(payload))
        return validate_file(path)

    def test_a_required_parameter_written_as_null_is_reported_missing(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            problems = self._problems(Path(tmp), {"CPU": {"MirrorTransform": {"allowed_axes": None}}})

        self.assertTrue(problems, "a required parameter set to null was accepted")
        self.assertIn("allowed_axes", " ".join(problems))

    def test_an_optional_parameter_may_still_be_null(self):
        """`SimulateLowResolutionTransform.allowed_channels` defaults to None."""
        import tempfile
        from pathlib import Path

        payload = {
            "CPU": {
                "SimulateLowResolutionTransform": {
                    "scale": [0.3, 1],
                    "synchronize_channels": True,
                    "synchronize_axes": False,
                    "ignore_axes": [],
                    "allowed_channels": None,
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._problems(Path(tmp), payload)

        self.assertEqual(problems, [])
