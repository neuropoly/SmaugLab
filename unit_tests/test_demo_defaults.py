"""`--backend cpu` has to demonstrate a CPU pipeline.

The demo fell back to `transform_params_gpu.json` whatever the backend. That
config's CPU section holds a single transform, so the documented CPU invocation

    python scripts/demo_augmentations.py --backend cpu --repeats 24 ...

rendered 24 near-identical tiles and looked like the pipeline was broken.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def section_size(path: Path, backend: str) -> int:
    payload = json.loads(path.read_text())
    return len([k for k in payload.get(backend, {}) if not k.startswith("_")])


class TestDemoDefaultConfig(unittest.TestCase):
    def test_each_backend_gets_a_config_that_populates_it(self):
        import demo_augmentations

        for backend, section in (("gpu", "GPU"), ("cpu", "CPU")):
            with self.subTest(backend=backend):
                path = demo_augmentations.default_config_path(backend)

                self.assertTrue(path.is_file(), f"{path.name} is not shipped")
                self.assertGreater(
                    section_size(path, section),
                    1,
                    f"the {backend} default config has nothing to demonstrate on that backend",
                )

    def test_the_cpu_default_is_not_the_gpu_config(self):
        import demo_augmentations

        self.assertNotEqual(
            demo_augmentations.default_config_path("cpu").name,
            demo_augmentations.default_config_path("gpu").name,
        )

    def test_the_parser_resolves_the_default_per_backend(self):
        import demo_augmentations

        args = demo_augmentations.build_parser().parse_args(["--backend", "cpu", "--image", "a.nii.gz", "--seg", "b.nii.gz"])

        self.assertIsNone(args.config)
        self.assertEqual(
            Path(str(demo_augmentations.default_config_path(args.backend))).name,
            "transform_params.json",
        )
