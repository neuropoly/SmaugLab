"""The built wheel must actually contain the package.

smauglab has no __init__.py anywhere, so it is picked up purely by the build
backend's namespace-package handling. That works, but it is easy to break
silently -- a wheel that is missing a subpackage installs fine and only fails
at import time for users. This test builds the real artifact and looks inside.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def source_modules() -> set[str]:
    """Every .py file under smauglab/, as a wheel-relative path."""
    package_root = REPO_ROOT / "smauglab"
    return {str(path.relative_to(REPO_ROOT)) for path in package_root.rglob("*.py") if "__pycache__" not in path.parts}


# Building a wheel is slow relative to the rest of the suite, so this class is
# marked for `pytest -m "not slow"`. The mark is applied at class level, which
# is the form pytest honours on unittest.TestCase subclasses.
@pytest.mark.slow
class TestWheelContents(unittest.TestCase):
    """Builds the wheel once for the whole class, then inspects it."""

    _tmpdir: tempfile.TemporaryDirectory
    wheel: Path

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if importlib.util.find_spec("build") is None:
            raise unittest.SkipTest("the `build` package is needed to test packaging")

        cls._tmpdir = tempfile.TemporaryDirectory()
        out_dir = Path(cls._tmpdir.name)
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(REPO_ROOT)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            cls._tmpdir.cleanup()
            raise AssertionError(f"wheel build failed:\n{result.stdout}\n{result.stderr}")

        wheels = list(out_dir.glob("*.whl"))
        if len(wheels) != 1:
            cls._tmpdir.cleanup()
            raise AssertionError(f"expected exactly one wheel, got {wheels}")
        cls.wheel = wheels[0]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def wheel_names(self) -> list[str]:
        with zipfile.ZipFile(self.wheel) as archive:
            return archive.namelist()

    def test_wheel_contains_every_module(self):
        shipped = {name for name in self.wheel_names() if name.endswith(".py")}
        missing = source_modules() - shipped
        self.assertFalse(missing, f"wheel is missing modules: {sorted(missing)}")

    def test_wheel_contains_the_config_data(self):
        """The JSON configs are the package's data; without them nothing runs."""
        configs = {n for n in self.wheel_names() if n.startswith("smauglab/configs/") and n.endswith(".json")}

        self.assertTrue(configs, "wheel ships no config JSONs")
        self.assertTrue(
            any("transform_params_gpu" in name for name in configs),
            "wheel is missing the default GPU transform config",
        )

    def test_wheel_excludes_scratch_directories(self):
        """Personal scratch configs should not be published to PyPI."""
        leaked = [name for name in self.wheel_names() if "configs_paul" in name]
        self.assertFalse(leaked, f"wheel ships personal scratch configs: {leaked}")


if __name__ == "__main__":
    unittest.main()
