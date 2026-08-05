"""The built wheel must actually contain the package.

auglab has no __init__.py anywhere in its tree, so it is picked up purely by
setuptools' namespace auto-discovery. That works, but it is easy to break
silently -- a wheel that is missing a subpackage installs fine and only fails
at import time for users. This test builds the real artifact and looks inside.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _source_modules() -> set[str]:
    """Every .py file under auglab/, as a wheel-relative path."""
    package_root = REPO_ROOT / "auglab"
    return {str(path.relative_to(REPO_ROOT)) for path in package_root.rglob("*.py") if "__pycache__" not in path.parts}


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    pytest.importorskip("build", reason="the `build` package is needed to test packaging")
    out_dir = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"wheel build failed:\n{result.stdout}\n{result.stderr}")
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@pytest.mark.slow
def test_wheel_contains_every_module(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        shipped = {name for name in archive.namelist() if name.endswith(".py")}

    missing = _source_modules() - shipped
    assert not missing, f"wheel is missing modules: {sorted(missing)}"


@pytest.mark.slow
def test_wheel_contains_the_config_data(built_wheel):
    """The JSON configs are the package's data; without them nothing runs."""
    with zipfile.ZipFile(built_wheel) as archive:
        configs = {name for name in archive.namelist() if name.startswith("auglab/configs/") and name.endswith(".json")}

    assert configs, "wheel ships no config JSONs"
    assert any("transform_params_gpu" in name for name in configs), "wheel is missing the default GPU transform config"


@pytest.mark.slow
def test_wheel_excludes_scratch_directories(built_wheel):
    """Personal scratch configs should not be published to PyPI."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()

    leaked = [name for name in names if "configs_paul" in name]
    assert not leaked, f"wheel ships personal scratch configs: {leaked}"
