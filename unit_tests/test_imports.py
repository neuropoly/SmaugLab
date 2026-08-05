"""Every module in the package must import cleanly.

This is the cheapest regression net there is. It catches undeclared
dependencies, syntax errors, and undefined names at module scope -- the class
of bug that ruff's F821 found in transforms_list.py.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import auglab

# Requires the optional `nnunetv2` extra; skipped rather than failed when absent.
OPTIONAL_PREFIXES = ("auglab.trainers", "auglab.add_trainer")


def _module_names() -> list[str]:
    """Every .py file under auglab/, as a dotted module name.

    Deliberately a filesystem walk rather than pkgutil.walk_packages: several
    subdirectories (transforms/, transforms/cpu/, transforms/gpu/, utils/) have
    no __init__.py, so auglab resolves as a PEP 420 namespace package and
    walk_packages only reaches 4 of the ~24 modules. Walking the tree keeps
    this test honest regardless of how the package is laid out.
    """
    roots = [Path(p) for p in auglab.__path__]
    names = set()
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).with_suffix("")
            parts = list(relative.parts)
            if parts[-1] == "__init__":
                parts.pop()
            if not parts:
                continue
            names.add(".".join(["auglab", *parts]))
    return sorted(names)


MODULES = _module_names()


def test_walk_found_modules():
    """Guard against the discovery itself silently returning too little.

    If this trips, either modules were deleted or the package layout changed in
    a way that hides them -- both worth noticing.
    """
    assert len(MODULES) >= 20, f"expected the full package, discovered only {len(MODULES)}: {MODULES}"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    if module_name.startswith(OPTIONAL_PREFIXES):
        pytest.importorskip("nnunetv2", reason=f"{module_name} needs the nnunetv2 extra")
    importlib.import_module(module_name)


def test_public_pipeline_entrypoints_are_importable():
    """The classes users actually construct must be reachable from the package."""
    from auglab.transforms.gpu.transforms import AugTransformsGPU
    from auglab.transforms.gpu.transforms_list import (
        AugTransformsGPURandomOrder,
        AugTransformsGPURandomOrderTA,
    )

    assert all(callable(cls) for cls in (AugTransformsGPU, AugTransformsGPURandomOrder, AugTransformsGPURandomOrderTA))
