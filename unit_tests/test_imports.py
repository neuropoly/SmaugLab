"""Every module in the package must import cleanly.

This is the cheapest regression net there is. It catches undeclared
dependencies, syntax errors, and undefined names at module scope -- the class
of bug that ruff's F821 found in transforms_list.py.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import unittest

import smauglab

# Requires the optional `nnunetv2` extra; skipped rather than failed when absent.
OPTIONAL_PREFIXES = ("smauglab.trainers", "smauglab.add_trainer")


def module_names() -> list[str]:
    """Every module and subpackage under smauglab/, as a dotted module name.

    smauglab is a regular package (every directory has an __init__.py), so
    walk_packages reaches the whole tree. Subpackages are kept, not filtered out:
    their __init__.py files hold real code now (smauglab/__init__.py resolves the
    version) and are worth importing too.

    `onerror` matters: without it, walk_packages swallows an ImportError raised
    while probing a subpackage and silently returns a short list, turning a real
    breakage into a quietly passing test. Re-raising surfaces it here instead.
    """

    def onerror(_name: str) -> None:
        raise  # noqa: PLE0704 -- re-raises whatever walk_packages was handling

    names = {module.name for module in pkgutil.walk_packages(smauglab.__path__, prefix="smauglab.", onerror=onerror)}
    names.add("smauglab")
    return sorted(names)


MODULES = module_names()


class TestModuleDiscovery(unittest.TestCase):
    def test_walk_found_modules(self):
        """Guard against the discovery itself silently returning too little.

        If this trips, either modules were deleted or the package layout changed
        in a way that hides them -- both worth noticing.
        """
        self.assertGreaterEqual(
            len(MODULES),
            20,
            f"expected the full package, discovered only {len(MODULES)}: {MODULES}",
        )


class TestModuleImports(unittest.TestCase):
    def test_every_module_imports(self):
        """Import each module in turn, reporting the module name on failure."""
        have_nnunet = importlib.util.find_spec("nnunetv2") is not None

        for module_name in MODULES:
            with self.subTest(module=module_name):
                if module_name.startswith(OPTIONAL_PREFIXES) and not have_nnunet:
                    self.skipTest(f"{module_name} needs the nnunetv2 extra")
                importlib.import_module(module_name)

    def test_public_pipeline_entrypoints_are_importable(self):
        """The classes users actually construct must be reachable from the package."""
        from smauglab.transforms.gpu.transforms import AugTransformsGPU
        from smauglab.transforms.gpu.transforms_list import (
            AugTransformsGPURandomOrder,
            AugTransformsGPURandomOrderTA,
        )

        for cls in (AugTransformsGPU, AugTransformsGPURandomOrder, AugTransformsGPURandomOrderTA):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(callable(cls))


if __name__ == "__main__":
    unittest.main()
