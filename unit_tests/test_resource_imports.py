"""`importlib.resources` has to be imported, not reached through `importlib`.

`import importlib` binds the package; it does not guarantee the `resources`
submodule is bound on it. Both trainers did

    import importlib
    ...
    importlib.resources.files(configs)

which worked only because something else in the import graph happened to import
`importlib.resources` first. Which import that is, and whether it happens at all,
is not ours to rely on: it varies with the torch and nnunetv2 versions installed.

`smauglab/add_trainer.py` always did this correctly, which is what makes the
other two a slip rather than a convention.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "smauglab"


def modules_touching_resources() -> list[Path]:
    """Every module that reads an attribute off `importlib.resources`."""
    return [path for path in sorted(PACKAGE.rglob("*.py")) if "importlib.resources" in path.read_text()]


class TestResourcesIsImportedExplicitly(unittest.TestCase):
    def test_there_is_something_to_check(self):
        self.assertTrue(modules_touching_resources())

    def test_every_user_imports_the_submodule(self):
        for path in modules_touching_resources():
            with self.subTest(module=str(path.relative_to(PACKAGE.parent))):
                tree = ast.parse(path.read_text())
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.update(f"{node.module}.{alias.name}" for alias in node.names)

                self.assertIn(
                    "importlib.resources",
                    imported,
                    "uses importlib.resources but only imports importlib; the submodule is bound only if something else imported it first",
                )
