"""`importlib.resources` has to be imported, not reached through `importlib`.

`import importlib` binds the package; it does not guarantee the `resources`
submodule is bound on it. Both trainers did

    import importlib
    ...
    importlib.resources.files(configs)

which worked only because something else in the import graph -- nnunetv2's, as it
turns out -- happened to import `importlib.resources` first. Verified on this
interpreter: the attribute is absent after `import importlib` and still absent
after `import torch`.

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

    def test_a_bare_importlib_does_not_expose_resources(self):
        """The premise, asserted rather than assumed.

        In a fresh interpreter neither `import importlib` nor `import torch` binds
        the submodule, so the old code depended entirely on some other import
        happening to pull it in.
        """
        import subprocess
        import sys

        probe = "import importlib, torch; print(hasattr(importlib, 'resources'))"
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        self.assertEqual(result.stdout.strip(), "False")
