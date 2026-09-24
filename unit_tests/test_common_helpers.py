"""`scripts/_common.py` helpers, which no test used to reach.

`parser2config` guarded its output directory with

    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

`os.path.dirname("config.json")` is `""`, `os.path.exists("")` is False and
`os.makedirs("")` raises, so writing a config next to the current directory --
the obvious invocation -- could not work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class TestParser2Config(unittest.TestCase):
    ARGS = argparse.Namespace(model="attunet", epochs=3, channels=[32, 64])

    def _written(self, path_out: str) -> dict:
        from _common import parser2config

        parser2config(self.ARGS, path_out)
        return json.loads(Path(path_out).read_text())

    def test_a_bare_filename_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                written = self._written("config.json")
            finally:
                os.chdir(cwd)

        self.assertEqual(written["model"], "attunet")

    def test_a_nested_path_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "a" / "b" / "config.json")

            written = self._written(target)

            self.assertEqual(written["epochs"], 3)

    def test_an_existing_directory_is_not_a_problem(self):
        """exist_ok, rather than a check that races with another worker."""
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "config.json")

            self._written(target)
            written = self._written(target)

            self.assertEqual(written["channels"], [32, 64])
