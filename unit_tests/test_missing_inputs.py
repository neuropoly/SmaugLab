"""Inputs that are not on disk have to be reported, not silently dropped.

`fetch_image_config` has always returned `(out_list, err)`, where `err` collects
every path that does not exist. Every caller threw `err` away:

* `generate_augmentations.py` unpacked it as `_`
* `train_monai.py` bound `err_train` / `err_val` and never used them

so a data config naming 500 subjects of which 400 were missing augmented 100 of
them and printed nothing at all.

The filter also only checked `IMAGE`. A missing `LABEL` passed it and failed
much later, inside a worker, with an opaque nibabel error.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def config_with(tmp: Path, pairs) -> dict:
    return {
        "TYPE": "LABEL",
        "DATASETS_PATH": str(tmp),
        "TRAINING": [{"IMAGE": image, "LABEL": label} for image, label in pairs],
    }


class TestMissingInputsAreReported(unittest.TestCase):
    def _fetch(self, tmp, pairs):
        from _common import fetch_image_config

        with redirect_stdout(io.StringIO()):  # the progress bar
            return fetch_image_config(config_with(tmp, pairs), split="TRAINING")

    def test_a_missing_label_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "a.nii.gz").write_bytes(b"")

            out_list, err = self._fetch(work, [("a.nii.gz", "a_seg.nii.gz")])

        self.assertEqual(out_list, [], "a pair with no label must not be handed to the workers")
        self.assertTrue(any("a_seg.nii.gz" in path for path, _ in err))

    def test_a_complete_pair_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "a.nii.gz").write_bytes(b"")
            (work / "a_seg.nii.gz").write_bytes(b"")

            out_list, err = self._fetch(work, [("a.nii.gz", "a_seg.nii.gz")])

        self.assertEqual(len(out_list), 1)
        self.assertEqual(err, [])

    def test_report_missing_names_the_paths(self):
        from _common import report_missing

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report_missing([["/data/a.nii.gz", "path error"]], "TRAINING")

        printed = buffer.getvalue()
        self.assertIn("1 TRAINING path(s)", printed)
        self.assertIn("/data/a.nii.gz", printed)

    def test_report_missing_says_nothing_when_nothing_is_missing(self):
        from _common import report_missing

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report_missing([], "TRAINING")

        self.assertEqual(buffer.getvalue(), "")

    def test_report_missing_truncates_a_long_list(self):
        from _common import report_missing

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report_missing([[f"/data/{i}.nii.gz", "path error"] for i in range(25)], "TRAINING", limit=3)

        printed = buffer.getvalue()
        self.assertIn("25 TRAINING path(s)", printed)
        self.assertIn("and 22 more", printed)

    def test_both_callers_report(self):
        """The list existed all along; what was missing was anyone looking at it."""
        for script in ("generate_augmentations.py", "train_monai.py"):
            with self.subTest(script=script):
                source = (SCRIPTS / script).read_text()

                self.assertIn("report_missing(", source)
