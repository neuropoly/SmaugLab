"""Resuming a run must not erase what the earlier epochs were trained with.

The trainer copies the config it was given to
`transform_params_used_for_training.json` in the run folder. It did so
unconditionally, so resuming with a different `SMAUGLAB_PARAMS_JSON` replaced
the record: the file then described a config that only the later epochs saw,
while claiming to describe the run.

That file is the only provenance a finished run carries. A differing config is
now written beside it under a numbered name, with a warning naming both.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path

if importlib.util.find_spec("nnunetv2") is None:
    raise unittest.SkipTest("the trainer needs the nnunetv2 extra")

from smauglab.trainers.nnUNetTrainerDAExt import _record_config  # noqa: E402

RECORD = "transform_params_used_for_training.json"


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


class TestRecordConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run = Path(self._tmp.name)
        self.destination = self.run / RECORD

    def test_the_first_run_writes_the_record(self):
        source = write(self.run / "a.json", {"GPU": {}})

        _record_config(str(source), str(self.destination))

        self.assertEqual(json.loads(self.destination.read_text()), {"GPU": {}})

    def test_resuming_with_the_same_config_is_a_no_op(self):
        source = write(self.run / "a.json", {"GPU": {}})
        _record_config(str(source), str(self.destination))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _record_config(str(source), str(self.destination))

        self.assertEqual(sorted(p.name for p in self.run.glob("transform_params_*")), [RECORD])

    def test_resuming_with_a_different_config_keeps_the_original(self):
        first = write(self.run / "a.json", {"GPU": {"RandomFlipTransformGPU": {"p": 1.0}}})
        _record_config(str(first), str(self.destination))
        second = write(self.run / "b.json", {"GPU": {}})

        with self.assertWarns(UserWarning) as caught:
            _record_config(str(second), str(self.destination))

        self.assertEqual(
            json.loads(self.destination.read_text()),
            {"GPU": {"RandomFlipTransformGPU": {"p": 1.0}}},
            "the record of the earlier epochs was overwritten",
        )
        kept = self.run / "transform_params_used_for_training_1.json"
        self.assertEqual(json.loads(kept.read_text()), {"GPU": {}})
        self.assertIn(kept.name, str(caught.warning))

    def test_a_third_config_does_not_clobber_the_second(self):
        _record_config(str(write(self.run / "a.json", {"GPU": {"a": 1}})), str(self.destination))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _record_config(str(write(self.run / "b.json", {"GPU": {"b": 2}})), str(self.destination))
            _record_config(str(write(self.run / "c.json", {"GPU": {"c": 3}})), str(self.destination))

        self.assertEqual(json.loads((self.run / "transform_params_used_for_training_1.json").read_text()), {"GPU": {"b": 2}})
        self.assertEqual(json.loads((self.run / "transform_params_used_for_training_2.json").read_text()), {"GPU": {"c": 3}})
