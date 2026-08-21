"""The `smauglab` command line.

Driven through `cli.main` with captured stdout rather than as a subprocess: the
registry import costs a few seconds, and paying it once per process keeps the suite
fast enough to gate every pull request.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from smauglab import cli

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GPU = REPO / "smauglab" / "configs" / "transform_params_gpu_default01-23.json"
LEGACY = REPO / "unit_tests" / "fixtures" / "legacy_configs" / "configs" / "transform_params_gpu_default01-23.json"


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = cli.main(list(argv))
    return code, out.getvalue()


class TestList(unittest.TestCase):
    def test_lists_gpu_augmentations_in_pipeline_order(self):
        code, out = run("list", "--backend", "gpu")
        self.assertEqual(code, 0)
        self.assertIn("RandomFlipTransformGPU", out)
        # order column ascends
        orders = [int(line.split()[0]) for line in out.splitlines() if line.startswith("  ") and line.split()[0].isdigit()]
        self.assertEqual(orders, sorted(orders))

    def test_group_filter(self):
        _, out = run("list", "--backend", "gpu", "--group", "geo")
        self.assertIn("RandomFlipTransformGPU", out)
        self.assertNotIn("RandomScharrGPU", out)

    def test_json_output_is_machine_readable(self):
        _, out = run("list", "--backend", "cpu", "--json")
        payload = json.loads(out)
        self.assertTrue(all({"name", "backend", "aug_id", "group", "position"} <= set(e) for e in payload))


class TestMatrix(unittest.TestCase):
    def test_markdown_shows_the_monai_column_as_empty(self):
        code, out = run("matrix", "--format", "md")
        self.assertEqual(code, 0)
        self.assertIn("| Augmentation | Group | GPU | CPU | MONAI |", out)

    def test_json_form_reports_missing_backends_as_null(self):
        _, out = run("matrix", "--format", "json")
        payload = json.loads(out)
        self.assertIsNone(payload["palette"]["CPU"])
        self.assertEqual(payload["palette"]["GPU"], "RandomPaletteGPU")
        self.assertTrue(all(row["MONAI"] is None for row in payload.values()))

    def test_check_passes_on_a_clean_tree(self):
        code, _ = run("matrix", "--check")
        self.assertEqual(code, 0)


class TestShow(unittest.TestCase):
    def test_reports_parameters_and_defaults(self):
        code, out = run("show", "RandomScharrGPU")
        self.assertEqual(code, 0)
        self.assertIn("group TA", out)
        self.assertIn("absolute", out)

    def test_marks_required_parameters(self):
        _, out = run("show", "MirrorTransform")
        self.assertIn("REQUIRED", out)

    def test_flags_trainer_supplied_parameters(self):
        _, out = run("show", "SpatialTransform")
        self.assertIn("supplied by the trainer", out)

    def test_unknown_name_exits_nonzero_with_a_suggestion(self):
        code, out = run("show", "ScharrTransform")
        self.assertEqual(code, 1)
        self.assertIn("RandomScharrGPU", out)


class TestValidate(unittest.TestCase):
    def test_shipped_config_is_valid(self):
        code, out = run("validate", str(DEFAULT_GPU))
        self.assertEqual(code, 0)
        self.assertIn("ok", out)

    def test_a_broken_config_exits_nonzero_and_reports_every_problem(self):
        payload = {"GPU": {"ScharrTransform": {"p": 0.1}, "RandomGaussianNoiseGPU": {"probability": 0.2}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text(json.dumps(payload))
            code, out = run("validate", str(path))
        self.assertEqual(code, 1)
        self.assertIn("2 problem(s)", out)
        self.assertIn("RandomScharrGPU", out)
        self.assertIn("'probability' -> p", out)

    def test_a_legacy_config_is_rejected_and_points_at_migrate(self):
        code, out = run("validate", str(LEGACY))
        self.assertEqual(code, 1)
        self.assertIn("migration/", out)


class TestTemplateAndHash(unittest.TestCase):
    def test_template_names_every_gpu_augmentation(self):
        from smauglab import registry
        from smauglab.registry import Backend

        _, out = run("template", "--backend", "gpu")
        self.assertEqual(set(json.loads(out)["GPU"]), set(registry.names(Backend.GPU)))

    def test_template_round_trips_through_validation(self):
        """A template that cannot be loaded would be worse than no template."""
        from smauglab.config import validate_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "template.json"
            run("template", "--backend", "gpu", "-o", str(path))
            self.assertEqual(validate_file(path), [])

    def test_hash_is_stable_and_content_addressed(self):
        code, first = run("hash", str(DEFAULT_GPU))
        self.assertEqual(code, 0)
        _, second = run("hash", str(DEFAULT_GPU))
        self.assertEqual(first, second)
        self.assertEqual(len(first.split()[0]), 8)


if __name__ == "__main__":
    unittest.main()
