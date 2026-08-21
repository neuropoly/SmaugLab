"""The sectioned config document: what it accepts, and what it says when it does not.

A config used to be a flat mapping of augmentation name to parameters, read as "GPU or
CPU, whichever the keys look like". The two namespaces overlap enough that
`GaussianBlurTransform` meant a different transform depending on which builder read it,
so sections are mandatory now and a flat document is rejected outright.

Every problem in a file is reported at once. The old behaviour surfaced one per run,
which for a 30-key config is 30 edit-run cycles.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smauglab.config import OrderSource, PipelineMode, SmaugConfig, validate_file, validate_section
from smauglab.registry import Backend, InvalidConfigError


def write(payload: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "config.json"
    path.write_text(json.dumps(payload))
    return path


class TestSections(unittest.TestCase):
    def test_a_minimal_gpu_config_validates(self):
        config = SmaugConfig({"GPU": {"RandomFlipTransformGPU": {"p": 0.5}}})
        self.assertEqual(config.names(Backend.GPU), ["RandomFlipTransformGPU"])

    def test_a_flat_section_less_config_is_rejected(self):
        """The hard break. It used to be guessed at from the key names."""
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"FlipTransform": {"probability": 0.5}})
        self.assertIn("pre-registry config", str(caught.exception))

    def test_an_unknown_top_level_key_is_rejected(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"GPU": {}, "GPUU": {}})
        self.assertIn("unknown top-level key", str(caught.exception))

    def test_underscore_keys_are_comments(self):
        config = SmaugConfig({"_comment": "anything", "GPU": {"_note": "ignored", "RandomFlipTransformGPU": {}}})
        self.assertEqual(config.names(Backend.GPU), ["RandomFlipTransformGPU"])

    def test_sections_are_handed_out_as_copies(self):
        """load_config caches documents, so a caller that mutates must not poison it."""
        config = SmaugConfig({"GPU": {"RandomFlipTransformGPU": {"p": 0.5}}})
        config.section(Backend.GPU)["RandomFlipTransformGPU"]["p"] = 99
        self.assertEqual(config.section(Backend.GPU)["RandomFlipTransformGPU"]["p"], 0.5)

    def test_a_missing_section_is_empty_not_an_error(self):
        self.assertEqual(SmaugConfig({"GPU": {}}).section(Backend.CPU), {})


class TestProblemsAreReportedTogether(unittest.TestCase):
    def test_every_problem_in_a_file_is_reported_at_once(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig(
                {
                    "GPU": {
                        "NotATransform": {},
                        "RandomFlipTransformGPU": {"nonsense": 1},
                        "RandomScharrGPU": {"probability": 0.5},
                    }
                }
            )
        self.assertEqual(len(caught.exception.problems), 3, caught.exception.problems)

    def test_an_unknown_augmentation_suggests_a_close_match(self):
        problems = validate_section({"RandomFlipTransformGP": {}}, Backend.GPU)
        self.assertTrue(any("RandomFlipTransformGPU" in p for p in problems), problems)

    def test_a_renamed_parameter_is_pointed_at_its_replacement(self):
        """`probability` -> `p` is the commonest migration mistake, and difflib
        cannot bridge it: the two strings score ~0.17."""
        problems = validate_section({"RandomFlipTransformGPU": {"probability": 0.5}}, Backend.GPU)
        self.assertTrue(any("'probability' -> p" in p for p in problems), problems)

    def test_a_block_that_is_not_an_object_is_reported(self):
        problems = validate_section({"RandomFlipTransformGPU": 0.5}, Backend.GPU)
        self.assertTrue(any("block of parameters" in p for p in problems), problems)

    def test_validate_file_returns_problems_without_raising(self):
        self.assertEqual(validate_file(write({"GPU": {"RandomFlipTransformGPU": {}}})), [])
        self.assertTrue(validate_file(write({"GPU": {"Nope": {}}})))

    def test_invalid_json_is_reported_as_such(self):
        path = Path(tempfile.mkdtemp()) / "broken.json"
        path.write_text("{not json")
        self.assertTrue(any("not valid JSON" in p for p in validate_file(path)))


class TestPipelineSection(unittest.TestCase):
    def test_mode_defaults_to_sequential(self):
        self.assertIs(SmaugConfig({"GPU": {}}).pipeline_mode(), PipelineMode.SEQUENTIAL)

    def test_mode_is_read_from_the_config(self):
        config = SmaugConfig({"GPU": {}, "pipeline": {"mode": "random_order"}})
        self.assertIs(config.pipeline_mode(), PipelineMode.RANDOM_ORDER)

    def test_an_unknown_mode_is_rejected_with_a_suggestion(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"GPU": {}, "pipeline": {"mode": "random_ordr"}})
        self.assertIn("random_order", str(caught.exception))

    def test_an_unknown_pipeline_key_is_rejected(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"GPU": {}, "pipeline": {"modes": "sequential"}})
        self.assertIn("unknown key", str(caught.exception))

    def test_pipeline_options_is_always_a_dict(self):
        """The old builder called .get() on the result and raised AttributeError
        on every config that omitted the block."""
        self.assertEqual(SmaugConfig({"GPU": {}}).pipeline_options("random_choose"), {})


class TestOrderSource(unittest.TestCase):
    """`pipeline.order`: registry order by default, config key order on request."""

    def test_it_defaults_to_the_registry(self):
        self.assertIs(SmaugConfig({"GPU": {}}).order_source(), OrderSource.REGISTRY)

    def test_config_key_order_can_be_asked_for(self):
        config = SmaugConfig({"GPU": {}, "pipeline": {"order": "config"}})
        self.assertIs(config.order_source(), OrderSource.CONFIG)

    def test_the_registry_default_can_be_stated_explicitly(self):
        config = SmaugConfig({"GPU": {}, "pipeline": {"order": "registry"}})
        self.assertIs(config.order_source(), OrderSource.REGISTRY)

    def test_an_unknown_order_source_is_rejected_with_a_suggestion(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"GPU": {}, "pipeline": {"order": "confgi"}})
        self.assertIn("pipeline.order", str(caught.exception))
        self.assertIn("config", str(caught.exception))

    def test_order_and_mode_are_independent(self):
        config = SmaugConfig({"GPU": {}, "pipeline": {"mode": "random_order", "order": "config"}})
        self.assertIs(config.pipeline_mode(), PipelineMode.RANDOM_ORDER)
        self.assertIs(config.order_source(), OrderSource.CONFIG)


if __name__ == "__main__":
    unittest.main()
