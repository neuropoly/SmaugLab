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
from typing import ClassVar

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
        """`probability` -> `p` is the commonest mistake in an old config, and difflib
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


class TestLossSection(unittest.TestCase):
    """The `loss` section, validated the same way `pipeline` is.

    A typo'd setting has to be an error rather than an ignored key: a config saying
    `"detatch_reference": false` would otherwise train with the reference detached and
    then be copied into the run folder as that run's only record, claiming the opposite.
    """

    BASE: ClassVar[dict] = {"GPU": {"RandomFlipTransformGPU": {"p": 1.0}}}

    def _config(self, loss) -> SmaugConfig:
        return SmaugConfig({**self.BASE, "loss": loss})

    def test_the_section_is_optional_and_the_term_defaults_to_off(self):
        config = SmaugConfig(dict(self.BASE))
        self.assertFalse(config.consistency_enabled())
        self.assertEqual(
            config.loss_options("consistency"),
            {"enabled": False, "weight": 0.5, "detach_reference": True},
        )

    def test_defaults_fill_in_around_a_partial_block(self):
        config = self._config({"consistency": {"enabled": True, "weight": 0.25}})
        self.assertTrue(config.consistency_enabled())
        self.assertEqual(
            config.loss_options("consistency"),
            {"enabled": True, "weight": 0.25, "detach_reference": True},
        )

    def test_options_are_handed_out_as_copies(self):
        config = self._config({"consistency": {"enabled": True}})
        config.loss_options("consistency")["weight"] = 99
        self.assertEqual(config.loss_options("consistency")["weight"], 0.5)

    def test_comment_keys_are_accepted(self):
        config = self._config({"_why": "ablation", "consistency": {"_note": "x", "enabled": True}})
        self.assertTrue(config.consistency_enabled())

    def test_an_unknown_term_is_rejected_with_a_suggestion(self):
        with self.assertRaises(InvalidConfigError) as caught:
            self._config({"consistancy": {"enabled": True}})
        self.assertIn("Did you mean: consistency?", caught.exception.problems[0])

    def test_an_unknown_setting_is_rejected_with_a_suggestion(self):
        with self.assertRaises(InvalidConfigError) as caught:
            self._config({"consistency": {"detatch_reference": True}})
        self.assertIn("Did you mean: detach_reference?", caught.exception.problems[0])

    def test_a_non_boolean_flag_is_rejected(self):
        for key in ("enabled", "detach_reference"):
            with self.subTest(key=key), self.assertRaises(InvalidConfigError) as caught:
                self._config({"consistency": {key: "yes"}})
            self.assertIn(f"loss.consistency.{key}", caught.exception.problems[0])

    def test_a_negative_weight_is_rejected(self):
        with self.assertRaises(InvalidConfigError) as caught:
            self._config({"consistency": {"weight": -0.5}})
        self.assertIn("non-negative", caught.exception.problems[0])

    def test_a_non_numeric_weight_is_rejected(self):
        for weight in ("0.5", True, None):
            with self.subTest(weight=weight), self.assertRaises(InvalidConfigError):
                self._config({"consistency": {"weight": weight}})

    def test_a_zero_weight_is_accepted(self):
        """The "pay the second forward, measure the term, change nothing" ablation."""
        config = self._config({"consistency": {"enabled": True, "weight": 0}})
        self.assertEqual(config.loss_options("consistency")["weight"], 0)

    def test_the_section_must_be_an_object(self):
        with self.assertRaises(InvalidConfigError) as caught:
            self._config(3)
        self.assertEqual(caught.exception.problems, ["loss: expected an object"])

    def test_a_terms_block_must_be_an_object(self):
        with self.assertRaises(InvalidConfigError) as caught:
            self._config({"consistency": 3})
        self.assertIn("loss.consistency: expected an object", caught.exception.problems[0])

    def test_every_problem_in_the_section_is_reported_at_once(self):
        with self.assertRaises(InvalidConfigError) as caught:
            self._config({"nope": {}, "consistency": {"bad": 1, "weight": -1}})
        self.assertEqual(len(caught.exception.problems), 3)

    def test_a_null_setting_is_rejected_rather_than_meaning_the_default(self):
        """`loss_options` merges the block over the defaults, so null would replace one.

        `"weight": null` would reach the trainer as None and raise in `float(...)`;
        `"enabled": null` is worse -- `bool(None)` is False, so the term would silently
        not run while the recorded config looked like it asked for it.
        """
        for key in ("enabled", "weight", "detach_reference"):
            with self.subTest(key=key), self.assertRaises(InvalidConfigError) as caught:
                self._config({"consistency": {key: None}})
            self.assertIn(f"loss.consistency.{key}", caught.exception.problems[0])
