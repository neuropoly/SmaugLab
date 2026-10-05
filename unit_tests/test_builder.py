"""The registry-driven builder: where transform order comes from, and what it rejects."""

from __future__ import annotations

from typing import ClassVar

from smauglab.config import OrderSource, SmaugConfig
from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import build_gpu_pipeline, build_transforms
from unit_tests.helpers import SmaugLabTestCase


class TestOrderSource(SmaugLabTestCase):
    """Registry order by default; config key order when the config asks for it."""

    #: Deliberately in the opposite order to PIPELINE_ORDER, so the two sources disagree.
    SECTION: ClassVar[dict] = {
        "ZscoreNormalizationGPU": {"p": 1.0},
        "RandomFlipTransformGPU": {"p": 1.0},
    }

    def test_registry_order_ignores_the_order_of_the_keys(self):
        built = build_transforms(self.SECTION, Backend.GPU)
        self.assertEqual([type(t).__name__ for t, _ in built], ["RandomFlipTransformGPU", "ZscoreNormalizationGPU"])

    def test_config_order_keeps_the_order_of_the_keys(self):
        built = build_transforms(self.SECTION, Backend.GPU, order_source=OrderSource.CONFIG)
        self.assertEqual([type(t).__name__ for t, _ in built], ["ZscoreNormalizationGPU", "RandomFlipTransformGPU"])

    def test_the_pipeline_honours_the_config_setting(self):
        payload = {"GPU": dict(self.SECTION), "pipeline": {"order": "config"}}
        config = SmaugConfig(payload)
        built = build_gpu_pipeline(config.section(Backend.GPU), order_source=config.order_source())
        self.assertEqual([type(t).__name__ for t in built], ["ZscoreNormalizationGPU", "RandomFlipTransformGPU"])


class TestValidation(SmaugLabTestCase):
    def test_an_unknown_augmentation_stops_the_build(self):
        with self.assertRaises(InvalidConfigError):
            build_transforms({"NotAThing": {}}, Backend.GPU)

    def test_an_unknown_parameter_stops_the_build(self):
        with self.assertRaises(InvalidConfigError):
            build_transforms({"RandomFlipTransformGPU": {"nope": 1}}, Backend.GPU)

    def test_comment_keys_are_skipped(self):
        built = build_transforms({"_note": "hi", "RandomFlipTransformGPU": {"p": 1.0}}, Backend.GPU)
        self.assertEqual(len(built), 1)

    def test_every_problem_is_reported_at_once(self):
        with self.assertRaises(InvalidConfigError) as caught:
            build_transforms({"Nope": {}, "RandomFlipTransformGPU": {"bad": 1}}, Backend.GPU)
        self.assertEqual(len(caught.exception.problems), 2)
