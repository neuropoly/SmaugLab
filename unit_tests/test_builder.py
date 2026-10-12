"""The registry-driven builder: where transform order comes from, and what it rejects."""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar

from smauglab import registry
from smauglab.config import OrderSource, PipelineMode, SmaugConfig, load_config
from smauglab.registry import AugType, Backend, InvalidConfigError
from smauglab.transforms.build import SHARED_GROUPS, build_gpu_pipeline, build_gpu_stages, build_transforms
from unit_tests.helpers import SmaugLabTestCase, all_config_paths, gpu_config_paths, requires_external_asset


def _signature(transform: Any) -> Any:
    """A structural fingerprint: class name, and a bucket's full contents."""
    inner = getattr(transform, "transforms_list", None)
    if inner is None:
        return type(transform).__name__
    return {
        "bucket": type(transform).__name__,
        "label": getattr(transform, "label", None),
        "num": getattr(transform, "num_transforms", None),
        "p": float(getattr(transform, "p", -1)),
        "random_order": bool(getattr(transform, "random_order", False)),
        "children": [_signature(child) for child in inner],
    }


def _buildable_configs():
    """Every shipped config with a GPU section that builds on this machine."""
    for path in all_config_paths():
        if requires_external_asset(path):
            continue
        try:
            config = load_config(str(path))
        except InvalidConfigError:
            continue
        if config.names(Backend.GPU):
            yield path, config


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


class TestSharedStageSplit(SmaugLabTestCase):
    """`build_gpu_stages` splits the pipeline for the consistency loss without moving anything."""

    def test_the_shared_stage_is_only_the_geometric_group(self):
        geometric = set(registry.names(Backend.GPU, group=AugType.GEO))
        for path, config in _buildable_configs():
            with self.subTest(config=path.name):
                shared, _ = build_gpu_stages(
                    config.section(Backend.GPU),
                    mode=config.pipeline_mode(),
                    options=config.pipeline_options("random_choose"),
                    order_source=config.order_source(),
                    split_shared=True,
                )
                names = {type(t).__name__ for t in shared}
                self.assertEqual(names, geometric & set(config.names(Backend.GPU)))

    def test_the_crop_never_enters_the_shared_stage(self):
        """The whole point of AugType.OCC: the occlusion stays behind the intensity stage."""
        occluders = set(registry.names(Backend.GPU, group=AugType.OCC))
        self.assertIn("RandomCropTransformGPU", occluders)
        for path, config in _buildable_configs():
            if not occluders & set(config.names(Backend.GPU)):
                continue
            for mode in PipelineMode:
                with self.subTest(config=path.name, mode=mode.value):
                    shared, augmented = build_gpu_stages(
                        config.section(Backend.GPU),
                        mode=mode,
                        options=config.pipeline_options("random_choose"),
                        order_source=config.order_source(),
                        split_shared=True,
                    )
                    self.assertFalse(occluders & {type(t).__name__ for t in shared})
                    flat = json.dumps([_signature(t) for t in augmented])
                    self.assertIn("RandomCropTransformGPU", flat)

    def test_the_two_stages_concatenate_to_the_unsplit_pipeline(self):
        """The split is a partition, not a rearrangement, in every pipeline mode."""
        for path, config in _buildable_configs():
            for mode in PipelineMode:
                with self.subTest(config=path.name, mode=mode.value):
                    kwargs = {
                        "mode": mode,
                        "options": config.pipeline_options("random_choose"),
                        "order_source": config.order_source(),
                    }
                    shared, augmented = build_gpu_stages(config.section(Backend.GPU), split_shared=True, **kwargs)
                    unsplit = build_gpu_pipeline(config.section(Backend.GPU), **kwargs)
                    self.assertEqual(
                        [_signature(t) for t in shared + augmented],
                        [_signature(t) for t in unsplit],
                    )

    def test_not_splitting_leaves_the_shared_stage_empty(self):
        shared, augmented = build_gpu_stages({"RandomFlipTransformGPU": {"p": 1.0}}, split_shared=False)
        self.assertEqual(shared, [])
        self.assertEqual([type(t).__name__ for t in augmented], ["RandomFlipTransformGPU"])

    def test_a_shared_transform_out_of_prefix_position_is_rejected(self):
        """Hoisting past a transform that stays behind would silently reorder the pipeline."""
        section = {"ZscoreNormalizationGPU": {"p": 1.0}, "RandomFlipTransformGPU": {"p": 1.0}}
        with self.assertRaises(InvalidConfigError) as caught:
            build_gpu_stages(section, order_source=OrderSource.CONFIG, split_shared=True)
        self.assertIn("RandomFlipTransformGPU", caught.exception.problems[0])
        self.assertIn("prefix", caught.exception.problems[0])

    def test_the_same_section_still_builds_unsplit(self):
        """The prefix rule is the consistency loss's constraint, not the pipeline's."""
        section = {"ZscoreNormalizationGPU": {"p": 1.0}, "RandomFlipTransformGPU": {"p": 1.0}}
        built = build_gpu_pipeline(section, order_source=OrderSource.CONFIG)
        self.assertEqual([type(t).__name__ for t in built], ["ZscoreNormalizationGPU", "RandomFlipTransformGPU"])

    def test_the_geometric_group_is_a_prefix_of_pipeline_order(self):
        """What makes the split free. `AugType.OCC` exists to keep it true."""
        order = registry.PIPELINE_ORDER[Backend.GPU]
        positions = [i for i, name in enumerate(order) if registry.get(name, Backend.GPU).group in SHARED_GROUPS]
        self.assertEqual(positions, list(range(len(positions))))


class TestShippedPipelinesAreUnchanged(SmaugLabTestCase):
    """A content hash over every shipped GPU pipeline, in every mode.

    `AugType.OCC` moved `RandomCropTransformGPU` out of `AugType.GEO`, and `_buckets`
    reads the group to decide what leads the pipeline. Nothing was supposed to move --
    `build.SEQUENTIAL_GROUPS` lists OCC beside GEO for exactly that reason -- and this
    is the assertion that says so for all 84 (config, mode) pairs at once.

    Regenerate the hash only when a config or a transform class name is *deliberately*
    changed; the failure message prints the new one.
    """

    SIGNATURE = "1a9bfbc31e5e15416c415189dae4aa70c0e9523534a21c1b05c185b37ffcad44"
    COUNT = 84

    def _structures(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for path, config in _buildable_configs():
            for mode in PipelineMode:
                out[f"{path.name}:{mode.value}"] = [
                    _signature(t)
                    for t in build_gpu_pipeline(
                        config.section(Backend.GPU),
                        mode=mode,
                        options=config.pipeline_options("random_choose"),
                        source=config.source,
                        order_source=config.order_source(),
                    )
                ]
        return out

    def test_the_structure_signature_is_stable(self):
        structures = self._structures()
        if len(structures) != self.COUNT:
            self.skipTest(f"{len(structures)} of {self.COUNT} pipelines built here; the hash is not comparable")
        blob = json.dumps(structures, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(blob.encode()).hexdigest()
        self.assertEqual(
            digest,
            self.SIGNATURE,
            "a shipped GPU pipeline changed shape. If that was intended, set SIGNATURE to " + digest,
        )


class TestGpuConfigsAreCoveredBySplitting(SmaugLabTestCase):
    def test_every_gpu_config_can_be_split(self):
        """No shipped GPU config trips the prefix rule, so none of them blocks the loss."""
        for path in gpu_config_paths():
            if requires_external_asset(path):
                continue
            config = load_config(str(path))
            with self.subTest(config=path.name):
                shared, augmented = build_gpu_stages(
                    config.section(Backend.GPU),
                    mode=config.pipeline_mode(),
                    options=config.pipeline_options("random_choose"),
                    order_source=config.order_source(),
                    split_shared=True,
                )
                self.assertTrue(shared or augmented)
