"""The registry-driven builder, and the strict loading in front of it.

These pin the behaviours the four hand-written ladders used to encode implicitly:
pipeline order, the TA/GE bucketing of the random-order modes, the runtime context
nnU-Net supplies, and the parameter adapters. They also pin the failure modes --
before this, a config naming a transform that did not exist simply did nothing.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from smauglab import registry
from smauglab.config import SmaugConfig, load_config, validate_file
from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import (
    PipelineMode,
    build_cpu_pipeline,
    build_gpu_pipeline,
    build_transforms,
    validate_section,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GPU = REPO / "smauglab" / "configs" / "transform_params_gpu_default01-23.json"
LIST_GPU = REPO / "smauglab" / "configs" / "transform_params_gpu_default01-23-List.json"
HYBRID = REPO / "smauglab" / "configs" / "transform_params_hybrid.json"
PATCH = (24, 24, 24)


def names(transforms) -> list[str]:
    return [type(t).__name__ for t in transforms]


class TestPipelineOrder(unittest.TestCase):
    def test_order_comes_from_the_registry_not_the_file(self):
        """A config's key order has never matched the pipeline order. Honouring the
        file would silently reorder every pipeline the moment someone tidied one."""
        section = load_config(str(DEFAULT_GPU)).section(Backend.GPU)
        shuffled = dict(reversed(list(section.items())))

        straight = build_transforms(section, Backend.GPU)
        reversed_ = build_transforms(shuffled, Backend.GPU)
        self.assertEqual([e.name for _, e in straight], [e.name for _, e in reversed_])

    def test_built_order_is_ascending_registry_order(self):
        built = build_transforms(load_config(str(DEFAULT_GPU)).section(Backend.GPU), Backend.GPU)
        orders = [entry.order for _, entry in built]
        self.assertEqual(orders, sorted(orders))


class TestPipelineModes(unittest.TestCase):
    def setUp(self):
        self.section = load_config(str(LIST_GPU)).section(Backend.GPU)

    def test_sequential_returns_every_transform(self):
        built = build_gpu_pipeline(self.section, mode=PipelineMode.SEQUENTIAL)
        self.assertEqual(len(built), len([k for k in self.section if not k.startswith("_")]))

    def test_random_order_buckets_transfer_and_enhancement(self):
        built = build_gpu_pipeline(self.section, mode=PipelineMode.RANDOM_ORDER)
        self.assertEqual(names(built).count("RandomChooseXTransformsGPU"), 2)
        # Geometry, plus the low-res transform, lead in order.
        self.assertEqual(names(built)[:3], ["RandomFlipTransformGPU", "RandomAffineGPU", "RandomLowResTransformGPU"])

    def test_random_order_ta_buckets_only_the_transfer_group(self):
        built = build_gpu_pipeline(self.section, mode=PipelineMode.RANDOM_ORDER_TA)
        self.assertEqual(names(built).count("RandomChooseXTransformsGPU"), 1)

    def test_low_res_is_hoisted_only_when_there_is_a_ge_bucket(self):
        """force_sequential means "never inside the GE bucket", so it only bites in
        RANDOM_ORDER. RANDOM_ORDER_TA leaves it in its ordinary GE position -- which
        is what the two hand-written pipelines did, differing from each other."""
        full = names(build_gpu_pipeline(self.section, mode=PipelineMode.RANDOM_ORDER))
        ta_only = names(build_gpu_pipeline(self.section, mode=PipelineMode.RANDOM_ORDER_TA))
        self.assertLess(full.index("RandomLowResTransformGPU"), full.index("RandomChooseXTransformsGPU"))
        self.assertGreater(ta_only.index("RandomLowResTransformGPU"), ta_only.index("RandomChooseXTransformsGPU"))


class TestCpuBuilder(unittest.TestCase):
    def setUp(self):
        self.section = load_config(str(HYBRID)).section(Backend.CPU)

    def test_probability_goes_on_the_wrapper_not_the_transform(self):
        built = build_cpu_pipeline(self.section, do_dummy_2d_data_aug=False, patch_size=PATCH, rotation=(-10, 10))
        wrapped = [t for t in built if type(t).__name__ == "RandomTransform"]
        self.assertTrue(wrapped, "expected batchgeneratorsv2 transforms to be wrapped")
        self.assertTrue(all(hasattr(t, "apply_probability") for t in wrapped))

    def test_context_is_injected_and_not_taken_from_the_config(self):
        built = build_cpu_pipeline(self.section, do_dummy_2d_data_aug=False, patch_size=PATCH, rotation=(-7, 7))
        spatial = next(t for t in built if type(t).__name__ == "SpatialTransform")
        self.assertEqual(tuple(spatial.patch_size), PATCH)
        self.assertEqual(spatial.rotation, (-7, 7))

    def test_dummy_2d_brackets_the_spatial_transform(self):
        """The converters are pipeline structure, not augmentations, so they carry
        no registry entry and are emitted around the slot."""
        built = names(build_cpu_pipeline(self.section, do_dummy_2d_data_aug=True, patch_size=PATCH, rotation=(-10, 10)))
        spatial = built.index("SpatialTransform")
        self.assertEqual(built[spatial - 1], "Convert3DTo2DTransform")
        self.assertEqual(built[spatial + 1], "Convert2DTo3DTransform")

    def test_dummy_2d_drops_the_leading_patch_axis(self):
        built = build_cpu_pipeline(self.section, do_dummy_2d_data_aug=True, patch_size=PATCH, rotation=(-10, 10))
        spatial = next(t for t in built if type(t).__name__ == "SpatialTransform")
        self.assertEqual(tuple(spatial.patch_size), PATCH[1:])

    def test_ranges_are_wrapped_by_their_adapter(self):
        """BGContrast samples 50/50 from [lo, 1] and [max(lo, 1), hi]; the bare tuple
        would be sampled uniformly, which is a different augmentation."""
        built = build_cpu_pipeline(self.section, do_dummy_2d_data_aug=False, patch_size=PATCH, rotation=(-10, 10))
        contrast = next(t.transform for t in built if type(getattr(t, "transform", None)).__name__ == "ContrastTransform")
        self.assertEqual(type(contrast.contrast_range).__name__, "BGContrast")


class TestStrictValidation(unittest.TestCase):
    def test_unknown_augmentation_is_rejected_with_a_suggestion(self):
        problems = validate_section({"ScharrTransform": {"p": 0.1}}, Backend.GPU)
        self.assertEqual(len(problems), 1)
        self.assertIn("RandomScharrGPU", problems[0])

    def test_unknown_parameter_is_rejected(self):
        problems = validate_section({"RandomGaussianNoiseGPU": {"p": 0.1, "stdd": 1.0}}, Backend.GPU)
        self.assertIn("Did you mean: std?", problems[0])

    def test_probability_is_rejected_and_points_at_p(self):
        problems = validate_section({"RandomGaussianNoiseGPU": {"probability": 0.1}}, Backend.GPU)
        self.assertIn("'probability' -> p", problems[0])

    def test_context_parameter_in_a_config_is_rejected(self):
        problems = validate_section(
            {"SpatialTransform": {"rotation": [0, 1], "patch_center_dist_from_border": 0, "random_crop": False}},
            Backend.CPU,
        )
        self.assertIn("supplied by the trainer", problems[0])

    def test_missing_required_parameter_is_reported(self):
        problems = validate_section({"SpatialTransform": {}}, Backend.CPU)
        self.assertIn("missing required parameter", problems[0])

    def test_every_problem_is_reported_at_once(self):
        """One pass to fix a broken file, not one pytest run per mistake."""
        payload = {
            "GPU": {
                "ScharrTransform": {"p": 0.1},
                "RandomGaussianNoiseGPU": {"probability": 0.2, "stdd": 1.0},
            }
        }
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig(payload, source="broken.json")
        self.assertEqual(len(caught.exception.problems), 3)
        self.assertIn("broken.json", str(caught.exception))

    def test_a_flat_config_is_no_longer_accepted(self):
        """`GaussianBlurTransform` meant different transforms to the two builders,
        so a document that does not say which backend it is cannot be resolved."""
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"FlipTransform": {"probability": 0.5}}, source="legacy.json")
        self.assertIn("migration/", str(caught.exception))

    def test_unknown_top_level_section_is_rejected(self):
        with self.assertRaises(InvalidConfigError):
            SmaugConfig({"GPU": {}, "typo": {}}, source="x.json")

    def test_underscore_keys_are_ignored(self):
        config = SmaugConfig({"_comment1": "note", "GPU": {"_note": "x", "RandomFlipTransformGPU": {"p": 0.5}}})
        self.assertEqual(config.names(Backend.GPU), ["RandomFlipTransformGPU"])


class TestConfigLoading(unittest.TestCase):
    def test_every_shipped_config_validates(self):
        for path in sorted((REPO / "smauglab" / "configs").glob("*.json")):
            with self.subTest(config=path.name):
                self.assertEqual(validate_file(path), [])

    def test_sections_are_copies_so_the_cache_cannot_be_poisoned(self):
        config = load_config(str(DEFAULT_GPU))
        section = config.section(Backend.GPU)
        section.clear()
        self.assertNotEqual(config.section(Backend.GPU), {})

    def test_the_same_path_is_parsed_once(self):
        """The nnU-Net trainer reads the same file from a staticmethod as well as
        from the instance; the cache is what stops it being parsed twice."""
        self.assertIs(load_config(str(DEFAULT_GPU)), load_config(str(DEFAULT_GPU)))

    def test_pipeline_options_default_to_empty(self):
        """The old builder did `.get("RandomChooseXTransforms")` then `.get()` on the
        result, which raised AttributeError on every config without the block."""
        self.assertEqual(load_config(str(DEFAULT_GPU)).pipeline_options("random_choose"), {})

    def test_the_shipped_template_names_every_registered_augmentation(self):
        template = json.loads((REPO / "smauglab" / "configs" / "all_augmentations.json").read_text())
        for backend in (Backend.GPU, Backend.CPU):
            with self.subTest(backend=backend.value):
                self.assertEqual(set(template[backend.value]), set(registry.names(backend)))


if __name__ == "__main__":
    unittest.main()
