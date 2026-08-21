"""The nnU-Net trainer, which is now one class driven entirely by the config.

Three trainers used to encode the CPU/GPU split in their class names. The config
already carries it -- which sections are populated says what runs -- so there is one
class, and these tests pin that it reproduces what each of the three used to build.

`get_training_transforms` is a staticmethod (nnU-Net's contract), so it can be driven
directly: no plans.json, no dataset, no GPU.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
import warnings
from pathlib import Path

from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from smauglab.config import load_config
from smauglab.registry import Backend
from smauglab.transforms.build import PipelineMode, build_cpu_pipeline

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "smauglab" / "configs"
PATCH = (24, 24, 24)
ROTATION = (-10, 10)
DS_SCALES = [[1, 1, 1], [0.5, 0.5, 0.5]]

# Raised at import so it works under pytest and `python -m unittest` alike; a
# `pytestmark` would only be understood by one of them.
if importlib.util.find_spec("nnunetv2") is None:
    raise unittest.SkipTest("the trainer needs the nnunetv2 extra")


def flatten(transform) -> list[str]:
    """Class names, descending into Compose and unwrapping RandomTransform."""
    if isinstance(transform, ComposeTransforms):
        return [name for child in transform.transforms for name in flatten(child)]
    wrapped = getattr(transform, "transform", None)
    if type(transform).__name__ == "RandomTransform" and wrapped is not None:
        return [type(wrapped).__name__]
    return [type(transform).__name__]


def training_transforms(config_name: str, **overrides) -> list[str]:
    from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

    os.environ["SMAUGLAB_PARAMS_JSON"] = str(CONFIGS / config_name)
    kwargs = {
        "patch_size": PATCH,
        "rotation_for_DA": ROTATION,
        "deep_supervision_scales": DS_SCALES,
        "mirror_axes": (0, 1, 2),
        "do_dummy_2d_data_aug": False,
        "use_mask_for_norm": None,
        "is_cascaded": False,
        "foreground_labels": None,
        "regions": None,
        "ignore_label": None,
    }
    kwargs.update(overrides)
    return flatten(nnUNetTrainerDAExtGPU.get_training_transforms(**kwargs))


def cpu_block(config_name: str) -> list[str]:
    """The CPU pipeline a config asks for, independent of the trainer."""
    config = load_config(str(CONFIGS / config_name))
    built = build_cpu_pipeline(config.section(Backend.CPU), do_dummy_2d_data_aug=False, patch_size=PATCH, rotation=ROTATION)
    return [name for transform in built for name in flatten(transform)]


class TestOnlyOneTrainerRemains(unittest.TestCase):
    def test_the_load_bearing_name_survives(self):
        """nnU-Net writes the class name into every checkpoint and resolves the class
        from it at inference, and several hundred trained runs record this one."""
        from smauglab.trainers import nnUNetTrainerDAExt

        self.assertTrue(hasattr(nnUNetTrainerDAExt, "nnUNetTrainerDAExtGPU"))

    def test_the_backend_specific_trainers_are_gone(self):
        """They differed only in which config they defaulted to, which the config
        itself now says. Neither had a single run on disk."""
        from smauglab.trainers import nnUNetTrainerDAExt

        for gone in ("nnUNetTrainerDAExtHybrid", "nnUNetTrainerDAExt"):
            with self.subTest(trainer=gone):
                self.assertFalse(hasattr(nnUNetTrainerDAExt, gone))


class TestCompositionMatchesTheOldTrainers(unittest.TestCase):
    """Each config must build what its dedicated trainer used to build."""

    def test_gpu_config_matches_the_old_gpu_trainer(self):
        self.assertEqual(training_transforms("transform_params_gpu.json"), ["SpatialTransform", "RemoveLabelTransform"])

    def test_hybrid_config_matches_the_old_hybrid_trainer(self):
        expected = [*cpu_block("transform_params_hybrid.json"), "RemoveLabelTransform"]
        self.assertEqual(training_transforms("transform_params_hybrid.json"), expected)

    def test_cpu_config_builds_the_whole_cpu_pipeline(self):
        got = training_transforms("transform_params.json")
        self.assertEqual(got[: len(cpu_block("transform_params.json"))], cpu_block("transform_params.json"))
        self.assertIn("SpatialTransform", got)

    def test_the_cpu_config_carries_the_spatial_transform_the_trainer_used_to_hardcode(self):
        """The old CPU trainer appended a SpatialTransform with every probability at
        0 -- a no-op that only enforces the patch size. The merged trainer builds only
        what the config names, so the config has to say it."""
        section = load_config(str(CONFIGS / "transform_params.json")).section(Backend.CPU)
        self.assertIn("SpatialTransform", section)
        spatial = section["SpatialTransform"]
        self.assertEqual(spatial["p_rotation"], 0)
        self.assertEqual(spatial["p_scaling"], 0)
        self.assertEqual(spatial["p_elastic_deform"], 0)
        self.assertEqual(spatial["mode_seg"], "nearest")


class TestDeepSupervisionPlacement(unittest.TestCase):
    """Downsampling must follow whatever last deformed the mask.

    With GPU augmentations that is `train_step`, so it happens there; without them
    nothing touches the mask after the dataloader and it belongs there. Getting this
    backwards would train against targets that no longer match the image.
    """

    def test_a_gpu_config_leaves_downsampling_to_train_step(self):
        for config in ("transform_params_gpu.json", "transform_params_hybrid.json"):
            with self.subTest(config=config):
                self.assertNotIn("DownsampleSegForDSTransform", training_transforms(config))

    def test_a_cpu_only_config_downsamples_in_the_dataloader(self):
        self.assertIn("DownsampleSegForDSTransform", training_transforms("transform_params.json"))

    def test_no_downsampling_when_no_scales_are_requested(self):
        got = training_transforms("transform_params.json", deep_supervision_scales=None)
        self.assertNotIn("DownsampleSegForDSTransform", got)


class TestDummy2D(unittest.TestCase):
    def test_the_converters_bracket_the_spatial_transform(self):
        got = training_transforms("transform_params_gpu.json", do_dummy_2d_data_aug=True)
        spatial = got.index("SpatialTransform")
        self.assertEqual(got[spatial - 1], "Convert3DTo2DTransform")
        self.assertEqual(got[spatial + 1], "Convert2DTo3DTransform")


class TestConfigResolution(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("SMAUGLAB_PARAMS_JSON", "SMAUGLAB_PARAMS_GPU_JSON")}
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_the_packaged_default_is_used_when_nothing_is_set(self):
        from smauglab.trainers.nnUNetTrainerDAExt import resolve_config_path

        self.assertTrue(resolve_config_path().endswith("transform_params_gpu.json"))

    def test_the_new_variable_wins(self):
        from smauglab.trainers.nnUNetTrainerDAExt import resolve_config_path

        os.environ["SMAUGLAB_PARAMS_JSON"] = "/new.json"
        os.environ["SMAUGLAB_PARAMS_GPU_JSON"] = "/old.json"
        self.assertEqual(resolve_config_path(), "/new.json")

    def test_the_old_variable_still_works_but_warns(self):
        """segtransferaug/run_trainings.py sets it, and it drives every historical run."""
        from smauglab.trainers.nnUNetTrainerDAExt import resolve_config_path

        os.environ["SMAUGLAB_PARAMS_GPU_JSON"] = "/old.json"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(resolve_config_path(), "/old.json")
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))


class TestPipelineMode(unittest.TestCase):
    def test_configs_default_to_the_sequential_pipeline(self):
        self.assertIs(load_config(str(CONFIGS / "transform_params_gpu.json")).pipeline_mode(), PipelineMode.SEQUENTIAL)

    def test_the_list_configs_ask_for_the_random_order_pipeline(self):
        """They carry a random_choose block, so they were written for the ChooseX
        trainers; the config says that now instead of the class name."""
        for config in sorted(CONFIGS.glob("*-List*.json")):
            with self.subTest(config=config.name):
                self.assertIs(load_config(str(config)).pipeline_mode(), PipelineMode.RANDOM_ORDER)


if __name__ == "__main__":
    unittest.main()
