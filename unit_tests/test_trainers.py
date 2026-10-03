"""The nnU-Net trainer, which is now one class driven entirely by the config.

Three trainers used to encode the CPU/GPU split in their class names. The config
already carries it -- which sections are populated says what runs -- so there is one
class, and these tests pin that it reproduces what each of the three used to build.

`get_training_transforms` is a staticmethod (nnU-Net's contract), so it can be driven
directly: no plans.json, no dataset, no GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

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


class TestConstructorMatchesNnUNet(unittest.TestCase):
    """The trainers are instantiated by nnU-Net, so their signature is a contract.

    `nnUNetv2_train` constructs the trainer itself, so a parameter that exists upstream
    and not here is a `TypeError` before the first batch, and one that exists here and
    not upstream breaks the `super().__init__` call instead. nnU-Net has moved this
    signature between releases -- 2.4 had `unpack_dataset` between `dataset_json` and
    `device`, 2.6 does not -- so which nnU-Net is installed decides what is correct,
    and pinning it against the installed one is the only check that stays true.

    Nothing caught this before: every other test here drives `get_training_transforms`,
    which is a staticmethod, so the constructor is never called. Comparing signatures
    needs neither plans, nor a dataset, nor a GPU.
    """

    TRAINERS = ("nnUNetTrainerDAExtGPU", "nnUNetTrainerTest", "nnUNetTrainerTestGPU")

    @staticmethod
    def _trainer(name):
        module = "nnUNetTrainerDAExt" if name == "nnUNetTrainerDAExtGPU" else "nnUNetTrainerTest"
        return getattr(importlib.import_module(f"smauglab.trainers.{module}"), name)

    def test_every_upstream_parameter_is_accepted(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        upstream = set(inspect.signature(nnUNetTrainer.__init__).parameters)
        for name in self.TRAINERS:
            with self.subTest(trainer=name):
                ours = set(inspect.signature(self._trainer(name).__init__).parameters)
                self.assertEqual(upstream - ours, set(), f"{name} would reject arguments nnU-Net passes")

    def test_the_parameter_order_matches(self):
        """They are forwarded to `super().__init__`, and a caller may pass positionally."""
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        upstream = list(inspect.signature(nnUNetTrainer.__init__).parameters)
        for name in self.TRAINERS:
            with self.subTest(trainer=name):
                ours = list(inspect.signature(self._trainer(name).__init__).parameters)
                self.assertEqual(ours[: len(upstream)], upstream)


class TestPipelineMode(unittest.TestCase):
    def test_configs_default_to_the_sequential_pipeline(self):
        self.assertIs(load_config(str(CONFIGS / "transform_params_gpu.json")).pipeline_mode(), PipelineMode.SEQUENTIAL)

    def test_the_list_configs_ask_for_the_random_order_pipeline(self):
        """They carry a random_choose block, so they were written for the ChooseX
        trainers; the config says that now instead of the class name."""
        for config in sorted(CONFIGS.glob("*-List*.json")):
            with self.subTest(config=config.name):
                self.assertIs(load_config(str(config)).pipeline_mode(), PipelineMode.RANDOM_ORDER)


class TestNonFiniteLossGuard(unittest.TestCase):
    """A NaN loss used to leave no trace at all.

    `GradScaler` skips the step when the gradients are not finite, silently, so a run
    poisoned by one bad augmentation keeps printing epochs and simply never learns. The
    only symptom was `train_loss nan` in the epoch line -- `np.mean` carrying a single
    bad batch through the whole mean -- which names neither the batch nor the cause.

    Driving this needs a trainer instance, which the rest of this file never builds:
    `__init__` wants plans, a dataset_json and an output folder to copy the config into.
    `__new__` sidesteps all of it, and `train_step` only ever touches seven attributes.
    """

    def _trainer(self, loss, transforms=None):
        import torch

        from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

        trainer = nnUNetTrainerDAExtGPU.__new__(nnUNetTrainerDAExtGPU)
        trainer.device = torch.device("cpu")  # so autocast is skipped via dummy_context
        trainer.transforms = transforms
        trainer.grad_scaler = None
        trainer.network = torch.nn.Conv3d(1, 2, 1)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.1)
        trainer.loss = loss
        trainer.current_epoch = 0
        trainer.num_iterations_per_epoch = 250
        trainer._nan_steps_this_epoch = 0
        trainer._nan_steps_total = 0
        trainer._get_deep_supervision_scales = lambda: None
        # An instance attribute shadowing the bound method: capturing the log this way
        # needs no output folder and no open file handle.
        trainer.logged = []
        trainer.print_to_log_file = lambda *args, **_kwargs: trainer.logged.append(" ".join(str(a) for a in args))
        return trainer

    def _batch(self, key="case_a"):
        import torch

        # Non-zero data deliberately: with an all-zero input a Conv3d's weight gradient
        # is zero too, so "did the step move the weights" would be unanswerable. The
        # target is empty, which is the shape this whole guard exists for.
        return {
            "data": torch.full((1, 1, 4, 4, 4), 0.5),
            "target": torch.zeros(1, 1, 4, 4, 4),
            "keys": [key],
        }

    def test_a_non_finite_loss_from_the_criterion_is_reported(self):
        # `out.sum() * nan` keeps a real autograd graph, so backward() still works.
        trainer = self._trainer(loss=lambda out, _tgt: out.sum() * float("nan"))

        trainer.train_step(self._batch())

        self.assertEqual(len(trainer.logged), 1, trainer.logged)
        self.assertIn("verdict=loss", trainer.logged[0])
        self.assertIn("case_a", trainer.logged[0])

    def test_a_non_finite_batch_is_blamed_on_the_augmentation(self):
        trainer = self._trainer(
            loss=lambda out, _tgt: out.mean(),
            transforms=lambda data, target: (data * float("nan"), target),
        )

        trainer.train_step(self._batch())

        self.assertIn("verdict=augmentation", trainer.logged[0])
        self.assertIn("nonfinite_samples=[0]", trainer.logged[0])

    def test_a_healthy_step_logs_nothing_and_still_updates_the_weights(self):
        """The guard reports; it must not become a filter on good steps."""
        import torch

        trainer = self._trainer(loss=lambda out, _tgt: (out - 1.0).pow(2).mean())
        before = trainer.network.weight.detach().clone()

        trainer.train_step(self._batch())

        self.assertEqual(trainer.logged, [])
        self.assertEqual(trainer._nan_steps_total, 0)
        self.assertFalse(bool(torch.equal(before, trainer.network.weight.detach())), "a healthy step was skipped")

    def test_a_non_finite_step_leaves_the_weights_alone_without_a_scaler(self):
        """CPU and MPS have no GradScaler to skip for them, so the guard has to."""
        import torch

        trainer = self._trainer(loss=lambda out, _tgt: out.sum() * float("nan"))
        before = trainer.network.weight.detach().clone()

        trainer.train_step(self._batch())

        self.assertTrue(bool(torch.equal(before, trainer.network.weight.detach())))

    def test_reports_are_rate_limited_and_summarised(self):
        from smauglab.trainers.nnUNetTrainerDAExt import NAN_REPORTS_PER_EPOCH

        trainer = self._trainer(loss=lambda out, _tgt: out.sum() * float("nan"))
        for i in range(10):
            trainer.train_step(self._batch(f"case_{i}"))

        details = [line for line in trainer.logged if "verdict=" in line]
        suppressed = [line for line in trainer.logged if "suppressed" in line]
        self.assertEqual(len(details), NAN_REPORTS_PER_EPOCH)
        self.assertEqual(len(suppressed), 1, "the suppression notice must be printed exactly once")
        self.assertEqual(trainer._nan_steps_this_epoch, 10)

        trainer.logged.clear()
        # Only the subclass's own contribution is under test; the parent hook wants a
        # logger and an lr_scheduler this bare instance does not have.
        parent = type(trainer).__mro__[1]
        with mock.patch.object(parent, "on_train_epoch_end", lambda _self, _outputs: None):
            type(trainer).on_train_epoch_end(trainer, [])

        self.assertEqual(len(trainer.logged), 1)
        self.assertIn("10/250", trainer.logged[0])


class TestTargetReachesTheDeviceEitherShape(unittest.TestCase):
    """`train_step` gets a tensor or a list, depending on the config.

    `get_training_transforms` passes the dataloader `deep_supervision_scales=None` when
    the config has a GPU section: the mask is still going to be augmented in `train_step`,
    so the downsampling has to happen after that and the dataloader yields one
    full-resolution mask. A CPU-only config has nothing touching the mask after the
    dataloader, so the downsampling stays there and the target arrives as the list of
    deep-supervision levels.

    `train_step` assumed the tensor, so `target.to(...)` raised `AttributeError` on the
    first batch of every CPU-only run. Upstream `validation_step` has always branched on
    it, which is why only training broke.
    """

    def _trainer(self):
        import torch

        from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

        trainer = nnUNetTrainerDAExtGPU.__new__(nnUNetTrainerDAExtGPU)
        trainer.device = torch.device("cpu")
        trainer.transforms = None  # a CPU-only config builds no GPU pipeline
        trainer.grad_scaler = None
        trainer.network = torch.nn.Conv3d(1, 2, 1)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.1)
        trainer.current_epoch = 0
        trainer._nan_steps_this_epoch = 0
        trainer._nan_steps_total = 0
        trainer.seen = []
        # Record what the loss was handed, which is the thing under test.
        trainer.loss = lambda out, tgt: (trainer.seen.append(tgt), out.mean())[1]
        return trainer

    def test_a_list_target_survives_the_device_move(self):
        import torch

        trainer = self._trainer()
        target = [torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 1, 2, 2, 2)]

        trainer.train_step({"data": torch.full((1, 1, 4, 4, 4), 0.5), "target": target, "keys": ["case_a"]})

        seen = trainer.seen[0]
        self.assertIsInstance(seen, list, "the deep-supervision levels were collapsed")
        self.assertEqual([tuple(t.shape) for t in seen], [(1, 1, 4, 4, 4), (1, 1, 2, 2, 2)])
        self.assertTrue(all(t.device == trainer.device for t in seen))

    def test_a_tensor_target_still_works(self):
        import torch

        trainer = self._trainer()

        trainer.train_step({"data": torch.full((1, 1, 4, 4, 4), 0.5), "target": torch.zeros(1, 1, 4, 4, 4), "keys": ["case_a"]})

        self.assertIsInstance(trainer.seen[0], torch.Tensor)


class TestTrainingTransformsMatchesNnUNet(unittest.TestCase):
    """The same contract for `get_training_transforms`, which nnU-Net also calls.

    `get_dataloaders` calls it by keyword, so a parameter upstream declares and the
    override does not is a TypeError before the first batch. nnU-Net 2.4 passed
    `order_resampling_data=` and `order_resampling_seg=`; 2.5 moved the pipeline to
    batchgeneratorsv2 and dropped them, which is why `pyproject.toml` pins
    `nnunetv2 >= 2.5`. Checking against the installed version makes an unsupported
    one fail here rather than an hour into a run.
    """

    def test_every_upstream_parameter_is_accepted(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        upstream = set(inspect.signature(nnUNetTrainer.get_training_transforms).parameters)
        for name in TestConstructorMatchesNnUNet.TRAINERS:
            with self.subTest(trainer=name):
                trainer = TestConstructorMatchesNnUNet._trainer(name)
                ours = set(inspect.signature(trainer.get_training_transforms).parameters)
                self.assertEqual(
                    upstream - ours,
                    set(),
                    f"{name}.get_training_transforms would reject arguments nnU-Net passes; this nnU-Net is older than the pinned >=2.5",
                )


if __name__ == "__main__":
    unittest.main()
