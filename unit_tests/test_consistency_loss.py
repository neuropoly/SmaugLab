"""The augmentation-consistency dice term, and the pipeline split it needs.

Heavy appearance augmentation teaches the network nothing about the fact that the
augmented and the original patch are the *same anatomy*. This term says so: both go
through the network and their predictions are compared with a soft dice. For that
comparison to mean anything the two images have to be voxel-aligned, which is why the
reference is the output of the geometric stage -- geometry runs once and is shared, so
one segmentation is correct for both branches.

What is pinned here: the arithmetic of the term, that only the augmented branch gets
gradients, that chaining the two stages reproduces the single container exactly, and
that the guardrails fire on the configs that deserve them.
"""

from __future__ import annotations

import importlib.util
import unittest
from typing import ClassVar
from unittest import mock

import torch

from smauglab.config import SmaugConfig, load_config
from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import build_gpu_stages
from unit_tests.helpers import SmaugLabTestCase, configs_dir, seed_everything

# Raised at import so it works under pytest and `python -m unittest` alike; a
# `pytestmark` would only be understood by one of them.
if importlib.util.find_spec("nnunetv2") is None:
    raise unittest.SkipTest("the consistency loss needs the nnunetv2 extra")

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402

from smauglab.trainers.nnUNetTrainerDAExt import (  # noqa: E402
    DICE_SMOOTH,
    _consistency_problems,
    _consistency_warnings,
)
from unit_tests.test_trainers import bare_trainer  # noqa: E402

SHAPE = (2, 1, 8, 8, 8)


def soft_dice(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    """Foreground soft dice between two probability maps, written out longhand.

    Deliberately not a call into nnU-Net: this is the independent arithmetic the
    trainer's term is checked against.
    """
    axes = tuple(range(2, prediction.ndim))
    x, y = prediction[:, 1:], reference[:, 1:]
    intersect = (x * y).sum(axes)
    dc = (2 * intersect + DICE_SMOOTH) / torch.clip(y.sum(axes) + x.sum(axes) + DICE_SMOOTH, 1e-8)
    return float(-dc.mean())


class TwoHeadNet(torch.nn.Module):
    """A deep-supervision network: two heads, and a record of each call's grad state."""

    def __init__(self, *, poison_low_head: bool = False) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 1)
        self.poison_low_head = poison_low_head
        self.grad_enabled: list[bool] = []

    def forward(self, x):
        self.grad_enabled.append(torch.is_grad_enabled())
        high = self.conv(x)
        low = torch.nn.functional.avg_pool3d(high, 2)
        if self.poison_low_head:
            low = low * float("nan")
        return [high, low]


def consistency_trainer(*, weight=0.5, detach=True, network=None, batch_dice=False):
    """A bare trainer with the term switched on, and identity augmentation stages."""
    trainer = bare_trainer(lambda out, _tgt: (out[0] if isinstance(out, list) else out).mean(), network=network)
    trainer.consistency_weight = weight
    trainer.consistency_detach = detach
    trainer.consistency_nonlin = torch.nn.Softmax(dim=1)
    trainer.consistency = MemoryEfficientSoftDiceLoss(
        apply_nonlin=trainer.consistency_nonlin,
        batch_dice=batch_dice,
        do_bg=False,
        smooth=DICE_SMOOTH,
        ddp=False,
    )
    # Identity stages: the split is exercised in TestTwoStagesMatchOneContainer; here
    # the point is what train_step does with the reference, not what the pipeline did.
    trainer.transforms_shared = lambda data, target: (data, target)
    trainer.transforms = lambda data, target: (data, target)
    return trainer


def batch(key: str = "case_a") -> dict:
    torch.manual_seed(7)
    return {
        "data": torch.rand(SHAPE),
        "target": torch.zeros(SHAPE, dtype=torch.long),
        "keys": [key, key],
    }


class TestTheTerm(unittest.TestCase):
    def test_it_is_the_soft_dice_of_the_two_predictions(self):
        """Checked against the longhand formula, not against another nnU-Net call."""
        network = TwoHeadNet()
        trainer = consistency_trainer(network=network)
        data = torch.rand(SHAPE)

        with torch.no_grad():
            logits = network(data)[0]
            probabilities = torch.softmax(logits, 1)
        term = trainer._consistency_term([logits], data)

        # Same input to both branches, so the reference prediction is the same tensor.
        self.assertAlmostEqual(float(term), soft_dice(probabilities, probabilities), places=5)

    def test_agreement_does_not_bottom_out_at_minus_one(self):
        """A soft target makes this a dice against probabilities, not an agreement score.

        Pinned so nobody later "fixes" a value that is not supposed to be -1: the term's
        minimiser over the augmented prediction is the *hard* argmax of the reference, so
        it sharpens as well as aligns. Documented on `_consistency_term`.
        """
        network = TwoHeadNet()
        trainer = consistency_trainer(network=network)
        data = torch.rand(SHAPE)
        with torch.no_grad():
            logits = network(data)[0]
        term = float(trainer._consistency_term([logits], data))
        self.assertGreater(term, -1.0)
        self.assertLess(term, 0.0)

    def test_disagreement_scores_worse_than_agreement(self):
        network = TwoHeadNet()
        trainer = consistency_trainer(network=network)
        data = torch.rand(SHAPE)
        with torch.no_grad():
            logits = network(data)[0]

        agreeing = float(trainer._consistency_term([logits], data))
        # Flip the two channels: the reference now predicts the opposite class everywhere.
        disagreeing = float(trainer._consistency_term([logits.flip(1)], data))
        self.assertGreater(disagreeing, agreeing, "disagreement must cost more than agreement")

    def test_only_the_augmented_branch_gets_gradients_when_detached(self):
        network = TwoHeadNet()
        trainer = consistency_trainer(network=network, detach=True)
        logits = network(torch.rand(SHAPE))[0]
        network.grad_enabled.clear()

        trainer._consistency_term([logits], torch.rand(SHAPE))
        self.assertEqual(network.grad_enabled, [False], "the reference forward must run under no_grad")

    def test_the_reference_forward_keeps_the_graph_when_not_detached(self):
        network = TwoHeadNet()
        trainer = consistency_trainer(network=network, detach=False)
        logits = network(torch.rand(SHAPE))[0]
        network.grad_enabled.clear()

        trainer._consistency_term([logits], torch.rand(SHAPE))
        self.assertEqual(network.grad_enabled, [True])

    def test_only_the_full_resolution_head_takes_part(self):
        """Head 1 is all NaN; a finite term proves it was never read."""
        network = TwoHeadNet(poison_low_head=True)
        trainer = consistency_trainer(network=network)
        output = network(torch.rand(SHAPE))

        term = trainer._consistency_term(output, torch.rand(SHAPE))
        self.assertTrue(torch.isfinite(term), "the deep-supervision heads must not reach the term")


class TestTrainStep(unittest.TestCase):
    def test_the_term_is_added_with_its_weight(self):
        for weight in (0.0, 0.5, 2.0):
            with self.subTest(weight=weight):
                trainer = consistency_trainer(weight=weight, network=TwoHeadNet())
                trainer.loss = lambda out, _tgt: out[0].mean() * 0.0 + 1.0
                result = trainer.train_step(batch())
                self.assertAlmostEqual(
                    float(result["loss"]),
                    1.0 + weight * result["consistency"],
                    places=5,
                )

    def test_a_step_reports_the_component(self):
        trainer = consistency_trainer(network=TwoHeadNet())
        result = trainer.train_step(batch())
        self.assertIn("consistency", result)
        self.assertTrue(torch.isfinite(torch.tensor(result["consistency"])))
        self.assertEqual(trainer._consistency_steps, 1)

    def test_nothing_happens_without_a_shared_stage(self):
        """A config with no geometric group leaves the term inert rather than guessing."""
        trainer = consistency_trainer(network=TwoHeadNet())
        trainer.transforms_shared = None
        result = trainer.train_step(batch())
        self.assertNotIn("consistency", result)
        self.assertEqual(trainer._consistency_steps, 0)

    def test_the_epoch_summary_names_the_contribution(self):
        trainer = consistency_trainer(weight=0.5, network=TwoHeadNet())
        for i in range(3):
            trainer.train_step(batch(f"case_{i}"))
        trainer.logged.clear()

        # Only the subclass's own contribution is under test; the parent hook wants a
        # logger and an lr_scheduler this bare instance does not have.
        parent = type(trainer).__mro__[1]
        with mock.patch.object(parent, "on_train_epoch_end", lambda _self, _outputs: None):
            type(trainer).on_train_epoch_end(trainer, [])

        self.assertEqual(len(trainer.logged), 1)
        self.assertIn("consistency: soft dice", trainer.logged[0])
        self.assertIn("over 3 batches", trainer.logged[0])
        self.assertIn("weight 0.5", trainer.logged[0])

    def test_the_epoch_counter_resets(self):
        trainer = consistency_trainer(network=TwoHeadNet())
        trainer.train_step(batch())
        parent = type(trainer).__mro__[1]
        with mock.patch.object(parent, "on_train_epoch_start", lambda _self: None):
            type(trainer).on_train_epoch_start(trainer)
        self.assertEqual(trainer._consistency_steps, 0)
        self.assertIsNone(trainer._consistency_sum)

    def test_a_non_finite_term_is_blamed_on_the_consistency_loss(self):
        """Otherwise the report says verdict=loss, pointing the reader at nnU-Net."""
        trainer = consistency_trainer(network=TwoHeadNet())
        trainer.consistency = lambda _x, _y: torch.tensor(float("nan"))
        trainer.train_step(batch())

        reports = [line for line in trainer.logged if "verdict=" in line]
        self.assertEqual(len(reports), 1)
        self.assertIn("verdict=consistency", reports[0])
        self.assertIn("consistency=nan", reports[0])


class TestTwoStagesMatchOneContainer(SmaugLabTestCase):
    """Chaining the split stages has to reproduce the unsplit container exactly."""

    CONFIG = "transform_params_gpu.json"

    def _stages(self, *, split: bool):
        from smauglab.transforms.gpu.base import AugmentationSequentialCustom

        config = load_config(str(configs_dir() / self.CONFIG))
        shared, augmented = build_gpu_stages(
            config.section(Backend.GPU),
            mode=config.pipeline_mode(),
            options=config.pipeline_options("random_choose"),
            order_source=config.order_source(),
            split_shared=split,
        )

        def container(*stage):
            return AugmentationSequentialCustom(
                *stage,
                data_keys=["input", "mask"],
                same_on_batch=True if config.same_on_batch() else None,
            )

        return (container(*shared) if shared else None), container(*augmented)

    def _inputs(self):
        volume = self.tiny_volume()
        seg = self.tiny_seg()
        return volume, seg

    def test_the_chained_stages_reproduce_the_single_container(self):
        volume, seg = self._inputs()

        seed_everything()
        _, single = self._stages(split=False)
        one_data, one_seg = single(volume.clone(), seg.clone())

        seed_everything()
        geometric, intensity = self._stages(split=True)
        self.assertIsNotNone(geometric, "transform_params_gpu.json has an active geometric group")
        staged_data, staged_seg = geometric(volume.clone(), seg.clone())
        staged_data, staged_seg = intensity(staged_data, staged_seg)

        self.assertTrue(torch.equal(one_data, staged_data), "splitting the pipeline changed the image")
        self.assertTrue(torch.equal(one_seg, staged_seg), "splitting the pipeline changed the mask")

    def test_the_intensity_stage_does_not_write_into_the_reference(self):
        """What justifies holding an alias in `train_step` instead of cloning.

        `test_no_inplace_mutation.py` pins the same property, but only for transforms
        built with default arguments; this runs the shipped configuration.
        """
        volume, seg = self._inputs()
        seed_everything()
        geometric, intensity = self._stages(split=True)

        reference, reference_seg = geometric(volume, seg)
        untouched = reference.clone()
        intensity(reference, reference_seg)

        self.assertTrue(torch.equal(reference, untouched), "the intensity stage wrote into the reference")

    def test_the_reference_differs_from_the_augmented_image(self):
        volume, seg = self._inputs()
        seed_everything()
        geometric, intensity = self._stages(split=True)

        reference, reference_seg = geometric(volume, seg)
        kept = reference.clone()
        augmented, _ = intensity(reference, reference_seg)

        self.assertEqual(kept.shape, augmented.shape)
        self.assertFalse(torch.allclose(kept, augmented), "the intensity stage did nothing")


class TestGuardrails(unittest.TestCase):
    GPU_ONLY: ClassVar[dict] = {"GPU": {"RandomFlipTransformGPU": {"p": 1.0}, "RandomGaussianNoiseGPU": {"p": 1.0}}}

    def test_a_cpu_only_config_is_a_hard_error(self):
        """Silently not running would leave the recorded config claiming otherwise."""
        config = load_config(str(configs_dir() / "transform_params.json"))
        problems = _consistency_problems(config)
        self.assertEqual(len(problems), 1)
        self.assertIn("GPU section", problems[0])

    def test_a_gpu_config_has_no_problems(self):
        for name in ("transform_params_gpu.json", "transform_params_hybrid.json"):
            with self.subTest(config=name):
                self.assertEqual(_consistency_problems(load_config(str(configs_dir() / name))), [])

    def test_an_empty_shared_stage_is_only_a_warning(self):
        """transform_params_hybrid.json has flip and affine at p=0; geometry is on the CPU."""
        config = load_config(str(configs_dir() / "transform_params_hybrid.json"))
        warnings_out = _consistency_warnings(config)
        self.assertTrue(any("shared stage is empty" in w for w in warnings_out))

    def test_cpu_intensity_augmentations_are_named(self):
        config = load_config(str(configs_dir() / "transform_params_hybrid.json"))
        warning = next(w for w in _consistency_warnings(config) if "CPU section" in w)
        self.assertIn("GaussianNoiseTransform", warning)
        # CPU geometry runs before the batch splits, so both branches inherit it and it
        # must not be reported.
        self.assertNotIn("SpatialTransform", warning)

    def test_the_occlusion_is_reported_with_its_probability(self):
        config = load_config(str(configs_dir() / "transform_params_gpu.json"))
        warning = next(w for w in _consistency_warnings(config) if "RandomCropTransformGPU" in w)
        self.assertIn("p=0.25", warning)
        self.assertIn("occluded region", warning)

    def test_an_inactive_occlusion_is_silent(self):
        payload = {"GPU": {**self.GPU_ONLY["GPU"], "RandomCropTransformGPU": {"p": 0}}}
        warnings_out = _consistency_warnings(SmaugConfig(payload))
        self.assertFalse([w for w in warnings_out if "RandomCropTransformGPU" in w])

    def test_a_clean_gpu_config_warns_about_nothing(self):
        self.assertEqual(_consistency_warnings(SmaugConfig(self.GPU_ONLY)), [])


class TestConfigSection(unittest.TestCase):
    def test_every_shipped_config_leaves_the_term_off(self):
        """Nothing shipped changes behaviour; the term is opt-in."""
        for path in sorted(configs_dir().glob("*.json")):
            with self.subTest(config=path.name):
                try:
                    config = load_config(str(path))
                except InvalidConfigError:
                    continue
                self.assertFalse(config.consistency_enabled())


if __name__ == "__main__":
    unittest.main()
