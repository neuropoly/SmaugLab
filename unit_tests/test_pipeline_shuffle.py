"""`pipeline.shuffle`: the order of a group is redrawn per batch, and nothing else moves."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from smauglab.config import SmaugConfig, load_config
from smauglab.registry import AugType, Backend, InvalidConfigError
from smauglab.transforms.build import build_gpu_pipeline, shuffle_slots
from smauglab.transforms.gpu.base import AugmentationSequentialCustom, record_applications
from smauglab.transforms.gpu.transforms import AugTransformsGPU
from unit_tests.helpers import SmaugLabTestCase, configs_dir

PAPER = "transform_params_one-sequence-to-segment-them-all.json"

#: Three of each group at p=1, so every draw applies all of them and only the order varies.
SECTION = {
    "RandomFlipTransformGPU": {"flip_axis": [0], "p": 1.0},
    "RandomInverseGPU": {"p": 1.0},
    "RandomGaussianNoiseGPU": {"p": 1.0},
    "RandomScharrGPU": {"p": 1.0},
    "RandomBrightnessGPU": {"p": 1.0},
    "RandomGammaGPU": {"p": 1.0},
    "RandomLowResTransformGPU": {"p": 1.0},
    "RandomBiasFieldGPU": {"p": 1.0},
}


def names(pipeline) -> list[str]:
    return [type(module).__name__ for _, module in pipeline.get_forward_sequence()]


class TestConfig(SmaugLabTestCase):
    def test_the_default_shuffles_nothing(self):
        self.assertEqual(SmaugConfig({"GPU": dict(SECTION)}).shuffle_groups(), ())

    def test_groups_are_read_in_order(self):
        config = SmaugConfig({"GPU": dict(SECTION), "pipeline": {"shuffle": ["GE", "TA"]}})
        self.assertEqual(config.shuffle_groups(), (AugType.GE, AugType.TA))

    def test_bad_values_are_rejected(self):
        for value in (["GEO"], ["TA", "TA"], "TA", ["ta"]):
            with self.subTest(shuffle=value), self.assertRaises(InvalidConfigError):
                SmaugConfig({"GPU": dict(SECTION), "pipeline": {"shuffle": value}})

    def test_the_bucket_modes_reject_it(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"GPU": dict(SECTION), "pipeline": {"mode": "random_order", "shuffle": ["TA"]}})
        self.assertIn("random_choose", str(caught.exception))


class TestSlots(SmaugLabTestCase):
    def test_paper_slots_are_the_group_positions(self):
        config = load_config(str(configs_dir() / PAPER))
        pipeline = build_gpu_pipeline(config.section(Backend.GPU))
        ta, ge = shuffle_slots(pipeline, (AugType.TA, AugType.GE))
        at = {type(t).__name__: i for i, t in enumerate(pipeline)}

        self.assertEqual(
            [type(pipeline[i]).__name__ for i in ta],
            [
                "RandomInverseGPU",
                "RandomHistogramEqualizationGPU",
                "RandomRedistributeSegGPU",
                "RandomScharrGPU",
                "RandomUnsharpMaskGPU",
                "RandomRandConvGPU",
                "RandomLog1pGPU",
                "RandomSqrtGPU",
                "RandomSinGPU",
                "RandomExpGPU",
                "RandomSigmoidGPU",
                "RandomBiasFieldGPU",
            ],
        )
        self.assertNotIn(at["RandomLowResTransformGPU"], ge, "force_sequential must keep its place")
        self.assertNotIn(at["RandomFlipTransformGPU"], ta + ge)
        self.assertIn(at["RandomGammaGPU"], ge)


class TestOrder(SmaugLabTestCase):
    def _pipeline(self, groups):
        built = build_gpu_pipeline(dict(SECTION))
        return AugmentationSequentialCustom(
            *built, data_keys=["input", "mask"], same_on_batch=True, shuffle_slots=shuffle_slots(built, groups)
        )

    def test_without_groups_the_order_is_the_pipeline_order(self):
        pipeline = self._pipeline(())
        fixed = [type(t).__name__ for t in pipeline.children()]
        for _ in range(5):
            self.assertEqual(names(pipeline), fixed)

    def test_only_the_named_group_moves_and_only_among_its_own_slots(self):
        pipeline = self._pipeline((AugType.TA,))
        fixed = [type(t).__name__ for t in pipeline.children()]
        ta = {"RandomInverseGPU", "RandomScharrGPU", "RandomBiasFieldGPU"}
        slots = [i for i, n in enumerate(fixed) if n in ta]

        seen = set()
        for _ in range(60):
            order = names(pipeline)
            self.assertEqual([n for i, n in enumerate(order) if i not in slots], [n for i, n in enumerate(fixed) if i not in slots])
            self.assertEqual({order[i] for i in slots}, ta)
            seen.add(tuple(order[i] for i in slots))
        self.assertEqual(len(seen), 6, "every permutation of three should turn up in 60 draws")

    def test_both_groups_move_independently(self):
        pipeline = self._pipeline((AugType.TA, AugType.GE))
        fixed = [type(t).__name__ for t in pipeline.children()]
        ge = {"RandomGaussianNoiseGPU", "RandomBrightnessGPU", "RandomGammaGPU"}
        self.assertEqual(names(pipeline).index("RandomLowResTransformGPU"), fixed.index("RandomLowResTransformGPU"))
        seen = {tuple(n for n in names(pipeline) if n in ge) for _ in range(60)}
        self.assertEqual(len(seen), 6)

    def test_the_applied_order_is_the_drawn_order_and_the_mask_is_untouched(self):
        pipeline = self._pipeline((AugType.TA, AugType.GE))
        image, seg = self.tiny_volume(), self.tiny_seg()
        orders = set()
        for _ in range(10):
            with record_applications(pipeline) as fired:
                _, mask = pipeline(image.clone(), seg.clone())
            applied = [e["transform"] for e in fired if e["target"] == "image"]
            self.assertEqual(applied, [p.name.rsplit("_", 1)[0] for p in pipeline._params])
            # The flip is the only transform that moves the mask, and this one is symmetric.
            self.assertTrue(torch.equal(mask, seg))
            orders.add(tuple(applied))
        self.assertGreater(len(orders), 1)


class TestDefaultUnchanged(SmaugLabTestCase):
    def test_an_empty_shuffle_is_the_unshuffled_pipeline(self):
        payload = json.loads((configs_dir() / PAPER).read_text())
        image, seg = self.tiny_volume(), self.tiny_seg()
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, pipeline_section in enumerate(({}, {"shuffle": []})):
                path = Path(tmp) / f"transform_params_shuffle_{i}.json"
                path.write_text(json.dumps({**payload, "pipeline": pipeline_section}))
                torch.manual_seed(0)
                outputs.append(AugTransformsGPU(json_path=str(path))(image.clone(), seg.clone())[0])
        self.assertTrue(torch.equal(*outputs))
