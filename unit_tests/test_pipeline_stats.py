"""The pipeline-level statistics bracket: `pipeline.retain_stats`.

What these pin, in the order the feature is assembled:

* the solver really does put the two drawn statistics back, for each of the six pairs,
  and leaves the other two free -- that is the whole premise, and it is the one thing a
  smoke test could not tell apart from a no-op;
* where the builder emits the bracket, in all three pipeline modes: after the geometry,
  and before the crop and the normalization that now follow the restore;
* the draw discipline the rest of the GPU package keeps -- a fixed number of uniforms per
  channel, independent of the configured ranges and of how many statistics are in the
  pool, so a jitter sweep is comparable against its own baseline at a fixed seed;
* the degenerate cases that have broken every stats restore in this repository before:
  a constant patch, an unlabelled mask, a channel the solver cannot determine.
"""

import json

import torch

from smauglab.config import PipelineMode, SmaugConfig
from smauglab.registry import AugType, Backend, InvalidConfigError, entries
from smauglab.transforms.build import build_gpu_pipeline
from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.contrast import RandomGammaGPU, _channel_stats, _restore_stats
from smauglab.transforms.gpu.stats import STAT_NAMES, StatsSpec, make_bracket, measure, parse_spec, spec_problems
from unit_tests.helpers import SmaugLabTestCase, configs_dir, first_output

#: Every pair the full pool can draw, by name.
ALL_PAIRS = (("mean", "std"), ("mean", "p1"), ("mean", "p99"), ("std", "p1"), ("std", "p99"), ("p1", "p99"))


def restored(image, spec, *, seg=None, transform=None):
    """Run `image` through a bracket, with one augmentation inside it by default."""
    snapshot, restore = make_bracket(spec)
    inner = RandomGammaGPU(p=1.0) if transform is None else transform
    keys = ["input", "mask"] if seg is not None else ["input"]
    pipeline = AugmentationSequentialCustom(snapshot, inner, restore, data_keys=keys, same_on_batch=True)
    out = pipeline(image.clone(), seg.clone()) if seg is not None else pipeline(image.clone())
    return first_output(out)


def stats_of(image, spec, channel=0, mask=None):
    """The four statistics of one channel, as a `[4, N]` tensor."""
    return measure(image[:, channel], mask, spec.probabilities)


class TestTheSolverHitsTheDrawnPair(SmaugLabTestCase):
    """Two statistics determine the map, and the solver puts exactly those two back."""

    def setUp(self):
        super().setUp()
        # Off-centre and wider than [0, 1], so a map that merely happened to be the
        # identity would be caught: every statistic here is far from its default.
        self.image = torch.rand(2, 1, 24, 24, 24) * 4.0 - 1.0

    def test_each_pair_is_restored_exactly(self):
        for pair in ALL_PAIRS:
            with self.subTest(pair=pair):
                spec = parse_spec({"stats": list(pair)})
                out = restored(self.image, spec)
                before, after = stats_of(self.image, spec), stats_of(out, spec)
                for name in pair:
                    index = STAT_NAMES.index(name)
                    torch.testing.assert_close(after[index], before[index], rtol=1e-4, atol=1e-5, msg=f"{name} was not restored")

    def test_the_statistics_outside_the_pair_are_free(self):
        """The counterpart: an over-determined restore would be a bug, not a feature."""
        moved = 0
        for pair in ALL_PAIRS:
            spec = parse_spec({"stats": list(pair)})
            before, after = stats_of(self.image, spec), stats_of(restored(self.image, spec), spec)
            for index, name in enumerate(STAT_NAMES):
                if name not in pair:
                    moved += int(bool((after[index] - before[index]).abs().max() > 1e-3))
        self.assertGreater(moved, 0, "no untargeted statistic moved, so the restore is pinning more than two")

    def test_it_generalizes_the_per_transform_retain_stats(self):
        """With the pool fixed to (mean, std) the bracket is `_restore_stats` over the pipeline.

        Compared loosely on purpose: `_restore_stats` divides by the sample standard
        deviation (`unbiased=True`) and everything in this module uses the population one,
        as nnU-Net and `ZscoreNormalizationGPU` do. On a 24^3 patch that is a factor of
        1.00004, which is the tolerance below and not a bug.
        """
        spec = parse_spec({"stats": ["mean", "std"]})
        torch.manual_seed(7)
        through_bracket = restored(self.image, spec)

        torch.manual_seed(7)
        snapshot, restore = make_bracket(spec)
        gamma = RandomGammaGPU(p=1.0)
        pipeline = AugmentationSequentialCustom(snapshot, gamma, restore, data_keys=["input"], same_on_batch=True)
        pipeline(self.image.clone())  # same draws, so the gamma below is the one it applied

        torch.manual_seed(7)
        reference = first_output(
            AugmentationSequentialCustom(RandomGammaGPU(p=1.0), data_keys=["input"], same_on_batch=True)(self.image.clone())
        )
        by_hand = _restore_stats(reference[:, 0], _channel_stats(self.image[:, 0]))
        torch.testing.assert_close(through_bracket[:, 0], by_hand, rtol=1e-3, atol=1e-3)


class TestTheJitter(SmaugLabTestCase):
    def setUp(self):
        super().setUp()
        self.image = torch.rand(1, 1, 24, 24, 24) * 2.0

    def test_an_identity_range_restores_the_snapshot_exactly(self):
        spec = parse_spec({"stats": ["mean", "std"], "std_scale": [1.0, 1.0]})
        before, after = stats_of(self.image, spec), stats_of(restored(self.image, spec), spec)
        torch.testing.assert_close(after[:2], before[:2], rtol=1e-4, atol=1e-5)

    def test_the_std_lands_inside_the_configured_range(self):
        spec = parse_spec({"stats": ["mean", "std"], "std_scale": [0.8, 1.25]})
        ratios = []
        for seed in range(25):
            torch.manual_seed(seed)
            ratios.append((stats_of(restored(self.image, spec), spec)[1] / stats_of(self.image, spec)[1]).item())
        self.assertGreaterEqual(min(ratios), 0.8 - 1e-4, f"below the configured range: {min(ratios)}")
        self.assertLessEqual(max(ratios), 1.25 + 1e-4, f"above the configured range: {max(ratios)}")
        self.assertGreater(max(ratios) - min(ratios), 0.1, "the range is configured but nothing is being drawn from it")

    def test_the_percentile_span_lands_inside_the_configured_range(self):
        spec = parse_spec({"stats": ["p1", "p99"], "percentile_scale": [0.8, 1.25]})
        before = stats_of(self.image, spec)
        spans = []
        for seed in range(25):
            torch.manual_seed(seed)
            after = stats_of(restored(self.image, spec), spec)
            spans.append(((after[3] - after[2]) / (before[3] - before[2])).item())
        # Each end is drawn independently against its own distance from the mean, so the
        # span is a convex combination of the two factors and stays inside their range.
        self.assertGreaterEqual(min(spans), 0.8 - 1e-4, f"span below the range: {min(spans)}")
        self.assertLessEqual(max(spans), 1.25 + 1e-4, f"span above the range: {max(spans)}")

    def test_the_mean_target_is_never_jittered(self):
        spec = parse_spec({"stats": ["mean", "std"], "std_scale": [0.5, 2.0], "percentile_scale": [0.5, 2.0]})
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                before, after = stats_of(self.image, spec), stats_of(restored(self.image, spec), spec)
                torch.testing.assert_close(after[0], before[0], rtol=1e-4, atol=1e-5)

    def test_p_zero_leaves_the_image_where_the_augmentations_left_it(self):
        spec = parse_spec({"stats": ["mean", "std"], "p": 0.0})
        torch.manual_seed(3)
        gated = restored(self.image, spec)
        torch.manual_seed(3)
        ungated = restored(self.image, parse_spec({"stats": ["mean", "std"], "p": 1.0}))
        self.assertFalse(bool(torch.allclose(gated, ungated)), "p=0 restored anyway")
        before, after = stats_of(self.image, spec), stats_of(gated, spec)
        self.assertGreater((after[0] - before[0]).abs().max().item(), 1e-4, "p=0 still put the mean back")


class TestTheDrawDiscipline(SmaugLabTestCase):
    """Five uniforms per channel, whatever the configuration says."""

    def _stream_after(self, spec, channels=1):
        image = torch.rand(1, channels, 24, 24, 24)
        _, restore = make_bracket(spec)
        snapshot, _ = make_bracket(spec)
        snapshot.slot = restore.slot
        snapshot.apply_transform(image, {}, {})
        torch.manual_seed(11)
        restore.apply_transform(image, {}, {})
        return torch.rand(3)

    def test_five_draws_per_channel(self):
        spec = parse_spec({})
        torch.manual_seed(11)
        torch.rand(5)
        expected = torch.rand(3)
        torch.testing.assert_close(self._stream_after(spec), expected)

    def test_the_count_does_not_depend_on_the_ranges(self):
        wide = self._stream_after(parse_spec({"std_scale": [0.5, 2.0], "percentile_scale": [0.5, 2.0]}))
        identity = self._stream_after(parse_spec({}))
        torch.testing.assert_close(wide, identity)

    def test_the_count_does_not_depend_on_the_size_of_the_pool(self):
        pair = self._stream_after(parse_spec({"stats": ["mean", "std"]}))
        full = self._stream_after(parse_spec({}))
        torch.testing.assert_close(pair, full)

    def test_two_channels_cost_twice(self):
        spec = parse_spec({})
        torch.manual_seed(11)
        torch.rand(10)
        expected = torch.rand(3)
        torch.testing.assert_close(self._stream_after(spec, channels=2), expected)

    def test_the_feature_off_emits_nothing_to_draw_with(self):
        section = json.loads((configs_dir() / "transform_params_gpu.json").read_text())["GPU"]
        built = [type(t).__name__ for t in build_gpu_pipeline(section)]
        self.assertNotIn("StatsSnapshotGPU", built)
        self.assertNotIn("StatsRestoreGPU", built)


class TestDegenerateInputs(SmaugLabTestCase):
    def test_a_constant_patch_is_left_alone(self):
        """No span, so no map is determined -- and the restore must not invent one."""
        image = self.constant_volume()
        for pair in ALL_PAIRS:
            with self.subTest(pair=pair):
                out = restored(image, parse_spec({"stats": list(pair)}), transform=torch.nn.Identity())
                self.assertIsImageLike(out, image, f"a constant patch through {pair}")
                torch.testing.assert_close(out, image)

    def test_an_unlabelled_mask_falls_back_to_the_whole_volume(self):
        image = self.tiny_volume()
        spec = parse_spec({"stats": ["mean", "std"], "stats_from": "seg"})
        torch.manual_seed(5)
        empty = restored(image, spec, seg=self.empty_seg())
        torch.manual_seed(5)
        everywhere = restored(image, parse_spec({"stats": ["mean", "std"], "stats_from": "all"}), seg=self.empty_seg())
        torch.testing.assert_close(empty, everywhere)

    def test_the_statistics_can_be_confined_to_the_labelled_anatomy(self):
        image = self.tiny_volume()
        seg = self.tiny_seg()
        image = image + seg * 5.0  # foreground mean far from the whole-volume mean
        spec = parse_spec({"stats": ["mean", "std"], "stats_from": "seg"})
        out = restored(image, spec, seg=seg)

        foreground = seg[:, 0] > 0
        before = image[:, 0][foreground].mean()
        after = out[:, 0][foreground].mean()
        torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)
        whole = (out[:, 0].mean() - image[:, 0].mean()).abs()
        self.assertGreater(whole.item(), 1e-4, "the whole-volume mean was restored, so the mask was ignored")

    def test_a_restore_without_a_snapshot_is_an_error(self):
        _, restore = make_bracket(parse_spec({}))
        with self.assertRaises(RuntimeError) as caught:
            restore.apply_transform(self.tiny_volume(), {}, {})
        self.assertIn("without a snapshot", str(caught.exception))

    def test_a_changed_batch_size_is_an_error(self):
        spec = parse_spec({})
        snapshot, restore = make_bracket(spec)
        snapshot.apply_transform(torch.rand(2, 1, 8, 8, 8), {}, {})
        with self.assertRaises(RuntimeError) as caught:
            restore.apply_transform(torch.rand(1, 1, 8, 8, 8), {}, {})
        self.assertIn("samples but the snapshot holds", str(caught.exception))

    def test_the_slot_is_consumed(self):
        spec = parse_spec({})
        snapshot, restore = make_bracket(spec)
        snapshot.apply_transform(self.tiny_volume(), {}, {})
        restore.apply_transform(self.tiny_volume(), {}, {})
        self.assertIsNone(restore.slot.channels, "a second pass could read the first one's statistics")


class TestTensorContract(SmaugLabTestCase):
    def test_the_caller_s_batch_is_not_mutated(self):
        image = self.tiny_volume()
        reference = image.clone()
        restored(image, parse_spec({}))
        torch.testing.assert_close(image, reference)

    def test_the_mask_comes_back_untouched(self):
        seg = self.multi_label_seg()
        snapshot, restore = make_bracket(parse_spec({}))
        pipeline = AugmentationSequentialCustom(snapshot, RandomGammaGPU(p=1.0), restore, data_keys=["input", "mask"], same_on_batch=True)
        _, mask = pipeline(self.tiny_volume(), seg.clone())
        torch.testing.assert_close(mask, seg)

    def test_both_kornia_routes_agree(self):
        """An open autocast region sends the leaves down kornia's path instead of the fast one."""
        image = torch.rand(1, 1, 24, 24, 24) * 3.0
        spec = parse_spec({})
        torch.manual_seed(2)
        direct = restored(image, spec, transform=torch.nn.Identity())
        torch.manual_seed(2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            through_kornia = restored(image, spec, transform=torch.nn.Identity())
        torch.testing.assert_close(direct, through_kornia, rtol=1e-5, atol=1e-5)

    def test_each_channel_is_restored_to_its_own_statistics(self):
        image = torch.stack([torch.rand(1, 24, 24, 24), torch.rand(1, 24, 24, 24) * 10.0 - 5.0], dim=1)
        spec = parse_spec({"stats": ["mean", "std"]})
        out = restored(image, spec)
        for channel in (0, 1):
            with self.subTest(channel=channel):
                before, after = stats_of(image, spec, channel), stats_of(out, spec, channel)
                torch.testing.assert_close(after[:2], before[:2], rtol=1e-4, atol=1e-4)

    def test_apply_to_channel_restricts_the_bracket(self):
        image = torch.stack([torch.rand(1, 24, 24, 24), torch.rand(1, 24, 24, 24)], dim=1)
        spec = parse_spec({"stats": ["mean", "std"], "apply_to_channel": [1], "std_scale": [2.0, 2.0]})
        snapshot, restore = make_bracket(spec)
        snapshot.apply_transform(image, {}, {})
        out = restore.apply_transform(image, {}, {})
        torch.testing.assert_close(out[:, 0], image[:, 0], msg="channel 0 was touched")
        self.assertGreater((out[:, 1] - image[:, 1]).abs().max().item(), 1e-3, "channel 1 was not touched")

    def test_an_out_of_range_channel_is_rejected(self):
        snapshot, _ = make_bracket(parse_spec({"apply_to_channel": [3]}))
        with self.assertRaises(IndexError):
            snapshot.apply_transform(self.tiny_volume(), {}, {})


class TestWhereTheBuilderPutsIt(SmaugLabTestCase):
    """`[geometry] [snapshot] [augmentations] [restore] [crop] [normalization]`, every mode."""

    def setUp(self):
        super().setUp()
        self.section = json.loads((configs_dir() / "transform_params_gpu.json").read_text())["GPU"]
        self.spec = parse_spec({})
        self.geometry = {e.name for e in entries(Backend.GPU) if e.group is AugType.GEO}

    def _names(self, mode, **kwargs):
        return [type(t).__name__ for t in build_gpu_pipeline(self.section, mode=mode, retain_stats=self.spec, **kwargs)]

    def test_the_snapshot_follows_the_geometry(self):
        for mode in PipelineMode:
            with self.subTest(mode=mode.value):
                names = self._names(mode)
                cut = names.index("StatsSnapshotGPU")
                self.assertTrue(
                    set(names[:cut]) <= self.geometry, f"something that is not geometry runs before the snapshot: {names[:cut]}"
                )
                self.assertTrue(
                    set(names[cut + 1 :]).isdisjoint(self.geometry - {"RandomCropTransformGPU"}),
                    "a geometric transform other than crop runs after the snapshot",
                )

    def test_the_restore_precedes_the_crop_and_the_normalization(self):
        for mode in PipelineMode:
            with self.subTest(mode=mode.value):
                names = self._names(mode)
                self.assertEqual(
                    names[-3:],
                    ["StatsRestoreGPU", "RandomCropTransformGPU", "ZscoreNormalizationGPU"],
                    f"unexpected tail: {names[-4:]}",
                )

    def test_the_low_resolution_hoist_stays_inside_the_bracket(self):
        """`random_order` hoists it next to the geometry, but it is an enhancement."""
        names = self._names(PipelineMode.RANDOM_ORDER)
        self.assertLess(names.index("StatsSnapshotGPU"), names.index("RandomLowResTransformGPU"))
        self.assertLess(names.index("RandomLowResTransformGPU"), names.index("StatsRestoreGPU"))

    def test_the_bracket_is_absent_without_the_block(self):
        for mode in PipelineMode:
            with self.subTest(mode=mode.value):
                plain = [type(t).__name__ for t in build_gpu_pipeline(self.section, mode=mode)]
                self.assertNotIn("StatsSnapshotGPU", plain)
                self.assertNotIn("StatsRestoreGPU", plain)

    def test_the_only_transform_the_bracket_moves_is_the_crop(self):
        """Everything else keeps the position it had; crop is staged after the restore."""
        for mode in PipelineMode:
            with self.subTest(mode=mode.value):
                plain = [type(t).__name__ for t in build_gpu_pipeline(self.section, mode=mode)]
                bracketed = self._names(mode)
                self.assertEqual(sorted(bracketed), sorted([*plain, "StatsSnapshotGPU", "StatsRestoreGPU"]))
                without_crop = [n for n in bracketed if not n.startswith("Stats") and n != "RandomCropTransformGPU"]
                self.assertEqual([n for n in plain if n != "RandomCropTransformGPU"], without_crop)
                self.assertIn("RandomCropTransformGPU", bracketed)

    def test_it_also_brackets_a_config_ordered_pipeline(self):
        from smauglab.config import OrderSource

        names = self._names(PipelineMode.SEQUENTIAL, order_source=OrderSource.CONFIG)
        self.assertEqual(names[-3:], ["StatsRestoreGPU", "RandomCropTransformGPU", "ZscoreNormalizationGPU"])
        self.assertEqual(names.index("StatsSnapshotGPU"), 2)

    def test_a_lone_normalization_is_staged_after_the_restore(self):
        """Without the bracket a single normalization keeps its position; with it, it trails."""
        section = {"RandomGammaGPU": {"p": 1.0}, "ZscoreNormalizationGPU": {"p": 1.0}}
        self.assertEqual(
            [type(t).__name__ for t in build_gpu_pipeline(section, retain_stats=self.spec)],
            ["StatsSnapshotGPU", "RandomGammaGPU", "StatsRestoreGPU", "ZscoreNormalizationGPU"],
        )


class TestTheConfigSurface(SmaugLabTestCase):
    def test_the_shipped_config_carries_the_block(self):
        payload = json.loads((configs_dir() / "transform_params_gpu_stats-restore.json").read_text())
        spec = SmaugConfig(payload, source="transform_params_gpu_stats-restore.json").retain_stats_spec()
        self.assertEqual(spec.std_scale, (0.9, 1.1))
        self.assertEqual(spec.percentile_scale, (0.9, 1.1))
        self.assertEqual(spec.stats, STAT_NAMES)
        self.assertFalse(
            any(block.get("retain_stats") for block in payload["GPU"].values() if isinstance(block, dict)),
            "a per-transform retain_stats is still on, and the two do not compose",
        )

    def test_the_paper_configs_do_not_carry_it(self):
        for name in ("transform_params_one-sequence-to-segment-them-all.json", "transform_params_paper-synthseg.json"):
            with self.subTest(config=name):
                payload = json.loads((configs_dir() / name).read_text())
                self.assertIsNone(SmaugConfig(payload, source=name).retain_stats_spec())

    def test_a_typo_is_reported_with_the_rest(self):
        with self.assertRaises(InvalidConfigError) as caught:
            SmaugConfig({"GPU": {"RandomGammaGPU": {"p": 1}}, "pipeline": {"retain_stats": {"std_scaled": [1, 2]}}})
        self.assertTrue(any("unknown key 'std_scaled'" in problem for problem in caught.exception.problems))

    def test_the_values_are_checked(self):
        for block, expected in (
            ({"stats": ["mean"]}, "at least two statistics"),
            ({"stats": ["median"]}, "unknown name"),
            ({"std_scale": [1.5, 0.5]}, "non-decreasing"),
            ({"std_scale": -1.0}, "must lie inside"),
            ({"percentiles": [99.0, 1.0]}, "0 <= low < high <= 100"),
            ({"stats_from": "bone"}, "stats_from must be one of"),
            ({"p": 2.0}, "must be a number in [0, 1]"),
            ({"apply_to_channel": []}, "non-empty list"),
            ({"nonzero_threshold": 0.0}, "must be positive"),
        ):
            with self.subTest(block=block):
                problems = spec_problems(block)
                self.assertTrue(any(expected in problem for problem in problems), f"{block} -> {problems}")

    def test_an_absent_block_is_off_and_an_empty_one_is_the_defaults(self):
        self.assertIsNone(parse_spec(None), "an absent block must emit no leaves at all")
        self.assertEqual(parse_spec({}), StatsSpec(), "a written-but-empty block is the defaults, not off")
        self.assertEqual(spec_problems(None), [])

    def test_the_pairs_follow_the_canonical_order(self):
        """The pair table is indexed by a draw, so the order of the keys must not decide it."""
        self.assertEqual(parse_spec({"stats": ["p99", "mean"]}).stats, ("mean", "p99"))
        self.assertEqual(StatsSpec().pairs, tuple((i, j) for i in range(4) for j in range(i + 1, 4)))
