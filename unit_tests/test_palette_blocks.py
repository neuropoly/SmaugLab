"""The composable PALETTE pipeline: what the blocks do, and what a config may say.

`RandomPaletteGPU` is covered by test_transforms_gpu.py and the config tests, which
between them prove the fixed composition still runs. These tests are about the part
that is new here -- that a config picks the partitioning, that picking a nonsensical
one fails at construction rather than on the first batch, and that every block hands
the remap the contiguous region ids it indexes with.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import build_transforms
from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.palette import (
    AnatomicalLabelOverlay,
    BlockContext,
    EMGMMInitial,
    EMGMMRefinement,
    IdentityRefinement,
    KMeans1DInitial,
    PaletteSynthesisGPU,
    VoronoiRefinement,
    signed_alpha_affine_remap,
)
from unit_tests.helpers import SmaugLabTestCase, first_output

SHAPE = (24, 24, 24)


def context(image: torch.Tensor | None = None) -> BlockContext:
    """A BlockContext over a 24^3 volume, built the way the transform builds one."""
    depth, height, width = SHAPE
    n_voxels = depth * height * width
    image01 = torch.rand(n_voxels) if image is None else image
    coords = torch.stack(
        torch.meshgrid(*(torch.arange(size, dtype=torch.float32) for size in SHAPE), indexing="ij"),
        dim=-1,
    ).reshape(n_voxels, 3)
    return BlockContext(
        image01=image01,
        fg_mask=(image01 > 0.01).float(),
        coords=coords,
        shape=SHAPE,
        device=torch.device("cpu"),
    )


class TestBlocksProduceUsableRegionMaps(SmaugLabTestCase):
    """Every block must return ids in [0, R), because the remap indexes with them.

    The EM blocks are the reason this is a test rather than an assumption:
    `em_subdivide_labels` encodes a fine id as `parent * mult + assign`, which is
    sparse, and indexing a length-R tensor with it would read out of bounds.
    """

    def test_initial_partitioners_return_contiguous_ids(self):
        for block in (KMeans1DInitial(skip_prob=0.0), EMGMMInitial()):
            with self.subTest(block=type(block).__name__):
                region_ids, n_regions = block.partition(context())
                self.assertGreater(n_regions, 1)
                self.assertEqual(set(region_ids.unique().tolist()), set(range(n_regions)))

    def test_refinements_return_contiguous_ids(self):
        ctx = context()
        region_ids, n_regions = EMGMMInitial().partition(ctx)
        for block in (VoronoiRefinement(skip_prob=0.0), EMGMMRefinement(), IdentityRefinement()):
            with self.subTest(block=type(block).__name__):
                finer, n_finer = block.refine(ctx, region_ids, n_regions)
                self.assertEqual(set(finer.unique().tolist()), set(range(n_finer)))

    def test_a_refinement_only_ever_subdivides(self):
        """A finer partition, never a coarser one -- that is what `refine` means."""
        ctx = context()
        region_ids, n_regions = KMeans1DInitial(skip_prob=0.0).partition(ctx)
        for block in (VoronoiRefinement(skip_prob=0.0), EMGMMRefinement()):
            with self.subTest(block=type(block).__name__):
                _finer, n_finer = block.refine(ctx, region_ids, n_regions)
                self.assertGreater(n_finer, n_regions)

    def test_identity_refinement_changes_nothing(self):
        ctx = context()
        region_ids, n_regions = KMeans1DInitial(skip_prob=0.0).partition(ctx)
        finer, n_finer = IdentityRefinement().refine(ctx, region_ids, n_regions)
        self.assertEqual(n_finer, n_regions)
        self.assertTrue(bool((finer == region_ids).all()))

    def test_kmeans_falls_back_to_one_region_when_told_to_skip(self):
        """`skip_prob=1.0` is the "no parcellation" path, which the remap reads as global."""
        region_ids, n_regions = KMeans1DInitial(skip_prob=1.0).partition(context())
        self.assertEqual(n_regions, 1)
        self.assertEqual(region_ids.unique().tolist(), [0])


class TestSignedAlphaRemap(SmaugLabTestCase):
    def test_background_stays_background(self):
        ctx = context()
        region_ids, n_regions = KMeans1DInitial(skip_prob=0.0).partition(ctx)
        remapped = signed_alpha_affine_remap(ctx.image01, ctx.fg_mask, region_ids, n_regions, (0.5, 2.0))
        self.assertTrue(bool((remapped[ctx.fg_mask == 0] == 0).all()))

    def test_output_stays_in_the_unit_range(self):
        ctx = context()
        region_ids, n_regions = KMeans1DInitial(skip_prob=0.0).partition(ctx)
        remapped = signed_alpha_affine_remap(ctx.image01, ctx.fg_mask, region_ids, n_regions, (0.5, 2.0))
        self.assertGreaterEqual(float(remapped.min()), 0.0)
        self.assertLessEqual(float(remapped.max()), 1.0)


class TestCompositionFromConfig(SmaugLabTestCase):
    """The blocks are chosen by the config, which is the whole point of the class."""

    def test_defaults_are_the_fixed_pipeline_they_generalise(self):
        transform = PaletteSynthesisGPU()
        self.assertIsInstance(transform.initial, KMeans1DInitial)
        self.assertEqual([type(block) for block in transform.refinements], [VoronoiRefinement])
        self.assertIsInstance(transform.overlay, AnatomicalLabelOverlay)

    def test_blocks_come_from_the_config(self):
        transform = PaletteSynthesisGPU(
            initial_partitioner={"type": "em_gmm", "n_foreground_clusters": 4},
            refinement_partitioners=[{"type": "voronoi"}, {"type": "em_gmm"}],
        )
        self.assertIsInstance(transform.initial, EMGMMInitial)
        self.assertEqual(transform.initial.n_foreground_clusters, 4)
        self.assertEqual([type(block) for block in transform.refinements], [VoronoiRefinement, EMGMMRefinement])

    def test_an_empty_refinement_list_means_no_refinement(self):
        """Distinct from omitting the key, which selects the default Voronoi step."""
        self.assertEqual(len(PaletteSynthesisGPU(refinement_partitioners=[]).refinements), 0)

    def test_already_built_blocks_are_accepted(self):
        block = EMGMMInitial()
        self.assertIs(PaletteSynthesisGPU(initial_partitioner=block).initial, block)

    def test_the_overlay_can_be_turned_off_without_losing_its_settings(self):
        transform = PaletteSynthesisGPU(overlay={"enabled": False, "label_remap_prob": 0.9})
        self.assertIsNone(transform.overlay)

    def test_overlay_settings_reach_the_overlay(self):
        transform = PaletteSynthesisGPU(overlay={"label_remap_prob": 0.9, "blend_strength": [0.2, 0.8]})
        self.assertEqual(transform.overlay.label_remap_prob, 0.9)
        self.assertEqual(transform.overlay.blend_strength, (0.2, 0.8))


class TestConfigMistakesFailAtConstruction(SmaugLabTestCase):
    """Not on the thousandth training step, and with the fix in the message."""

    def test_a_refinement_in_the_initial_slot_says_where_it_belongs(self):
        with self.assertRaises(ValueError) as caught:
            PaletteSynthesisGPU(initial_partitioner={"type": "voronoi"})
        self.assertIn("refinement_partitioners", str(caught.exception))

    def test_an_initial_in_the_refinement_slot_says_where_it_belongs(self):
        with self.assertRaises(ValueError) as caught:
            PaletteSynthesisGPU(refinement_partitioners=[{"type": "kmeans1d"}])
        self.assertIn("initial_partitioner", str(caught.exception))

    def test_an_unknown_type_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as caught:
            PaletteSynthesisGPU(initial_partitioner={"type": "kmeans3d"})
        self.assertIn("em_gmm", str(caught.exception))

    def test_a_block_without_a_type_is_rejected(self):
        with self.assertRaises(ValueError):
            PaletteSynthesisGPU(initial_partitioner={"n_foreground_clusters": 4})

    def test_the_offending_refinement_is_identified_by_index(self):
        with self.assertRaises(ValueError) as caught:
            PaletteSynthesisGPU(refinement_partitioners=[{"type": "voronoi"}, {"type": "nope"}])
        self.assertIn("refinement_partitioners[1]", str(caught.exception))

    def test_an_unknown_block_parameter_is_rejected(self):
        with self.assertRaises(TypeError):
            PaletteSynthesisGPU(initial_partitioner={"type": "kmeans1d", "not_a_parameter": 1})

    def test_a_descending_blend_range_is_rejected(self):
        with self.assertRaises(ValueError):
            AnatomicalLabelOverlay(blend_strength=[0.9, 0.2])


class TestTheRegistryBuildsItFromAConfigSection(SmaugLabTestCase):
    """The nested blocks have to survive the signature-derived config validation."""

    SECTION: ClassVar[dict] = {
        "PaletteSynthesisGPU": {
            "p": 1.0,
            "initial_partitioner": {"type": "em_gmm", "background_clusters_range": [3, 8]},
            "refinement_partitioners": [{"type": "voronoi", "skip_prob": 0.4}],
            "overlay": {"enabled": True, "blend_strength": 1.0},
        }
    }

    def test_a_nested_config_section_builds(self):
        built = build_transforms(self.SECTION, Backend.GPU)
        self.assertEqual([type(transform).__name__ for transform, _ in built], ["PaletteSynthesisGPU"])
        self.assertIsInstance(built[0][0].initial, EMGMMInitial)

    def test_an_unknown_top_level_key_is_still_rejected(self):
        section = {"PaletteSynthesisGPU": {"partitioner": {"type": "em_gmm"}}}
        with self.assertRaises(InvalidConfigError):
            build_transforms(section, Backend.GPU)


class TestTheTransformRuns(SmaugLabTestCase):
    """Each composition, end to end, on a tiny volume."""

    COMPOSITIONS: ClassVar[dict] = {
        "defaults": {},
        "kmeans only": {"refinement_partitioners": []},
        "em then voronoi": {"initial_partitioner": {"type": "em_gmm"}},
        "em then em": {"initial_partitioner": {"type": "em_gmm"}, "refinement_partitioners": [{"type": "em_gmm"}]},
        "two refinements": {"refinement_partitioners": [{"type": "voronoi"}, {"type": "em_gmm"}]},
        "no overlay": {"overlay": {"enabled": False}},
        "partial blend": {"overlay": {"blend_strength": [0.2, 0.8]}},
    }

    def _run(self, **kwargs):
        pipeline = AugmentationSequentialCustom(
            PaletteSynthesisGPU(p=1.0, **kwargs),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )
        return first_output(pipeline(self.tiny_volume(), self.tiny_seg()))

    def test_every_composition_produces_a_finite_volume(self):
        for label, kwargs in self.COMPOSITIONS.items():
            with self.subTest(composition=label):
                volume = self.tiny_volume()
                self.assertIsImageLike(self._run(**kwargs), volume, f"PaletteSynthesisGPU ({label})")

    def test_it_runs_without_a_segmentation(self):
        """The overlay is the only step that needs one, and it is skipped when absent."""
        transform = PaletteSynthesisGPU(p=1.0)
        volume = self.tiny_volume()
        output = transform.apply_transform(volume, params={}, flags={})
        self.assertIsImageLike(output, volume, "PaletteSynthesisGPU without seg")

    def test_the_foreground_is_z_scored(self):
        volume = self.tiny_volume()
        output = self._run()
        foreground = output[volume > 0]
        self.assertAlmostEqual(float(foreground.mean()), 0.0, places=1)
        self.assertAlmostEqual(float(foreground.std()), 1.0, places=1)
