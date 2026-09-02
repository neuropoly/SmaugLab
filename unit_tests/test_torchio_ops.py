"""The torchio wrapper table that replaced eleven copy-pasted functions.

`aug_motion`, `aug_ghosting`, `aug_spike`, `aug_bias_field`, `aug_blur`, `aug_noise`,
`aug_swap`, `aug_flip`, `aug_affine`, `aug_elastic` and `aug_anisotropy` were the same
seventeen lines each, differing only in which `tio.Random*` they built. What matters
here is the two input layouts they all had to handle, and that `select` reproduces the
random_pick logic both call sites had written out separately.
"""

import torch
import torchio as tio

from smauglab.transforms.cpu.artifact import ARTIFACTS, ArtifactTransform
from smauglab.transforms.cpu.spatial import SPATIAL_TRANSFORMS, SpatialCustomTransform
from smauglab.transforms.cpu.torchio_ops import apply_enabled, apply_tio, select
from unit_tests.helpers import SmaugLabTestCase


def _pair(channels: int = 1, size: int = 16):
    img = torch.rand(channels, size, size, size)
    seg = torch.zeros(1, size, size, size)
    seg[0, 4:12, 4:12, 4:12] = 1.0
    return img, seg


class TestApplyTio(SmaugLabTestCase):
    def test_the_single_channel_layout_round_trips(self):
        img, seg = _pair(channels=1)

        img_out, seg_out = apply_tio(tio.RandomNoise(), img, seg)

        self.assertEqual(tuple(img_out.shape), tuple(img.shape))
        self.assertEqual(tuple(seg_out.shape), tuple(seg.shape))

    def test_the_two_channel_step2_layout_round_trips(self):
        """Channel 0 is the image, channel 1 an odd-disc segmentation."""
        img, seg = _pair(channels=2)

        img_out, seg_out = apply_tio(tio.RandomNoise(), img, seg)

        self.assertEqual(tuple(img_out.shape), tuple(img.shape))
        self.assertEqual(tuple(seg_out.shape), tuple(seg.shape))

    def test_the_second_channel_is_treated_as_labels(self):
        """It goes in as a LabelMap, so a resampling transform must not interpolate it."""
        img, seg = _pair(channels=2)
        img[1] = (img[1] > 0.5).float()

        img_out, _ = apply_tio(tio.RandomAffine(degrees=15), img, seg)

        self.assertTrue(bool(torch.isin(img_out[1], torch.tensor([0.0, 1.0])).all()), "the disc labels were interpolated")

    def test_the_segmentation_stays_a_label_map(self):
        img, seg = _pair()

        _, seg_out = apply_tio(tio.RandomAffine(degrees=10), img, seg)

        self.assertTrue(bool(torch.isin(seg_out, torch.tensor([0.0, 1.0])).all()), "the segmentation was interpolated")


class TestEveryTableEntryIsWellFormed(SmaugLabTestCase):
    """What the eleven functions actually differed in: the transform they built.

    Deliberately does not *run* every entry. `apply_tio` is the same code whichever
    transform it is handed, so running all eleven would exercise torchio rather than
    this change -- and RandomElasticDeformation(max_displacement=40) on a test-sized
    volume takes minutes. The cheap entries are run end to end below.
    """

    def test_every_entry_builds_a_torchio_transform(self):
        for table in (ARTIFACTS, SPATIAL_TRANSFORMS):
            for name, factory in table.items():
                with self.subTest(name=name):
                    self.assertIsInstance(factory(), tio.Transform)

    def test_each_call_builds_a_fresh_transform(self):
        """The table holds factories, not instances, so sampling state is not shared
        between calls -- the hand-written functions constructed one per call too."""
        for table in (ARTIFACTS, SPATIAL_TRANSFORMS):
            for name, factory in table.items():
                with self.subTest(name=name):
                    self.assertIsNot(factory(), factory())

    def test_the_tables_cover_exactly_the_old_function_names(self):
        self.assertEqual(set(ARTIFACTS), {"motion", "ghosting", "spike", "bias_field", "blur", "noise", "swap"})
        self.assertEqual(set(SPATIAL_TRANSFORMS), {"flip", "affine", "elastic", "anisotropy"})

    def test_the_cheap_entries_run_end_to_end(self):
        for name in ("blur", "noise", "bias_field"):
            with self.subTest(artifact=name):
                img, seg = _pair()
                img_out, seg_out = apply_tio(ARTIFACTS[name](), img, seg)
                self.assertEqual(tuple(img_out.shape), tuple(img.shape))
                self.assertEqual(tuple(seg_out.shape), tuple(seg.shape))

        for name in ("flip", "affine"):
            with self.subTest(transform=name):
                img, seg = _pair()
                img_out, seg_out = apply_tio(SPATIAL_TRANSFORMS[name](), img, seg)
                self.assertEqual(tuple(img_out.shape), tuple(img.shape))
                self.assertEqual(tuple(seg_out.shape), tuple(seg.shape))


class TestSelect(SmaugLabTestCase):
    def test_without_random_pick_every_enabled_flag_survives(self):
        flags = {"a": True, "b": False, "c": True}
        self.assertEqual(select(flags, random_pick=False), flags)

    def test_random_pick_keeps_exactly_one(self):
        flags = {"a": True, "b": True, "c": True}

        chosen = select(flags, random_pick=True)

        self.assertEqual(sum(chosen.values()), 1)
        self.assertEqual(set(chosen), set(flags))

    def test_random_pick_only_ever_picks_an_enabled_one(self):
        flags = {"a": False, "b": True, "c": False}
        for _ in range(20):
            self.assertEqual(select(flags, random_pick=True), {"a": False, "b": True, "c": False})

    def test_random_pick_with_nothing_enabled_is_a_no_op(self):
        flags = {"a": False, "b": False}
        self.assertEqual(select(flags, random_pick=True), flags)

    def test_it_does_not_mutate_the_caller_s_mapping(self):
        flags = {"a": True, "b": True}
        select(flags, random_pick=True)
        self.assertEqual(flags, {"a": True, "b": True})


class TestApplyEnabled(SmaugLabTestCase):
    def test_a_disabled_entry_is_skipped(self):
        img, seg = _pair()

        img_out, seg_out = apply_enabled(ARTIFACTS, img, seg, dict.fromkeys(ARTIFACTS, False))

        self.assertTrue(torch.equal(img_out, img))
        self.assertTrue(torch.equal(seg_out, seg))

    def test_an_enabled_entry_changes_the_image(self):
        img, seg = _pair()

        img_out, _ = apply_enabled(ARTIFACTS, img, seg, {"noise": True})

        self.assertFalse(torch.allclose(img_out, img))


class TestTheTransformsStillWorkEndToEnd(SmaugLabTestCase):
    def test_artifact_transform(self):
        img, seg = _pair()
        data = {"image": img.clone(), "segmentation": seg.clone()}

        out = ArtifactTransform(noise=True, blur=True)(**data)

        self.assertEqual(tuple(out["image"].shape), tuple(img.shape))

    def test_spatial_custom_transform(self):
        img, seg = _pair()
        data = {"image": img.clone(), "segmentation": seg.clone()}

        out = SpatialCustomTransform(flip=True, affine=True)(**data)

        self.assertEqual(tuple(out["image"].shape), tuple(img.shape))

    def test_random_pick_applies_exactly_one_artifact(self):
        transform = ArtifactTransform(motion=True, ghosting=True, noise=True, random_pick=True)

        params = transform.get_parameters(image=None)

        self.assertEqual(sum(bool(v) for v in params.values()), 1)
