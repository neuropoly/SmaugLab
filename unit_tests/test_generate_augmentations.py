"""The offline augmentation script has to actually run.

`augment` built its pipeline with

    train_transforms = AugTransforms(json_path=str(train_transforms_path))

but `AugTransforms.__init__` takes `do_dummy_2d_data_aug`, `patch_size` and
`rotation_for_DA` with no defaults. Every call raised `TypeError` inside a
`process_map` worker, so the whole script was dead -- and nothing under
`unit_tests/` imported anything from `scripts/`, so nothing said so.

`demo_augmentations.py` passes all four, which is what the call should look
like.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

CONFIG = REPO / "smauglab" / "configs" / "transform_params.json"


def write_pair(directory: Path) -> tuple[Path, Path]:
    """A tiny image/segmentation pair on disk, 1mm isotropic."""
    affine = np.eye(4)
    image = np.random.default_rng(0).random((12, 12, 12)).astype(np.float32)
    seg = np.zeros((12, 12, 12), dtype=np.uint8)
    seg[3:9, 3:9, 3:9] = 1

    image_path = directory / "sub-01_T2w.nii.gz"
    seg_path = directory / "sub-01_T2w_seg.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), str(image_path))
    nib.save(nib.Nifti1Image(seg, affine), str(seg_path))
    return image_path, seg_path


class TestAugmentRuns(unittest.TestCase):
    def test_it_produces_the_augmented_pair(self):
        import generate_augmentations

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            image_path, seg_path = write_pair(work)
            out = work / "out"

            generate_augmentations.augment(
                {"image": str(image_path), "segmentation": str(seg_path)},
                augmentations_per_image=2,
                train_transforms_path=CONFIG,
                ofolder=out,
                overwrite=True,
            )

            produced = sorted(p.name for p in (out / "img").glob("*.nii.gz"))
            self.assertEqual(produced, ["sub-01_T2w_a1.nii.gz", "sub-01_T2w_a2.nii.gz"])
            self.assertEqual(len(list((out / "seg").glob("*.nii.gz"))), 2)

    def test_the_augmented_image_is_usable(self):
        import generate_augmentations

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            image_path, seg_path = write_pair(work)
            out = work / "out"

            generate_augmentations.augment(
                {"image": str(image_path), "segmentation": str(seg_path)},
                augmentations_per_image=1,
                train_transforms_path=CONFIG,
                ofolder=out,
                overwrite=True,
            )

            written = nib.load(str(out / "img" / "sub-01_T2w_a1.nii.gz")).get_fdata()
            labels = nib.load(str(out / "seg" / "sub-01_T2w_seg_a1.nii.gz")).get_fdata()

            self.assertTrue(np.isfinite(written).all(), "the augmented image holds NaN or Inf")
            self.assertTrue(set(np.unique(labels)) <= {0.0, 1.0}, "the segmentation was interpolated")
