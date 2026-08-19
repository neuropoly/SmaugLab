"""The CPU augmentation pipeline, built from a config section via the registry.

The ~250-line `if` ladder this replaces also read three values from outside the
config: `patch_size` and `rotation` come from nnU-Net at runtime (declared as
`context_params` on the registry entries), a top-level `retain_stats` was pushed
into several blocks, and `mode_seg="nearest"` was hardcoded. The first is injected
by the builder; the other two were folded into the configs by the `migration/` script.
"""

from typing import Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from smauglab.config import load_config
from smauglab.registry import Backend
from smauglab.transforms.build import build_cpu_pipeline
from smauglab.transforms.cpu.contrast import ScharrConvTransform
from smauglab.transforms.cpu.spatial import SpatialCustomTransform


class AugTransforms(ComposeTransforms):
    """Dataloader-side augmentations, in registry order."""

    def __init__(
        self,
        json_path: str,
        do_dummy_2d_data_aug: bool,
        # tuple[int, ...], not tuple[int]: these are 3D patch shapes and axis
        # tuples, so the one-element form was never what callers pass.
        patch_size: Union[np.ndarray, tuple[int, ...]],
        rotation_for_DA: RandomScalar,
        # Accepted for signature compatibility with the nnU-Net trainers but unused:
        # the mirror axes come from the config's MirrorTransform block, which is
        # where the old builder read them from too (transform_params["mirror_axes"],
        # never its own argument).
        mirror_axes: tuple[int, ...] | None = None,
    ):
        config = load_config(str(json_path))
        self.transform_params = config.section(Backend.CPU)
        self.transforms = build_cpu_pipeline(
            self.transform_params,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            patch_size=patch_size,
            rotation=rotation_for_DA,
            source=config.source,
        )
        super().__init__(transforms=self.transforms)


class AugTransformsTest(ComposeTransforms):
    def __init__(self):
        self.transforms = self._build_transforms()
        super().__init__(transforms=self.transforms)

    def _build_transforms(self):
        transforms = []

        # Scharr filter
        transforms.append(
            RandomTransform(
                ScharrConvTransform(absolute=True),
                apply_probability=0.9,
            )
        )

        # Affine transforms
        transforms.append(
            RandomTransform(
                SpatialCustomTransform(
                    affine=True,
                ),
                apply_probability=0.9,
            )
        )

        return transforms


if __name__ == "__main__":
    # Example usage
    import importlib

    import cv2

    from smauglab import configs
    from smauglab.transforms.gpu.transforms import AugTransformsGPU
    from smauglab.utils.image import Image, resample_nib
    from smauglab.utils.utils import normalize

    configs_path = importlib.resources.files(configs)
    json_path = str(configs_path / "transform_params_hybrid_TAGE.json")

    # Load images and masks tensors
    img_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/sub-amu02/anat/sub-amu02_T1w.nii.gz"
    img = Image(img_path).change_orientation("RSP")
    img = resample_nib(img, new_size=[1, 1, 1], new_size_type="mm", interpolation="linear")
    img_tensor = torch.from_numpy(img.data.copy()).to(torch.float32).unsqueeze(0)

    seg_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/derivatives/labels/sub-amu02/anat/sub-amu02_T1w_label-spine_dseg.nii.gz"
    seg = Image(seg_path).change_orientation("RSP")
    seg = resample_nib(seg, new_size=[1, 1, 1], new_size_type="mm", interpolation="nn")
    seg_tensor_all = torch.from_numpy(seg.data.copy())

    # Add segmentation values to different channels
    seg_tensor = torch.zeros((5, *seg_tensor_all.shape))
    for i, value in enumerate([12, 13, 14, 15, 16]):
        seg_tensor[i] = seg_tensor_all == value

    # Example usage
    aug_transforms = AugTransforms(
        json_path=json_path, do_dummy_2d_data_aug=False, patch_size=(128, 128, 128), rotation_for_DA=(-10, 10), mirror_axes=None
    )

    augmentor_gpu = AugTransformsGPU(json_path)

    # Apply transforms
    tensor_dict = {}
    gpu = False
    for i in range(24):
        tensor_dict[f"transfo_{i + 1!s}"] = aug_transforms(image=img_tensor.detach().clone(), segmentation=seg_tensor.detach().clone())

        if gpu:
            augmented_img, augmented_seg = augmentor_gpu(
                tensor_dict[f"transfo_{i + 1!s}"]["image"].cuda().unsqueeze(0).clone(),
                tensor_dict[f"transfo_{i + 1!s}"]["segmentation"].cuda().unsqueeze(0).clone(),
            )
            tensor_dict[f"transfo_{i + 1!s}"]["image"] = augmented_img.cpu().squeeze(0)
            tensor_dict[f"transfo_{i + 1!s}"]["segmentation"] = augmented_seg.cpu().squeeze(0)

    nb_img = len(tensor_dict.keys())
    nb_col = 6
    for key in ["image", "segmentation"]:
        output = []
        line: list[np.ndarray] = []
        aug: list[list[str]] = [[]]
        for _idx, (augment, _dic) in enumerate(tensor_dict.items()):
            if len(line) < nb_col:
                img = 255 * normalize(np.sum(tensor_dict[augment][key].detach().numpy(), axis=0, keepdims=True)[0, 64])
                line.append(img)
                aug[-1].append(augment)
            else:
                output.append(np.concatenate(line, axis=1))
                img = 255 * normalize(np.sum(tensor_dict[augment][key].detach().numpy(), axis=0, keepdims=True)[0, 64])
                line = [img]
                aug.append([augment])
        output.append(np.concatenate(line, axis=1))

        out_img = np.concatenate(output, axis=0)
        cv2.imwrite(f"img/transforms_default+plus_{key}.png", out_img)
    print(aug_transforms)
