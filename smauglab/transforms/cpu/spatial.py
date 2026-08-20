import random

import torch
import torchio as tio
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform

from smauglab.transforms.cpu.torchio_ops import TransformFactory, apply_enabled, select

#: Transform name -> the torchio transform it runs, in application order.
SPATIAL_TRANSFORMS: dict[str, TransformFactory] = {
    # torchio accepts anatomical axis names ("LR", "AP", "IS") as well as indices, but
    # types `axes` as int | tuple[int, ...]. Flipping left/right is the point here, and
    # that is only expressible by name.
    "flip": lambda: tio.RandomFlip(axes=("LR",)),  # type: ignore[arg-type]
    "affine": lambda: tio.RandomAffine(degrees=10, translation=(0.1, 0.1, 0.1), scales=(0.9, 1.1)),
    "elastic": lambda: tio.RandomElasticDeformation(max_displacement=40),
    "anisotropy": lambda: tio.RandomAnisotropy(downsampling=7),
}


class SpatialCustomTransform(BasicTransform):
    def __init__(self, flip=False, affine=False, elastic=False, anisotropy=False, random_pick=False):
        """
        Apply all selected spatial transformation (flip, affine, elastic and anisotropy) to the image if they are enabled (set to True).
        If `random_pick` is True, randomly select and apply ONE of the enabled transformation.

        Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
        """
        super().__init__()
        self.flip = flip
        self.affine = affine
        self.elastic = elastic
        self.anisotropy = anisotropy
        self.random_pick = random_pick

    def get_parameters(self, **data_dict) -> dict:
        return select({name: getattr(self, name) for name in SPATIAL_TRANSFORMS}, self.random_pick)

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
            data_dict["image"], data_dict["segmentation"] = self._apply_to_image(data_dict["image"], data_dict["segmentation"], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_enabled(SPATIAL_TRANSFORMS, img, seg, params)


### Shape transform


class ShapeTransform(ImageOnlyTransform):
    def __init__(self, shape_min=1, ignore_axes=()):
        """
        shape_min: minimal shape size along allowed axis

        Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
        """
        super().__init__()
        self.shape_min = shape_min
        self.ignore_axes = ignore_axes

    def get_parameters(self, **data_dict) -> dict:
        return {"shape_min": self.shape_min, "ignore_axes": self.ignore_axes}

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
            data_dict["image"], data_dict["segmentation"] = self._apply_to_image(data_dict["image"], data_dict["segmentation"], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute random shape
        img_shape = img.shape[1:]
        new_shape = [random.randint(params["shape_min"], s) if i not in params["ignore_axes"] else s for i, s in enumerate(img_shape)]

        # Find image center
        img_center = [s // 2 for s in img_shape]

        # Compute start and end crop indices per axis
        starts = [max(0, c - ns // 2) for c, ns in zip(img_center, new_shape)]
        ends = [start + ns for start, ns in zip(starts, new_shape)]

        # Crop using advanced slicing
        slices = tuple(slice(start, end) for start, end in zip(starts, ends))
        img_cropped = img[(slice(None), *slices)]  # Keep channel dim intact
        seg_cropped = seg[(slice(None), *slices)]
        return img_cropped, seg_cropped
