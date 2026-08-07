import gc
import random

import torch
import torchio as tio
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform


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
        transfo = {"flip": self.flip, "affine": self.affine, "elastic": self.elastic, "anisotropy": self.anisotropy}

        enabled_transfo = {k: v for k, v in transfo.items() if v}

        if self.random_pick and enabled_transfo:
            selected_transfo = random.choice(list(enabled_transfo.keys()))
            transfo = {k: (k == selected_transfo) for k, v in transfo.items()}

        return transfo

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
            data_dict["image"], data_dict["segmentation"] = self._apply_to_image(data_dict["image"], data_dict["segmentation"], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor:
        if params["flip"]:
            img, seg = aug_flip(img, seg)
        if params["affine"]:
            img, seg = aug_affine(img, seg)
        if params["elastic"]:
            img, seg = aug_elastic(img, seg)
        if params["anisotropy"]:
            img, seg = aug_anisotropy(img, seg)
        return img, seg


def aug_flip(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomFlip(axes=("LR",))(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomFlip(axes=("LR",))(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_affine(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomAffine(degrees=10, translation=(0.1, 0.1, 0.1), scales=(0.9, 1.1))(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomAffine(degrees=10, translation=(0.1, 0.1, 0.1), scales=(0.9, 1.1))(
            tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg))
        )
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_elastic(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomElasticDeformation(max_displacement=40)(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomElasticDeformation(max_displacement=40)(
            tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg))
        )
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_anisotropy(img, seg, downsampling=7):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomAnisotropy(downsampling=downsampling)(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomAnisotropy(downsampling=downsampling)(
            tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg, axis=0))
        )
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


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

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor:
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
