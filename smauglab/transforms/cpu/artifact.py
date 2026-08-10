import gc
import random

import torch
import torchio as tio
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class ArtifactTransform(BasicTransform):
    def __init__(self, motion=False, ghosting=False, spike=False, bias_field=False, blur=False, noise=False, swap=False, random_pick=False):
        """
        Apply all selected artifacts (motion, ghosting, spike, bias field, blur, noise, and swap) to the image if they are enabled (set to True).
        If `random_pick` is True, randomly select and apply ONE of the enabled artifacts.

        Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
        """
        super().__init__()
        self.motion = motion
        self.ghosting = ghosting
        self.spike = spike
        self.bias_field = bias_field
        self.blur = blur
        self.noise = noise
        self.swap = swap
        self.random_pick = random_pick

    def get_parameters(self, **data_dict) -> dict:

        artifacts = {
            "motion": self.motion,
            "ghosting": self.ghosting,
            "spike": self.spike,
            "bias_field": self.bias_field,
            "blur": self.blur,
            "noise": self.noise,
            "swap": self.swap,
        }

        enabled_artifacts = {k: v for k, v in artifacts.items() if v}

        if self.random_pick and enabled_artifacts:
            selected_artifact = random.choice(list(enabled_artifacts.keys()))
            artifacts = {k: (k == selected_artifact) for k, v in artifacts.items()}

        return artifacts

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
            data_dict["image"], data_dict["segmentation"] = self._apply_to_image(data_dict["image"], data_dict["segmentation"], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> tuple[torch.Tensor, torch.Tensor]:
        if params["motion"]:
            img, seg = aug_motion(img, seg)
        if params["ghosting"]:
            img, seg = aug_ghosting(img, seg)
        if params["spike"]:
            img, seg = aug_spike(img, seg)
        if params["bias_field"]:
            img, seg = aug_bias_field(img, seg)
        if params["blur"]:
            img, seg = aug_blur(img, seg)
        if params["noise"]:
            img, seg = aug_noise(img, seg)
        if params["swap"]:
            img, seg = aug_swap(img, seg)
        return img, seg


def aug_motion(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomMotion()(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomMotion()(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_ghosting(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomGhosting()(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomGhosting()(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_spike(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomSpike(intensity=(1, 2))(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomSpike(intensity=(1, 2))(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_bias_field(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomBiasField()(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomBiasField()(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_blur(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomBlur()(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomBlur()(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_noise(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomNoise()(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomNoise()(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def aug_swap(img, seg):
    if img.shape[0] == 2:  # Step2: channel 1 --> image / channel 2 --> odd discs segmentation
        subject = tio.RandomSwap()(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((subject.image.data, subject.discs.data), axis=0)
        seg_out = subject.seg.data
    else:
        subject = tio.RandomSwap()(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = subject.image.data, subject.seg.data
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out
