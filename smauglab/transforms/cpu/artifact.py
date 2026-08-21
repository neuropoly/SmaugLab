import torch
import torchio as tio
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.cpu.torchio_ops import TransformFactory, apply_enabled, select

#: Artifact name -> the torchio transform that produces it, in application order.
#: The order is the one the seven hand-written `if params[...]` lines used.
ARTIFACTS: dict[str, TransformFactory] = {
    "motion": tio.RandomMotion,
    "ghosting": tio.RandomGhosting,
    "spike": lambda: tio.RandomSpike(intensity=(1, 2)),
    "bias_field": tio.RandomBiasField,
    "blur": tio.RandomBlur,
    "noise": tio.RandomNoise,
    "swap": tio.RandomSwap,
}


@register(
    aug_id=AugId.ARTIFACT,
    backend=Backend.CPU,
    group=AugType.TA,
)
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
        return select({name: getattr(self, name) for name in ARTIFACTS}, self.random_pick)

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
            data_dict["image"], data_dict["segmentation"] = self._apply_to_image(data_dict["image"], data_dict["segmentation"], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_enabled(ARTIFACTS, img, seg, params)
