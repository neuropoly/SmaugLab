from functools import partial

import scipy.ndimage as ndi
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from scipy.stats import norm


class RedistributeTransform(BasicTransform):
    """
    Redistribute image values using segmentation regions.

    Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
    """

    def __init__(self, classes=None, in_seg=0.2, retain_stats=False):
        super().__init__()
        self.classes = classes
        self.in_seg = in_seg
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        return {"classes": self.classes, "in_seg": self.in_seg, "retain_stats": self.retain_stats}

    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
            data_dict["image"], data_dict["segmentation"] = self._apply_to_image(data_dict["image"], data_dict["segmentation"], **params)
        return data_dict

    def _apply_to_image(self, img: torch.Tensor, seg: torch.Tensor, **params) -> torch.Tensor:
        for c in range(1):  # Works on the first channel only
            img[c], seg[c] = aug_redistribute_seg(
                img[c], seg[c], classes=params["classes"], in_seg=params["in_seg"], retain_stats=params["retain_stats"]
            )
        return img, seg


def aug_redistribute_seg(img, seg, classes=None, in_seg=0.2, retain_stats=False):
    """
    Augment the image by redistributing the values of the image within the
    regions defined by the segmentation.

    Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
    """
    device = img.device
    _seg = seg
    in_seg_bool = 1 - torch.rand(1, device=device) <= in_seg

    if classes:
        _seg = combine_classes(_seg, classes)

    if retain_stats:
        # Compute original mean, std and min/max values
        original_mean, original_std = img.mean(), img.std()

    # Normalize image
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min)

    # Get the unique label values (excluding 0)
    labels = torch.unique(_seg)
    labels = labels[labels != 0]

    to_add = torch.zeros_like(img, device=device)

    # Loop over each label value
    for l in labels:
        # Get the mask for the current label
        l_mask = _seg == l

        # Get mean and std of the current label
        l_mean, l_std = img[l_mask].mean(), img[l_mask].std()

        # Convert to NumPy for dilation operations (not supported in PyTorch)
        l_mask_np = l_mask.cpu().numpy()
        struct = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 3)
        l_mask_dilate_np = ndi.binary_dilation(l_mask_np, structure=struct)

        # Convert back to PyTorch
        l_mask_dilate = torch.tensor(l_mask_dilate_np, device=device)

        # Create mask of the dilated mask excluding the original mask
        l_mask_dilate_excl = l_mask_dilate & ~l_mask

        # Compute mean and std for the dilated region
        if l_mask_dilate_excl.any():
            l_mean_dilate = img[l_mask_dilate_excl].mean()
            l_std_dilate = img[l_mask_dilate_excl].std()
        else:
            l_mean_dilate, l_std_dilate = l_mean, l_std  # Fallback to original values

        redist_std = max(
            torch.rand(1, device=device) * 0.2 + 0.4 * abs((l_mean - l_mean_dilate) * l_std / (l_std_dilate + 1e-6)),
            torch.tensor([0.01], device=device),
        )

        redist = partial(norm.pdf, loc=l_mean.cpu().numpy(), scale=redist_std.cpu().numpy())

        if in_seg_bool:
            to_add[l_mask] += torch.tensor(redist(img[l_mask].cpu().numpy()), device=device) * (2 * torch.rand(1, device=device) - 1)
        else:
            to_add += torch.tensor(redist(img.cpu().numpy()), device=device) * (2 * torch.rand(1, device=device) - 1)

    # Normalize to_add and apply it to the image
    to_add_min, to_add_max = to_add.min(), to_add.max()
    img += 2 * (to_add - to_add_min) / (to_add_max - to_add_min + 1e-6)

    if retain_stats:
        # Return to original range
        mean = torch.mean(img)
        std = torch.std(img)
        img = (img - mean) / torch.clamp(std, min=1e-7)
        img = img * original_std + original_mean

    return img, seg


def combine_classes(seg, classes):
    _seg = torch.zeros_like(seg)
    for i, c in enumerate(classes):
        _seg[torch.isin(seg, c)] = i + 1
    return _seg
