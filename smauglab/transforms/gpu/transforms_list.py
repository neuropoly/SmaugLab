"""The random-order GPU pipelines, and the combinator they are built from.

`AugTransformsGPURandomOrder` and `AugTransformsGPURandomOrderTA` used to carry a
near-verbatim copy each of the ~300-line dispatch ladder in `transforms.py`, which
had already drifted from it (they passed a `crop=` argument no transform accepts,
and ordered SimulateLowRes differently). Both are now three lines: the only thing
that distinguishes them from `AugTransformsGPU` is how the registry's TA and GE
groups get bucketed, which `smauglab.transforms.build` handles.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from smauglab.transforms.build import PipelineMode
from smauglab.transforms.gpu.base import ImageOnlyTransform
from smauglab.transforms.gpu.transforms import AugTransformsGPU


class AugTransformsGPURandomOrder(AugTransformsGPU):
    """Geometry in order, then the TA and GE groups each shuffled in their own bucket."""

    mode = PipelineMode.RANDOM_ORDER


class AugTransformsGPURandomOrderTA(AugTransformsGPU):
    """Only the transfer augmentations are bucketed; everything else keeps its order."""

    mode = PipelineMode.RANDOM_ORDER_TA


class RandomChooseXTransformsGPU(ImageOnlyTransform):
    """Randomly choose X transforms to apply from a given list of ImageOnlyTransform transforms (GPU version).

    Args:
        transforms_list: List of initialized ImageOnlyTransform to choose from.
        num_transforms: Number of transforms to randomly select and apply.
        same_on_batch: apply the same transformation across the batch.
        p: probability for applying the X transforms to a batch. This param controls the augmentation
          probabilities batch-wise.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    """

    def __init__(
        self,
        transforms_list: list[ImageOnlyTransform],
        num_transforms: int = 1,
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
        random_order: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if not isinstance(num_transforms, int) or num_transforms < 0:
            raise ValueError(f"num_transforms must be a non-negative int. Got {num_transforms!r}.")
        self.transforms_list = nn.ModuleList(transforms_list)
        self.num_transforms = num_transforms
        self.random_order = random_order

    def _apply_mix(self, x: Tensor, seg: Tensor | None) -> Tensor:
        if self.num_transforms == 0 or len(self.transforms_list) == 0:
            return x

        k = min(self.num_transforms, len(self.transforms_list))
        # sample without replacement
        if self.random_order:
            idx = torch.randperm(len(self.transforms_list), device=x.device)[:k]
        else:
            idx = torch.arange(len(self.transforms_list), device=x.device)[:k]

        child_params: dict[str, Tensor] = {}
        if seg is not None:
            child_params["seg"] = seg

        for j in idx.tolist():
            t = self.transforms_list[j]
            if torch.rand(1, device=x.device, dtype=x.dtype) > t.p:
                continue
            if not hasattr(t, "apply_transform"):
                raise TypeError(f"All transforms must implement apply_transform like ImageOnlyTransform. Got {type(t)}")
            # Most contrast transforms perform their random sampling inside apply_transform.
            t_flags = getattr(t, "flags", {})
            x = t.apply_transform(x, child_params, t_flags, transform=None)
        return x

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        seg = params.get("seg")

        if self.same_on_batch:
            return self._apply_mix(input, seg)

        batch_size = input.shape[0]
        out = input
        for i in range(batch_size):
            xi = out[i : i + 1]
            seg_i = None
            seg_i = seg[i : i + 1] if seg is not None and isinstance(seg, torch.Tensor) and seg.shape[0] == batch_size else seg
            xi = self._apply_mix(xi, seg_i)
            out[i : i + 1] = xi
        return out


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a tensor to the range [0, 1].
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val + 1e-8)
    return normalized_arr


def pad_numpy_array(arr, shape):
    """
    Pad a numpy array to the desired shape with zeros.
    """
    # Calculate padding needed for each dimension
    pad_width = [
        (max(0, shape[i] - arr.shape[i]) // 2, max(0, shape[i] - arr.shape[i]) - max(0, shape[i] - arr.shape[i]) // 2)
        for i in range(len(shape))
    ]
    padded_arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return padded_arr


if __name__ == "__main__":
    # Example usage
    import importlib

    from smauglab import configs
    from smauglab.transforms.gpu.transforms import AugTransformsGPU
    from smauglab.utils.image import Image, resample_nib

    configs_path = importlib.resources.files(configs)
    json_path = str(configs_path / "transform_params_gpu.json")
    augmentor = AugTransformsGPU(json_path)

    # Load images and masks tensors
    img_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/sub-amu02/anat/sub-amu02_T1w.nii.gz"
    img = Image(img_path).change_orientation("RSP")
    img = resample_nib(img, new_size=[1, 1, 1], new_size_type="mm", interpolation="linear")
    img_tensor = torch.from_numpy(img.data.copy()).to(torch.float32)

    seg_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/derivatives/labels/sub-amu02/anat/sub-amu02_T1w_label-spine_dseg.nii.gz"
    seg = Image(seg_path).change_orientation("RSP")
    seg = resample_nib(seg, new_size=[1, 1, 1], new_size_type="mm", interpolation="nn")
    seg_tensor_all = torch.from_numpy(seg.data.copy())

    img2_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/spider-challenge-2023/sub-002/anat/sub-002_acq-lowresSag_T2w.nii.gz"
    img2 = Image(img2_path).change_orientation("RSP")
    img2 = resample_nib(img2, new_size=[1, 1, 1], new_size_type="mm", interpolation="linear")
    img2_tensor = torch.from_numpy(img2.data.copy()).to(torch.float32)

    seg2_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/spider-challenge-2023/derivatives/labels/sub-002/anat/sub-002_acq-lowresSag_T2w_label-spine_dseg.nii.gz"
    seg2 = Image(seg2_path).change_orientation("RSP")
    seg2 = resample_nib(seg2, new_size=[1, 1, 1], new_size_type="mm", interpolation="nn")
    seg2_tensor_all = torch.from_numpy(seg2.data.copy())

    # Combine two images to same size
    new_shape = []
    for dim in range(3):
        size1 = img_tensor.shape[dim]
        size2 = img2_tensor.shape[dim]
        min_size = min(size1, size2)
        new_shape.append(min_size)

    new_img_tensor = torch.zeros(new_shape)
    new_img2_tensor = torch.zeros(new_shape)
    new_seg_tensor_all = torch.zeros(new_shape)
    new_seg2_tensor_all = torch.zeros(new_shape)

    gap = (torch.tensor(img_tensor.shape) - torch.tensor(new_shape)) // 2
    gap2 = (torch.tensor(img2_tensor.shape) - torch.tensor(new_shape)) // 2
    new_img_tensor = img_tensor[gap[0] : gap[0] + new_shape[0], gap[1] : gap[1] + new_shape[1], gap[2] : gap[2] + new_shape[2]]
    new_img2_tensor = img2_tensor[gap2[0] : gap2[0] + new_shape[0], gap2[1] : gap2[1] + new_shape[1], gap2[2] : gap2[2] + new_shape[2]]
    new_seg_tensor_all = seg_tensor_all[gap[0] : gap[0] + new_shape[0], gap[1] : gap[1] + new_shape[1], gap[2] : gap[2] + new_shape[2]]
    new_seg2_tensor_all = seg2_tensor_all[
        gap2[0] : gap2[0] + new_shape[0], gap2[1] : gap2[1] + new_shape[1], gap2[2] : gap2[2] + new_shape[2]
    ]

    # Add segmentation values to different channels
    seg_tensor = torch.zeros((1, 5, *new_seg_tensor_all.shape))
    for i, value in enumerate([12, 13, 14, 15, 16]):
        seg_tensor[0, i] = new_seg_tensor_all == value

    seg2_tensor = torch.zeros((1, 5, *new_seg2_tensor_all.shape))
    for i, value in enumerate([50, 45, 44, 43, 42]):
        seg2_tensor[0, i] = new_seg2_tensor_all == value

    # Format tensors to match expected input shape (B, C, D, H, W)
    img_tensor = torch.cat([new_img_tensor.unsqueeze(0), new_seg_tensor_all.bool().int().unsqueeze(0)], dim=0).unsqueeze(
        0
    )  # Add batch dimension and second channel
    img2_tensor = torch.cat([new_img2_tensor.unsqueeze(0), new_seg2_tensor_all.bool().int().unsqueeze(0)], dim=0).unsqueeze(
        0
    )  # Add batch dimension and second channel

    # Add batch
    img_tensor = torch.cat([img_tensor, img2_tensor], dim=0)
    seg_tensor = torch.cat([seg_tensor, seg2_tensor], dim=0)

    # Move to GPU
    img_tensor = img_tensor.cuda()
    seg_tensor = seg_tensor.cuda()
    augmentor = augmentor.cuda()

    # Apply augmentations
    augmented_img, augmented_seg = augmentor(img_tensor.clone(), seg_tensor.clone())

    if augmented_img.shape != img_tensor.shape:
        raise ValueError("Augmented image shape does not match input shape.")
    if augmented_seg.shape != seg_tensor.shape:
        raise ValueError("Augmented segmentation shape does not match input shape.")
    # Check if nans are present
    if torch.isnan(augmented_img).any():
        raise ValueError("NaNs found in augmented image.")
    if torch.isnan(augmented_seg).any():
        raise ValueError("NaNs found in augmented segmentation.")

    import os
    import warnings

    import cv2
    import numpy as np

    warnings.simplefilter("always")

    # Convert tensors to numpy arrays
    img_tensor_np = img_tensor.cpu().detach().numpy()
    seg_tensor_np = seg_tensor.cpu().detach().numpy()
    augmented_img_np = augmented_img.cpu().detach().numpy()
    augmented_seg_np = augmented_seg.cpu().detach().numpy()

    # Concatenate segmentation channels for visualization
    seg_tensor_np = np.sum(seg_tensor_np, axis=1)
    augmented_seg_np = np.sum(augmented_seg_np, axis=1)

    pad_shape = 2 * (np.max(img_tensor_np.shape[2:]),)

    # Combine tensors into single output for visualization
    os.makedirs("img", exist_ok=True)
    img_line = np.concatenate(
        [
            normalize(pad_numpy_array(img_tensor_np[0, 0, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(img_tensor_np[0, 0, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(img_tensor_np[0, 0, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    augmented_img_line = np.concatenate(
        [
            normalize(pad_numpy_array(augmented_img_np[0, 0, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[0, 0, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[0, 0, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    seg_line = np.concatenate(
        [
            normalize(pad_numpy_array(seg_tensor_np[0, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(seg_tensor_np[0, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(seg_tensor_np[0, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    augmented_seg_line = np.concatenate(
        [
            normalize(pad_numpy_array(augmented_seg_np[0, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_seg_np[0, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_seg_np[0, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    not_augmented_channel_line = np.concatenate(
        [
            normalize(pad_numpy_array(augmented_img_np[0, 1, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[0, 1, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[0, 1, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    combined_img = np.concatenate([img_line, seg_line, augmented_img_line, augmented_seg_line, not_augmented_channel_line], axis=0)
    cv2.imwrite("img/combined.png", combined_img * 255)

    img_line2 = np.concatenate(
        [
            normalize(pad_numpy_array(img_tensor_np[1, 0, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(img_tensor_np[1, 0, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(img_tensor_np[1, 0, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    augmented_img_line2 = np.concatenate(
        [
            normalize(pad_numpy_array(augmented_img_np[1, 0, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[1, 0, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[1, 0, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    seg_line2 = np.concatenate(
        [
            normalize(pad_numpy_array(seg_tensor_np[1, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(seg_tensor_np[1, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(seg_tensor_np[1, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    augmented_seg_line2 = np.concatenate(
        [
            normalize(pad_numpy_array(augmented_seg_np[1, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_seg_np[1, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_seg_np[1, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    not_augmented_channel_line2 = np.concatenate(
        [
            normalize(pad_numpy_array(augmented_img_np[1, 1, img_tensor_np.shape[2] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[1, 1, :, :, img_tensor_np.shape[4] // 2], pad_shape)),
            normalize(pad_numpy_array(augmented_img_np[1, 1, :, img_tensor_np.shape[3] // 2, :], pad_shape)),
        ],
        axis=1,
    )
    combined_img2 = np.concatenate([img_line2, seg_line2, augmented_img_line2, augmented_seg_line2, not_augmented_channel_line2], axis=0)
    cv2.imwrite("img/combined2.png", combined_img2 * 255)

    print(augmentor)
