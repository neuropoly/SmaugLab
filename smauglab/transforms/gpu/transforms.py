import numpy as np
import torch

from smauglab.config import load_config
from smauglab.registry import Backend
from smauglab.transforms.build import PipelineMode, build_gpu_pipeline
from smauglab.transforms.gpu.base import AugmentationSequentialCustom


class AugTransformsGPU(AugmentationSequentialCustom):
    """GPU augmentation pipeline, built from a config section via the registry.

    The ~370-line `if` ladder this replaces decided the class, the parameters and
    the pipeline position of every augmentation inline; all three now come from the
    registry, and `smauglab.transforms.build` does the dispatch once for all three
    GPU pipeline modes.
    """

    mode: PipelineMode = PipelineMode.SEQUENTIAL

    def __init__(self, json_path: str):
        config = load_config(str(json_path))
        self.transform_params = config.section(Backend.GPU)
        transforms = build_gpu_pipeline(
            self.transform_params,
            mode=self.mode,
            options=config.pipeline_options("random_choose"),
            source=config.source,
        )
        # same_on_batch keeps the mask aligned with the image; see
        # AugmentationSequentialOpsCustom in base.py.
        super().__init__(*transforms, data_keys=["input", "mask"], same_on_batch=True)


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
    from smauglab.utils.image import Image, resample_nib

    configs_path = importlib.resources.files(configs)
    json_path = str(configs_path / "transform_params_gpu.json")
    augmentor = AugTransformsGPU(json_path)

    # Load images and masks tensors
    img_path = "/home/ge.polymtl.ca/p118739/data/datasets/data-multi-subject/sub-amu02/anat/sub-amu02_T1w.nii.gz"
    img = Image(img_path).change_orientation("RSP")
    img = resample_nib(img, new_size=[1, 1, 1], new_size_type="mm", interpolation="linear")
    img_tensor = torch.from_numpy(img.data.copy()).to(torch.float32)

    seg_path = "/home/ge.polymtl.ca/p118739/data/datasets/data-multi-subject/derivatives/labels/sub-amu02/anat/sub-amu02_T1w_label-spine_dseg.nii.gz"
    seg = Image(seg_path).change_orientation("RSP")
    seg = resample_nib(seg, new_size=[1, 1, 1], new_size_type="mm", interpolation="nn")
    seg_tensor_all = torch.from_numpy(seg.data.copy())

    img2_path = "/home/ge.polymtl.ca/p118739/data/datasets/spider-challenge-2023/sub-002/anat/sub-002_acq-lowresSag_T2w.nii.gz"
    img2 = Image(img2_path).change_orientation("RSP")
    img2 = resample_nib(img2, new_size=[1, 1, 1], new_size_type="mm", interpolation="linear")
    img2_tensor = torch.from_numpy(img2.data.copy()).to(torch.float32)

    seg2_path = "/home/ge.polymtl.ca/p118739/data/datasets/spider-challenge-2023/derivatives/labels/sub-002/anat/sub-002_acq-lowresSag_T2w_label-spine_dseg.nii.gz"
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
    img_tensor = img_tensor.cuda(device=7)
    seg_tensor = seg_tensor.cuda(device=7)
    augmentor = augmentor.cuda(device=7)

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

    # cv2.imwrite('img/orig_img.png', normalize(pad_numpy_array(img_tensor_np[1, 0, img_tensor_np.shape[2] // 2], pad_shape))*255)
    # cv2.imwrite('img/aug_img.png', normalize(pad_numpy_array(augmented_img_np[1, 0, img_tensor_np.shape[2] // 2], pad_shape))*255)

    print(augmentor)
