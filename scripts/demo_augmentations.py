"""Render a before/after montage of a SmaugLab pipeline on a real volume.

This replaces three `if __name__ == "__main__":` blocks that used to live inside the
shipped package -- one each in `transforms/gpu/transforms.py`,
`transforms/gpu/transforms_list.py` and `transforms/cpu/transforms.py`. The two GPU
blocks were ~205 lines of near-identical copy (they differed only in a hardcoded home
directory and `cuda(device=7)` vs `cuda()`), and all three hardcoded absolute paths
into one person's machine, so nobody else could run them. They also imported `cv2`,
which is not a SmaugLab dependency; this script writes PNGs through torchvision,
which is.

    # GPU pipeline, one subject
    python scripts/demo_augmentations.py --image sub-01_T1w.nii.gz --seg sub-01_dseg.nii.gz

    # GPU pipeline, two subjects batched together (what the old GPU demos did)
    python scripts/demo_augmentations.py \
        --image sub-01_T1w.nii.gz --seg sub-01_dseg.nii.gz \
        --image sub-02_T2w.nii.gz --seg sub-02_dseg.nii.gz \
        --config smauglab/configs/transform_params_gpu.json --device cuda:0

    # CPU pipeline, a grid of repeated draws (what the old CPU demo did)
    python scripts/demo_augmentations.py --backend cpu --repeats 24 \
        --image sub-01_T1w.nii.gz --seg sub-01_dseg.nii.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from smauglab.config import default_config_path
from smauglab.utils.image import Image, resample_nib

#: Segmentation values pulled into their own channels. The old demos hardcoded two
#: different sets, one per dataset; they are only used to build a multi-channel mask
#: so the pipeline's mask handling gets exercised, so any distinct labels will do.
DEFAULT_SEG_LABELS = (12, 13, 14, 15, 16)


# --- small array helpers ----------------------------------------------------------
#
# `normalize` here is min-max, and is the one the GPU demos carried (identically, in
# two files). It is deliberately NOT the percentile-based `normalize` that used to sit
# in smauglab/utils/utils.py -- three functions shared that name and computed two
# different things.


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val + 1e-8)


def pad_to(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Zero-pad `arr` up to `shape`, centred. Axes already at or over size are left alone."""
    pad_width = []
    for i in range(len(shape)):
        total = max(0, shape[i] - arr.shape[i])
        pad_width.append((total // 2, total - total // 2))
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def mid_slices(volume: np.ndarray, pad_shape: tuple[int, int]) -> np.ndarray:
    """The three orthogonal centre slices of a [D, H, W] volume, side by side."""
    return np.concatenate(
        [
            normalize_minmax(pad_to(volume[volume.shape[0] // 2, :, :], pad_shape)),
            normalize_minmax(pad_to(volume[:, :, volume.shape[2] // 2], pad_shape)),
            normalize_minmax(pad_to(volume[:, volume.shape[1] // 2, :], pad_shape)),
        ],
        axis=1,
    )


def write_png(image: np.ndarray, path: Path) -> None:
    """Write a 2-D float array in [0, 1] as an 8-bit greyscale PNG."""
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.from_numpy(np.ascontiguousarray(image)).float().clamp(0, 1).unsqueeze(0), str(path))
    print(f"wrote {path}")


# --- loading ----------------------------------------------------------------------


def load_volume(path: str, interpolation: str) -> torch.Tensor:
    """Load a NIfTI, reorient to RSP and resample to 1 mm isotropic."""
    image = Image(path).change_orientation("RSP")
    image = resample_nib(image, new_size=[1, 1, 1], new_size_type="mm", interpolation=interpolation)
    return torch.from_numpy(image.data.copy())


def centre_crop(tensor: torch.Tensor, shape: list[int]) -> torch.Tensor:
    """Centre-crop a [D, H, W] tensor down to `shape`."""
    gap = (torch.tensor(tensor.shape) - torch.tensor(shape)) // 2
    return tensor[gap[0] : gap[0] + shape[0], gap[1] : gap[1] + shape[1], gap[2] : gap[2] + shape[2]]


def load_subject(image_path: str, seg_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One subject as (image [D,H,W] float, label map [D,H,W])."""
    return load_volume(image_path, "linear").to(torch.float32), load_volume(seg_path, "nn")


def stack_subjects(subjects: list[tuple[torch.Tensor, torch.Tensor]], labels: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch subjects into (image [B,2,D,H,W], mask [B,C,D,H,W]).

    Subjects rarely share a shape, so everything is centre-cropped to the smallest
    common one first. Channel 1 of the image is the binarised segmentation: the
    pipeline must leave it alone, which is what the montage's last row shows.
    """
    common = [min(subject[0].shape[dim] for subject in subjects) for dim in range(3)]

    images, masks = [], []
    for image, seg_all in subjects:
        image_c = centre_crop(image, common)
        seg_c = centre_crop(seg_all, common)

        mask = torch.zeros((1, len(labels), *seg_c.shape))
        for i, value in enumerate(labels):
            mask[0, i] = seg_c == value

        images.append(torch.cat([image_c.unsqueeze(0), seg_c.bool().int().unsqueeze(0)], dim=0).unsqueeze(0))
        masks.append(mask)

    return torch.cat(images, dim=0), torch.cat(masks, dim=0)


# --- the two demos ----------------------------------------------------------------


def demo_gpu(args: argparse.Namespace) -> int:
    from smauglab.transforms.gpu.transforms import AugTransformsGPU

    subjects = [load_subject(image, seg) for image, seg in zip(args.image, args.seg)]
    image_tensor, mask_tensor = stack_subjects(subjects, args.labels)

    augmentor = AugTransformsGPU(args.config).to(args.device)
    image_tensor = image_tensor.to(args.device)
    mask_tensor = mask_tensor.to(args.device)

    augmented_image, augmented_mask = augmentor(image_tensor.clone(), mask_tensor.clone())

    # The old demos asserted these inline and are the reason the script is worth
    # keeping: a pipeline that changes shape or emits NaN is broken in a way the unit
    # tests' 24-voxel volumes do not always surface.
    if augmented_image.shape != image_tensor.shape:
        raise ValueError(f"augmented image shape {tuple(augmented_image.shape)} != input {tuple(image_tensor.shape)}")
    if augmented_mask.shape != mask_tensor.shape:
        raise ValueError(f"augmented mask shape {tuple(augmented_mask.shape)} != input {tuple(mask_tensor.shape)}")
    if torch.isnan(augmented_image).any():
        raise ValueError("NaNs in the augmented image")
    if torch.isnan(augmented_mask).any():
        raise ValueError("NaNs in the augmented mask")

    image_np = image_tensor.cpu().detach().numpy()
    augmented_image_np = augmented_image.cpu().detach().numpy()
    # Collapse the one-hot mask channels so the montage shows one picture per subject.
    mask_np = mask_tensor.cpu().detach().numpy().sum(axis=1)
    augmented_mask_np = augmented_mask.cpu().detach().numpy().sum(axis=1)

    pad_shape = 2 * (max(image_np.shape[2:]),)
    out_dir = Path(args.out_dir)

    for b in range(image_np.shape[0]):
        montage = np.concatenate(
            [
                mid_slices(image_np[b, 0], pad_shape),
                mid_slices(mask_np[b], pad_shape),
                mid_slices(augmented_image_np[b, 0], pad_shape),
                mid_slices(augmented_mask_np[b], pad_shape),
                # Channel 1 is the untouched segmentation channel; it should look
                # exactly like the input mask row above.
                mid_slices(augmented_image_np[b, 1], pad_shape),
            ],
            axis=0,
        )
        write_png(montage, out_dir / f"combined_{b}.png")

    print(augmentor)
    return 0


def demo_cpu(args: argparse.Namespace) -> int:
    from smauglab.transforms.cpu.transforms import AugTransforms

    image, seg_all = load_subject(args.image[0], args.seg[0])
    image_tensor = image.unsqueeze(0)

    mask = torch.zeros((len(args.labels), *seg_all.shape))
    for i, value in enumerate(args.labels):
        mask[i] = seg_all == value

    augmentor = AugTransforms(
        json_path=args.config,
        do_dummy_2d_data_aug=False,
        patch_size=tuple(args.patch_size),
        rotation_for_DA=(-10, 10),
    )

    draws = [augmentor(image=image_tensor.detach().clone(), segmentation=mask.detach().clone()) for _ in range(args.repeats)]

    out_dir = Path(args.out_dir)
    slice_index = image_tensor.shape[-3] // 2
    for key in ("image", "segmentation"):
        tiles = [normalize_minmax(draw[key].detach().numpy().sum(axis=0)[slice_index]) for draw in draws]
        rows = [np.concatenate(tiles[i : i + args.columns], axis=1) for i in range(0, len(tiles), args.columns)]
        # A short final row would not concatenate against the full-width ones.
        rows = [row for row in rows if row.shape[1] == rows[0].shape[1]]
        write_png(np.concatenate(rows, axis=0), out_dir / f"transforms_{key}.png")

    print(augmentor)
    return 0


# --- wiring -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", action="append", required=True, help="NIfTI image; repeat for a multi-subject batch")
    parser.add_argument("--seg", action="append", required=True, help="matching NIfTI segmentation; repeat alongside --image")
    parser.add_argument("--config", default=None, help="config JSON (default: the packaged GPU config)")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="img", help="where the PNGs go (default: img/)")
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEG_LABELS),
        help=f"segmentation values to split into mask channels (default: {' '.join(map(str, DEFAULT_SEG_LABELS))})",
    )
    parser.add_argument("--repeats", type=int, default=24, help="cpu backend: how many draws to render")
    parser.add_argument("--columns", type=int, default=6, help="cpu backend: tiles per row")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128], help="cpu backend: patch size")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if len(args.image) != len(args.seg):
        print(f"got {len(args.image)} --image but {len(args.seg)} --seg; they pair up one to one")
        return 2
    if args.config is None:
        args.config = str(default_config_path())
    args.labels = tuple(args.labels)

    return demo_gpu(args) if args.backend == "gpu" else demo_cpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
