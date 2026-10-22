"""
SemRandConvTransform — augmentation-swap wrapper that exposes the SRCSM operator
(sem_rand_conv_3d.SemRandConv3D) through the batchgenerators transform contract
used by nnU-Net / AugLab.

This is the thin adapter between nnU-Net's dataloader and the operator. It does
NOT reimplement the augmentation — it only:
  - pulls the image + label map out of the batch dict (whatever the key names),
  - normalises the tensor layout (batch vs per-sample, numpy vs torch),
  - calls SemRandConv3D(image, label) with probability p,
  - writes the augmented image back and returns the dict unchanged otherwise.

It is written to be the SAME kind of object as a GIN-3D swap transform: a callable
`__call__(**data_dict) -> data_dict`. Register it in your AugLab augmentation list
in the intensity-augmentation slot (after spatial augmentation, before the patch
enters the network) — exactly where GIN-3D goes.

--------------------------------------------------------------------------- #
INTEGRATION NOTES  (read before wiring in)
--------------------------------------------------------------------------- #
* Two nnU-Net conventions are supported automatically:
    - v1 (batchgenerators):   data_dict['data'] (B,C,X,Y,Z), ['seg'] (B,C,X,Y,Z), numpy
    - v2 (batchgeneratorsv2): data_dict['image'] (C,X,Y,Z), ['segmentation'] (C,X,Y,Z), torch
  The wrapper detects key names and ndim (4 -> per-sample, 5 -> batch) and adapts.
  Override with image_key / seg_key if your harness uses different names.

* Single modality only (C == 1), matching the operator (in_channels=1), the same
  restriction as the planned GIN-3D swap. Multi-channel input raises.

* The label map MUST be the training segmentation for the crop. It is present in
  the dict at this pipeline stage in nnU-Net. If nnU-Net uses an ignore label
  (-1) or region-based training, negatives are folded into background (label 0)
  before one-hot; set num_labels to the number of foreground classes + 1.

* Device: defaults to auto (device=None -> cuda if visible, else cpu). The
  operator is device-agnostic and validated on an L40S at nnU-Net patch size
  (160x128x160): ~39 ms/call per-label (B=2, L=5), ~580 MB peak, fp16-safe.
  Force device='cpu' only when constructing inside forked dataloader workers,
  where spawning CUDA contexts per worker is undesirable; device='cuda' forces
  GPU regardless.

* CRITICAL — match the intensity-augmentation budget to PALETTE. Their pipeline
  used ONLY this operator (no gamma/brightness/contrast/noise). Disable nnU-Net's
  default intensity augmentations for this arm so the comparison isolates the
  mechanism (see README).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from sem_rand_conv_3d import SemRandConv3D


_IMAGE_KEYS = ("data", "image", "images")
_SEG_KEYS = ("seg", "segmentation", "target", "seg_onehot")


def _find_key(data_dict: dict, candidates: Sequence[str],
              override: Optional[str]) -> str:
    if override is not None:
        if override not in data_dict:
            raise KeyError(f"key '{override}' not in data_dict "
                           f"(have {list(data_dict.keys())})")
        return override
    for k in candidates:
        if k in data_dict:
            return k
    raise KeyError(f"none of {candidates} in data_dict "
                   f"(have {list(data_dict.keys())})")


class SemRandConvTransform:
    """batchgenerators-style augmentation-swap transform for the SRCSM operator.

    Args:
        num_labels:   foreground classes + 1 (background). One-hot spans range(num_labels).
        p:            probability of applying the augmentation to a given call.
        per_label:    True -> semantic-aware (per-region); False -> global RCNet.
        smoothing:    Gaussian-smoothed (soft) label masks.
        device:       None (default) auto-selects cuda if available else cpu;
                      pass 'cpu' explicitly inside forked dataloader workers.
        image_key/seg_key: override auto-detected dict keys.
        num_filters_base, kernel_candidates, leaky_slope: passed to the operator.
    """

    def __init__(self, num_labels: int, p: float = 1.0, per_label: bool = True,
                 smoothing: bool = True, device: Optional[str] = None,
                 image_key: Optional[str] = None, seg_key: Optional[str] = None,
                 num_filters_base: int = 2, kernel_candidates=(1, 3),
                 leaky_slope: float = 0.1, max_retries: int = 20):
        self.p = float(p)
        # device=None -> auto: GPU when visible, else CPU. Pass device='cpu'
        # explicitly if you construct this inside forked dataloader workers,
        # where CUDA-in-worker is undesirable; pass device='cuda' to force GPU.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.image_key = image_key
        self.seg_key = seg_key
        self.op = SemRandConv3D(
            num_labels=num_labels, per_label=per_label, smoothing=smoothing,
            num_filters_base=num_filters_base, kernel_candidates=kernel_candidates,
            leaky_slope=leaky_slope, max_retries=max_retries,
        ).to(self.device)

    # -- layout helpers ----------------------------------------------------- #
    @staticmethod
    def _to_torch(x):
        """Return (tensor, was_numpy, orig_dtype)."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x), True, x.dtype
        return x, False, x.dtype

    def __call__(self, **data_dict):
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return data_dict

        ik = _find_key(data_dict, _IMAGE_KEYS, self.image_key)
        sk = _find_key(data_dict, _SEG_KEYS, self.seg_key)

        img_raw, img_np, img_dt = self._to_torch(data_dict[ik])
        seg_raw, _, _ = self._to_torch(data_dict[sk])

        # normalise to (B, 1, D, H, W)
        added_batch = False
        if img_raw.dim() == 4:            # (C, D, H, W) per-sample -> add batch
            img_raw = img_raw.unsqueeze(0)
            seg_raw = seg_raw.unsqueeze(0)
            added_batch = True
        elif img_raw.dim() != 5:
            raise ValueError(f"image must be 4D or 5D, got {img_raw.dim()}D")

        if img_raw.shape[1] != 1:
            raise ValueError(f"single-channel only (C==1), got C={img_raw.shape[1]}")

        # seg -> (B, 1, D, H, W) integer, ignore/negative labels -> background
        if seg_raw.dim() == 5 and seg_raw.shape[1] != 1:
            seg_raw = seg_raw[:, :1]      # first channel is the label map in nnU-Net
        seg_lab = seg_raw.clone()
        seg_lab[seg_lab < 0] = 0          # fold ignore label (-1) into background

        img_f = img_raw.to(self.device).float()
        seg_l = seg_lab.to(self.device).long()

        aug = self.op(img_f, seg_l)       # (B,1,D,H,W)

        aug = aug.to(dtype=img_raw.dtype)
        if added_batch:
            aug = aug.squeeze(0)
        aug = aug.cpu() if img_np else aug

        data_dict[ik] = aug.numpy().astype(img_dt) if img_np else aug
        return data_dict


# GIN-3D shares this exact wrapper shape; to swap operators, construct the
# transform with a different `op`. Exposed as a factory for the harness:
def build_semrandconv_transform(num_labels: int, **kwargs) -> SemRandConvTransform:
    return SemRandConvTransform(num_labels=num_labels, **kwargs)
