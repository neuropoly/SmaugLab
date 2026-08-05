"""Per-anatomical-label overlay: fixed algorithm, tunable frequency and blend."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


BlendSpec = Union[float, Sequence[float]]


class AnatomicalLabelOverlay(nn.Module):
    """PALETTE per-anatomical-label affine remap with a tunable blend.

    Reproduces fromSeg.py:414-447 algorithmically. For each foreground label,
    with probability ``label_remap_prob``, samples a fresh (μ, α) and computes
    ``new_vals = μ + α·(synth − mean_label)``, then blends into the current
    synthesised image via a per-label blend strength.

    Args:
        label_remap_prob: per-label per-sample probability of applying the remap.
        min_label_voxels: minimum voxel count for a label to be eligible.
        label_classes: if set, restrict overlay to these class indices.
        blend_strength: scalar in [0,1] (full overwrite = 1.0) or [lo, hi]
            sampled per label per sample. Applied as
            ``write_mask = c_mask · apply · blend``.
        alpha_magnitude_range: [lo, hi] for |α| of the signed-alpha remap.
    """

    def __init__(
        self,
        label_remap_prob: float = 0.5,
        min_label_voxels: int = 4,
        label_classes: Optional[List[int]] = None,
        blend_strength: BlendSpec = 1.0,
        alpha_magnitude_range: Sequence[float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        self.label_remap_prob = float(label_remap_prob)
        self.min_label_voxels = int(min_label_voxels)
        self.label_classes = None if label_classes is None else list(label_classes)
        self.blend_strength = _normalize_blend(blend_strength)
        self.alpha_magnitude_range = tuple(alpha_magnitude_range)

    def apply(
        self,
        synth: torch.Tensor,        # (B, N)
        labels: torch.Tensor,       # (B, 1, D, H, W) long
        shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        B, N = synth.shape
        device = synth.device
        alpha_lo, alpha_hi = self.alpha_magnitude_range
        blend_lo, blend_hi = self.blend_strength
        D, H, W = shape

        if labels.shape[2:] != (D, H, W):
            labels = F.interpolate(labels.float(), size=(D, H, W), mode="nearest").long()
        lbl = labels[:, 0].reshape(B, N).clamp(min=0)

        unique_classes = lbl.unique()
        unique_classes = unique_classes[unique_classes > 0]
        if self.label_classes is not None:
            keep = torch.tensor(self.label_classes, device=device)
            unique_classes = unique_classes[torch.isin(unique_classes, keep)]

        for c in unique_classes:
            c_val = int(c.item())
            c_mask = (lbl == c_val).float()
            c_cnt = c_mask.sum(dim=1, keepdim=True)

            apply = (
                (torch.rand(B, 1, device=device) < self.label_remap_prob)
                & (c_cnt >= self.min_label_voxels)
            ).float()

            if apply.sum() == 0:
                continue

            c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)

            mu_c = torch.rand(B, 1, device=device)
            mag_c = torch.rand(B, 1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
            sign_c = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1
            alp_c = mag_c * sign_c

            if blend_lo == blend_hi:
                blend = torch.full((B, 1), blend_lo, device=device)
            else:
                blend = torch.rand(B, 1, device=device) * (blend_hi - blend_lo) + blend_lo

            new_vals = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)
            write_mask = c_mask * apply * blend
            synth = synth * (1.0 - write_mask) + new_vals * write_mask

        return synth


def _normalize_blend(spec: BlendSpec) -> Tuple[float, float]:
    if isinstance(spec, (int, float)):
        v = float(spec)
        return (v, v)
    lo, hi = float(spec[0]), float(spec[1])
    if lo > hi:
        raise ValueError(f"blend_strength range must be non-decreasing, got [{lo}, {hi}]")
    return (lo, hi)
