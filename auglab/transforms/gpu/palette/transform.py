"""Composed PALETTE synthesis transform."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

import torch
from kornia.core import Tensor
from torch import nn

from auglab.transforms.gpu.base import ImageOnlyTransform
from auglab.transforms.gpu.fromSeg import _gaussian_blur_3d, collapse_onehot_to_index
from auglab.transforms.gpu.palette.base import (
    BlockContext,
    InitialPartitioner,
    RefinementPartitioner,
    signed_alpha_affine_remap,
)
from auglab.transforms.gpu.palette.overlay import AnatomicalLabelOverlay


class PaletteSynthesisGPU(ImageOnlyTransform):
    """PALETTE contrast synthesis composed from swappable partition blocks.

    Pipeline: min-max normalise → initial_partitioner → refinement_partitioners →
    signed-alpha per-region affine remap → optional blur → anatomical-label
    overlay → optional blur → foreground z-score.

    The initial partitioner produces the region map from raw intensities
    (k-means, EM). Refinement partitioners subdivide that partition further
    (Voronoi, EM). The intensity remap step is fixed. The overlay algorithm is
    fixed; only its frequency and blend amount are tunable.
    """

    def __init__(
        self,
        initial_partitioner: InitialPartitioner,
        refinement_partitioners: Optional[List[RefinementPartitioner]] = None,
        overlay: Optional[AnatomicalLabelOverlay] = None,
        alpha_magnitude_range: Sequence[float] = (0.5, 2.0),
        dark_threshold: float = 0.01,
        blur_sigmas_pre: Sequence[float] = (0.0, 0.0, 0.0, 0.3, 0.5, 0.8),
        blur_sigmas_post: Sequence[float] = (0.0, 0.0, 0.0, 0.3, 0.5, 0.8),
        p: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, **kwargs)
        if not isinstance(initial_partitioner, InitialPartitioner):
            raise TypeError(
                f"initial_partitioner must be an InitialPartitioner, "
                f"got {type(initial_partitioner).__name__}"
            )
        refinements = list(refinement_partitioners or [])
        for r in refinements:
            if not isinstance(r, RefinementPartitioner):
                raise TypeError(
                    f"refinement_partitioners must be RefinementPartitioner "
                    f"instances, got {type(r).__name__}"
                )
        self.initial = initial_partitioner
        self.refinements = nn.ModuleList(refinements)
        self.overlay = overlay
        self.alpha_magnitude_range = tuple(alpha_magnitude_range)
        self.dark_threshold = float(dark_threshold)
        self.blur_sigmas_pre = list(blur_sigmas_pre)
        self.blur_sigmas_post = list(blur_sigmas_post)

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Any],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        seg_raw: Optional[torch.Tensor] = params.get("seg", None)
        labels: Optional[torch.Tensor] = None
        if seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] > 1:
            labels = collapse_onehot_to_index(seg_raw)
        elif seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] == 1:
            labels = seg_raw.long()

        B, _C, D, H, W = input.shape
        N = D * H * W
        device = input.device
        eps = 1e-7

        # 1. Per-sample min-max normalize (channel 0) to [0, 1]
        flat_all = input[:, 0].float().reshape(B, N)
        v_min = flat_all.min(dim=1).values.view(B, 1)
        v_max = flat_all.max(dim=1).values.view(B, 1)
        images_01 = ((flat_all - v_min) / (v_max - v_min + eps)).clamp(0, 1)
        flat_m_all = (images_01 > self.dark_threshold).float()

        coords = torch.stack(torch.meshgrid(
            torch.arange(D, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"), dim=-1).reshape(N, 3)

        # 2. Per sample: partition stack, then fixed signed-alpha remap
        synth_list = []
        for i in range(B):
            ctx = BlockContext(
                image01=images_01[i],
                fg_mask=flat_m_all[i],
                coords=coords,
                shape=(D, H, W),
                device=device,
            )
            rid, R = self.initial.partition(ctx)
            for r in self.refinements:
                rid, R = r.refine(ctx, rid, R)
            synth_i = signed_alpha_affine_remap(
                ctx.image01, ctx.fg_mask, rid, R, self.alpha_magnitude_range,
            )
            synth_list.append(synth_i)

        synth = torch.stack(synth_list)               # (B, N)
        synth_01 = synth.reshape(B, 1, D, H, W)

        # 3. Optional pre-overlay blur
        sigma = random.choice(self.blur_sigmas_pre) if self.blur_sigmas_pre else 0.0
        if sigma > 0.0:
            synth_01 = _gaussian_blur_3d(synth_01, sigma)
            synth = synth_01.reshape(B, N)

        # 4. Anatomical-label overlay (fixed algorithm, tunable knobs)
        if self.overlay is not None and labels is not None:
            synth = self.overlay.apply(synth, labels, (D, H, W))

        # 5. Optional post-overlay blur
        synth_01 = synth.reshape(B, 1, D, H, W)
        sigma2 = random.choice(self.blur_sigmas_post) if self.blur_sigmas_post else 0.0
        if sigma2 > 0.0:
            synth_01 = _gaussian_blur_3d(synth_01, sigma2)
            synth = synth_01.reshape(B, N)

        # 6. Foreground z-score
        b_sum = (synth * flat_m_all).sum(dim=1, keepdim=True)
        b_cnt = flat_m_all.sum(dim=1, keepdim=True).clamp(min=1)
        b_mean = b_sum / b_cnt
        b_sq = ((synth - b_mean) * flat_m_all).pow(2).sum(dim=1, keepdim=True)
        b_std = (b_sq / b_cnt + eps).sqrt()
        synth_z = ((synth - b_mean) / b_std * flat_m_all).reshape(B, 1, D, H, W)

        out = input.clone()
        out[:, 0:1] = synth_z.to(input.dtype)
        return out
