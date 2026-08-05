"""Base contracts and shared helpers for the composable PALETTE pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class BlockContext:
    """Per-sample state shared across partitioners and remap steps."""
    image01: torch.Tensor          # (N,) float, min-max normalized to [0,1]
    fg_mask: torch.Tensor          # (N,) float 0/1
    coords: torch.Tensor           # (N, 3) ijk voxel coords
    shape: Tuple[int, int, int]    # (D, H, W)
    device: torch.device


class InitialPartitioner(nn.Module):
    """Produce the initial region map from the raw image."""

    def partition(self, ctx: BlockContext) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError


class RefinementPartitioner(nn.Module):
    """Subdivide an existing region map. May optionally consult the image."""

    def refine(
        self,
        ctx: BlockContext,
        region_ids: torch.Tensor,
        n_regions: int,
    ) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError


def signed_alpha_affine_remap(
    image01: torch.Tensor,
    fg_mask: torch.Tensor,
    region_ids: torch.Tensor,
    n_regions: int,
    alpha_magnitude_range: Tuple[float, float],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Signed-alpha per-region affine remap: y = μ_c + α_c · (x − mean_c).

    Mirrors fromSeg.py:392-401. Foreground-only means; output clamped to [0,1]
    and multiplied by the foreground mask.
    """
    device = image01.device
    alpha_lo, alpha_hi = alpha_magnitude_range

    s_c = torch.zeros(n_regions, device=device).scatter_add_(0, region_ids, image01 * fg_mask)
    n_c = torch.zeros(n_regions, device=device).scatter_add_(0, region_ids, fg_mask)
    mean_c = s_c / n_c.clamp(min=eps)

    mu_c = torch.rand(n_regions, device=device)
    mag_c = torch.rand(n_regions, device=device) * (alpha_hi - alpha_lo) + alpha_lo
    sign_c = (torch.rand(n_regions, device=device) > 0.5).float() * 2 - 1
    alp_c = mag_c * sign_c

    return (mu_c[region_ids] + alp_c[region_ids] * (image01 - mean_c[region_ids])).clamp(0, 1) * fg_mask
