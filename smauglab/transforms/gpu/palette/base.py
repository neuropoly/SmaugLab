"""Base contracts and the one shared helper for the composable PALETTE pipeline.

`RandomPaletteGPU` in `gpu/fromSeg.py` hard-codes one composition: 1-D K-means,
then Voronoi, then a per-region affine remap. This package takes that pipeline
apart so the two partitioning steps become swappable blocks, while the remap --
the part that actually synthesises contrast -- stays fixed.

The split is `partition` (build a region map from the image) versus `refine`
(subdivide an existing one), because those are genuinely different signatures: a
refinement is handed the running partition and must return a finer one, and a
Voronoi refinement never looks at intensities at all. Keeping them as separate
base classes is what lets the config reject "voronoi as an initial partitioner"
at build time rather than at the first `partition()` call in the training loop.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class BlockContext:
    """Per-sample state every block reads, computed once by the transform.

    Bundled rather than passed as five arguments: `coords` in particular is a
    (N, 3) tensor shared across the whole batch, and threading it through each
    block's signature by hand is how the two copies of this pipeline drifted apart
    in the first place.
    """

    image01: torch.Tensor  # (N,) float, min-max normalised to [0, 1]
    fg_mask: torch.Tensor  # (N,) float 0/1
    coords: torch.Tensor  # (N, 3) ijk voxel coordinates
    shape: tuple[int, int, int]  # (D, H, W)
    device: torch.device


class InitialPartitioner(nn.Module):
    """Produce the initial region map from the raw image."""

    def partition(self, ctx: BlockContext) -> tuple[torch.Tensor, int]:
        """Return (per-voxel region id in [0, R), R)."""
        raise NotImplementedError


class RefinementPartitioner(nn.Module):
    """Subdivide an existing region map. May optionally consult the image."""

    def refine(self, ctx: BlockContext, region_ids: torch.Tensor, n_regions: int) -> tuple[torch.Tensor, int]:
        """Return (finer per-voxel region id in [0, R'), R')."""
        raise NotImplementedError


def signed_alpha_affine_remap(
    image01: torch.Tensor,
    fg_mask: torch.Tensor,
    region_ids: torch.Tensor,
    n_regions: int,
    alpha_magnitude_range: Sequence[float],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Signed-alpha per-region affine remap: y = mu_c + alpha_c * (x - mean_c).

    The same formula `RandomPaletteGPU` applies to its K-means/Voronoi regions,
    lifted out so every composition gets it verbatim. Region means are
    foreground-only, and the result is clamped to [0, 1] and masked, so background
    stays background whatever the partition looks like.
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
