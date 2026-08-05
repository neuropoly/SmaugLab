"""Concrete partitioner blocks for the composable PALETTE pipeline."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from auglab.transforms.gpu.fromSeg import _kmeans_1d, _voronoi_region_ids
from auglab.transforms.gpu.palette.base import (
    BlockContext,
    InitialPartitioner,
    RefinementPartitioner,
)
from auglab.transforms.synthseg.functional import em_subdivide_labels


# ── initial partitioners ────────────────────────────────────────────────────

class KMeans1DInitial(InitialPartitioner):
    """1-D K-means on foreground intensities, then bucketize the whole image.

    Mirrors fromSeg.py:374-385. With probability ``skip_prob`` (or when there
    are fewer than 4 foreground voxels) returns a single-region partition; the
    downstream signed-alpha remap handles that as a global remap.
    """

    def __init__(
        self,
        c_choices: Sequence[int] = (2, 3, 4, 5, 6),
        n_kmeans_subsample: int = 10_000,
        skip_prob: float = 0.10,
        dark_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.c_choices = list(c_choices)
        self.n_kmeans_subsample = int(n_kmeans_subsample)
        self.skip_prob = float(skip_prob)
        self.dark_threshold = float(dark_threshold)

    def partition(self, ctx: BlockContext) -> Tuple[torch.Tensor, int]:
        device = ctx.device
        flat = ctx.image01
        N = flat.shape[0]
        n_fg = int(ctx.fg_mask.sum().item())

        if n_fg < 4 or torch.rand(1, device=device).item() < self.skip_prob:
            return torch.zeros(N, dtype=torch.long, device=device), 1

        C_k = self.c_choices[int(torch.rand(1, device=device).item() * len(self.c_choices))]
        idx = torch.randint(0, N, (min(N, 40_000),), device=device)
        samp = flat[idx]
        sub_fg = samp[samp > self.dark_threshold][: self.n_kmeans_subsample]
        if sub_fg.numel() < 4:
            sub_fg = samp[: self.n_kmeans_subsample]

        centroids = _kmeans_1d(sub_fg, C_k)
        sorted_c, sort_idx = torch.sort(centroids)
        boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
        lbl_s = torch.bucketize(flat, boundaries)
        lbl_l = sort_idx[lbl_s].long()
        return lbl_l, C_k


class EMGMMInitial(InitialPartitioner):
    """SynthSeg-style EM/GMM clustering on the raw image.

    Calls ``em_subdivide_labels`` with a two-label bg/fg map (``fg_mask`` as
    the label input). Background is split into ``background_clusters_range``
    subclusters; foreground into ``n_foreground_clusters``.
    """

    def __init__(
        self,
        n_foreground_clusters: int = 3,
        background_clusters_range: Sequence[int] = (3, 10),
        background_label: int = 0,
        n_iters: int = 20,
        max_fit_voxels: int = 100_000,
    ) -> None:
        super().__init__()
        self.n_foreground_clusters = int(n_foreground_clusters)
        self.background_clusters_range = tuple(background_clusters_range)
        self.background_label = int(background_label)
        self.n_iters = int(n_iters)
        self.max_fit_voxels = int(max_fit_voxels)

    def partition(self, ctx: BlockContext) -> Tuple[torch.Tensor, int]:
        D, H, W = ctx.shape
        image = ctx.image01.view(1, 1, D, H, W)
        label_map = ctx.fg_mask.view(1, 1, D, H, W).long()
        fine, _gen, _out = em_subdivide_labels(
            image=image,
            label_map=label_map,
            n_foreground_clusters=self.n_foreground_clusters,
            background_clusters_range=self.background_clusters_range,
            background_label=self.background_label,
            n_iters=self.n_iters,
            max_fit_voxels=self.max_fit_voxels,
            channel=0,
        )
        rid_flat = fine.view(-1).long()
        return _densify_region_ids(rid_flat)


# ── refinement partitioners ────────────────────────────────────────────────

class VoronoiRefinement(RefinementPartitioner):
    """Spatially subdivide each existing region into S seed-nearest cells.

    Wraps _voronoi_region_ids (fromSeg.py:44) — coord-only, does not read the
    image. With per-region probability ``skip_prob`` the region is left intact.
    """

    def __init__(
        self,
        s_choices: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
        skip_prob: float = 0.40,
    ) -> None:
        super().__init__()
        self.s_choices = list(s_choices)
        self.skip_prob = float(skip_prob)

    def refine(
        self,
        ctx: BlockContext,
        region_ids: torch.Tensor,
        n_regions: int,
    ) -> Tuple[torch.Tensor, int]:
        return _voronoi_region_ids(
            ctx.coords, region_ids, ctx.fg_mask,
            n_regions, ctx.device, self.s_choices, self.skip_prob,
        )


class EMGMMRefinement(RefinementPartitioner):
    """Subdivide each existing region by intensity via SynthSeg's EM/GMM.

    Each incoming region is treated as one "label" fed to ``em_subdivide_labels``
    with ``n_foreground_clusters`` sub-clusters per region.
    """

    def __init__(
        self,
        n_foreground_clusters: int = 2,
        background_clusters_range: Sequence[int] = (2, 4),
        background_label: int = 0,
        n_iters: int = 20,
        max_fit_voxels: int = 100_000,
    ) -> None:
        super().__init__()
        self.n_foreground_clusters = int(n_foreground_clusters)
        self.background_clusters_range = tuple(background_clusters_range)
        self.background_label = int(background_label)
        self.n_iters = int(n_iters)
        self.max_fit_voxels = int(max_fit_voxels)

    def refine(
        self,
        ctx: BlockContext,
        region_ids: torch.Tensor,
        n_regions: int,
    ) -> Tuple[torch.Tensor, int]:
        D, H, W = ctx.shape
        image = ctx.image01.view(1, 1, D, H, W)
        label_map = region_ids.view(1, 1, D, H, W).long()
        fine, _gen, _out = em_subdivide_labels(
            image=image,
            label_map=label_map,
            n_foreground_clusters=self.n_foreground_clusters,
            background_clusters_range=self.background_clusters_range,
            background_label=self.background_label,
            n_iters=self.n_iters,
            max_fit_voxels=self.max_fit_voxels,
            channel=0,
        )
        return _densify_region_ids(fine.view(-1).long())


class IdentityRefinement(RefinementPartitioner):
    """Passthrough — leaves the running partition unchanged."""

    def refine(
        self,
        ctx: BlockContext,
        region_ids: torch.Tensor,
        n_regions: int,
    ) -> Tuple[torch.Tensor, int]:
        return region_ids, n_regions


# ── helpers ────────────────────────────────────────────────────────────────

def _densify_region_ids(region_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Remap sparse integer region ids to a contiguous [0, R) range.

    ``em_subdivide_labels`` encodes fine ids as ``parent_idx * mult + assign``,
    which is sparse. The downstream signed-alpha remap uses region ids as
    indices into per-region tensors of size ``R``, so they must be contiguous.
    """
    unique, inverse = torch.unique(region_ids, return_inverse=True)
    return inverse.long(), int(unique.numel())
