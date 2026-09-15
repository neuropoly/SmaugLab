"""Concrete partitioner blocks for the composable PALETTE pipeline.

Each block wraps machinery that already exists elsewhere in the package rather
than reimplementing it: `KMeans1DInitial` and `VoronoiRefinement` call the same
`_kmeans_1d` / `_voronoi_region_ids` helpers `RandomPaletteGPU` does, and the two
EM blocks call SynthSeg's `em_subdivide_labels`. The blocks are the composition,
not a second copy of the algorithms.
"""

from collections.abc import Sequence

import torch

from smauglab.transforms.gpu.fromSeg import _kmeans_1d, _voronoi_region_ids
from smauglab.transforms.gpu.palette.base import BlockContext, InitialPartitioner, RefinementPartitioner
from smauglab.transforms.synthseg.functional import em_subdivide_labels

# ── initial partitioners ──────────────────────────────────────────────────


class KMeans1DInitial(InitialPartitioner):
    """1-D K-means on foreground intensities, then bucketize the whole image.

    The partitioning step of `RandomPaletteGPU`, verbatim. With probability
    `skip_prob` -- or when there are fewer than four foreground voxels -- it
    returns a single region, which makes the downstream remap a global one; that
    is the same escape hatch `skip_parcellation_prob` was.
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

    def partition(self, ctx: BlockContext) -> tuple[torch.Tensor, int]:
        device = ctx.device
        flat = ctx.image01
        n_voxels = flat.shape[0]
        n_fg = int(ctx.fg_mask.sum().item())

        if n_fg < 4 or torch.rand(1, device=device).item() < self.skip_prob:
            return torch.zeros(n_voxels, dtype=torch.long, device=device), 1

        n_clusters = self.c_choices[int(torch.rand(1, device=device).item() * len(self.c_choices))]
        idx = torch.randint(0, n_voxels, (min(n_voxels, 40_000),), device=device)
        samp = flat[idx]
        sub_fg = samp[samp > self.dark_threshold][: self.n_kmeans_subsample]
        if sub_fg.numel() < 4:
            sub_fg = samp[: self.n_kmeans_subsample]

        centroids = _kmeans_1d(sub_fg, n_clusters)
        sorted_c, sort_idx = torch.sort(centroids)
        boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
        return sort_idx[torch.bucketize(flat, boundaries)].long(), n_clusters


class EMGMMInitial(InitialPartitioner):
    """SynthSeg-style EM/GMM clustering of the raw intensities.

    Where K-means sees only the 1-D histogram, this fits a Gaussian mixture, so
    two tissues that overlap in intensity but differ in spread end up in different
    regions. The foreground mask is handed over as a two-label bg/fg map: the
    background is split into `background_clusters_range` subclusters and the
    foreground into `n_foreground_clusters`.
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

    def partition(self, ctx: BlockContext) -> tuple[torch.Tensor, int]:
        depth, height, width = ctx.shape
        fine, _generation, _merge = em_subdivide_labels(
            image=ctx.image01.view(1, 1, depth, height, width),
            label_map=ctx.fg_mask.view(1, 1, depth, height, width).long(),
            n_foreground_clusters=self.n_foreground_clusters,
            background_clusters_range=self.background_clusters_range,
            background_label=self.background_label,
            n_iters=self.n_iters,
            max_fit_voxels=self.max_fit_voxels,
            channel=0,
        )
        return _densify_region_ids(fine.view(-1).long())


# ── refinement partitioners ───────────────────────────────────────────────


class VoronoiRefinement(RefinementPartitioner):
    """Spatially subdivide each existing region into seed-nearest cells.

    The sub-parcellation step of `RandomPaletteGPU`, verbatim. Reads coordinates
    only, never intensities, so the same region can end up with two different
    contrasts either side of a boundary the image does not show -- which is the
    point. Each region is left intact with probability `skip_prob`.
    """

    def __init__(
        self,
        s_choices: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
        skip_prob: float = 0.40,
    ) -> None:
        super().__init__()
        self.s_choices = list(s_choices)
        self.skip_prob = float(skip_prob)

    def refine(self, ctx: BlockContext, region_ids: torch.Tensor, n_regions: int) -> tuple[torch.Tensor, int]:
        return _voronoi_region_ids(
            ctx.coords,
            region_ids,
            ctx.fg_mask,
            n_regions,
            ctx.device,
            self.s_choices,
            self.skip_prob,
        )


class EMGMMRefinement(RefinementPartitioner):
    """Subdivide each existing region by intensity, via SynthSeg's EM/GMM.

    Every incoming region is treated as one "label" and split into
    `n_foreground_clusters` sub-clusters, so this refines along intensity where
    `VoronoiRefinement` refines along space.
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

    def refine(self, ctx: BlockContext, region_ids: torch.Tensor, n_regions: int) -> tuple[torch.Tensor, int]:
        # `n_regions` is unused: the incoming partition is carried by the label map
        # itself, and the new count comes back from `_densify_region_ids`.
        depth, height, width = ctx.shape
        fine, _generation, _merge = em_subdivide_labels(
            image=ctx.image01.view(1, 1, depth, height, width),
            label_map=region_ids.view(1, 1, depth, height, width).long(),
            n_foreground_clusters=self.n_foreground_clusters,
            background_clusters_range=self.background_clusters_range,
            background_label=self.background_label,
            n_iters=self.n_iters,
            max_fit_voxels=self.max_fit_voxels,
            channel=0,
        )
        return _densify_region_ids(fine.view(-1).long())


class IdentityRefinement(RefinementPartitioner):
    """Passthrough -- leaves the running partition unchanged."""

    def refine(self, ctx: BlockContext, region_ids: torch.Tensor, n_regions: int) -> tuple[torch.Tensor, int]:
        return region_ids, n_regions


# ── helpers ───────────────────────────────────────────────────────────────


def _densify_region_ids(region_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Remap sparse integer region ids onto a contiguous [0, R) range.

    `em_subdivide_labels` encodes a fine id as `parent_idx * mult + assign`, which
    is sparse. `signed_alpha_affine_remap` indexes per-region tensors of length R
    with these, so a sparse id would index out of bounds.
    """
    unique, inverse = torch.unique(region_ids, return_inverse=True)
    return inverse.long(), int(unique.numel())
