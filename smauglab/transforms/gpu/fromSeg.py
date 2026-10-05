import functools
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform, segmentation_from
from smauglab.transforms.kernels import gaussian_blur3d
from smauglab.transforms.rng import shared_choice

# ── PALETTE AUG helpers ──────────────────────────────────────────────────


def _kmeans_1d(values: torch.Tensor, C: int, n_iter: int = 10) -> torch.Tensor:
    """1-D K-means on foreground values. Returns (C,) centroids.

    Lloyd's algorithm on sorted data, which is the same algorithm the dense version
    ran but with each step costing O(C) instead of O(n*C):

    * In one dimension with sorted centroids, nearest-centroid assignment is an
      interval split at the midpoints, so the cluster boundaries come from a
      `searchsorted` over the sorted values rather than an `argmin` over an
      `n x C` distance matrix.
    * Cluster sums then come from differences of one prefix sum, instead of a
      `scatter_add_` of n values into C bins -- which on CUDA means every thread
      contending for one of a handful of addresses, and measured as the single most
      expensive operation in `RandomPaletteGPU`.
    * The `torch.allclose` convergence test is gone. It was a host synchronisation
      on every iteration, and the loop body it guards is now a few kernels over C
      elements; waiting for the GPU cost more than the iterations it saved.

    The prefix sum is accumulated in float64. A float32 running total over ten
    thousand values loses enough low-order bits that two nearby cluster boundaries
    can disagree about which side a value fell on, and the result is an augmentation
    whose region map depends on the subsample size.
    """
    device = values.device
    v_min, v_max = values.min(), values.max()
    if C <= 1:
        return v_min.reshape(1)

    # `torch.linspace(a, b, C)`, written out: the last point is `b` exactly, not
    # `a + (C-1)*step`, and starting K-means from a different point can land it in a
    # different local optimum.
    step = (v_max - v_min) / (C - 1)
    centroids = v_min + step * torch.arange(C, device=device, dtype=values.dtype)
    centroids = torch.cat([centroids[:-1], v_max.reshape(1)])

    ordered, _ = torch.sort(values)
    prefix = torch.cat([torch.zeros(1, device=device, dtype=torch.float64), ordered.double().cumsum(0)])
    n = ordered.numel()
    tail = torch.full((1,), n, dtype=torch.long, device=device)
    head = torch.zeros(1, dtype=torch.long, device=device)

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        # `right=True`: a value exactly on a midpoint joins the lower cluster, which
        # is what `argmin` did (it returns the first of two equal distances) and what
        # the `torch.bucketize` the caller runs afterwards does.
        edges = torch.searchsorted(ordered, boundaries, right=True)
        starts = torch.cat([head, edges])
        ends = torch.cat([edges, tail])
        counts = ends - starts
        sums = prefix[ends] - prefix[starts]
        means = (sums / counts.clamp_min(1)).to(centroids.dtype)
        centroids = torch.where(counts > 0, means, centroids)
    return centroids


def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur of a [0, 1] volume. x: (B, 1, D, H, W).

    The blur itself is the shared one; this wrapper only keeps the clamp, which is a
    no-op guard for the already-normalised `synth_01` inputs. The previous local copy
    zero-padded (via conv3d's `padding=`) rather than reflecting, which darkened the
    volume border.
    """
    return gaussian_blur3d(x, sigma).clamp(0, 1)


def _voronoi_region_ids(
    coords: torch.Tensor,
    lbl_l: torch.Tensor,
    fg: torch.Tensor,
    C: int,
    device: torch.device,
    s_choices: Sequence[int],
    skip_sub_parc_prob: float,
) -> tuple[torch.Tensor, int]:
    """Spatially subdivide each K-means cluster into Voronoi sub-regions.

    Returns (rid, R): per-voxel sub-region id and total region count.
    """
    N = lbl_l.shape[0]
    rid = torch.zeros(N, dtype=torch.long, device=device)

    # Everything the loop needs off the GPU, in two reads instead of three per
    # cluster. Each `.item()` inside the loop drained the queue, and with up to six
    # clusters per sample and two samples per batch that was the dominant cost of
    # the whole transform on a 128^3 patch.
    #
    # Drawing the decisions up front consumes a fixed two uniforms per cluster where
    # the scalar version drew the second one only when the first said "subdivide",
    # so a seeded run no longer reproduces the previous release's region maps. The
    # draws are the same independent uniforms either way.
    counts = torch.bincount(lbl_l[fg > 0], minlength=C)[:C].tolist()
    decisions = torch.rand(C, 2, device=device).tolist()

    offset = 0
    for c in range(C):
        n_fg = int(counts[c])
        if n_fg == 0:
            continue
        c_mask = lbl_l == c
        if n_fg < 2 or decisions[c][0] < skip_sub_parc_prob:
            S = 1
        else:
            s_idx = int(decisions[c][1] * len(s_choices))
            S = min(s_choices[s_idx], n_fg)
        if S <= 1:
            rid.masked_fill_(c_mask, offset)
            offset += 1
            continue
        # Only this cluster's voxels take part: the result is written under `c_mask`
        # anyway, so the full-volume `cdist` computed C times as many distances as
        # the output has rows.
        member = c_mask.nonzero(as_tuple=True)[0]
        fg_member = member[fg[member] > 0]
        # The S largest of n iid uniforms is a uniformly random S-subset, and `topk`
        # for S <= 10 is a reduction where `randperm` is a full sort of n: 0.11 ms
        # against 0.49 ms at a million foreground voxels.
        seed_idx = fg_member[torch.topk(torch.rand(fg_member.numel(), device=device), S).indices]
        d = torch.cdist(coords.index_select(0, member), coords.index_select(0, seed_idx))
        rid[member] = offset + torch.argmin(d, dim=1)
        offset += S
    # At least one region, always. Every cluster can be skipped above -- a constant image
    # has no foreground at all, so `n_fg` is zero for all of them -- and `offset` then
    # stays at 0 while `rid` is a valid all-zero id map. The caller sizes its scatter
    # target with this count, so returning 0 means scattering index 0 into a zero-length
    # tensor: a RuntimeError on CPU and a device-side assert on CUDA, which does not just
    # fail the batch, it poisons the context and ends the run.
    return rid, max(offset, 1)


# ─────────────────────────────────────────────────────────────────────────────


def segment_sum(ids: torch.Tensor, values: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Sum `values` into `n_segments` bins indexed by `ids`. Returns `(n_segments,)`.

    `torch.zeros(R).scatter_add_(0, ids, values)` is the obvious spelling and the
    slow one: with a couple of dozen bins and two million values, every thread in
    the grid contends for one of a handful of global addresses. `bincount` keeps its
    accumulator in shared memory per block, and measured 0.16 ms against 0.52 ms for
    a 128^3 volume over 24 regions.

    `bincount` sizes its output from the largest id present, so a trailing segment
    that no voxel landed in would come back missing; the pad restores it as a zero.
    """
    summed = torch.bincount(ids, weights=values, minlength=n_segments).to(values.dtype)
    if summed.numel() < n_segments:
        summed = F.pad(summed, (0, n_segments - summed.numel()))
    return summed[:n_segments]


def foreground_classes(labels: torch.Tensor) -> torch.Tensor:
    """The distinct positive values in an integer label volume, ascending.

    `labels.unique()` radix-sorts the whole volume to answer a question about a
    handful of small integers; counting them and keeping the non-empty bins is half
    the time on a 2 x 128^3 batch, and the result is identical.
    """
    flat = labels.reshape(-1)
    counts = torch.bincount(flat.clamp_min(0))
    present = counts.nonzero().flatten()
    return present[present > 0]


@functools.lru_cache(maxsize=8)
def voxel_coordinates(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """`[D*H*W, 3]` ijk voxel coordinates for a patch shape, built once and shared.

    This is a function of the patch shape alone -- fixed for a training run -- but
    was rebuilt on every call, and at 128^3 the `meshgrid`/`stack` writes 24 MB each
    time. The returned tensor is shared; treat it as read-only.
    """
    depth, height, width = shape
    return torch.stack(
        torch.meshgrid(
            torch.arange(depth, device=device, dtype=torch.float32),
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(depth * height * width, 3)


#: sqrt(2*pi), as a Python float. It used to be built with `torch.sqrt(torch.tensor(
#: 2*pi, device=...))` inside `_normal_pdf`, which is a host-to-device copy and a
#: kernel launch for a compile-time constant -- once per region per sample.
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _normal_pdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    inv = 1.0 / (std + 1e-6)
    return (inv / _SQRT_TWO_PI) * torch.exp(-0.5 * ((x - mean) * inv) ** 2)


def _region_moments(region_mask: torch.Tensor, design: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxel count, mean and standard deviation per region, in one matrix multiply.

    `design` is the `[S, 3]` stack of `1`, `x` and `x**2`, so the product holds the
    three sums each region's statistics need. The variance comes from
    `E[x^2] - E[x]^2`, which is the same quantity the explicit
    `((x - mean) * mask) ** 2` pass computed -- the mask is binary, so squaring it
    changes nothing -- without materialising the centred array.
    """
    sums = region_mask.to(design.dtype) @ design  # (R, 3)
    counts = sums[:, 0].clamp_min(1)
    means = sums[:, 1] / counts
    variances = (sums[:, 2] / counts - means * means).clamp_min(0)
    return counts, means, variances.sqrt()


## Redistribute segmentation values transform (GPU)
@register(
    aug_id=AugId.REDISTRIBUTE_SEG,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomRedistributeSegGPU(ImageOnlyTransform):
    """Redistribute image values using segmentation regions (GPU version).

    Mirrors the CPU `RedistributeTransform` behavior using GPU-friendly ops.
    Works with inputs shaped [N, C, H, W] or [N, C, D, H, W].

    `retain_stats` defaults to True, and wants to stay that way. The whole method
    operates in a per-sample [0, 1] min-max space and adds a perturbation of up to
    2.0 *in that space* -- twice the input's full dynamic range. With
    `retain_stats=False` nothing maps the result back, so the output is
    independent of the input's scale entirely: a z-scored patch, the same patch
    scaled by 100, and anything else all come out in the same [0.5, 2.9] band with
    mean 2.2. For a network fed z-scored patches that is finite, silent and badly
    out of distribution -- the same defect the no-foreground branch below carries a
    comment about.

    Mapping back is not a fix on its own: the perturbation is defined in
    normalised units, so rescaling it by the input range makes it far larger
    (measured, mean 18.4 rather than 2.2). Bounding the amplitude in input units
    would be a redesign of an augmentation inherited from totalspineseg, so this
    only changes which setting you get by default. `transform_params_hybrid.json`
    and `transform_params_hybrid_TAGE.json` ask for False explicitly and are
    unaffected.
    """

    def __init__(
        self,
        in_seg: float = 0.2,
        apply_to_channel: Sequence[int] = (0,),
        retain_stats: bool = True,
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
        std_noise_range: Sequence[float] = (0.1, 0.3),
        dilation_iterations_range: Sequence[int] = (1, 3),
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.in_seg = in_seg
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.std_noise_range = std_noise_range
        self.dilation_iterations_range = dilation_iterations_range

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # Expect segmentation provided in params: shape [N, 1, ...] or [N, C_seg, ...]
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg = segmentation_from(params)
        if seg is None:
            return input
        if seg.dim() != input.dim():
            # Allow seg [N, ...] by adding channel dim
            if seg.dim() == input.dim() - 1:
                seg = seg.unsqueeze(1)
            else:
                return input

        spatial_dims = input.dim() - 2
        if spatial_dims not in (2, 3):
            return input

        N = input.shape[0]

        # One region per label, whichever layout the caller used. Reading regions off
        # the channel axis alone collapses a single-channel label map -- which is what
        # nnU-Net's trainer passes -- to one foreground blob.
        regions = seg_region_masks(seg)

        # Apply per selected image channel and per batch sample
        for c in self.apply_to_channel:
            img_batch = input[:, c]  # (N, [...])
            # Sanitize incoming values to prevent NaN/Inf propagation
            img_batch = torch.nan_to_num(img_batch, nan=0.0, posinf=0.0, neginf=0.0)

            # Optionally retain original stats (vectorized per sample)
            if self.retain_stats:
                flat = img_batch.view(N, -1)
                orig_mean = flat.mean(dim=1)
                # Use unbiased=False to avoid NaNs for tiny tensors
                orig_std = flat.std(dim=1, unbiased=False)

            # Normalize entire batch to [0,1] per sample
            img_min = img_batch.view(N, -1).min(dim=1)[0].view(N, *([1] * (img_batch.dim() - 1)))
            img_max = img_batch.view(N, -1).max(dim=1)[0].view(N, *([1] * (img_batch.dim() - 1)))
            denom = (img_max - img_min).clamp_min(1e-6)
            x_batch = (img_batch - img_min) / denom

            # Everything the per-sample loop needs to read back to Python, read once.
            # These were three `.item()` calls inside the loop -- the empty-mask test,
            # the redistribution mode and the dilation count -- and each one stalls the
            # host until the queue drains. Drawing the two random ones for the whole
            # batch up front gives each sample its own independent draw exactly as
            # before, but no longer in the interleaved order the loop produced, so a
            # seeded run differs from the previous release's.
            has_foreground = (seg.reshape(N, -1) > 0).any(dim=1).tolist()
            in_seg_draws = (torch.rand(N, device=input.device) <= self.in_seg).tolist()
            dilation_draws = torch.randint(
                self.dilation_iterations_range[0], self.dilation_iterations_range[1] + 1, (N,), device=input.device
            ).tolist()

            # Iterate per sample (seg can differ in shape or labels per sample)
            for b in range(N):
                x = x_batch[b]
                # No regions to redistribute between, so leave the sample exactly as it
                # came in. `x_batch` is the [0,1] min-max normalisation this transform
                # works in, and writing *that* back -- which is what this did -- turns a
                # z-scored air patch at about [-2.7, -2.67] into [0, 1] and then skips
                # the `retain_stats` restore below that would have undone it. It is not a
                # rare shape: nnU-Net leaves some samples of every batch unconstrained,
                # and a few cases are effectively unlabelled, so this fires on real
                # anatomy and produces a perfectly finite, badly out-of-distribution
                # patch that no NaN guard can see.
                if not has_foreground[b]:
                    input[b, c] = img_batch[b]
                    continue

                # Decide redistribution mode once per sample
                in_seg_bool = in_seg_draws[b]

                # Binary masks for regions
                masks = regions[b]  # (R, ...)
                R = masks.shape[0]

                # Vectorized dilation for all regions (3 iterations)
                dilated = masks.float()
                for _ in range(int(dilation_draws[b])):
                    if spatial_dims == 3:
                        dilated = F.max_pool3d(dilated.unsqueeze(0), 3, 1, 1).squeeze(0)
                    else:
                        dilated = F.max_pool2d(dilated.unsqueeze(0), 3, 1, 1).squeeze(0)
                dilated_excl = (dilated > 0) & (~masks)

                # Flatten for stats
                x_flat = x.view(1, -1)  # (1, S)
                mask_flat = masks.view(R, -1)
                dil_flat = dilated_excl.view(R, -1)

                # Count, mean and variance per region, as one [R, S] x [S, 3] matrix
                # multiply each. Spelled out, that was six passes over an [R, S]
                # array and two [R, S] temporaries -- 32 MB apiece for four regions
                # of a 128^3 patch -- to produce six numbers per region.
                design = torch.stack([torch.ones_like(x_flat[0]), x_flat[0], x_flat[0] * x_flat[0]], dim=1)  # (S, 3)
                counts, means, stds = _region_moments(mask_flat, design)
                dil_counts, dil_means, dil_stds = _region_moments(dil_flat, design)

                # redist_std per region
                std_noise_range = (
                    torch.rand(1, device=input.device)[0] * (self.std_noise_range[1] - self.std_noise_range[0]) + self.std_noise_range[0]
                )
                redist_std = torch.maximum(
                    torch.rand(R, device=input.device) * std_noise_range + 0.4 * torch.abs((means - dil_means) * stds / (dil_stds + 1e-6)),
                    torch.full((R,), 0.01, device=input.device, dtype=input.dtype),
                )

                # Build additive term.
                #
                # Accumulated in place rather than stacked: the global branch used to
                # build a list of R full-volume tensors and `torch.stack` them before
                # summing, which is an extra R-volume allocation and an extra pass over
                # it. The in-region branch indexed each region out with a boolean mask,
                # and a boolean index is a `nonzero` -- a host synchronisation per
                # region, on top of the gather. Multiplying by the mask writes the same
                # values at the same voxels, because the regions are summed either way.
                #
                # The `counts[r] == 0` guards both branches carried could not fire:
                # `counts` is `clamp_min(1)`.
                to_add = torch.zeros_like(x)
                rand_sign = 2 * torch.rand(R, device=input.device) - 1  # random sign factor per region
                for r in range(R):
                    contribution = _normal_pdf(x, means[r], redist_std[r]) * rand_sign[r]
                    if in_seg_bool:
                        contribution = contribution * mask_flat[r].view(x.shape)
                    to_add += contribution

                # Normalize to_add if non-zero. `torch.where`, not `if`: the test is one
                # more host read, and both sides are cheap.
                tmin, tmax = to_add.min(), to_add.max()
                x = torch.where(tmax - tmin > 1e-8, x + 2 * (to_add - tmin) / (tmax - tmin + 1e-6), x)

                # Restore stats
                if self.retain_stats:
                    mean = x.mean()
                    # Use unbiased=False to avoid NaNs on degenerate shapes
                    std = x.std(unbiased=False)
                    x = (x - mean) / torch.clamp(std, min=1e-7)
                    x = x * orig_std[b] + orig_mean[b]

                # Final safety: check if nan/inf appeared
                if not bool(torch.isfinite(x).all()):
                    print(f"Warning nan: {self.__class__.__name__}", flush=True)
                    continue
                input[b, c] = x

        return input


@register(
    aug_id=AugId.PALETTE,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomPaletteGPU(ImageOnlyTransform):
    """
    SmaugLab GPU augmentation implementing PALETTE synthesis.

    Pipeline (mirrors src/synthesis/PALETTE_synthesis.py, self-contained):
      1. Min-max normalise input to [0, 1] per sample.
      2. PALETTE whole-image synthesis: 1-D K-means intensity parcellation →
         Voronoi spatial sub-parcellation → per-region signed-alpha affine remap
         y = μ + α·(x − mean_region), optional Gaussian blur.
      3. Per-anatomical-label affine remap (PALETTE step): for each foreground
         label, with probability `label_remap_prob`, independently remap that
         label's voxels with a fresh (μ, α).
      4. Optional second Gaussian blur, then foreground z-score.

    Segmentation is read from params['seg'] (injected by SmaugLab's pipeline).
    Supported formats: one-hot [B, C_seg, D, H, W] or index [B, 1, D, H, W].

    Args:
        c_choices: candidate K-means cluster counts.
        s_choices: candidate Voronoi sub-region counts per cluster.
        blur_sigmas: Gaussian blur sigma options (first pass and second pass each
            sample independently; weighted toward 0 = no blur).
        dark_threshold: voxels below this are treated as background.
        n_kmeans_subsample: max foreground voxels used to fit K-means.
        skip_parcellation_prob: probability of single global remap (no K-means).
        skip_sub_parc_prob: per-cluster probability of skipping Voronoi sub-split.
        alpha_magnitude_range: [min, max] for |alpha| in every affine remap.
        label_remap_prob: per-label per-sample probability of applying label remap.
        min_label_voxels: minimum voxels in a label to attempt the remap.
        label_classes: if set, restrict label remap to these class indices
            (None = all foreground classes present in the batch).
        p: probability of applying the transform.
    """

    def __init__(
        self,
        c_choices: Sequence[int] = (2, 3, 4, 5, 6),
        s_choices: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
        blur_sigmas: Sequence[float] = (0.0, 0.0, 0.0, 0.3, 0.5, 0.8),
        dark_threshold: float = 0.01,
        n_kmeans_subsample: int = 10_000,
        skip_parcellation_prob: float = 0.10,
        skip_sub_parc_prob: float = 0.40,
        alpha_magnitude_range: Sequence[float] = (0.5, 2.0),
        label_remap_prob: float = 0.5,
        min_label_voxels: int = 4,
        label_classes: list[int] | None = None,
        p: float = 1.0,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        # Note the default is False, not the True its siblings use. This class
        # previously forwarded **kwargs straight to super(), so keepdim fell through
        # to kornia's own default -- and nothing ever passed it. Spelling that out
        # rather than "fixing" it keeps the transform behaving exactly as before.
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.c_choices = c_choices
        self.s_choices = s_choices
        self.blur_sigmas = blur_sigmas
        self.dark_threshold = dark_threshold
        self.n_kmeans_subsample = n_kmeans_subsample
        self.skip_parcellation_prob = skip_parcellation_prob
        self.skip_sub_parc_prob = skip_sub_parc_prob
        self.alpha_magnitude_range = alpha_magnitude_range
        self.label_remap_prob = label_remap_prob
        self.min_label_voxels = min_label_voxels
        self.label_classes = label_classes

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Any],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_raw: torch.Tensor | None = segmentation_from(params)

        labels: torch.Tensor | None = None
        if seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] > 1:
            labels = collapse_onehot_to_index(seg_raw)
        elif seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] == 1:
            labels = seg_raw.long()

        B, _C, D, H, W = input.shape
        N = D * H * W
        device = input.device
        eps = 1e-7
        alpha_lo, alpha_hi = self.alpha_magnitude_range

        # Min-max normalise channel-0 to [0, 1] per sample
        flat_all = input[:, 0].float().reshape(B, N)
        v_min = flat_all.min(dim=1).values.view(B, 1)
        v_max = flat_all.max(dim=1).values.view(B, 1)
        images_01 = ((flat_all - v_min) / (v_max - v_min + eps)).clamp(0, 1)

        flat_m_all = (images_01 > self.dark_threshold).float()  # foreground mask

        # Voxel coordinates (shared — same spatial dims for every sample, and the
        # same from one call to the next, so they are built once per patch shape).
        coords = voxel_coordinates((D, H, W), device)

        # ── Step 1: PALETTE K-means + Voronoi per-region affine remap ──────────
        synth_list = []
        for i in range(B):
            flat = images_01[i]
            flat_m = flat_m_all[i]
            n_fg = flat_m.sum()

            if n_fg < 4 or torch.rand(1, device=device).item() < self.skip_parcellation_prob:
                # Single global remap
                b_mean_i = (flat * flat_m).sum() / n_fg.clamp(min=1)
                mu = torch.rand(1, device=device).item()
                sign = (torch.rand(1, device=device) > 0.5).float() * 2 - 1
                mag = torch.rand(1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
                alpha = (sign * mag).item()
                synth_i = (mu + alpha * (flat - b_mean_i)).clamp(0, 1) * flat_m
            else:
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

                rid, R = _voronoi_region_ids(
                    coords,
                    lbl_l,
                    flat_m,
                    C_k,
                    device,
                    self.s_choices,
                    self.skip_sub_parc_prob,
                )

                s_c = segment_sum(rid, flat * flat_m, R)
                n_c = segment_sum(rid, flat_m, R)
                mean_c = s_c / n_c.clamp(min=eps)

                mu_c = torch.rand(R, device=device)
                mag_c = torch.rand(R, device=device) * (alpha_hi - alpha_lo) + alpha_lo
                sign_c = (torch.rand(R, device=device) > 0.5).float() * 2 - 1
                alp_c = mag_c * sign_c

                synth_i = (mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m

            synth_list.append(synth_i)

        synth = torch.stack(synth_list)  # (B, N)
        synth_01 = synth.reshape(B, 1, D, H, W)

        sigma = shared_choice(self.blur_sigmas)
        if sigma > 0.0:
            synth_01 = _gaussian_blur_3d(synth_01, sigma)
            synth = synth_01.reshape(B, N)

        # ── Step 2: per-anatomical-label affine remap (PALETTE) ───────────────
        if labels is not None:
            if labels.shape[2:] != (D, H, W):
                # "nearest-exact", not "nearest": the latter maps src = floor(dst * scale)
                # with no half-pixel offset, which walks the label map about half a voxel
                # toward higher indices relative to the intensities synthesised from it.
                labels = F.interpolate(labels.float(), size=(D, H, W), mode="nearest-exact").long()
            lbl = labels[:, 0].reshape(B, N).clamp(min=0)

            unique_classes = foreground_classes(lbl)
            if self.label_classes is not None:
                keep = torch.tensor(self.label_classes, device=device)
                unique_classes = unique_classes[torch.isin(unique_classes, keep)]

            for c in unique_classes:
                c_val = int(c.item())
                c_mask = (lbl == c_val).float()  # (B, N)
                c_cnt = c_mask.sum(dim=1, keepdim=True)  # (B, 1)

                apply = ((torch.rand(B, 1, device=device) < self.label_remap_prob) & (c_cnt >= self.min_label_voxels)).float()

                if apply.sum() == 0:
                    continue

                c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)

                mu_c = torch.rand(B, 1, device=device)
                mag_c = torch.rand(B, 1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
                sign_c = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1
                alp_c = mag_c * sign_c

                new_vals = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)
                write_mask = c_mask * apply
                synth = synth * (1.0 - write_mask) + new_vals * write_mask

        # ── Step 3: optional second blur, then foreground z-score ─────────────
        synth_01 = synth.reshape(B, 1, D, H, W)
        sigma2 = shared_choice(self.blur_sigmas)
        if sigma2 > 0.0:
            synth_01 = _gaussian_blur_3d(synth_01, sigma2)
            synth = synth_01.reshape(B, N)

        b_sum = (synth * flat_m_all).sum(dim=1, keepdim=True)
        b_cnt = flat_m_all.sum(dim=1, keepdim=True).clamp(min=1)
        b_mean = b_sum / b_cnt
        b_sq = ((synth - b_mean) * flat_m_all).pow(2).sum(dim=1, keepdim=True)
        b_std = (b_sq / b_cnt + eps).sqrt()
        synth_z = ((synth - b_mean) / b_std * flat_m_all).reshape(B, 1, D, H, W)

        out = input.clone()
        out[:, 0:1] = synth_z.to(input.dtype)
        return out


def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample min-max normalise to [0, 1]. Returns (normed, min, max)."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    vmin = x_flat.min(dim=1).values.view(B, 1, 1, 1, 1)
    vmax = x_flat.max(dim=1).values.view(B, 1, 1, 1, 1)
    return (x - vmin) / (vmax - vmin + eps), vmin, vmax


def _minmax_denorm(x_norm: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x_norm * (vmax - vmin + eps) + vmin


def _zscore_renorm(x: torch.Tensor, bg_threshold: float = 1e-6) -> torch.Tensor:
    """Per-sample foreground-masked z-score. Mirrors nnUNet's use_mask_for_norm=True.

    Background voxels (abs ≈ 0, zeroed by nnUNet masking) stay at 0.
    Eliminates the train/inference distribution mismatch that would occur
    because nnUNet always z-scores at inference time.
    """
    fg = x.abs() > bg_threshold
    fg_f = fg.float()
    n = fg_f.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)
    mean = (x * fg_f).sum(dim=(2, 3, 4), keepdim=True) / n
    var = ((x - mean).pow(2) * fg_f).sum(dim=(2, 3, 4), keepdim=True) / n
    std = var.sqrt().clamp(min=1e-8)
    return torch.where(fg, (x - mean) / std, torch.zeros_like(x))


def seg_region_masks(seg: torch.Tensor, max_regions: int | None = None) -> torch.Tensor:
    """Per-region binary masks, from either segmentation layout.

    A transform is handed the mask in whichever layout its caller happens to use, and
    the two are not interchangeable. A one-hot tensor carries one region per
    *channel*; nnU-Net's trainer passes a single-channel integer label map, where the
    regions are distinct *values*. Anything that reads `seg.shape[1]` as its region
    count therefore sees exactly one region in the second case and silently treats the
    whole foreground as a single blob -- which is what `RandomRedistributeSegGPU` and
    `RandomDomainTransferGPU` both did, in training, for every run.

    * ``[B, C, *spatial]`` with ``C > 1`` -- already one region per channel, returned
      as bool unchanged, so one-hot callers keep their exact previous behaviour.
    * ``[B, 1, *spatial]`` -- one channel per distinct value present, background
      included, values ascending. Taken over the whole batch rather than per sample so
      every sample of a batch gets the same channel ordering.

    `max_regions` truncates, matching how a consumer with a fixed class count used to
    slice the one-hot channel axis.

    Values are rounded first: a mask that has been through a geometric transform comes
    back as float, and `torch.unique` on unrounded floats would invent regions.
    """
    if seg.dim() < 3:
        raise ValueError(f"expected [B, C, *spatial], got {tuple(seg.shape)}")
    if seg.shape[1] > 1:
        masks = seg.bool()
        return masks if max_regions is None else masks[:, :max_regions]

    labels = seg.round()
    values = torch.unique(labels)
    if max_regions is not None:
        values = values[:max_regions]
    return labels == values.view((1, values.numel()) + (1,) * (seg.dim() - 2))


def collapse_onehot_to_index(seg_raw: torch.Tensor) -> torch.Tensor:
    """
    Convert a one-hot segmentation mask to a single-channel integer index mask.

    Args:
        seg_raw: One-hot tensor [B, C_seg, D, H, W], bool or float.
                 Channel 0 is assumed to be *absent* (background is implicit).
                 Each foreground channel c encodes class index (c + 1).

    Returns:
        labels: Integer index tensor [B, 1, D, H, W].
                Background voxels (all-zero across channels) map to 0.
                Foreground voxels map to argmax(seg_raw, dim=1) + 1.
    """
    foreground_mask = seg_raw.any(dim=1, keepdim=True)  # [B,1,D,H,W] bool
    labels = torch.argmax(seg_raw, dim=1, keepdim=True).long() + 1  # 0-based → 1-based
    labels = torch.where(foreground_mask, labels, torch.zeros_like(labels))
    return labels


class DifferentiableHistogram3D(nn.Module):
    """Differentiable soft histogram for 3D volumes returning RAW VOXEL COUNTS."""

    def __init__(self, num_bins: int = 64, value_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.min_value = float(value_range[0])
        self.max_value = float(value_range[1])
        self.eps = eps

        # No bin_centers buffer: `forward` derives every index it needs from
        # min_value and bin_width arithmetically and never read it, so the buffer
        # was dead state that still showed up in state_dict() and moved with
        # .to(device) on every call.
        self.bin_width = (self.max_value - self.min_value) / max(num_bins - 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected a 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}")

        b, c, *_ = x.shape
        flat_x = x.reshape(b, c, -1)

        scaled = (flat_x - self.min_value) / (self.bin_width + self.eps)
        left_idx = torch.floor(scaled).to(torch.long)
        right_idx = left_idx + 1

        wl = (right_idx.to(flat_x.dtype) - scaled).clamp(0.0, 1.0)
        wr = (scaled - left_idx.to(flat_x.dtype)).clamp(0.0, 1.0)

        left_idx = left_idx.clamp(0, self.num_bins - 1)
        right_idx = right_idx.clamp(0, self.num_bins - 1)

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("Mask shape must match the input tensor shape.")
            flat_mask = mask.reshape(b, c, -1).to(dtype=flat_x.dtype)
            wl = wl * flat_mask
            wr = wr * flat_mask

        hist = torch.zeros((b, c, self.num_bins), device=x.device, dtype=x.dtype)
        hist.scatter_add_(2, left_idx, wl)
        hist.scatter_add_(2, right_idx, wr)

        return hist
