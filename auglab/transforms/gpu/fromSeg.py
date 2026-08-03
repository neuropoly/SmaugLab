import random

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, Optional, Tuple, Union, List, Protocol
from kornia.core import Tensor
import torch.distributed as dist

from auglab.transforms.gpu.base import ImageOnlyTransform


# ── PALETTE AUG helpers ──────────────────────────────────────────────────

def _kmeans_1d(values: torch.Tensor, C: int, n_iter: int = 10) -> torch.Tensor:
    """1-D K-means on foreground values. Returns (C,) centroids."""
    centroids = torch.linspace(values.min().item(), values.max().item(), C, device=values.device)
    for _ in range(n_iter):
        d = torch.abs(values.unsqueeze(1) - centroids.unsqueeze(0))
        lbl = torch.argmin(d, dim=1)
        s = torch.zeros(C, device=values.device).scatter_add_(0, lbl, values)
        n = torch.zeros(C, device=values.device).scatter_add_(0, lbl, torch.ones_like(values))
        new_c = torch.where(n > 0, s / n, centroids)
        if torch.allclose(centroids, new_c):
            break
        centroids = new_c
    return centroids


def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur. x: (B, 1, D, H, W)."""
    k_r = max(1, int(3.0 * sigma + 0.5))
    k1d = torch.arange(-k_r, k_r + 1, dtype=x.dtype, device=x.device)
    k1d = torch.exp(-0.5 * (k1d / sigma) ** 2)
    k1d = k1d / k1d.sum()
    pad = len(k1d) // 2
    y = F.conv3d(x, k1d.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
    y = F.conv3d(y, k1d.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
    y = F.conv3d(y, k1d.view(1, 1, 1, 1, -1), padding=(0, 0, pad))
    return y.clamp(0, 1)


def _voronoi_region_ids(
    coords: torch.Tensor,
    lbl_l: torch.Tensor,
    fg: torch.Tensor,
    C: int,
    device: torch.device,
    s_choices: List[int],
    skip_sub_parc_prob: float,
) -> tuple[torch.Tensor, int]:
    """Spatially subdivide each K-means cluster into Voronoi sub-regions.

    Returns (rid, R): per-voxel sub-region id and total region count.
    """
    N = lbl_l.shape[0]
    rid = torch.zeros(N, dtype=torch.long, device=device)
    offset = 0
    for c in range(C):
        c_mask = lbl_l == c
        c_fg_mask = c_mask & (fg > 0)
        n_fg = int(c_fg_mask.sum().item())
        if n_fg == 0:
            continue
        if n_fg < 2 or torch.rand(1, device=device).item() < skip_sub_parc_prob:
            S = 1
        else:
            s_idx = int(torch.rand(1, device=device).item() * len(s_choices))
            S = min(s_choices[s_idx], n_fg)
        if S <= 1:
            rid = torch.where(c_mask, torch.full_like(rid, offset), rid)
            offset += 1
            continue
        seed_idx = torch.multinomial(c_fg_mask.float(), S, replacement=False)
        centroids_v = coords.index_select(0, seed_idx)
        d = torch.cdist(coords, centroids_v)
        sub = torch.argmin(d, dim=1)
        rid = torch.where(c_mask, offset + sub, rid)
        offset += S
    return rid, offset


# ─────────────────────────────────────────────────────────────────────────────

def _normal_pdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    inv = 1.0 / (std + 1e-6)
    return (inv / (torch.sqrt(torch.tensor(2.0 * 3.141592653589793, device=x.device, dtype=x.dtype)))) * torch.exp(
        -0.5 * ((x - mean) * inv) ** 2
    )

## Redistribute segmentation values transform (GPU)
class RandomRedistributeSegGPU(ImageOnlyTransform):
    """Redistribute image values using segmentation regions (GPU version).

    Mirrors the CPU `RedistributeTransform` behavior using GPU-friendly ops.
    Works with inputs shaped [N, C, H, W] or [N, C, D, H, W].
    """

    def __init__(
        self,
        in_seg: float = 0.2,
        apply_to_channel: list[int] = [0],
        retain_stats: bool = False,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = True,
        std_noise_range: list[float] = [0.1, 0.3],
        dilation_iterations_range: list[int] = [1, 3],
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.in_seg = in_seg
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.std_noise_range = std_noise_range
        self.dilation_iterations_range = dilation_iterations_range

    @torch.no_grad()
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Expect segmentation provided in params: shape [N, 1, ...] or [N, C_seg, ...]
        if 'seg' not in params:
            return input
        seg = params['seg']
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
            img_min = img_batch.view(N, -1).min(dim=1)[0].view(N, *([1] * (img_batch.dim()-1)))
            img_max = img_batch.view(N, -1).max(dim=1)[0].view(N, *([1] * (img_batch.dim()-1)))
            denom = (img_max - img_min).clamp_min(1e-6)
            x_batch = (img_batch - img_min) / denom

            # Iterate per sample (seg can differ in shape or labels per sample)
            for b in range(N):
                x = x_batch[b]
                seg_b = seg[b]  # shape (R, ...)

                # Quick skip if no foreground
                if (seg_b > 0).sum() == 0:
                    input[b, c] = x
                    continue

                # Decide redistribution mode once per sample
                # Scalar random flag for redistribution mode
                in_seg_bool = torch.rand((), device=input.device) <= self.in_seg

                # Binary masks for regions
                masks = seg_b.bool()  # (R, ...)
                R = masks.shape[0]

                # Vectorized dilation for all regions (3 iterations)
                dilated = masks.float()
                dilation_iterations = torch.randint(self.dilation_iterations_range[0], self.dilation_iterations_range[1]+1, (1,), device=input.device)[0].item()
                for _ in range(dilation_iterations):
                    if spatial_dims == 3:
                        dilated = F.max_pool3d(dilated.unsqueeze(0), 3, 1, 1).squeeze(0)
                    else:
                        dilated = F.max_pool2d(dilated.unsqueeze(0), 3, 1, 1).squeeze(0)
                dilated_excl = (dilated > 0) & (~masks)

                # Flatten for stats
                x_flat = x.view(1, -1)  # (1, S)
                mask_flat = masks.view(R, -1)
                dil_flat = dilated_excl.view(R, -1)

                # Region counts
                counts = mask_flat.sum(dim=1).clamp_min(1)
                # Means
                means = (mask_flat * x_flat).sum(dim=1) / counts
                # Std (compute variance then sqrt) avoid indexing overhead
                diffs = (x_flat - means.view(R,1)) * mask_flat
                vars = (diffs * diffs).sum(dim=1) / counts.clamp_min(1)
                stds = vars.sqrt()

                # Dilated stats
                dil_counts = dil_flat.sum(dim=1).clamp_min(1)
                dil_means = (dil_flat * x_flat).sum(dim=1) / dil_counts
                dil_diffs = (x_flat - dil_means.view(R,1)) * dil_flat
                dil_vars = (dil_diffs * dil_diffs).sum(dim=1) / dil_counts
                dil_stds = dil_vars.sqrt()

                # redist_std per region
                std_noise_range = torch.rand(1, device=input.device)[0] * (self.std_noise_range[1] - self.std_noise_range[0]) + self.std_noise_range[0]
                redist_std = torch.maximum(
                    torch.rand(R, device=input.device) * std_noise_range + 0.4 * torch.abs((means - dil_means) * stds / (dil_stds + 1e-6)),
                    torch.full((R,), 0.01, device=input.device, dtype=input.dtype)
                )

                # Build additive term
                to_add = torch.zeros_like(x)
                rand_sign = (2 * torch.rand(R, device=input.device) - 1)  # random sign factor per region
                if in_seg_bool.item():
                    # Only inside region
                    for r in range(R):
                        if counts[r] == 0:  # skip empty
                            continue
                        region_vals = x[mask_flat[r].view(x.shape)]
                        pdf_vals = _normal_pdf(region_vals, means[r], redist_std[r]) * rand_sign[r]
                        to_add[mask_flat[r].view(x.shape)] += pdf_vals
                else:
                    # Global additive influence per region
                    pdf_all = []
                    for r in range(R):
                        if counts[r] == 0:
                            continue
                        pdf_all.append(_normal_pdf(x, means[r], redist_std[r]) * rand_sign[r])
                    if pdf_all:
                        to_add += torch.stack(pdf_all, dim=0).sum(dim=0)

                # Normalize to_add if non-zero
                tmin, tmax = to_add.min(), to_add.max()
                if (tmax - tmin) > 1e-8:
                    x = x + 2 * (to_add - tmin) / (tmax - tmin + 1e-6)

                # Restore stats
                if self.retain_stats:
                    mean = x.mean()
                    # Use unbiased=False to avoid NaNs on degenerate shapes
                    std = x.std(unbiased=False)
                    x = (x - mean) / torch.clamp(std, min=1e-7)
                    x = x * orig_std[b] + orig_mean[b]

                # Final safety: check if nan/inf appeared
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning nan: {self.__class__.__name__}", flush=True)
                    continue
                input[b, c] = x

        return input


class RandomPALETTEGPU(ImageOnlyTransform):
    """
    AugLab GPU augmentation implementing PALETTE synthesis.

    Pipeline (mirrors src/synthesis/PALETTE_synthesis.py, self-contained):
      1. Min-max normalise input to [0, 1] per sample.
      2. PALETTE whole-image synthesis: 1-D K-means intensity parcellation →
         Voronoi spatial sub-parcellation → per-region signed-alpha affine remap
         y = μ + α·(x − mean_region), optional Gaussian blur.
      3. Per-anatomical-label affine remap (PALETTE step): for each foreground
         label, with probability `label_remap_prob`, independently remap that
         label's voxels with a fresh (μ, α).
      4. Optional second Gaussian blur, then foreground z-score.

    Segmentation is read from params['seg'] (injected by AugLab's pipeline).
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
        c_choices: List[int] = [2, 3, 4, 5, 6],
        s_choices: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
        blur_sigmas: List[float] = [0.0, 0.0, 0.0, 0.3, 0.5, 0.8],
        dark_threshold: float = 0.01,
        n_kmeans_subsample: int = 10_000,
        skip_parcellation_prob: float = 0.10,
        skip_sub_parc_prob: float = 0.40,
        alpha_magnitude_range: List[float] = [0.5, 2.0],
        label_remap_prob: float = 0.5,
        min_label_voxels: int = 4,
        label_classes: Optional[List[int]] = None,
        p: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, **kwargs)
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
        alpha_lo, alpha_hi = self.alpha_magnitude_range

        # Min-max normalise channel-0 to [0, 1] per sample
        flat_all = input[:, 0].float().reshape(B, N)
        v_min = flat_all.min(dim=1).values.view(B, 1)
        v_max = flat_all.max(dim=1).values.view(B, 1)
        images_01 = ((flat_all - v_min) / (v_max - v_min + eps)).clamp(0, 1)

        flat_m_all = (images_01 > self.dark_threshold).float()  # foreground mask

        # Voxel coordinates (shared — same spatial dims for every sample)
        coords = torch.stack(torch.meshgrid(
            torch.arange(D, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"), dim=-1).reshape(N, 3)

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
                sub_fg = samp[samp > self.dark_threshold][:self.n_kmeans_subsample]
                if sub_fg.numel() < 4:
                    sub_fg = samp[:self.n_kmeans_subsample]

                centroids = _kmeans_1d(sub_fg, C_k)
                sorted_c, sort_idx = torch.sort(centroids)
                boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
                lbl_s = torch.bucketize(flat, boundaries)
                lbl_l = sort_idx[lbl_s].long()

                rid, R = _voronoi_region_ids(
                    coords, lbl_l, flat_m, C_k, device,
                    self.s_choices, self.skip_sub_parc_prob,
                )

                s_c = torch.zeros(R, device=device).scatter_add_(0, rid, flat * flat_m)
                n_c = torch.zeros(R, device=device).scatter_add_(0, rid, flat_m)
                mean_c = s_c / n_c.clamp(min=eps)

                mu_c = torch.rand(R, device=device)
                mag_c = torch.rand(R, device=device) * (alpha_hi - alpha_lo) + alpha_lo
                sign_c = (torch.rand(R, device=device) > 0.5).float() * 2 - 1
                alp_c = mag_c * sign_c

                synth_i = (mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m

            synth_list.append(synth_i)

        synth = torch.stack(synth_list)       # (B, N)
        synth_01 = synth.reshape(B, 1, D, H, W)

        sigma = random.choice(self.blur_sigmas)
        if sigma > 0.0:
            synth_01 = _gaussian_blur_3d(synth_01, sigma)
            synth = synth_01.reshape(B, N)

        # ── Step 2: per-anatomical-label affine remap (PALETTE) ───────────────
        if labels is not None:
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
                c_mask = (lbl == c_val).float()                         # (B, N)
                c_cnt = c_mask.sum(dim=1, keepdim=True)                 # (B, 1)

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

                new_vals = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)
                write_mask = c_mask * apply
                synth = synth * (1.0 - write_mask) + new_vals * write_mask

        # ── Step 3: optional second blur, then foreground z-score ─────────────
        synth_01 = synth.reshape(B, 1, D, H, W)
        sigma2 = random.choice(self.blur_sigmas)
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


_SHARED_RNG_COUNTER = 0


def _next_shared_seed() -> int:
    global _SHARED_RNG_COUNTER
    _SHARED_RNG_COUNTER += 1
    seed = (int(torch.initial_seed()) + _SHARED_RNG_COUNTER) % (2**63 - 1)
    if dist.is_available() and dist.is_initialized():
        seed_tensor = torch.tensor([seed], dtype=torch.long)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    return seed


@staticmethod
def _minmax_norm(x: torch.Tensor, eps: float = 1e-8
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample min-max normalise to [0, 1]. Returns (normed, min, max)."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    vmin = x_flat.min(dim=1).values.view(B, 1, 1, 1, 1)
    vmax = x_flat.max(dim=1).values.view(B, 1, 1, 1, 1)
    return (x - vmin) / (vmax - vmin + eps), vmin, vmax

@staticmethod
def _minmax_denorm(x_norm: torch.Tensor, vmin: torch.Tensor,
                    vmax: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x_norm * (vmax - vmin + eps) + vmin

@staticmethod
def _zscore_renorm(x: torch.Tensor, bg_threshold: float = 1e-6) -> torch.Tensor:
    """Per-sample foreground-masked z-score. Mirrors nnUNet's use_mask_for_norm=True.

    Background voxels (abs ≈ 0, zeroed by nnUNet masking) stay at 0.
    Eliminates the train/inference distribution mismatch that would occur
    because nnUNet always z-scores at inference time.
    """
    fg   = x.abs() > bg_threshold
    fg_f = fg.float()
    n    = fg_f.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)
    mean = (x * fg_f).sum(dim=(2, 3, 4), keepdim=True) / n
    var  = ((x - mean).pow(2) * fg_f).sum(dim=(2, 3, 4), keepdim=True) / n
    std  = var.sqrt().clamp(min=1e-8)
    return torch.where(fg, (x - mean) / std, torch.zeros_like(x))

def _shared_cpu_generator() -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_next_shared_seed())
    return generator

def _shared_rand(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.rand(shape, device=device, dtype=dtype)
    rand_cpu = torch.rand(shape, generator=_shared_cpu_generator(), device="cpu", dtype=dtype)
    return rand_cpu.to(device=device, dtype=dtype)


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
    foreground_mask = seg_raw.any(dim=1, keepdim=True)               # [B,1,D,H,W] bool
    labels = torch.argmax(seg_raw, dim=1, keepdim=True).long() + 1   # 0-based → 1-based
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

        bin_centers = torch.linspace(self.min_value, self.max_value, num_bins)
        self.register_buffer("bin_centers", bin_centers.view(1, 1, num_bins, 1), persistent=False)
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
