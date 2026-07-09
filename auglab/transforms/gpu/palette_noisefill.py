"""
PALETTE-NoiseFill — Level-3 causal ablation transform.

Identical to RandomV26_6_2ContrastGPU (PALETTE) in EVERY way — same per-sample min-max
normalisation, same 1-D K-means intensity parcellation, same Voronoi spatial
sub-parcellation, same region target means μ_c, same per-anatomical-label affine remap
(step 2), same optional blur, same foreground z-score — EXCEPT the per-region FILL.

PALETTE fills each region by remapping the REAL source intensities:
    y = μ_c + α_c · (x − mean_c)        (carries the source's within-region texture)
PALETTE-NoiseFill replaces that with spatially-independent Gaussian NOISE (SynthSeg-style):
    y = μ_c + σ_c · N(0, 1),   σ_c ~ U(noise_std_range)   per region

This holds PALETTE's partition fixed and swaps ONLY the fill (real content → parametric
noise), isolating texture preservation as the single causal variable for the Level-3
downstream ablation. PALETTE's own class is untouched; this subclass overrides
`apply_transform`, with the partition/μ-sampling lines copied verbatim from the parent so the
partition distribution and region means are provably identical — only the fill differs.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F
from kornia.core import Tensor

from auglab.transforms.gpu.fromSeg import (
    RandomV26_6_2ContrastGPU,
    _kmeans_1d,
    _voronoi_region_ids,
    _gaussian_blur_3d,
    collapse_onehot_to_index,
)


class RandomV26_6_2NoiseFillContrastGPU(RandomV26_6_2ContrastGPU):
    """PALETTE partition + SynthSeg-style Gaussian fill (Level-3 causal ablation).

    v1 (label_fill_noise=False, default): only the K-means/Voronoi region fill
    (step 1) is swapped to noise. The per-anatomical-label affine remap (step 2)
    is inherited from PALETTE UNCHANGED — still `mu + alpha*(x-mean)` on the real
    image, applied directly to the lesion label ~50% of the time regardless of
    arm. This leaves a real-texture leak at the lesion in BOTH arms, diluting
    the ablation (confirmed on open-ms 2026-07-07: OURS vs v1-noisefill showed
    only a modest gap, most visible at the hardest cross-contrast test).

    v2 (label_fill_noise=True): step 2 ALSO fills with noise
    (`mu_c + sigma_c*N(0,1)`, same noise_std_range), so no path in the transform
    preserves real texture. This is the version that actually tests "remove all
    texture preservation," which v1 didn't fully do.
    """

    def __init__(
        self,
        *args: Any,
        noise_std_range: List[float] = [0.05, 0.25],
        label_fill_noise: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.noise_std_range = noise_std_range
        self.label_fill_noise = label_fill_noise

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Any],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # ── identical to parent up to the fill ────────────────────────────────────
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
        std_lo, std_hi = self.noise_std_range                       # ← ablation param

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

        # ── Step 1: PALETTE partition, NOISE fill ─────────────────────────────────
        synth_list = []
        for i in range(B):
            flat = images_01[i]
            flat_m = flat_m_all[i]
            n_fg = flat_m.sum()

            if n_fg < 4 or torch.rand(1, device=device).item() < self.skip_parcellation_prob:
                # Single global region — mean drawn identically to PALETTE, fill = noise
                mu = torch.rand(1, device=device).item()
                sigma = torch.rand(1, device=device).item() * (std_hi - std_lo) + std_lo
                noise = torch.randn(N, device=device)
                synth_i = (mu + sigma * noise).clamp(0, 1) * flat_m
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

                # region means computed identically to PALETTE (partition parity; mean_c unused by fill)
                s_c = torch.zeros(R, device=device).scatter_add_(0, rid, flat * flat_m)
                n_c = torch.zeros(R, device=device).scatter_add_(0, rid, flat_m)
                mean_c = s_c / n_c.clamp(min=eps)                    # noqa: F841 (parity with PALETTE)

                mu_c = torch.rand(R, device=device)                 # region target means — identical to PALETTE

                # ── FILL SWAP: real remap  α_c·(x−mean_c)  →  Gaussian noise  σ_c·N(0,1) ──
                sigma_c = torch.rand(R, device=device) * (std_hi - std_lo) + std_lo
                noise = torch.randn(N, device=device)
                synth_i = (mu_c[rid] + sigma_c[rid] * noise).clamp(0, 1) * flat_m

            synth_list.append(synth_i)

        synth = torch.stack(synth_list)
        synth_01 = synth.reshape(B, 1, D, H, W)

        sigma_blur = random.choice(self.blur_sigmas)
        if sigma_blur > 0.0:
            synth_01 = _gaussian_blur_3d(synth_01, sigma_blur)
            synth = synth_01.reshape(B, N)

        # ── Step 2: per-anatomical-label fill ──────────────────────────────────────
        # v1 (label_fill_noise=False): real-texture affine remap, IDENTICAL to
        #   PALETTE — leaves a texture leak at the label (see class docstring).
        # v2 (label_fill_noise=True): SAME noise fill as step 1, so this step no
        #   longer references the real source intensities at all.
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
                c_mask = (lbl == c_val).float()
                c_cnt = c_mask.sum(dim=1, keepdim=True)

                apply = (
                    (torch.rand(B, 1, device=device) < self.label_remap_prob)
                    & (c_cnt >= self.min_label_voxels)
                ).float()
                if apply.sum() == 0:
                    continue

                mu_c = torch.rand(B, 1, device=device)
                if self.label_fill_noise:
                    sigma_c = torch.rand(B, 1, device=device) * (std_hi - std_lo) + std_lo
                    noise_c = torch.randn(B, N, device=device)
                    new_vals = (mu_c + sigma_c * noise_c).clamp(0, 1)
                else:
                    c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)
                    mag_c = torch.rand(B, 1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
                    sign_c = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1
                    alp_c = mag_c * sign_c
                    new_vals = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)
                write_mask = c_mask * apply
                synth = synth * (1.0 - write_mask) + new_vals * write_mask

        # ── Step 3: optional second blur, then foreground z-score (IDENTICAL) ─────
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
