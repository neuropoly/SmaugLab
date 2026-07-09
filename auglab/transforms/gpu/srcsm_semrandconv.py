"""
SemRandConvGPU — AugLab Kornia ``ImageOnlyTransform`` wrapper around the ported
SRCSM semantic-random-convolution operator (:class:`SemRandConv3D`).

Purpose
-------
Run the SRCSM augmentation (Thaler et al., IEEE Access 2025;
github.com/imigraz/SRCSM_Domain_Generalization) through AugLab's GPU/Kornia
augmentation path (``nnUNetTrainerDAExtGPU``) so it can be trained on the SAME
nnU-Net backbone as PALETTE, for a fair augmentation-swap comparison.

This wrapper is *self-contained*: it reproduces SRCSM's exact per-training-step
intensity operator internally, regardless of what normalization AugLab feeds it,
so the augmentation is faithful to their published recipe. It is modelled on
``auglab.transforms.gpu.fromSeg.RandomV26_6_2ContrastGPU`` (PALETTE's own Kornia
op): reads seg from ``params['seg']``, operates on channel 0, ends with a
foreground z-score write-back into the nnU-Net input domain.

SRCSM per-step MR operator reproduced here (confirmed from their
dataset.py / ssc.py / normalize.py defaults; train scripts override none of
these), applied in this exact order:

  1. normalize_robust: map the 5th/95th intensity percentiles to [-1, 1]
     (affine, unclamped).  consideration_factors=(0.1, 0.1).
  2. ShiftScaleClamp(random_shift=0.2, random_scale=0.4, clamp_min=-1.0,
     clamp_max=None):
         x += U(-0.2, 0.2)
         x *= 1 + U(-0.4, 0.4)          # multiplier in [0.6, 1.4]
         x  = clamp(x, min=-1.0)        # no upper clamp
  3. SemRandConv3D (RCNet) per-label + Gaussian mask smoothing, applied EVERY
     step (their train_step calls it unconditionally => p = 1.0).

Faithfulness notes
------------------
* RCNet is positive-homogeneous of degree 1 (conv + leaky_relu) and Frobenius-
  renormalised to the *input* norm, so ``RCNet(a*x) == a*RCNet(x)`` exactly
  (verified empirically). Therefore the choice of intensity domain does NOT
  change RCNet's texture, only its amplitude. The self-contained robust
  re-normalisation in step 1 makes the *additive* shift in step 2 land at the
  correct magnitude relative to SRCSM's [-1, 1] domain — the additive shift is
  the only intensity term that changes texture; the multiplicative term (step 2)
  and the domain scale are washed out by RCNet's renorm and the trailing
  z-score.
* Trailing foreground z-score (step 5, below) returns the image to the nnU-Net
  backbone's expected input distribution WITHOUT altering RCNet's texture
  (z-score is affine; positive-homogeneity => texture preserved). This is the
  same output contract PALETTE's V26 op uses, giving true backbone parity.

Harness gaps (documented, not silently hidden)
----------------------------------------------
* SRCSM's spatial augmentation (rotation +-0.35 rad, scale, translation +-20 mm,
  elastic Output([8,8,8], mag=15)) is configured OUTSIDE this op, in the
  AffineTransform / nnUNetSpatialTransform config blocks (see the companion
  config JSON). MDAT translation is in physical (mm) space and MDAT elastic is a
  b-spline field with no exact Kornia equivalent — these map only approximately.
  Rotation maps exactly (+-0.35 rad = +-20.05 deg per axis).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from kornia.core import Tensor

from auglab.transforms.gpu.base import ImageOnlyTransform

# The ported SRCSM operator. Installed as an editable package `srcsm_auglab_port`
# (see SRCSM_PORT_README.md); a sys.path fallback keeps it importable if the
# package is only present as a sibling sub-workspace directory.
try:
    from srcsm_auglab_port.sem_rand_conv_3d import SemRandConv3D
except Exception:  # pragma: no cover - fallback for non-installed layouts
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.environ.get("SRCSM_PORT_DIR", ""),
        os.path.join(_here, "..", "..", "..", "..", "srcsm_auglab_port"),
    ]
    for _c in _candidates:
        if _c and os.path.isdir(_c) and _c not in sys.path:
            sys.path.insert(0, os.path.dirname(_c) if os.path.basename(_c) == "srcsm_auglab_port" else _c)
    try:
        from srcsm_auglab_port.sem_rand_conv_3d import SemRandConv3D
    except Exception:
        from sem_rand_conv_3d import SemRandConv3D  # last resort: flat module


def _collapse_onehot_to_index(seg_raw: torch.Tensor) -> torch.Tensor:
    """One-hot [B, C>1, D, H, W] -> index [B, 1, D, H, W].

    Mirrors auglab.transforms.gpu.fromSeg.collapse_onehot_to_index: channel 0 is
    an *implicit-background* foreground channel, so foreground voxels map to
    argmax(dim=1) + 1 and all-zero voxels map to background 0.
    """
    foreground_mask = seg_raw.any(dim=1, keepdim=True)
    labels = torch.argmax(seg_raw, dim=1, keepdim=True).long() + 1
    return torch.where(foreground_mask, labels, torch.zeros_like(labels))


class SemRandConvGPU(ImageOnlyTransform):
    """SRCSM semantic-random-convolution augmentation as an AugLab GPU transform.

    Reproduces SRCSM's exact per-step MR intensity operator internally
    (robust-normalise -> ShiftScaleClamp -> per-label RCNet), then returns the
    result to the nnU-Net backbone's z-score input domain. The operator is
    applied every step in SRCSM, i.e. p should be 1.0.

    Args:
        num_labels: number of GT label values INCLUDING background (range(num_labels)).
            If None, derived at runtime as ``int(label.max()) + 1`` per batch —
            set it explicitly (per task) to match SRCSM's static num_labels and
            guarantee every class gets its own RCNet even when absent from a patch.
        per_label: True -> semantic-aware (fresh RCNet per label region, SRCSM
            default); False -> single global RCNet.
        smoothing: Gaussian-smooth the one-hot masks (SRCSM per-label-smoothing).
        random_shift: SRCSM ShiftScaleClamp random_shift (x += U(-s, s)).
        random_scale: SRCSM ShiftScaleClamp random_scale (x *= 1 + U(-s, s)).
        clamp_min: SRCSM ShiftScaleClamp clamp_min (lower clamp after SSC).
        clamp_max: SRCSM ShiftScaleClamp clamp_max (None = no upper clamp).
        robust_quantiles: (lo, hi) quantiles for normalize_robust; SRCSM
            consideration_factors=(0.1, 0.1) -> (0.05, 0.95).
        smooth_kernel_size, smooth_sigma: SRCSM per-label mask smoothing kernel.
        num_filters_base, kernel_candidates, leaky_slope: RCNet hyperparameters.
        zscore_output: apply trailing foreground z-score write-back (backbone
            parity). Leave True for AugLab / nnU-Net.
        p: probability of applying the transform (SRCSM = 1.0).
    """

    def __init__(
        self,
        num_labels: Optional[int] = None,
        per_label: bool = True,
        smoothing: bool = True,
        random_shift: float = 0.2,
        random_scale: float = 0.4,
        clamp_min: Optional[float] = -1.0,
        clamp_max: Optional[float] = None,
        robust_quantiles: List[float] = [0.05, 0.95],
        smooth_kernel_size: int = 5,
        smooth_sigma: float = 1.0,
        num_filters_base: int = 2,
        kernel_candidates: List[int] = [1, 3],
        leaky_slope: float = 0.1,
        zscore_output: bool = True,
        p: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, **kwargs)
        self.num_labels = num_labels
        self.per_label = per_label
        self.smoothing = smoothing
        self.random_shift = float(random_shift)
        self.random_scale = float(random_scale)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.robust_quantiles = robust_quantiles
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
        self.num_filters_base = num_filters_base
        self.kernel_candidates = tuple(kernel_candidates)
        self.leaky_slope = leaky_slope
        self.zscore_output = zscore_output

    # -- SRCSM intensity primitives -------------------------------------- #
    def _normalize_robust(self, x: Tensor) -> Tensor:
        """Per-sample affine map of (q_lo, q_hi) percentiles -> [-1, 1] (unclamped).

        Reproduces MDAT normalize_robust: old_range=(p_lo, p_hi), new_range=(-1, 1),
        result = 2*(x - p_lo)/(p_hi - p_lo) - 1.
        """
        B = x.shape[0]
        flat = x.reshape(B, -1).float()
        q = torch.tensor(self.robust_quantiles, device=x.device, dtype=torch.float32)
        qs = torch.quantile(flat, q, dim=1)  # (2, B)
        p_lo = qs[0].view(B, 1, 1, 1, 1)
        p_hi = qs[1].view(B, 1, 1, 1, 1)
        denom = (p_hi - p_lo)
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        return (2.0 * (x - p_lo) / denom - 1.0).to(x.dtype)

    def _shift_scale_clamp(self, x: Tensor) -> Tensor:
        """SRCSM ShiftScaleClamp, per-sample random draws.

        x += U(-random_shift, random_shift)
        x *= 1 + U(-random_scale, random_scale)
        x  = clamp(x, clamp_min, clamp_max)
        """
        B = x.shape[0]
        shp = (B, 1, 1, 1, 1)
        shift = (torch.rand(shp, device=x.device, dtype=x.dtype) * 2 - 1) * self.random_shift
        scale = 1.0 + (torch.rand(shp, device=x.device, dtype=x.dtype) * 2 - 1) * self.random_scale
        x = (x + shift) * scale
        if self.clamp_min is not None or self.clamp_max is not None:
            x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x

    def _foreground_zscore(self, x: Tensor, labels: Optional[Tensor]) -> Tensor:
        """Per-sample foreground z-score write-back (nnU-Net input contract).

        Foreground = labels > 0 when a label map is available, else the whole
        volume. Statistics computed per sample; texture preserved (affine).
        """
        B = x.shape[0]
        out = x.clone()
        for i in range(B):
            xi = x[i, 0]
            if labels is not None:
                fg = labels[i, 0] > 0
                vals = xi[fg] if fg.any() else xi.reshape(-1)
            else:
                vals = xi.reshape(-1)
            mean = vals.float().mean()
            std = vals.float().std()
            std = std if std > 1e-8 else torch.ones_like(std)
            out[i, 0] = ((xi - mean) / std).to(x.dtype)
        return out

    # -- Kornia entry point ---------------------------------------------- #
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
            labels = _collapse_onehot_to_index(seg_raw)
        elif seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] == 1:
            labels = seg_raw.long()

        if self.per_label and labels is None:
            # No usable segmentation -> fall back to global RCNet (SRCSM
            # apply_randnet_globally) rather than silently doing nothing.
            per_label = False
        else:
            per_label = self.per_label

        num_labels = self.num_labels
        if per_label and num_labels is None:
            num_labels = int(labels.max().item()) + 1
            num_labels = max(num_labels, 2)

        # Operate on channel 0 (single-source MR), matching PALETTE's V26 op.
        x = input[:, 0:1].clone()

        # Step 1: SRCSM robust-normalise to [-1, 1].
        x = self._normalize_robust(x)
        # Step 2: SRCSM ShiftScaleClamp.
        x = self._shift_scale_clamp(x)

        # Step 3: per-label (or global) RCNet, fresh weights this call.
        op = SemRandConv3D(
            num_labels=num_labels if per_label else 2,
            per_label=per_label,
            smoothing=self.smoothing,
            num_filters_base=self.num_filters_base,
            kernel_candidates=self.kernel_candidates,
            leaky_slope=self.leaky_slope,
            smooth_kernel_size=self.smooth_kernel_size,
            smooth_sigma=self.smooth_sigma,
        ).to(x.device)
        x = op(x, labels if per_label else None)

        # Step 5: foreground z-score write-back (backbone parity).
        if self.zscore_output:
            x = self._foreground_zscore(x, labels)

        out = input.clone()
        out[:, 0:1] = x.to(input.dtype)
        return out
