"""
SemRandConv3D — PyTorch reimplementation of the semantic-aware random-convolution
augmentation operator from SRCSM (Thaler et al., IEEE Access 2025;
github.com/imigraz/SRCSM_Domain_Generalization).

This is a faithful port of the *augmentation operator only* (RCNet +
apply_randnet_per_label) from their TensorFlow/Keras code, reimplemented so it can
be dropped into an AugLab / nnU-Net PyTorch training pipeline as an
augmentation-swap baseline. Their U-Net backbone, training framework, data
pipeline and source-matching step are intentionally NOT reproduced here — the
point is to isolate the augmentation mechanism on a common (nnU-Net) backbone.

Reference (their code):
  network.py :: RCNet, frobenius_norm
  main.py    :: apply_randnet_globally, apply_randnet_per_label,
                get_smooth_per_label_mask, apply_gaussian_smoothing_3d

Load-bearing details preserved from their implementation (see README notes):
  1. Fresh random conv weights (RandomNormal mean=0 std=1) sampled on EVERY call,
     never trained, never persisted.
  2. 4 conv layers, num_filters_base=2, last layer 1 filter.
  3. Per-layer kernel size drawn from {1, 3} independently each instantiation.
  4. leaky_relu(0.1) after every conv (incl. the last).
  5. alpha ~ U(0,1) sampled once per RCNet instance: out = net(x)*a + x*(1-a).
  6. Frobenius energy renorm: out = out / ||out|| * ||x||, with a 1e-5 denom floor.
     Computed PER SAMPLE (their code assumed batch_size==1; we vectorize).
  7. Per-label mode: one-hot split the GT label map over range(num_labels)
     INCLUDING background (label 0); optionally Gaussian-smooth each mask
     (kernel=5, sigma=1.0) -> SOFT masks; a FRESH RCNet per label applied to the
     whole image, composited as sum_l rcnet_l(x) * mask_l. Masks are NOT
     renormalised to sum to 1 (matches their code).
  8. Zero-output retry: if a random field collapses to ~0 energy, resample.

Deliberate differences from their paper setup (state these in the manuscript):
  - Operates on nnU-Net-normalised inputs (z-score), not their [-1, 1] range.
    The Frobenius renorm makes the operator scale-tolerant; this keeps the
    backbone byte-identical to the PALETTE runs.
  - Backbone is nnU-Net, not UnetAvgLinear3D -> paper numbers are NOT reproduced;
    this is a mechanism-isolation comparison, not a reproduction.
  - Source matching (their target-side CDF histogram matching) is out of scope of
    the augmentation and not included here.

Author note: reimplemented from the public SRCSM code (GPL-3.0). If this operator
is distributed, respect the upstream licence.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Gaussian smoothing kernel (fixed, non-trainable) — matches their
#  apply_gaussian_smoothing_3d(kernel_size=5, sigma=1.0), applied per one-hot
#  channel independently (depthwise).
# --------------------------------------------------------------------------- #
def _gaussian_kernel_3d(kernel_size: int = 5, sigma: float = 1.0,
                        dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    g3d = g3d / g3d.sum()
    return g3d  # (k, k, k)


def _smooth_onehot(masks: torch.Tensor, kernel_size: int = 5,
                   sigma: float = 1.0) -> torch.Tensor:
    """Depthwise Gaussian smoothing of one-hot masks.
    masks: (B, L, D, H, W) -> (B, L, D, H, W), each label channel smoothed
    independently (matches their per-channel conv3d loop)."""
    b, l, d, h, w = masks.shape
    k = _gaussian_kernel_3d(kernel_size, sigma, dtype=masks.dtype).to(masks.device)
    weight = k[None, None].repeat(l, 1, 1, 1, 1)  # (L,1,k,k,k) depthwise
    pad = kernel_size // 2
    return F.conv3d(masks, weight, padding=pad, groups=l)


def _frobenius_per_sample(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """||x|| computed per sample over (C,D,H,W); returns (B,1,1,1,1).
    Their frobenius_norm floors a zero norm at 1e-5."""
    b = x.shape[0]
    n = torch.sqrt((x.float() ** 2).reshape(b, -1).sum(dim=1))
    n = torch.clamp(n, min=eps)
    return n.reshape(b, 1, 1, 1, 1).to(x.dtype)


# --------------------------------------------------------------------------- #
#  RCNet — the random-convolution field. Weights are re-sampled on every
#  forward() and are NOT nn.Parameters (never trained, never persisted).
# --------------------------------------------------------------------------- #
class RCNet(nn.Module):
    """Random 3D conv stack, faithful to SRCSM network.py::RCNet.

    - 4 conv layers, num_filters_base=2, last layer num_filters_last=1
    - per-layer kernel size drawn from {1,3}
    - RandomNormal(0, 1) weights, resampled every call, requires_grad=False
    - leaky_relu(0.1) after every conv
    - alpha ~ U(0,1); out = net(x)*alpha + x*(1-alpha)
    - Frobenius energy renorm (per sample)
    """

    def __init__(self, num_filters_base: int = 2, num_filters_last: int = 1,
                 num_layers: int = 4, kernel_candidates=(1, 3),
                 leaky_slope: float = 0.1):
        super().__init__()
        self.num_filters_base = num_filters_base
        self.num_filters_last = num_filters_last
        self.num_layers = num_layers
        self.kernel_candidates = tuple(kernel_candidates)
        self.leaky_slope = leaky_slope

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W). Their RandomNormal init uses std=1.0 regardless of
        # fan-in (Keras RandomNormal is literal), so we replicate that exactly:
        # weight ~ N(0,1), bias = 0 (Keras Conv3D default use_bias=True, bias
        # initialised to zeros).
        b, c_in = x.shape[0], x.shape[1]
        alpha = random.uniform(0.0, 1.0)

        h = x
        in_ch = c_in
        for i in range(self.num_layers):
            ksize = random.choice(self.kernel_candidates)
            out_ch = self.num_filters_last if i == self.num_layers - 1 else self.num_filters_base
            weight = torch.randn(out_ch, in_ch, ksize, ksize, ksize,
                                 device=x.device, dtype=x.dtype)
            bias = torch.zeros(out_ch, device=x.device, dtype=x.dtype)
            pad = ksize // 2  # 'same' for odd kernels (1->0, 3->1)
            h = F.conv3d(h, weight, bias, padding=pad)
            # leaky_relu(0.1) after EVERY conv (their last conv has activation=None
            # but the trailing Activation(leaky_relu) applies it anyway)
            h = F.leaky_relu(h, negative_slope=self.leaky_slope)
            in_ch = out_ch

        out = h * alpha + x * (1.0 - alpha)
        out = out / _frobenius_per_sample(out) * _frobenius_per_sample(x)
        return out


# --------------------------------------------------------------------------- #
#  SemRandConv3D — the augmentation transform (global or per-label).
# --------------------------------------------------------------------------- #
class SemRandConv3D(nn.Module):
    """Semantic-aware random-convolution augmentation (SRCSM operator).

    Call as:  aug_image = op(image, label)   during training only.

    Args:
        num_labels:  number of GT label values incl. background (range(num_labels)).
        per_label:   True -> semantic-aware (fresh RCNet per label region);
                     False -> single global RCNet (their apply_randnet_globally).
        smoothing:   True -> Gaussian-smooth the one-hot masks (soft blending).
        num_filters_base, kernel_candidates, leaky_slope: RCNet hyperparameters.
        max_retries: zero-output resample guard.

    Shapes:
        image: (B, 1, D, H, W)  float
        label: (B, 1, D, H, W)  or (B, D, H, W) integer label map
    Returns:
        (B, 1, D, H, W) augmented image (same dtype/device as input).
    """

    def __init__(self, num_labels: int, per_label: bool = True,
                 smoothing: bool = True, num_filters_base: int = 2,
                 kernel_candidates=(1, 3), leaky_slope: float = 0.1,
                 smooth_kernel_size: int = 5, smooth_sigma: float = 1.0,
                 max_retries: int = 20):
        super().__init__()
        self.num_labels = int(num_labels)
        self.per_label = per_label
        self.smoothing = smoothing
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
        self.max_retries = max_retries
        self._rc_kwargs = dict(num_filters_base=num_filters_base,
                               kernel_candidates=kernel_candidates,
                               leaky_slope=leaky_slope)

    def _new_rcnet(self, x: torch.Tensor) -> torch.Tensor:
        """One RCNet application with the zero-output retry loop (per-sample:
        retry until every sample has non-trivial energy)."""
        for _ in range(self.max_retries):
            rc = RCNet(**self._rc_kwargs)
            out = rc(x)
            energy = (out.float() ** 2).reshape(out.shape[0], -1).sum(dim=1)
            if torch.all(energy > 0):
                return out
        return out  # last attempt

    @torch.no_grad()
    def forward(self, image: torch.Tensor,
                label: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image.dim() != 5 or image.shape[1] != 1:
            raise ValueError(f"expected image (B,1,D,H,W), got {tuple(image.shape)}")

        if not self.per_label:
            return self._new_rcnet(image)

        if label is None:
            raise ValueError("per_label=True requires a label map")
        if label.dim() == 4:
            label = label.unsqueeze(1)  # (B,1,D,H,W)
        if label.shape[0] != image.shape[0]:
            raise ValueError("image/label batch mismatch")

        # one-hot over range(num_labels), INCLUDING background (label 0)
        lab = label.long().clamp(0, self.num_labels - 1)  # (B,1,D,H,W)
        onehot = F.one_hot(lab.squeeze(1), num_classes=self.num_labels)  # (B,D,H,W,L)
        onehot = onehot.permute(0, 4, 1, 2, 3).to(image.dtype)  # (B,L,D,H,W)

        if self.smoothing:
            onehot = _smooth_onehot(onehot, self.smooth_kernel_size,
                                    self.smooth_sigma)
            # NOTE: masks are intentionally NOT renormalised to sum to 1
            # (matches their code).

        aug = torch.zeros_like(image)
        for l in range(self.num_labels):
            mask = onehot[:, l:l + 1]  # (B,1,D,H,W)
            aug = aug + self._new_rcnet(image) * mask
        return aug


if __name__ == "__main__":
    # Minimal smoke test / self-check.
    torch.manual_seed(0)
    random.seed(0)
    B, D, H, W, L = 2, 16, 24, 24, 4
    img = torch.randn(B, 1, D, H, W)
    lab = torch.randint(0, L, (B, 1, D, H, W))

    op = SemRandConv3D(num_labels=L, per_label=True, smoothing=True)
    a1 = op(img, lab)
    a2 = op(img, lab)
    print("output shape:", tuple(a1.shape))
    print("two calls differ (fresh weights):", not torch.allclose(a1, a2))
    e_in = (img ** 2).reshape(B, -1).sum(1)
    e_out = (a1 ** 2).reshape(B, -1).sum(1)
    print("per-sample energy in :", e_in.tolist())
    print("per-sample energy out:", e_out.tolist())
    print("requires_grad on output:", a1.requires_grad)
