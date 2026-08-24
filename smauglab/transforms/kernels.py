"""Convolution kernels and smooth random fields, shared by every backend.

Four independent 3-D Gaussian implementations, three bias fields and two copies of
the Laplace/Scharr constant tables used to live in five different modules. They were
not equivalent, and the divergence is what let two of them go wrong unnoticed: the
uncentred Gaussian in `gpu/contrast.py` and the malformed 2-D Scharr x-kernel in
`cpu/contrast.py`, both corrected earlier in this series. Each was fixed on its own
because there was no shared implementation to fix instead. This module is that
implementation; anything that convolves or blurs should import from here rather than
growing a fifth copy.

The consolidation is not bit-for-bit for the two blur call sites, and deliberately so:

* Radius is `ceil(3*sigma)`. `domain_transfer` and `fromSeg` used `round(3*sigma)`,
  which is never wider, so their kernels may now be one tap larger.
* Padding is `reflect` everywhere. `domain_transfer` used `replicate` and `fromSeg`
  relied on conv3d's implicit zero padding. Zero padding darkens the volume border,
  which is the one difference here that was wrong rather than merely different.

Everything else -- the dense Gaussian, both derivative tables, the bias field -- is
the same arithmetic as the copy it replaces.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "LAPLACE_2D",
    "LAPLACE_3D",
    "SCHARR_2D",
    "SCHARR_3D",
    "gaussian_blur3d",
    "gaussian_kernel1d",
    "gaussian_kernel3d",
    "laplace_kernel",
    "random_bias_field3d",
    "scharr_kernels",
]


# --- Gaussian kernels -------------------------------------------------------------


def gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """A normalised 1-D Gaussian, centred, with radius `ceil(3*sigma)`.

    A non-positive sigma means "do not blur", and returns the identity kernel `[1.0]`
    rather than raising -- `blurring_sigma_for_downsampling` legitimately produces
    zeros for axes that are already at the target resolution.
    """
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_kernel3d(
    kernel_size: int,
    sigma: Union[float, Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """A dense `[kernel_size]*3` Gaussian as the outer product of three 1-D kernels.

    Fixed-size rather than sigma-derived, because the caller
    (`_RandomConvBaseGPU.get_kernel`) hands the kernel to a generic convolution path
    shared with Scharr and RandConv and needs a tensor of a known shape.

    The sample points are centred on the kernel: `linspace(-(k-1)/2, (k-1)/2, k)`.
    Sampling at `arange(k)` -- what this used to do -- puts the peak at index 0 and
    turns the blur into a blur plus a translation.
    """
    if isinstance(sigma, (int, float)):
        sigma_t = torch.tensor([float(sigma)] * 3, device=device, dtype=dtype)
    elif isinstance(sigma, Tensor):
        if sigma.shape != (3,):
            raise ValueError(f"sigma must be a float or a tensor of three floats, got shape {tuple(sigma.shape)}")
        sigma_t = sigma.to(device=device, dtype=dtype)
    else:
        raise TypeError(f"sigma must be a float or a tensor of three floats, got {type(sigma).__name__}")

    half = (kernel_size - 1) / 2.0
    x = torch.linspace(-half, half, kernel_size, device=device, dtype=dtype)

    axes = []
    for axis in range(3):
        s = sigma_t[axis]
        # A zero sigma degenerates to a delta at the centre; exp(-inf) would be 0
        # everywhere and the normalisation would divide by zero.
        if float(s) <= 0:
            delta = torch.zeros(kernel_size, device=device, dtype=dtype)
            delta[kernel_size // 2] = 1.0
            axes.append(delta)
            continue
        pdf = torch.exp(-0.5 * (x / s).pow(2))
        axes.append(pdf / pdf.sum())

    kernel = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return kernel / kernel.sum()


def gaussian_blur3d(
    image: Tensor,
    sigma: Union[float, Tensor],
    *,
    blur_range: float = 1.0,
    padding_mode: str = "reflect",
) -> Tensor:
    """Separable, optionally anisotropic Gaussian blur of a `[B, C, D, H, W]` volume.

    `sigma` is either a scalar or a `(3,)` tensor of per-axis sigmas. `blur_range > 1`
    multiplies every sigma by `U(1/blur_range, blur_range)`, which is SynthSeg's
    `DynamicGaussianBlur` jitter; the default of 1.0 disables it.

    Three 1-D convolutions rather than one dense 3-D kernel: for a radius-r kernel that
    is 3*(2r+1) multiply-adds per voxel instead of (2r+1)^3.
    """
    if image.dim() != 5:
        raise ValueError(f"expected a 5D [B, C, D, H, W] tensor, got shape {tuple(image.shape)}")

    channels = image.shape[1]
    device, dtype = image.device, image.dtype

    if isinstance(sigma, Tensor):
        sigmas = sigma.detach().to(device=device, dtype=torch.float32).flatten()
        if sigmas.numel() == 1:
            sigmas = sigmas.repeat(3)
    else:
        sigmas = torch.full((3,), float(sigma), device=device, dtype=torch.float32)

    if blur_range and blur_range > 1.0:
        jitter = (1.0 / blur_range) + torch.rand(3, device=device) * (blur_range - 1.0 / blur_range)
        sigmas = sigmas * jitter

    out = image
    for axis, s in enumerate(sigmas.tolist()):
        if s <= 0:
            continue
        kernel = gaussian_kernel1d(s, device, dtype)
        ksize = kernel.numel()
        if ksize == 1:
            continue
        pad = ksize // 2

        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = ksize
        weight = kernel.view(shape).repeat(channels, 1, 1, 1, 1)

        # F.pad's tuple runs last spatial axis first: (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi).
        pad_full = [0, 0, 0, 0, 0, 0]
        pad_full[(2 - axis) * 2] = pad
        pad_full[(2 - axis) * 2 + 1] = pad

        out = F.conv3d(F.pad(out, pad_full, mode=padding_mode), weight, groups=channels)
    return out


# --- smooth random fields ---------------------------------------------------------


def random_bias_field3d(
    shape: tuple[int, int, int],
    std: float,
    scale: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    batch: int = 1,
    channels: int = 1,
) -> Tensor:
    """A smooth, positive, multiplicative bias field of shape `[batch, channels, *shape]`.

    Sample `N(0, U(0, std))` on a coarse `ceil(shape * scale)` grid, trilinear-upsample
    to full resolution and exponentiate. Gaussian in log space, so the field is
    strictly positive and multiplies rather than shifts.

    This is lab2im's `BiasFieldCorruption`, and was written out three times: in
    `synthseg/functional.py::bias_field`, in `gpu/domain_transfer.py::_random_bias_field3d`
    and (in a different, polynomial form) in `gpu/contrast.py::RandomBiasFieldGPU`. The
    first two were line-for-line identical.
    """
    if std <= 0:
        return torch.ones(batch, channels, *shape, device=device, dtype=dtype)

    small = [max(2, math.ceil(s * scale)) for s in shape]
    # One std per batch element, shared across channels, matching lab2im.
    sampled_std = torch.rand(batch, 1, 1, 1, 1, device=device, dtype=dtype) * std
    field = torch.randn(batch, channels, *small, device=device, dtype=dtype) * sampled_std
    field = F.interpolate(field, size=tuple(shape), mode="trilinear", align_corners=True)
    return torch.exp(field)


# --- fixed derivative kernels -----------------------------------------------------
#
# Held as nested lists rather than tensors so there is no import-time device or dtype
# choice; `laplace_kernel` and `scharr_kernels` materialise them on demand.

#: 8-neighbour 2-D Laplacian.
LAPLACE_2D = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
]

#: 26-neighbour 3-D Laplacian: -1 everywhere, +26 in the centre. Sums to 0.
LAPLACE_3D = [[[-1] * 3 for _ in range(3)] for _ in range(3)]
LAPLACE_3D[1][1][1] = 26

#: 2-D Scharr, (x, y). The x-kernel's middle row is [-10, 0, 10]; the CPU copy of this
#: table had [-10, 0, -10], which made it sum to -20 instead of 0.
SCHARR_2D = [
    [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
    [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
]

#: 3-D Scharr, (x, y, z).
SCHARR_3D = [
    [
        [[9, 0, -9], [30, 0, -30], [9, 0, -9]],
        [[30, 0, -30], [100, 0, -100], [30, 0, -30]],
        [[9, 0, -9], [30, 0, -30], [9, 0, -9]],
    ],
    [
        [[9, 30, 9], [0, 0, 0], [-9, -30, -9]],
        [[30, 100, 30], [0, 0, 0], [-30, -100, -30]],
        [[9, 30, 9], [0, 0, 0], [-9, -30, -9]],
    ],
    [
        [[9, 30, 9], [30, 100, 30], [9, 30, 9]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]],
    ],
]


def laplace_kernel(spatial_dims: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """The Laplacian for 2-D or 3-D data."""
    if spatial_dims == 2:
        return torch.tensor(LAPLACE_2D, dtype=dtype, device=device)
    if spatial_dims == 3:
        return torch.tensor(LAPLACE_3D, dtype=dtype, device=device)
    raise ValueError(f"Laplace kernel is defined for 2 or 3 spatial dimensions, got {spatial_dims}")


def scharr_kernels(spatial_dims: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> list[Tensor]:
    """The directional Scharr kernels: two for 2-D data, three for 3-D."""
    if spatial_dims == 2:
        return [torch.tensor(k, dtype=dtype, device=device) for k in SCHARR_2D]
    if spatial_dims == 3:
        return [torch.tensor(k, dtype=dtype, device=device) for k in SCHARR_3D]
    raise ValueError(f"Scharr kernels are defined for 2 or 3 spatial dimensions, got {spatial_dims}")
