"""Convolution kernels and smooth random fields, shared by every backend.

Four 3-D Gaussian implementations, three bias fields and two copies of the Laplace/Scharr
constant tables used to live in five different modules, and the divergence is what let two
of them go wrong unnoticed: an uncentred Gaussian in `gpu/contrast.py`, a malformed 2-D
Scharr x-kernel in `cpu/contrast.py`. Anything that convolves or blurs imports from here.

The consolidation is deliberately not bit-for-bit for the two blur call sites: the radius is
`ceil(3*sigma)` where `domain_transfer` and `fromSeg` used `round(3*sigma)`, which is never
wider, so their kernels may now be one tap larger; and padding is `reflect` everywhere,
where `domain_transfer` used `replicate` and `fromSeg` relied on conv3d's implicit zero
padding -- which darkens the volume border, the one difference here that was a bug rather
than a choice. Everything else is the same arithmetic as the copy it replaces.
"""

from __future__ import annotations

import functools
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
    "box_filter",
    "depthwise_conv3d",
    "gaussian_blur3d",
    "gaussian_kernel1d",
    "gaussian_kernel3d",
    "laplace_kernel",
    "laplacian_response",
    "random_bias_field3d",
    "scharr_kernels",
    "stacked_scharr_kernels",
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


def _pad_axis(volume: Tensor, axis: int, pad: int, mode: str) -> Tensor:
    """Pad one spatial axis by `pad` on both sides, however wide `pad` is.

    `F.pad(mode="reflect")` requires the padding to be smaller than the axis it
    pads, and a Gaussian wide enough to blur a small patch exceeds that
    routinely: SynthSeg's `blurring_sigma_for_downsampling` reaches sigma ~6, so
    `ksize // 2` is 18 against a 16-voxel axis. That raised RuntimeError on about
    2% of generator calls on 16-cube patches -- an intermittent crash mid-run
    rather than a reproducible failure.

    Reflect is applied in as many legal passes as it takes, each at most
    `extent - 1`, which keeps the existing behaviour wherever one pass was
    already enough. Whatever is left over, and an axis of length 1 where reflect
    is undefined at all, falls back to replicate.
    """
    remaining = pad
    while remaining > 0:
        step = min(remaining, volume.shape[2 + axis] - 1) if mode == "reflect" else remaining
        step_mode = mode
        if step <= 0:
            step, step_mode = remaining, "replicate"
        pad_full = [0, 0, 0, 0, 0, 0]
        # F.pad's tuple runs last spatial axis first: (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi).
        pad_full[(2 - axis) * 2] = step
        pad_full[(2 - axis) * 2 + 1] = step
        volume = F.pad(volume, pad_full, mode=step_mode)
        remaining -= step
    return volume


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

    batch = image.shape[0]
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
        out = depthwise_conv3d(_pad_axis(out, axis, pad, padding_mode), kernel.view(shape))
    return out.view(batch, channels, *out.shape[2:])


def depthwise_conv3d(volume: Tensor, kernel: Tensor, filters_per_plane: int = 1) -> Tensor:
    """Convolve every (sample, channel) plane of a pre-padded volume, in one call.

    `volume` is `[B, C, D, H, W]`, already padded. `kernel` is
    `[filters_per_plane, 1, kd, kh, kw]` to apply the same filter bank to every
    plane, or `[B*C*filters_per_plane, 1, ...]` to give each plane its own. The
    result is `[B*C, filters_per_plane, ...]`.

    Folding the batch into the channel axis is the whole point. `F.conv3d` over a
    single-channel volume (`groups == 1`) goes through cuDNN's implicit-GEMM path,
    which on an A40 takes **4.25 ms** for one 3x3x3 kernel over a 128^3 patch;
    with `groups == B*C > 1` it dispatches to ATen's depthwise kernel and the same
    convolution takes **0.12 ms**. The depthwise path is also the more accurate of
    the two -- implicit GEMM runs in TF32 by default and lands 6e-3 from a float64
    reference where the depthwise kernel lands 4e-6 away.

    A genuinely single-plane call (one sample, one channel, which is what the
    per-sample `RandomChooseXTransformsGPU` bucket hands down) cannot reach that
    path, so the plane is duplicated to make `groups == 2` and the spare output
    dropped. Twice the arithmetic, a twentieth of the time.
    """
    if volume.dim() != 5:
        raise ValueError(f"expected a 5D [B, C, D, H, W] tensor, got shape {tuple(volume.shape)}")
    planes = volume.shape[0] * volume.shape[1]
    merged = volume.reshape(1, planes, *volume.shape[2:])

    if kernel.shape[0] == filters_per_plane:
        weight = kernel.repeat(planes, 1, 1, 1, 1)
    elif kernel.shape[0] == planes * filters_per_plane:
        weight = kernel
    else:
        raise ValueError(f"kernel must hold {filters_per_plane} or {planes * filters_per_plane} filters, got {kernel.shape[0]}")

    groups = planes
    if planes == 1:
        merged = merged.expand(1, 2, *merged.shape[2:]).contiguous()
        weight = weight.repeat(2, 1, 1, 1, 1)
        groups = 2

    out = F.conv3d(merged, weight, groups=groups)
    return out[0, : planes * filters_per_plane].reshape(planes, filters_per_plane, *out.shape[2:])


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
#:
#: Signed as SCHARR_2D above and as the usual Scharr/Sobel convention: the derivative
#: runs [-1, 0, +1], so the positive lobe is on the far side of each axis. This table was
#: the exact negation, so a 2-D and a 3-D run produced opposite-signed gradients.
#:
#: Every shipped config sets `absolute: true`, and |(-k) * x| == |k * x|, so no shipped
#: pipeline changes. It matters for `absolute=False`, the default of `RandomScharrGPU`.
SCHARR_3D = [
    [
        [[-9, 0, 9], [-30, 0, 30], [-9, 0, 9]],
        [[-30, 0, 30], [-100, 0, 100], [-30, 0, 30]],
        [[-9, 0, 9], [-30, 0, 30], [-9, 0, 9]],
    ],
    [
        [[-9, -30, -9], [0, 0, 0], [9, 30, 9]],
        [[-30, -100, -30], [0, 0, 0], [30, 100, 30]],
        [[-9, -30, -9], [0, 0, 0], [9, 30, 9]],
    ],
    [
        [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[9, 30, 9], [30, 100, 30], [9, 30, 9]],
    ],
]


# These tables are compile-time constants, but `torch.tensor(nested_list)` walks the list
# in Python and then copies host memory to the device -- every call, from inside
# `apply_transform`, for at most 27 numbers. Caching makes later calls a dict lookup.
#
# The cached tensors are shared, so callers must treat them as read-only. Every caller in
# this repository does: they expand, reshape, or convolve with them.


@functools.cache
def _laplace_table(spatial_dims: int, device: torch.device | None, dtype: torch.dtype) -> Tensor:
    table = LAPLACE_2D if spatial_dims == 2 else LAPLACE_3D
    return torch.tensor(table, dtype=dtype, device=device)


@functools.cache
def _scharr_table(spatial_dims: int, index: int, device: torch.device | None, dtype: torch.dtype) -> Tensor:
    table = SCHARR_2D if spatial_dims == 2 else SCHARR_3D
    return torch.tensor(table[index], dtype=dtype, device=device)


def box_filter(volume: Tensor, spatial_dims: int) -> Tensor:
    """Sum over a 3-wide window on every spatial axis, zero-padded outside.

    Separable, so `spatial_dims` one-dimensional passes -- `3 * d` reads per voxel
    instead of `3 ** d`.
    """
    conv = F.conv3d if spatial_dims == 3 else F.conv2d
    out = volume
    for axis in range(spatial_dims):
        shape = [1, 1] + [1] * spatial_dims
        shape[2 + axis] = 3
        padding = [0] * (2 * spatial_dims)
        padding[(spatial_dims - 1 - axis) * 2] = 1
        padding[(spatial_dims - 1 - axis) * 2 + 1] = 1
        out = conv(F.pad(out, padding), torch.ones(shape, device=volume.device, dtype=volume.dtype))
    return out


def laplacian_response(volume: Tensor, spatial_dims: int) -> Tensor:
    """Convolve `[1, 1, *spatial]` with the Laplacian, without the dense kernel.

    `LAPLACE_3D` is `27 * delta - ones(3, 3, 3)` (and `LAPLACE_2D` is
    `9 * delta - ones(3, 3)`), convolution is linear, and the all-ones window is
    separable -- so the whole filter is a box filter subtracted from a scaled copy.
    That is nine multiply-free passes in 3-D rather than twenty-seven taps per
    voxel, and measured 8.6 ms against 32.9 ms for a 96^3 patch on eight CPU
    threads. Zero padding, matching `padding="same"`.

    Float addition is not associative, so the result differs from the dense
    convolution in the last couple of digits (2e-5 absolute on a response of
    magnitude 100).
    """
    return (3**spatial_dims) * volume - box_filter(volume, spatial_dims)


def laplace_kernel(spatial_dims: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """The Laplacian for 2-D or 3-D data. Cached and shared; do not mutate the result."""
    if spatial_dims not in (2, 3):
        raise ValueError(f"Laplace kernel is defined for 2 or 3 spatial dimensions, got {spatial_dims}")
    return _laplace_table(spatial_dims, device, dtype)


def scharr_kernels(spatial_dims: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> list[Tensor]:
    """The directional Scharr kernels: two for 2-D data, three for 3-D.

    Cached and shared; do not mutate the results.
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"Scharr kernels are defined for 2 or 3 spatial dimensions, got {spatial_dims}")
    count = 2 if spatial_dims == 2 else 3
    return [_scharr_table(spatial_dims, i, device, dtype) for i in range(count)]


@functools.cache
def stacked_scharr_kernels(spatial_dims: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """The Scharr kernels as one `[n_dims, 1, *kernel_shape]` conv weight.

    Lets a caller run all the directional filters in a single convolution instead of
    one per direction. Cached and shared; do not mutate the result.
    """
    return torch.stack(scharr_kernels(spatial_dims, device, dtype)).unsqueeze(1)
