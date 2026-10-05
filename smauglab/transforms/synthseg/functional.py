"""Functional building blocks for the SynthSeg generative model (GPU / torch).

This module re-implements, as standalone differentiable-free torch ops, the
individual layers of the SynthSeg "brain generator" described in:

    B. Billot et al., "SynthSeg: Segmentation of brain MRI scans of any contrast
    and resolution without retraining", Medical Image Analysis, 2023.
    (and the earlier MICCAI-2020 contrast-agnostic / PV-segmentation papers)

The reference TensorFlow implementation lives in ``BBillot/SynthSeg`` and
``BBillot/lab2im``. Each function below cites the corresponding reference layer.
Everything operates on 3D volumes stored as ``(B, C, D, H, W)`` torch tensors
(label maps as ``(B, 1, D, H, W)`` integer tensors), which is the convention
used throughout SmaugLab's GPU transforms.

Spatial conventions
--------------------
* Spatial axes ``(D, H, W)`` map to torch dims ``(2, 3, 4)``. Internally we work
  with voxel coordinates in ``(i, j, k) = (D, H, W)`` order and only convert to
  the ``(x, y, z) = (W, H, D)`` order expected by ``F.grid_sample`` at the very
  end, with ``align_corners=True`` so that integer voxel indices map exactly.
* Affine transforms are applied about the volume centre (standard practice and
  matching SmaugLab's existing ``RandomAffineGPU``), so small rotations /
  scalings keep the anatomy in frame.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Union

import torch
import torch.nn.functional as F

from smauglab.transforms.kernels import gaussian_blur3d, random_bias_field3d

Number = Union[int, float]

__all__ = [
    "bias_field",
    "blurring_sigma_for_downsampling",
    "convert_labels",
    "distinct_values",
    "em_subdivide_labels",
    "flip_lr_with_swap",
    "gaussian_blur_3d",
    "infer_label_values",
    "intensity_augmentation",
    "labels_to_image_gmm",
    "mimic_acquisition",
    "random_merge_classes",
    "random_svf_field",
    "sample_affine_matrices",
    "sample_gmm_parameters",
    "sample_resolution",
    "to_label_map",
    "warp_volume",
]


# ---------------------------------------------------------------------------
# Label map helpers
# ---------------------------------------------------------------------------
def to_label_map(seg: torch.Tensor) -> torch.Tensor:
    """Coerce a segmentation tensor into a single-channel integer label map.

    Accepts:
      * ``(B, 1, D, H, W)``  -> rounded to integer labels (used directly).
      * ``(B, C, D, H, W)`` one-hot (C > 1) -> argmax + 1, background (all-zero
        across channels) stays 0. This matches SmaugLab's
        :func:`collapse_onehot_to_index` convention where channel ``c`` encodes
        label ``c + 1``.
      * ``(B, D, H, W)``     -> unsqueezed to ``(B, 1, D, H, W)``.

    Returns a ``(B, 1, D, H, W)`` ``long`` tensor.
    """
    if seg.dim() == 4:
        seg = seg.unsqueeze(1)
    if seg.dim() != 5:
        raise ValueError(f"Expected a 4D or 5D segmentation tensor, got {seg.dim()}D.")

    if seg.shape[1] == 1:
        return seg.round().long()

    # One-hot -> integer index (channel c -> label c + 1, background -> 0).
    foreground = seg.any(dim=1, keepdim=True)
    labels = torch.argmax(seg, dim=1, keepdim=True).long() + 1
    return torch.where(foreground, labels, torch.zeros_like(labels))


def infer_label_values(label_map: torch.Tensor) -> torch.Tensor:
    """Return the sorted unique label values present in ``label_map``."""
    return torch.unique(label_map).long()


# ---------------------------------------------------------------------------
# GMM intensity model  (lab2im.layers.SampleConditionalGMM
#                       + SynthSeg.model_inputs.build_model_inputs)
# ---------------------------------------------------------------------------
def _draw_value(
    prior: Union[Number, Sequence[Number], torch.Tensor] | None,
    size: tuple[int, int],
    distribution: str,
    centre: float,
    default_range: float,
    device: torch.device,
    positive_only: bool = False,
) -> torch.Tensor:
    """Port of ``lab2im.utils.draw_value_from_distribution``.

    ``size`` is ``(batch, n_classes)``. Returns a tensor of that shape.

    ``prior`` interpretations:
      * ``None``           -> ``uniform``: U(centre-range, centre+range);
                              ``normal``:  N(centre, range).
      * scalar ``s``       -> ``uniform``: U(centre-s, centre+s);
                              ``normal``:  N(centre, s).
      * length-2 ``[a, b]``-> ``uniform``: U(a, b); ``normal``: N(a, b).
        (shared across classes)
      * array ``(2, K)``   -> per-class ``[a, b]`` rows.
    """
    batch, n_classes = size

    # Resolve the two distribution parameters (a, b) of shape ``size``.
    #   uniform -> (low, high);  normal -> (mean, std).
    if prior is None:
        if distribution == "uniform":
            a = torch.full(size, centre - default_range, device=device)
            b = torch.full(size, centre + default_range, device=device)
        else:
            a = torch.full(size, centre, device=device)
            b = torch.full(size, default_range, device=device)
    elif isinstance(prior, (int, float)):
        if distribution == "uniform":
            a = torch.full(size, centre - float(prior), device=device)
            b = torch.full(size, centre + float(prior), device=device)
        else:
            a = torch.full(size, centre, device=device)
            b = torch.full(size, float(prior), device=device)
    else:
        prior_t = torch.as_tensor(prior, dtype=torch.float32, device=device)
        if prior_t.numel() == 2 and prior_t.dim() == 1:
            a = prior_t[0].expand(size).clone()
            b = prior_t[1].expand(size).clone()
        elif prior_t.dim() == 2 and prior_t.shape[0] == 2:
            if prior_t.shape[1] != n_classes:
                raise ValueError(f"Prior array has {prior_t.shape[1]} classes, expected {n_classes}.")
            a = prior_t[0].unsqueeze(0).expand(size).clone()
            b = prior_t[1].unsqueeze(0).expand(size).clone()
        else:
            raise ValueError(f"Unsupported prior shape {tuple(prior_t.shape)}.")

    out = _sample(distribution, a, b, device)
    return out.clamp_min(0.0) if positive_only else out


def _sample(distribution: str, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> torch.Tensor:
    if distribution == "uniform":
        return a + (b - a) * torch.rand(a.shape, device=device)
    if distribution == "normal":
        return a + b * torch.randn(a.shape, device=device)
    raise ValueError(f"Unknown distribution '{distribution}' (use 'uniform' or 'normal').")


def sample_gmm_parameters(
    n_labels: int,
    n_channels: int,
    batch: int,
    device: torch.device,
    prior_means=None,
    prior_stds=None,
    prior_distributions: str = "uniform",
    generation_classes: Sequence[int] | torch.Tensor | None = None,
    background_label_index: int | None = 0,
    randomise_background: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw per-label Gaussian means/stds for one minibatch.

    Mirrors ``SynthSeg.model_inputs.build_model_inputs``. With the default
    ``prior_means=None`` / ``prior_stds=None`` the *effective* priors are
    ``means ~ U(0, 250)`` and ``stds ~ U(0, 30)`` per class (the code uses
    ``centre=125, range=125`` and ``centre=15, range=15``; the often-quoted
    ``[25, 225]`` / ``[5, 25]`` come from a stale docstring).

    ``generation_classes`` lets several labels share the same Gaussian (e.g. to
    tie left/right homologues). When ``None`` every label is independent. A
    ``(batch, n_labels)`` array ties different labels together in each sample,
    which is what :func:`random_merge_classes` draws.

    Returns ``means, stds`` of shape ``(batch, n_labels, n_channels)``.
    """
    if generation_classes is None:
        classes = torch.arange(n_labels, device=device)
    else:
        classes = torch.as_tensor(generation_classes, device=device, dtype=torch.long)
        if classes.dim() not in (1, 2) or classes.shape[-1] != n_labels:
            raise ValueError("generation_classes must have one entry per generation label.")
        if classes.dim() == 2 and classes.shape[0] != batch:
            raise ValueError("per-sample generation_classes must have one row per batch element.")
    n_classes = int(classes.max().item()) + 1

    means = torch.empty(batch, n_classes, n_channels, device=device)
    stds = torch.empty(batch, n_classes, n_channels, device=device)
    for ch in range(n_channels):
        means[:, :, ch] = _draw_value(prior_means, (batch, n_classes), prior_distributions, 125.0, 125.0, device, positive_only=True)
        stds[:, :, ch] = _draw_value(prior_stds, (batch, n_classes), prior_distributions, 15.0, 15.0, device, positive_only=True)

    # Scatter class parameters to per-label parameters.
    if classes.dim() == 2:
        # Per-sample classes: every row picks from its own draws, so gather rather
        # than index -- `means[:, classes, :]` would broadcast one row over all.
        index = classes.unsqueeze(-1).expand(-1, -1, n_channels)
        means_lab = torch.gather(means, 1, index)
        stds_lab = torch.gather(stds, 1, index)
    else:
        means_lab = means[:, classes, :]
        stds_lab = stds[:, classes, :]

    # Background special-casing (build_model_inputs): per subject, 5% pure black,
    # 25% very dark/low-variance, 70% normal draw.
    if randomise_background and background_label_index is not None and 0 <= background_label_index < n_labels:
        # One draw per sample and two candidate parameter sets, selected with
        # `torch.where`. The loop this replaces read `float(torch.rand(()))` back to
        # the host once per sample, which stalls the whole queue; the branch
        # probabilities and the per-sample independence are unchanged, but the dark
        # draws are now made for every sample rather than only the ones that take
        # that branch, so the stream differs from the previous release's.
        draw = torch.rand(batch, 1, device=device)
        dark_means = torch.rand(batch, n_channels, device=device) * 15.0
        dark_stds = torch.rand(batch, n_channels, device=device) * 5.0
        black = draw > 0.95
        dark = (draw > 0.70) & ~black

        background_means = torch.where(
            black, torch.zeros_like(dark_means), torch.where(dark, dark_means, means_lab[:, background_label_index, :])
        )
        background_stds = torch.where(
            black, torch.zeros_like(dark_stds), torch.where(dark, dark_stds, stds_lab[:, background_label_index, :])
        )
        means_lab = means_lab.clone()
        stds_lab = stds_lab.clone()
        means_lab[:, background_label_index, :] = background_means
        stds_lab[:, background_label_index, :] = background_stds

    return means_lab, stds_lab


def labels_to_image_gmm(
    label_map: torch.Tensor,
    label_values: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """Render an image from a label map with a per-label Gaussian mixture.

    ``image[v] = mean[label_v] + std[label_v] * N(0, 1)`` (independent noise per
    voxel and channel). Mirrors ``lab2im.layers.SampleConditionalGMM``.

    Args:
        label_map: ``(B, 1, D, H, W)`` integer labels.
        label_values: ``(K,)`` the generation label values (sorted unique).
        means, stds: ``(B, K, C)`` per-label parameters.

    Returns ``(B, C, D, H, W)`` float image.
    """
    B = label_map.shape[0]
    spatial = label_map.shape[2:]
    C = means.shape[-1]
    device = label_map.device
    dtype = means.dtype

    max_label = int(label_values.max().item())
    # LUT: label value -> contiguous class index 0..K-1.
    lut = torch.zeros(max_label + 1, dtype=torch.long, device=device)
    lut[label_values] = torch.arange(label_values.numel(), device=device)
    idx = lut[label_map.clamp(min=0, max=max_label)].squeeze(1)  # (B, D, H, W)

    # One gather and one noise draw for the whole batch. The nested loop this
    # replaces indexed `means[b, :, ch][idx_b]` once per (sample, channel) and drew
    # its noise volume separately, so a four-channel batch of two paid eight small
    # `randn` launches and eight gathers for what is one of each.
    flat_idx = idx.reshape(B, 1, -1).expand(B, C, -1)  # (B, C, N)
    mean_map = torch.gather(means.transpose(1, 2), 2, flat_idx)  # (B, C, N)
    std_map = torch.gather(stds.transpose(1, 2), 2, flat_idx)
    noise = torch.randn((B, C, *spatial), device=device, dtype=dtype)
    return (mean_map + std_map * noise.reshape(B, C, -1)).reshape(B, C, *spatial)


# ---------------------------------------------------------------------------
# Spatial deformation  (lab2im.layers.RandomSpatialDeformation
#                       + utils.sample_affine_transform + neuron VecInt)
# ---------------------------------------------------------------------------
def _as_3vec(value, device: torch.device, default: float = 0.0) -> torch.Tensor:
    if value is None or value is False:
        return torch.full((3,), float(default), device=device)
    if isinstance(value, (int, float)):
        return torch.full((3,), float(value), device=device)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def sample_affine_matrices(
    batch: int,
    device: torch.device,
    scaling_bounds: Union[bool, Number, Sequence[Number]] = 0.2,
    rotation_bounds: Union[bool, Number, Sequence[Number]] = 15.0,
    shearing_bounds: Union[bool, Number, Sequence[Number]] = 0.012,
    translation_bounds: Union[bool, Number, Sequence[Number]] = False,
) -> torch.Tensor:
    """Sample a batch of 3D affine matrices.

    Port of ``lab2im.utils.sample_affine_transform`` /
    ``create_affine_transformation_matrix`` for ``n_dims=3``. Each parameter is
    drawn independently per axis:

      * scaling ``~ U(1 - s, 1 + s)`` (``scaling_bounds=0.2`` -> U(0.8, 1.2))
      * rotation ``~ U(-r, r)`` degrees (``rotation_bounds=15``)
      * shearing ``~ U(-sh, sh)`` (``shearing_bounds=0.012``)
      * translation ``~ U(-t, t)`` voxels (``False`` -> disabled)

    The composition is ``T = T_scaling @ T_shearing @ T_rotation`` with the
    translation placed in the last column. Returns ``(B, 4, 4)``. The matrix is
    applied about the volume centre by :func:`warp_volume`.
    """

    def draw(bounds, centre):
        vec = _as_3vec(bounds, device, default=0.0)
        return centre + (2.0 * torch.rand(batch, 3, device=device) - 1.0) * vec.view(1, 3)

    scaling = draw(scaling_bounds, 1.0) if scaling_bounds not in (False, None) else torch.ones(batch, 3, device=device)
    rotation_deg = draw(rotation_bounds, 0.0) if rotation_bounds not in (False, None) else torch.zeros(batch, 3, device=device)
    # 6 shear parameters for the off-diagonal entries of the 3x3 linear part.
    shear_vec = _as_3vec(shearing_bounds, device, default=0.0)
    shear_vec6 = torch.cat([shear_vec, shear_vec]) if shearing_bounds not in (False, None) else torch.zeros(6, device=device)
    shearing = (2.0 * torch.rand(batch, 6, device=device) - 1.0) * shear_vec6.view(1, 6)
    translation = draw(translation_bounds, 0.0) if translation_bounds not in (False, None) else torch.zeros(batch, 3, device=device)

    rot = math.pi / 180.0 * rotation_deg
    cx, cy, cz = torch.cos(rot[:, 0]), torch.cos(rot[:, 1]), torch.cos(rot[:, 2])
    sx, sy, sz = torch.sin(rot[:, 0]), torch.sin(rot[:, 1]), torch.sin(rot[:, 2])

    zeros = torch.zeros(batch, device=device)
    ones = torch.ones(batch, device=device)

    def stack3(rows):
        return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)  # (B, 3, 3)

    rx = stack3([[ones, zeros, zeros], [zeros, cx, -sx], [zeros, sx, cx]])
    ry = stack3([[cy, zeros, sy], [zeros, ones, zeros], [-sy, zeros, cy]])
    rz = stack3([[cz, -sz, zeros], [sz, cz, zeros], [zeros, zeros, ones]])
    rotation_m = torch.bmm(torch.bmm(rx, ry), rz)

    shear_m = torch.eye(3, device=device).unsqueeze(0).repeat(batch, 1, 1)
    # off-diagonal positions (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)
    pos = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    for n, (i, j) in enumerate(pos):
        shear_m[:, i, j] = shearing[:, n]

    scale_m = torch.zeros(batch, 3, 3, device=device)
    scale_m[:, 0, 0] = scaling[:, 0]
    scale_m[:, 1, 1] = scaling[:, 1]
    scale_m[:, 2, 2] = scaling[:, 2]

    linear = torch.bmm(torch.bmm(scale_m, shear_m), rotation_m)  # (B, 3, 3)

    affine = torch.zeros(batch, 4, 4, device=device)
    affine[:, :3, :3] = linear
    affine[:, :3, 3] = translation
    affine[:, 3, 3] = 1.0
    return affine


@functools.lru_cache(maxsize=8)
def _identity_grid(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Voxel-coordinate identity grid in ``(i, j, k)`` order, shape ``(3, D, H, W)``.

    Cached: it depends only on the patch shape, and `_integrate_velocity` calls
    `warp_volume` seven times per field, each of which rebuilt it -- 24 MB of
    `meshgrid` and `stack` at 128^3, seven times over, for a constant. The returned
    tensor is shared; callers must not write into it (none do -- `warp_volume`
    expands and then reads).
    """
    d, h, w = shape
    zs = torch.arange(d, device=device, dtype=torch.float32)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    ii, jj, kk = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([ii, jj, kk], dim=0)


def _coords_to_grid_sample(coords: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Convert ``(B, 3, D, H, W)`` voxel coords (i,j,k) to a grid_sample grid.

    Output ``(B, D, H, W, 3)`` with last-dim order ``(x, y, z) = (k, j, i)``
    normalised to ``[-1, 1]`` for ``align_corners=True``.
    """
    d, h, w = shape
    i = coords[:, 0]
    j = coords[:, 1]
    k = coords[:, 2]
    norm_k = 2.0 * k / max(w - 1, 1) - 1.0
    norm_j = 2.0 * j / max(h - 1, 1) - 1.0
    norm_i = 2.0 * i / max(d - 1, 1) - 1.0
    return torch.stack([norm_k, norm_j, norm_i], dim=-1)


def warp_volume(
    volume: torch.Tensor,
    affine: torch.Tensor | None = None,
    displacement: torch.Tensor | None = None,
    interp: str = "linear",
    center: bool = True,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Resample ``volume`` by an affine matrix and/or a dense displacement field.

    Mirrors ``ext.neuron.layers.SpatialTransformer`` composing
    ``[affine, dense_field]``: the sampling location of each output voxel is
    ``affine(identity) + displacement``.

    Args:
        volume: ``(B, C, D, H, W)``.
        affine: ``(B, 4, 4)`` linear+translation applied about the centre (voxel
            units, ``(i, j, k)`` order). ``None`` -> identity.
        displacement: ``(B, 3, D, H, W)`` per-voxel shift in ``(i, j, k)`` voxel
            units. ``None`` -> no elastic term.
        interp: ``"linear"`` (trilinear) for images / fields, ``"nearest"`` for
            label maps.
        center: apply the affine about the volume centre.
        padding_mode: ``grid_sample`` padding (``"zeros"`` for label/image,
            ``"border"`` for field composition).
    """
    B, C, D, H, W = volume.shape
    shape = (D, H, W)
    device = volume.device

    grid = _identity_grid(shape, device).unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B,3,D,H,W)
    coords = grid

    if affine is not None:
        flat = coords.reshape(B, 3, -1)  # (B, 3, N)
        if center:
            centre = torch.tensor([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0], device=device).view(1, 3, 1)
            flat = flat - centre
        linear = affine[:, :3, :3]
        translation = affine[:, :3, 3:4]
        flat = torch.bmm(linear, flat) + translation
        if center:
            flat = flat + centre
        coords = flat.reshape(B, 3, D, H, W)

    if displacement is not None:
        coords = coords + displacement

    sample_grid = _coords_to_grid_sample(coords, shape)
    mode = "nearest" if interp == "nearest" else "bilinear"  # 3D 'bilinear' == trilinear
    return F.grid_sample(volume, sample_grid, mode=mode, align_corners=True, padding_mode=padding_mode)


def _integrate_velocity(velocity: torch.Tensor, int_steps: int = 7) -> torch.Tensor:
    """Scaling-and-squaring integration of a stationary velocity field.

    Port of ``ext.neuron.layers.VecInt`` (``method='ss'``, ``int_steps=7``):
    ``phi = v / 2**N`` then ``phi <- phi + warp(phi, phi)`` repeated ``N`` times,
    yielding a diffeomorphic displacement field. ``velocity`` and the returned
    displacement are ``(B, 3, D, H, W)`` in voxel units.
    """
    disp = velocity / (2**int_steps)
    for _ in range(int_steps):
        disp = disp + warp_volume(disp, displacement=disp, interp="linear", padding_mode="border")
    return disp


def random_svf_field(
    batch: int,
    shape: tuple[int, int, int],
    device: torch.device,
    nonlin_std: float = 4.0,
    nonlin_scale: float = 0.04,
    int_steps: int = 7,
) -> torch.Tensor:
    """Sample a smooth diffeomorphic displacement field.

    Port of ``RandomSpatialDeformation``'s nonlinear branch: draw a small
    stationary velocity field ``~ N(0, U(0, nonlin_std))`` on a coarse grid
    (``ceil(shape * nonlin_scale)``), upsample it (trilinear) to full resolution,
    then integrate it by scaling-and-squaring. Returns ``(B, 3, D, H, W)``.
    """
    if nonlin_std <= 0:
        return torch.zeros(batch, 3, *shape, device=device)

    small = [max(2, math.ceil(s * nonlin_scale)) for s in shape]
    std = torch.rand(batch, 1, 1, 1, 1, device=device) * nonlin_std
    velocity = torch.randn(batch, 3, *small, device=device) * std
    velocity = F.interpolate(velocity, size=shape, mode="trilinear", align_corners=True)
    return _integrate_velocity(velocity, int_steps=int_steps)


# ---------------------------------------------------------------------------
# Bias field  (lab2im.layers.BiasFieldCorruption)
# ---------------------------------------------------------------------------
def bias_field(
    image: torch.Tensor,
    bias_field_std: float = 0.7,
    bias_scale: float = 0.025,
) -> torch.Tensor:
    """Apply a smooth multiplicative bias field.

    Port of ``BiasFieldCorruption``: sample a small Gaussian field
    ``~ N(0, U(0, bias_field_std))`` on a coarse grid (``ceil(shape*bias_scale)``),
    upsample (trilinear) to full resolution, exponentiate, and multiply. The
    field is Gaussian in log space -> positive and multiplicative in intensity
    space. A separate field is drawn per channel.
    """
    if bias_field_std <= 0:
        return image
    B, C, D, H, W = image.shape
    return image * random_bias_field3d((D, H, W), bias_field_std, bias_scale, image.device, image.dtype, batch=B, channels=C)


# ---------------------------------------------------------------------------
# Intensity augmentation  (lab2im.layers.IntensityAugmentation)
# ---------------------------------------------------------------------------
def intensity_augmentation(
    image: torch.Tensor,
    clip: float = 300.0,
    gamma_std: float = 0.5,
    normalise: bool = True,
) -> torch.Tensor:
    """Clip -> per-channel min-max normalise to [0, 1] -> gamma.

    Port of ``IntensityAugmentation(clip=300, normalise=True, gamma_std=.5,
    separate_channels=True)``. Gamma is log-normal:
    ``image <- image ** exp(N(0, gamma_std))`` (one exponent per channel).
    """
    B, C = image.shape[:2]
    reduce_dims = tuple(range(2, image.dim()))

    if clip and clip > 0:
        image = image.clamp(0.0, clip)

    if normalise:
        m = image.amin(dim=reduce_dims, keepdim=True)
        M = image.amax(dim=reduce_dims, keepdim=True)
        image = (image - m) / (M - m + 1e-7)

    if gamma_std and gamma_std > 0:
        gamma = torch.exp(torch.randn(B, C, *([1] * (image.dim() - 2)), device=image.device) * gamma_std)
        image = image.clamp_min(0.0) ** gamma

    return image


# ---------------------------------------------------------------------------
# Resolution randomisation  (lab2im.edit_tensors + layers.GaussianBlur /
#                            DynamicGaussianBlur / SampleResolution /
#                            MimicAcquisition)
# ---------------------------------------------------------------------------
def blurring_sigma_for_downsampling(
    current_res: torch.Tensor,
    downsample_res: torch.Tensor,
    thickness: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-axis Gaussian blur sigma for a target acquisition resolution.

    Port of ``edit_tensors.blurring_sigma_for_downsampling`` with
    ``mult_coef=None``: ``sigma = 0.75 * min(downsample_res, thickness) /
    current_res``; ``sigma = 0.5`` where ``downsample_res == current_res``;
    ``sigma = 0`` where ``downsample_res == 0``. All inputs are ``(3,)`` tensors.
    """
    if thickness is None:
        thickness = downsample_res
    effective = torch.minimum(downsample_res, thickness)
    sigma = 0.75 * effective / current_res
    sigma = torch.where(downsample_res == current_res, torch.full_like(sigma, 0.5), sigma)
    sigma = torch.where(downsample_res == 0, torch.zeros_like(sigma), sigma)
    return sigma


def gaussian_blur_3d(
    image: torch.Tensor,
    sigma: torch.Tensor,
    blur_range: float = 1.03,
) -> torch.Tensor:
    """Separable anisotropic Gaussian blur with random sigma jitter.

    Port of ``GaussianBlur`` / ``DynamicGaussianBlur``: the per-axis ``sigma`` is
    multiplied by ``U(1/blur_range, blur_range)`` (``blur_range=1.03`` in
    SynthSeg, ``1.15`` in the 2020 lab2im model) and applied as three 1D
    convolutions (reflect padding). ``sigma`` is a ``(3,)`` tensor.
    """
    return gaussian_blur3d(image, sigma, blur_range=blur_range)


def sample_resolution(
    min_res: torch.Tensor,
    max_res_iso: float = 4.0,
    max_res_aniso: float = 8.0,
    prob_iso: float = 0.1,
    prob_min: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random target acquisition resolution and slice thickness.

    Port of ``lab2im.layers.SampleResolution`` (with ``return_thickness=True``):

      * with prob ``prob_iso``: isotropic, ``res ~ U(min_res, max_res_iso)`` (same
        on all axes);
      * else: anisotropic, one random axis gets ``U(min_res, max_res_aniso)`` and
        the others stay at ``min_res``;
      * with prob ``prob_min``: override to ``min_res`` (no downsampling);
      * thickness ``~ U(min_res, res)`` per axis.

    ``min_res`` is a ``(3,)`` tensor (the native/atlas resolution). Returns
    ``(resolution, thickness)`` as ``(3,)`` tensors.
    """
    device = min_res.device
    if float(torch.rand((), device=device)) < prob_iso:
        r = float(torch.rand((), device=device))
        res = min_res + (max_res_iso - min_res) * r  # same scalar factor -> isotropic-ish
    else:
        res = min_res.clone()
        axis = int(torch.randint(0, 3, (1,), device=device))
        res[axis] = min_res[axis] + (max_res_aniso - float(min_res[axis])) * float(torch.rand((), device=device))

    if float(torch.rand((), device=device)) < prob_min:
        res = min_res.clone()

    thickness = min_res + (res - min_res) * torch.rand(3, device=device)
    return res, thickness


def mimic_acquisition(
    image: torch.Tensor,
    current_res: torch.Tensor,
    downsample_res: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Downsample to a target resolution, then resample to the output grid.

    Port of ``lab2im.layers.MimicAcquisition``: nearest-neighbour downsampling to
    the sampled ``downsample_res`` grid (the partial-volume step) followed by
    trilinear resampling to ``output_shape``.

    The downsampling uses ``"nearest-exact"``, not ``"nearest"``. Torch's
    ``"nearest"`` maps ``src = floor(dst * scale)`` with no half-pixel offset, so
    it does not sample the centre of each output voxel -- it samples the left
    edge, and the content drifts toward higher indices. Composed with the
    trilinear upsampling here that is a mean edge displacement of +0.76 voxels
    (sd 0.45, measured over the 0.25-0.9 factor range). The label map this image
    is synthesised from is *not* resampled, so that drift is a straight
    image-to-label misregistration on every sample. ``"nearest-exact"`` is
    half-pixel centred and brings it to -0.10 voxels, which is the unavoidable
    nearest-neighbour tie-break rather than a bias. It is also what the lab2im
    original does: ``tf.image.resize(..., method="nearest")`` uses half-pixel
    centres, so ``"nearest"`` was never the faithful port.
    """
    B, C, D, H, W = image.shape
    in_shape = (D, H, W)
    factor = (current_res / downsample_res).tolist()
    down_shape = [max(1, round(in_shape[i] * factor[i])) for i in range(3)]
    x = F.interpolate(image, size=down_shape, mode="nearest-exact")
    x = F.interpolate(x, size=tuple(output_shape), mode="trilinear", align_corners=True)
    return x


# ---------------------------------------------------------------------------
# EM label completion for sparse label maps  (SynthSeg paper, Sec. 5.4)
# ---------------------------------------------------------------------------
def _quadratic_log_likelihood_terms(means: torch.Tensor, var: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    """The `(3, K)` coefficients of the per-component log-density as a quadratic in x.

    ``log N(x | m, v) + log w`` expands to ``a x^2 + b x + c`` with
    ``a = -1/(2v)``, ``b = m/v`` and ``c = log w - (log 2pi + log v)/2 - m^2/(2v)``.
    Stacking those lets a whole `(n, K)` responsibility matrix come out of one
    matrix multiply against ``[x^2, x, 1]``, instead of half a dozen elementwise
    kernels each writing an `(n, K)` temporary.
    """
    inv_var = 1.0 / var
    const = torch.log(weights.clamp_min(eps)) - 0.5 * (math.log(2.0 * math.pi) + torch.log(var)) - 0.5 * means * means * inv_var
    return torch.stack([-0.5 * inv_var, means * inv_var, const])


def distinct_values(volume: torch.Tensor) -> torch.Tensor:
    """The distinct non-negative integer values in `volume`, ascending.

    `torch.unique` radix-sorts the input; counting into bins and keeping the
    non-empty ones answers the same question for small label alphabets in about
    half the time on a 2 x 128^3 volume, and the two results are identical.
    """
    return torch.bincount(volume.reshape(-1).clamp_min(0)).nonzero().flatten()


def _label_counts_per_sample(label_map: torch.Tensor) -> torch.Tensor:
    """`(B, max_label + 1)` voxel counts per label, per sample."""
    batch = label_map.shape[0]
    flat = label_map.reshape(batch, -1).clamp_min(0)
    n_bins = int(flat.max()) + 1
    # One `bincount` over row-offset labels: a `scatter_add_` into a handful of bins
    # has every thread fighting for the same addresses, where bincount accumulates
    # in shared memory per block.
    offset = flat + torch.arange(batch, device=flat.device).view(-1, 1) * n_bins
    return torch.bincount(offset.reshape(-1), minlength=batch * n_bins).view(batch, n_bins)


def _em_gmm_1d(x_fit: torch.Tensor, n_components: int, n_iters: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fit a 1D Gaussian mixture by Expectation-Maximization.

    ``x_fit`` is a 1D tensor of intensities. Returns ``(means, vars, weights)``
    each of shape ``(K,)`` (``K = min(n_components, len(x_fit))``), or ``None`` if
    ``x_fit`` is empty. Responsibilities use a numerically-stable log-sum-exp.

    Each iteration is two matrix multiplies and a softmax rather than a dozen
    elementwise passes over an ``(n, K)`` array: the E-step log-densities are a
    quadratic in x (see :func:`_quadratic_log_likelihood_terms`) and the M-step only
    needs the responsibility-weighted sums of ``1``, ``x`` and ``x^2``, which is the
    same design matrix transposed. With ``n_iters=20`` and up to ten components that
    is the difference between ~240 kernel launches per fit and ~60.

    ``x`` is centred on its own mean before the moments are accumulated, so the
    ``E[x^2] - E[x]^2`` variance cannot lose its significant digits to cancellation
    when a component is tight and far from zero. The means are shifted back at the
    end.
    """
    n = x_fit.numel()
    if n == 0:
        return None
    k = max(1, min(int(n_components), n))
    device = x_fit.device

    # `float(x.min())` twice was two host synchronisations per fit, and
    # `em_subdivide_labels` calls this once per label per sample.
    x_min, x_max = x_fit.min(), x_fit.max()
    step = (x_max - x_min) / max(k - 1, 1)
    means = x_min + step * torch.arange(k, device=device, dtype=x_fit.dtype)
    # Degenerate (constant region): spread the means slightly.
    means = torch.where(x_max > x_min, means, x_min + torch.arange(k, device=device, dtype=x_fit.dtype) * eps)
    var = x_fit.var(unbiased=False).clamp_min(eps).repeat(k)
    weights = torch.full((k,), 1.0 / k, device=device)

    shift = x_fit.mean()
    centred = (x_fit - shift).view(n, 1)
    design = torch.cat([centred * centred, centred, torch.ones_like(centred)], dim=1)  # (n, 3)
    means = means - shift

    for _ in range(int(n_iters)):
        resp = torch.softmax(design @ _quadratic_log_likelihood_terms(means, var, weights, eps), dim=1)
        moments = resp.transpose(0, 1) @ design  # (k, 3): sums of r*x^2, r*x, r
        nk = moments[:, 2].clamp_min(eps)
        weights = nk / n
        means = moments[:, 1] / nk
        var = (moments[:, 0] / nk - means * means).clamp_min(eps)
    return means + shift, var, weights


def _assign_gmm(
    x_full: torch.Tensor,
    means: torch.Tensor,
    var: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
    chunk: int = 2_000_000,
) -> torch.Tensor:
    """Hard-assign each value in ``x_full`` to its most likely mixture component."""
    n = x_full.numel()
    out = torch.empty(n, dtype=torch.long, device=x_full.device)
    # Same quadratic form as the E-step: one matmul per chunk instead of the
    # subtract / square / divide / subtract chain, each of which wrote its own
    # (chunk, K) array out to memory.
    terms = _quadratic_log_likelihood_terms(means, var, weights, eps)
    for s in range(0, n, chunk):
        xc = x_full[s : s + chunk].view(-1, 1)
        design = torch.cat([xc * xc, xc, torch.ones_like(xc)], dim=1)
        out[s : s + chunk] = (design @ terms).argmax(dim=1)
    return out


def em_subdivide_labels(
    image: torch.Tensor,
    label_map: torch.Tensor,
    n_foreground_clusters: int = 2,
    background_clusters_range: Sequence[int] = (3, 10),
    background_label: int = 0,
    n_iters: int = 20,
    max_fit_voxels: int = 100000,
    channel: int = 0,
    same_on_batch: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Subdivide each label into intensity-coherent subregions via EM (SynthSeg §5.4).

    Reproduces SynthSeg's handling of sparse / incomplete label maps: "we enhance
    the training segmentations by subdividing all their labels (background and
    foreground) into finer subregions [...] by clustering the intensities of the
    associated image with the Expectation Maximisation algorithm." Each foreground
    label is split into ``n_foreground_clusters`` (2 in the paper); the background
    label is split into a random ``N`` in ``background_clusters_range`` ([3, 10]).
    The resulting fine labels are the *generation* labels (each gets its own
    Gaussian), while the original labels are recovered for the segmentation target
    via the returned merge map.

    Args:
        image: ``(B, C, D, H, W)`` paired real intensities (channel ``channel`` is
            used as the clustering reference).
        label_map: ``(B, 1, D, H, W)`` integer parent labels.
        n_foreground_clusters: sub-clusters per non-background label.
        background_clusters_range: ``(min, max)`` for the random background split.
        background_label: the label value treated as background.
        n_iters: EM iterations.
        max_fit_voxels: subsample this many voxels to *fit* each EM (the full
            region is still assigned); ``0`` / ``None`` -> use all voxels.
        same_on_batch: draw a single background ``N`` shared across the batch.
        eps: numerical floor.

    Returns:
        ``(fine_labels, generation_labels, output_labels)`` where ``fine_labels``
        is ``(B, 1, D, H, W)`` long, ``generation_labels`` is the sorted list of
        sub-label values, and ``output_labels[i]`` is the parent label that
        ``generation_labels[i]`` merges back to.
    """
    B = image.shape[0]
    device = image.device
    ref = image[:, channel]  # (B, D, H, W)
    # Per-sample label histogram, in one pass. It answers three questions the loop
    # below used to ask one at a time with a host synchronisation each: which labels
    # exist batch-wide, which is the largest, and how many voxels each holds in each
    # sample.
    label_counts = _label_counts_per_sample(label_map)
    totals = label_counts.sum(dim=0)
    parents = totals.nonzero().flatten().tolist()  # sorted, batch-wide
    counts_per_sample = label_counts.tolist()
    parent_to_idx = {p: i for i, p in enumerate(parents)}
    lo, hi = int(background_clusters_range[0]), int(background_clusters_range[1])
    mult = max(hi, int(n_foreground_clusters)) + 1  # collision-free encoding

    # If the configured background label is absent (e.g. a *complete* one-hot whose
    # decoding shifted every label by +1, so the real background is no longer 0),
    # fall back to the largest-area label so it still receives the richer
    # [min, max] background split rather than the 2-cluster foreground split.
    if background_label not in parents and len(parents) > 0:
        background_label = int(totals.argmax())

    bg_n_shared = int(torch.randint(lo, hi + 1, (1,), device=device)) if same_on_batch else None

    fine = torch.zeros_like(label_map)
    for b in range(B):
        lab_b = label_map[b, 0]
        ref_b = ref[b]
        for p in parents:
            pi = parent_to_idx[p]
            cnt = int(counts_per_sample[b][p])
            if cnt == 0:
                continue
            mask = lab_b == p
            if p == background_label:
                k = bg_n_shared if bg_n_shared is not None else int(torch.randint(lo, hi + 1, (1,), device=device))
            else:
                k = int(n_foreground_clusters)
            k = max(1, min(k, cnt))

            x = ref_b[mask]
            if k == 1:
                assign = torch.zeros(cnt, dtype=torch.long, device=device)
            else:
                if max_fit_voxels and cnt > max_fit_voxels:
                    sel = torch.randint(0, cnt, (int(max_fit_voxels),), device=device)
                    x_fit = x[sel]
                else:
                    x_fit = x
                fit = _em_gmm_1d(x_fit, k, n_iters, eps)
                assign = torch.zeros(cnt, dtype=torch.long, device=device) if fit is None else _assign_gmm(x, *fit, eps=eps)
            fine[b, 0][mask] = pi * mult + assign

    gen_values = distinct_values(fine).tolist()
    out_values = [parents[g // mult] for g in gen_values]
    return fine, gen_values, out_values


def random_merge_classes(
    parent_labels: Sequence[int],
    merge_prob: float,
    batch: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Randomly tie sibling EM subregions to a shared generation Gaussian.

    :func:`em_subdivide_labels` gives every subregion its own Gaussian, so an EM
    split is always visible in the synthetic image as an intensity boundary the
    segmentation target does not have. Merging siblings back at generation time
    makes each such boundary show up in only a fraction of the samples, so the
    network can rely neither on it nor on its absence. This is an extension: the
    paper gives every subregion its own Gaussian, i.e. ``merge_prob=0``.

    Each subregion after the first of its parent joins one of that parent's
    existing classes with probability ``merge_prob``, and otherwise opens a new
    one. ``0`` leaves every subregion independent; ``1`` collapses each parent to
    a single Gaussian, as if its label had never been split. Siblings only:
    letting a foreground subregion share the background's Gaussian would hide a
    structure the target still asks for.

    A separate pattern is drawn per sample, matching the per-sample GMM draws.

    Args:
        parent_labels: the parent label of each generation label, in generation
            order (``output_labels`` from :func:`em_subdivide_labels`).
        merge_prob: per-subregion probability of joining a sibling's class.
        batch: number of independent merge patterns to draw.
        device: device for the draws and the returned tensor.

    Returns:
        ``(batch, len(parent_labels))`` class indices, contiguous from 0 in each
        row, for :func:`sample_gmm_parameters`.
    """
    n_labels = len(parent_labels)
    classes = torch.empty((batch, n_labels), dtype=torch.long, device=device)
    for b in range(batch):
        taken: dict[int, list[int]] = {}
        n_classes = 0
        for i, parent in enumerate(parent_labels):
            siblings = taken.setdefault(int(parent), [])
            if siblings and float(torch.rand((), device=device)) < merge_prob:
                classes[b, i] = siblings[int(torch.randint(len(siblings), (1,), device=device))]
            else:
                classes[b, i] = n_classes
                siblings.append(n_classes)
                n_classes += 1
    return classes


# ---------------------------------------------------------------------------
# Label utilities  (lab2im.layers.RandomFlip / ConvertLabels)
# ---------------------------------------------------------------------------
def flip_lr_with_swap(
    label_map: torch.Tensor,
    flip_axis: int,
    label_values: torch.Tensor | None = None,
    n_neutral_labels: int | None = None,
) -> torch.Tensor:
    """Flip the label map along ``flip_axis`` and (optionally) swap L/R labels.

    Port of ``lab2im.layers.RandomFlip(swap_labels=True)``. When
    ``n_neutral_labels`` is provided, ``label_values`` is assumed ordered as
    ``[neutral..., left-hemisphere..., right-hemisphere...]`` and the two
    hemispheres are relabelled into each other after flipping (so anatomical
    left/right stays correct). When ``n_neutral_labels`` is ``None`` a plain flip
    is performed (no relabelling).

    ``flip_axis`` is the spatial axis index in ``(0, 1, 2) = (D, H, W)``.
    """
    flipped = torch.flip(label_map, dims=(2 + flip_axis,))

    if n_neutral_labels is None or label_values is None:
        return flipped

    values = label_values.tolist()
    n_labels = len(values)
    if n_neutral_labels >= n_labels:
        return flipped
    n_sided = (n_labels - n_neutral_labels) // 2
    if n_sided == 0:
        return flipped

    neutral = values[:n_neutral_labels]
    left = values[n_neutral_labels : n_neutral_labels + n_sided]
    right = values[n_neutral_labels + n_sided : n_neutral_labels + 2 * n_sided]
    source = neutral + left + right
    dest = neutral + right + left
    return convert_labels(flipped, source, dest)


def convert_labels(
    label_map: torch.Tensor,
    source_values: Sequence[int],
    dest_values: Sequence[int],
) -> torch.Tensor:
    """Relabel ``label_map`` mapping ``source_values[i] -> dest_values[i]``.

    Port of ``lab2im.layers.ConvertLabels``. Labels not present in
    ``source_values`` are left unchanged.
    """
    device = label_map.device
    source = torch.as_tensor(source_values, dtype=torch.long, device=device)
    dest = torch.as_tensor(dest_values, dtype=torch.long, device=device)
    max_label = int(max(int(label_map.max().item()), int(source.max().item())))
    lut = torch.arange(max_label + 1, dtype=torch.long, device=device)
    lut[source] = dest
    return lut[label_map.clamp(min=0, max=max_label)]
