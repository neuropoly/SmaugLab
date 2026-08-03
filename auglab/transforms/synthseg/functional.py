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
used throughout AugLab's GPU transforms.

Spatial conventions
--------------------
* Spatial axes ``(D, H, W)`` map to torch dims ``(2, 3, 4)``. Internally we work
  with voxel coordinates in ``(i, j, k) = (D, H, W)`` order and only convert to
  the ``(x, y, z) = (W, H, D)`` order expected by ``F.grid_sample`` at the very
  end, with ``align_corners=True`` so that integer voxel indices map exactly.
* Affine transforms are applied about the volume centre (standard practice and
  matching AugLab's existing ``RandomAffine3DCustom``), so small rotations /
  scalings keep the anatomy in frame.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

Number = Union[int, float]

__all__ = [
    "to_label_map",
    "infer_label_values",
    "sample_gmm_parameters",
    "labels_to_image_gmm",
    "sample_affine_matrices",
    "random_svf_field",
    "warp_volume",
    "bias_field",
    "intensity_augmentation",
    "blurring_sigma_for_downsampling",
    "gaussian_blur_3d",
    "sample_resolution",
    "mimic_acquisition",
    "em_subdivide_labels",
    "flip_lr_with_swap",
    "convert_labels",
]


# ---------------------------------------------------------------------------
# Label map helpers
# ---------------------------------------------------------------------------
def to_label_map(seg: torch.Tensor) -> torch.Tensor:
    """Coerce a segmentation tensor into a single-channel integer label map.

    Accepts:
      * ``(B, 1, D, H, W)``  -> rounded to integer labels (used directly).
      * ``(B, C, D, H, W)`` one-hot (C > 1) -> argmax + 1, background (all-zero
        across channels) stays 0. This matches AugLab's
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
    prior: Optional[Union[Number, Sequence[Number], torch.Tensor]],
    size: Tuple[int, int],
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
                raise ValueError(
                    f"Prior array has {prior_t.shape[1]} classes, expected {n_classes}."
                )
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
    generation_classes: Optional[Sequence[int]] = None,
    background_label_index: Optional[int] = 0,
    randomise_background: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw per-label Gaussian means/stds for one minibatch.

    Mirrors ``SynthSeg.model_inputs.build_model_inputs``. With the default
    ``prior_means=None`` / ``prior_stds=None`` the *effective* priors are
    ``means ~ U(0, 250)`` and ``stds ~ U(0, 30)`` per class (the code uses
    ``centre=125, range=125`` and ``centre=15, range=15``; the often-quoted
    ``[25, 225]`` / ``[5, 25]`` come from a stale docstring).

    ``generation_classes`` lets several labels share the same Gaussian (e.g. to
    tie left/right homologues). When ``None`` every label is independent.

    Returns ``means, stds`` of shape ``(batch, n_labels, n_channels)``.
    """
    if generation_classes is None:
        classes = torch.arange(n_labels, device=device)
    else:
        classes = torch.as_tensor(generation_classes, device=device, dtype=torch.long)
        if classes.numel() != n_labels:
            raise ValueError("generation_classes must have one entry per generation label.")
    n_classes = int(classes.max().item()) + 1

    means = torch.empty(batch, n_classes, n_channels, device=device)
    stds = torch.empty(batch, n_classes, n_channels, device=device)
    for ch in range(n_channels):
        means[:, :, ch] = _draw_value(
            prior_means, (batch, n_classes), prior_distributions, 125.0, 125.0, device, positive_only=True
        )
        stds[:, :, ch] = _draw_value(
            prior_stds, (batch, n_classes), prior_distributions, 15.0, 15.0, device, positive_only=True
        )

    # Scatter class parameters to per-label parameters.
    means_lab = means[:, classes, :]
    stds_lab = stds[:, classes, :]

    # Background special-casing (build_model_inputs): per subject, 5% pure black,
    # 25% very dark/low-variance, 70% normal draw.
    if randomise_background and background_label_index is not None and 0 <= background_label_index < n_labels:
        for b in range(batch):
            r = float(torch.rand((), device=device))
            if r > 0.95:
                means_lab[b, background_label_index, :] = 0.0
                stds_lab[b, background_label_index, :] = 0.0
            elif r > 0.70:
                means_lab[b, background_label_index, :] = torch.rand(n_channels, device=device) * 15.0
                stds_lab[b, background_label_index, :] = torch.rand(n_channels, device=device) * 5.0
            # else: keep the normal draw

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

    image = torch.empty((B, C, *spatial), device=device, dtype=dtype)
    for b in range(B):
        idx_b = idx[b]
        for ch in range(C):
            mean_map = means[b, :, ch][idx_b]
            std_map = stds[b, :, ch][idx_b]
            image[b, ch] = mean_map + std_map * torch.randn(spatial, device=device, dtype=dtype)
    return image


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


def _identity_grid(shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Voxel-coordinate identity grid in ``(i, j, k)`` order, shape ``(3, D, H, W)``."""
    d, h, w = shape
    zs = torch.arange(d, device=device, dtype=torch.float32)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    ii, jj, kk = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([ii, jj, kk], dim=0)


def _coords_to_grid_sample(coords: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
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
    affine: Optional[torch.Tensor] = None,
    displacement: Optional[torch.Tensor] = None,
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
            centre = torch.tensor(
                [(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0], device=device
            ).view(1, 3, 1)
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
    return F.grid_sample(
        volume, sample_grid, mode=mode, align_corners=True, padding_mode=padding_mode
    )


def _integrate_velocity(velocity: torch.Tensor, int_steps: int = 7) -> torch.Tensor:
    """Scaling-and-squaring integration of a stationary velocity field.

    Port of ``ext.neuron.layers.VecInt`` (``method='ss'``, ``int_steps=7``):
    ``phi = v / 2**N`` then ``phi <- phi + warp(phi, phi)`` repeated ``N`` times,
    yielding a diffeomorphic displacement field. ``velocity`` and the returned
    displacement are ``(B, 3, D, H, W)`` in voxel units.
    """
    disp = velocity / (2 ** int_steps)
    for _ in range(int_steps):
        disp = disp + warp_volume(disp, displacement=disp, interp="linear", padding_mode="border")
    return disp


def random_svf_field(
    batch: int,
    shape: Tuple[int, int, int],
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

    small = [max(2, int(math.ceil(s * nonlin_scale))) for s in shape]
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
    device = image.device
    small = [max(2, int(math.ceil(s * bias_scale))) for s in (D, H, W)]
    std = torch.rand(B, 1, 1, 1, 1, device=device) * bias_field_std
    field = torch.randn(B, C, *small, device=device) * std
    field = F.interpolate(field, size=(D, H, W), mode="trilinear", align_corners=True)
    return image * torch.exp(field)


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
    thickness: Optional[torch.Tensor] = None,
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


def _gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], device=device)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


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
    B, C, D, H, W = image.shape
    device = image.device
    sigma = sigma.clone().float()
    if blur_range and blur_range > 1.0:
        jitter = (1.0 / blur_range) + torch.rand(3, device=device) * (blur_range - 1.0 / blur_range)
        sigma = sigma * jitter

    out = image
    for axis, s in enumerate(sigma.tolist()):
        if s <= 0:
            continue
        kernel = _gaussian_kernel1d(s, device)
        ksize = kernel.numel()
        pad = ksize // 2
        # shape the separable kernel for conv3d along the given spatial axis
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = ksize
        weight = kernel.view(shape).repeat(C, 1, 1, 1, 1)
        padding = [0, 0, 0]
        padding[axis] = pad
        pad_full = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        out = F.pad(out, pad_full, mode="reflect")
        out = F.conv3d(out, weight, groups=C)
    return out


def sample_resolution(
    min_res: torch.Tensor,
    max_res_iso: float = 4.0,
    max_res_aniso: float = 8.0,
    prob_iso: float = 0.1,
    prob_min: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    output_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Downsample to a target resolution, then resample to the output grid.

    Port of ``lab2im.layers.MimicAcquisition``: nearest-neighbour downsampling to
    the sampled ``downsample_res`` grid (the partial-volume step) followed by
    trilinear resampling to ``output_shape``.
    """
    B, C, D, H, W = image.shape
    in_shape = (D, H, W)
    factor = (current_res / downsample_res).tolist()
    down_shape = [max(1, int(round(in_shape[i] * factor[i]))) for i in range(3)]
    x = F.interpolate(image, size=down_shape, mode="nearest")
    x = F.interpolate(x, size=tuple(output_shape), mode="trilinear", align_corners=True)
    return x


# ---------------------------------------------------------------------------
# EM label completion for sparse label maps  (SynthSeg paper, Sec. 5.4)
# ---------------------------------------------------------------------------
def _em_gmm_1d(
    x_fit: torch.Tensor, n_components: int, n_iters: int, eps: float
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fit a 1D Gaussian mixture by Expectation-Maximization.

    ``x_fit`` is a 1D tensor of intensities. Returns ``(means, vars, weights)``
    each of shape ``(K,)`` (``K = min(n_components, len(x_fit))``), or ``None`` if
    ``x_fit`` is empty. Responsibilities use a numerically-stable log-sum-exp.
    """
    n = x_fit.numel()
    if n == 0:
        return None
    k = max(1, min(int(n_components), n))
    device = x_fit.device
    xmin, xmax = float(x_fit.min()), float(x_fit.max())
    means = torch.linspace(xmin, xmax, k, device=device)
    if xmax <= xmin:  # degenerate (constant region): spread the means slightly
        means = means + torch.arange(k, device=device, dtype=means.dtype) * eps
    var = x_fit.var(unbiased=False).clamp_min(eps).repeat(k)
    weights = torch.full((k,), 1.0 / k, device=device)

    x = x_fit.view(n, 1)
    log2pi = math.log(2.0 * math.pi)
    for _ in range(int(n_iters)):
        logp = (
            torch.log(weights.clamp_min(eps)).view(1, k)
            - 0.5 * (log2pi + torch.log(var).view(1, k))
            - 0.5 * (x - means.view(1, k)) ** 2 / var.view(1, k)
        )
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
        resp = logp.exp()                       # (n, k)
        nk = resp.sum(0).clamp_min(eps)         # (k,)
        weights = nk / n
        means = (resp * x).sum(0) / nk
        var = (resp * (x - means.view(1, k)) ** 2).sum(0) / nk
        var = var.clamp_min(eps)
    return means, var, weights


def _assign_gmm(
    x_full: torch.Tensor, means: torch.Tensor, var: torch.Tensor, weights: torch.Tensor,
    eps: float, chunk: int = 2_000_000,
) -> torch.Tensor:
    """Hard-assign each value in ``x_full`` to its most likely mixture component."""
    n = x_full.numel()
    k = means.numel()
    out = torch.empty(n, dtype=torch.long, device=x_full.device)
    logw = torch.log(weights.clamp_min(eps)).view(1, k)
    half_logvar = 0.5 * torch.log(var).view(1, k)
    m = means.view(1, k)
    v = var.view(1, k)
    for s in range(0, n, chunk):
        xc = x_full[s:s + chunk].view(-1, 1)
        logp = logw - half_logvar - 0.5 * (xc - m) ** 2 / v
        out[s:s + chunk] = logp.argmax(dim=1)
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
) -> Tuple[torch.Tensor, List[int], List[int]]:
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
    ref = image[:, channel]                                   # (B, D, H, W)
    parents = torch.unique(label_map).long().tolist()         # sorted, batch-wide
    parent_to_idx = {p: i for i, p in enumerate(parents)}
    lo, hi = int(background_clusters_range[0]), int(background_clusters_range[1])
    mult = max(hi, int(n_foreground_clusters)) + 1            # collision-free encoding

    # If the configured background label is absent (e.g. a *complete* one-hot whose
    # decoding shifted every label by +1, so the real background is no longer 0),
    # fall back to the largest-area label so it still receives the richer
    # [min, max] background split rather than the 2-cluster foreground split.
    if background_label not in parents and len(parents) > 0:
        background_label = int(torch.bincount(label_map.flatten().clamp_min(0)).argmax())

    bg_n_shared = int(torch.randint(lo, hi + 1, (1,), device=device)) if same_on_batch else None

    fine = torch.zeros_like(label_map)
    for b in range(B):
        lab_b = label_map[b, 0]
        ref_b = ref[b]
        for p in parents:
            pi = parent_to_idx[p]
            mask = lab_b == p
            cnt = int(mask.sum())
            if cnt == 0:
                continue
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
                assign = (
                    torch.zeros(cnt, dtype=torch.long, device=device)
                    if fit is None else _assign_gmm(x, *fit, eps=eps)
                )
            fine[b, 0][mask] = pi * mult + assign

    gen_values = torch.unique(fine).long().tolist()
    out_values = [parents[g // mult] for g in gen_values]
    return fine, gen_values, out_values


# ---------------------------------------------------------------------------
# Label utilities  (lab2im.layers.RandomFlip / ConvertLabels)
# ---------------------------------------------------------------------------
def flip_lr_with_swap(
    label_map: torch.Tensor,
    flip_axis: int,
    label_values: Optional[torch.Tensor] = None,
    n_neutral_labels: Optional[int] = None,
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
    left = values[n_neutral_labels:n_neutral_labels + n_sided]
    right = values[n_neutral_labels + n_sided:n_neutral_labels + 2 * n_sided]
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
