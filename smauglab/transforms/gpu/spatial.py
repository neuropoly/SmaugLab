from collections.abc import Sequence
from typing import Any, Union

import torch
import torch.nn.functional as F
from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D
from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _tuple_range_reader
from kornia.constants import DataKey, Resample
from kornia.geometry import deg2rad, get_affine_matrix3d, warp_affine3d
from torch import Tensor

try:  # kornia < 0.8.3
    from kornia.utils.helpers import _extract_device_dtype
except ImportError:  # kornia >= 0.8.3 moved it and dropped kornia.utils.helpers
    # A conditional import is a redefinition by construction. The fallback is what
    # the kornia-compat matrix in tests.yml exercises, so it stays.
    from kornia.core.utils import _extract_device_dtype  # type: ignore[no-redef]

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform


# Affine transform
@register(
    aug_id=AugId.AFFINE,
    backend=Backend.GPU,
    group=AugType.GEO,
)
class RandomAffineGPU(RigidAffineAugmentationBase3D):
    r"""Apply affine transformation 3D volumes (5D tensor).

    Based on :class:`kornia.augmentation.RandomAffine3D`.

    The transformation is computed so that the center is kept invariant.

    Args:
        degrees: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal, vertical and
            depthical translations (dx,dy,dz). For example translate=(a, b, c), then
            horizontal shift will be randomly sampled in the range -img_width * a < dx < img_width * a
            vertical shift will be randomly sampled in the range -img_height * b < dy < img_height * b.
            depthical shift will be randomly sampled in the range -img_depth * c < dz < img_depth * c.
            Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If ((a, b), (c, d), (e, f)), the scale is randomly sampled from the range a <= scale_x <= b,
            c <= scale_y <= d, e <= scale_z <= f. Will keep original scale by default.
        shears: Range of degrees to select from.
            If shear is a number, a shear to the 6 facets in the range (-shear, +shear) will be applied.
            If shear is a tuple of 2 values, a shear to the 6 facets in the range (shear[0], shear[1]) will be applied.
            If shear is a tuple of 6 values, a shear to the i-th facet in the range (-shear[i], shear[i])
            will be applied.
            If shear is a tuple of 6 tuples, a shear to the i-th facet in the range (-shear[i, 0], shear[i, 1])
            will be applied.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False). Default: False.

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomAffine3D((15.0, 20.0, 20.0), p=1.0)
        >>> aug(input), aug.transform_matrix
        (tensor([[[[[0.4503, 0.4763, 0.1680],
                   [0.2029, 0.4267, 0.3515],
                   [0.3195, 0.5436, 0.3706]],
        <BLANKLINE>
                  [[0.5255, 0.3508, 0.4858],
                   [0.0795, 0.1689, 0.4220],
                   [0.5306, 0.7234, 0.6879]],
        <BLANKLINE>
                  [[0.2971, 0.2746, 0.3471],
                   [0.4924, 0.4960, 0.6460],
                   [0.3187, 0.4556, 0.7596]]]]]), tensor([[[ 0.9722, -0.0603,  0.2262, -0.1381],
                 [ 0.1131,  0.9669, -0.2286,  0.1486],
                 [-0.2049,  0.2478,  0.9469,  0.0102],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]]))

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomAffine3D((15.0, 20.0, 20.0), p=1.0)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)

    """

    def __init__(
        self,
        degrees: Union[
            Tensor,
            float,
            tuple[float, float],
            tuple[float, float, float],
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        ] = 10,
        translate: Union[Tensor, tuple[float, float, float]] | None = (0.1, 0.1, 0.1),
        scale: Union[Tensor, tuple[float, float], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None = (0.9, 1.1),
        shears: Union[
            Tensor,
            float,
            tuple[float, float],
            tuple[float, float, float, float, float, float],
            tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ],
            None,
        ] = (-10, 10, -10, 10, -10, 10),
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 0.5,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.degrees = degrees
        self.shears = shears
        self.translate = translate
        self.scale = scale

        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}
        self._param_generator = rg.AffineGenerator3D(degrees, translate, scale, shears)

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        transform: Tensor = get_affine_matrix3d(
            params["translations"],
            params["center"],
            params["scale"],
            params["angles"],
            deg2rad(params["sxy"]),
            deg2rad(params["sxz"]),
            deg2rad(params["syx"]),
            deg2rad(params["syz"]),
            deg2rad(params["szx"]),
            deg2rad(params["szy"]),
        ).to(input)
        return transform

    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the transform to be a Tensor. Gotcha {type(transform)}")

        # Ensure align_corners is a boolean (avoid passing None to affine_grid/grid_sample)
        align = flags.get("align_corners", True)
        if align is None:
            align = True

        return warp_affine3d(
            input,
            transform[:, :3, :],
            (input.shape[-3], input.shape[-2], input.shape[-1]),
            flags["resample"].name.lower(),
            align_corners=bool(align),
        )

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.

        """
        # `resample_method` was declared but only *assigned* inside the `if`, so the
        # restore below raised UnboundLocalError whenever flags carried no "resample".
        resample_method: Resample | None = flags.get("resample")
        if resample_method is not None:
            flags["resample"] = Resample.get("nearest")
        output = self.apply_transform(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output


# Low resolution transform
@register(
    aug_id=AugId.LOW_RES,
    backend=Backend.GPU,
    group=AugType.GE,
    force_sequential=True,
)
class RandomLowResTransformGPU(RigidAffineAugmentationBase3D):
    """
    Apply low resolution simulation to 3D volumes (5D tensor).
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = ScaleGenerator3D(scale=scale)

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # input shape: (B, C, D, H, W)
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Expected input to be a Tensor. Got {type(input)}")

        batch_size, C, D, H, W = input.shape

        # params expected to contain 'scale' as a tensor of shape (B, 3)
        if params is None or "scale" not in params:
            raise ValueError("params must contain 'scale' tensor")

        scales = params["scale"]  # shape [B, 3]

        # Only MaskSequentialOpsCustom injects "data_keys" (see gpu/base.py), so a bare
        # `flags["data_keys"]` raised KeyError for every other caller: a standalone call,
        # or RandomChooseXTransformsGPU, which passes the transform's own `flags`.
        data_keys = flags.get("data_keys") or [DataKey.INPUT]
        if data_keys[0] in (DataKey.INPUT, DataKey.IMAGE):
            resample = "trilinear"
        elif data_keys[0] is DataKey.MASK:
            # `apply_transform_mask` no longer routes a mask here, so this branch is
            # reachable only by a direct call. The mode is "nearest-exact", not "nearest":
            # torch's "nearest" is not half-pixel centred (src = floor(dst * scale)) and
            # the down and the up step each drop ~0.5 voxels, so the composed map is
            # out[i] = in[i - 1] almost regardless of the scale factor. That measured a
            # mean edge displacement of +1.16 voxels (sd 0.44) over the shipped 0.5-1.0
            # range against -0.06 for "nearest-exact", while the trilinear image path did
            # not move -- the label-to-image misregistration behind "the foreground grew".
            resample = "nearest-exact"
        else:
            raise ValueError(f"Unsupported data key {data_keys[0]} for RandomLowResTransformGPU. Expected IMAGE or MASK.")

        # Define interpolation modes
        interp_down = resample
        interp_up = resample

        # `scales` lives on the device, so each `float(...)` below blocked the host until
        # the queue drained, three times per sample. One `.tolist()` reads [B, 3] in one go.
        scale_rows = scales.tolist()
        align = False if "linear" in interp_down else None

        # Samples that drew the same target size resample together: one `F.interpolate`
        # over a [k, C, D, H, W] slab instead of k of them. With `same_on_batch` the whole
        # batch is a single group and the loop that remains has one iteration.
        groups: dict[tuple[int, int, int], list[int]] = {}
        for b in range(batch_size):
            sx, sy, sz = scale_rows[b]
            size = (max(1, round(sz * D)), max(1, round(sy * H)), max(1, round(sx * W)))
            groups.setdefault(size, []).append(b)

        # `empty_like`, not `clone`: every row is written below.
        out = torch.empty_like(input)
        for size, members in groups.items():
            # A one-member group is a view, not a gather: with a small batch most groups are
            # singletons, and `index_select`/`index_copy_` would copy in and out for nothing.
            if len(members) == 1:
                b = members[0]
                chunk = F.interpolate(input[b : b + 1], size=size, mode=interp_down, align_corners=align)
                out[b : b + 1] = F.interpolate(chunk, size=(D, H, W), mode=interp_up, align_corners=align)
                continue
            index = torch.as_tensor(members, device=input.device)
            chunk = F.interpolate(input.index_select(0, index), size=size, mode=interp_down, align_corners=align)
            out.index_copy_(0, index, F.interpolate(chunk, size=(D, H, W), mode=interp_up, align_corners=align))

        return out

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Leave the segmentation alone: this simulates resolution, not motion.

        Resampling the image down and back models a thicker slice or a coarser
        acquisition. The anatomy does not move, so the label map must not change --
        nnU-Net's `SimulateLowResolutionTransform` is image-only for the same
        reason, and so is `RandomAcqTransformGPU` below, which is this operation
        restricted to a single axis.

        It is also what the CPU backend already did: `AugId.LOW_RES` maps to nnU-Net's
        image-only `SimulateLowResolutionTransform` there (transforms/cpu/external.py),
        so before this change `low_res` meant two different things depending on
        `Backend`, and any GPU-vs-CPU comparison carried that confound.

        This used to call `apply_transform`, which nearest-resampled the mask along with
        the image. The visible symptom was a foreground count that moved by a few percent,
        but that is not what the damage was: the volume change is unbiased (mean -0.02%,
        sd 0.85% over 200 spheres at the shipped scale range, growing about half the
        time). What was biased was the *position*. Torch's `F.interpolate(mode="nearest")`
        is not half-pixel centred, so the label came back displaced by +1.16 voxels
        (sd 0.44) on every axis while the trilinear image path stayed put -- a systematic
        ~1 voxel label-to-image misregistration on a quarter of the samples, in the same
        direction every time. See `unit_tests/test_resample_alignment.py`.

        The override exists at all only because the class inherits
        `RigidAffineAugmentationBase3D` rather than `ImageOnlyTransform`, so
        `MaskSequentialOpsCustom` routes the mask through it.
        """
        return input


def _choose_axis(batch_size: int, device: torch.device, same_on_batch: bool) -> torch.Tensor:
    """Pick the single axis to act on, per batch element. Returns `[B]` indices.

    Drawn here, from `forward`, rather than in `make_samplers`. kornia calls
    `make_samplers` once and caches the samplers it builds, so an axis picked there was
    fixed for the transform's lifetime -- "degrade a random axis" degraded the *same*
    axis for a whole training run.
    """
    keep = torch.randint(0, 3, (1 if same_on_batch else batch_size,), device=device)
    return keep.expand(batch_size) if same_on_batch else keep


def _keep_one_axis(values: torch.Tensor, keep: torch.Tensor, neutral: float) -> torch.Tensor:
    """Keep column `keep[b]` of a `[B, 3]` draw and set the other two to `neutral`."""
    selected = torch.arange(3, device=values.device).unsqueeze(0) == keep.unsqueeze(1)
    return torch.where(selected, values, torch.full_like(values, neutral))


class ScaleGenerator3D(RandomGeneratorBase):
    def __init__(self, scale: tuple[float, float], one_dim: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.one_dim = one_dim

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        scale = _tuple_range_reader(self.scale, 3, device, dtype)
        self.scalex_sampler = UniformDistribution(scale[0, 0], scale[0, 1], validate_args=False)
        self.scaley_sampler = UniformDistribution(scale[1, 0], scale[1, 1], validate_args=False)
        self.scalez_sampler = UniformDistribution(scale[2, 0], scale[2, 1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]

        _device, _dtype = _extract_device_dtype([self.scalex_sampler, self.scaley_sampler, self.scalez_sampler])

        scalex = _adapted_rsampling((batch_size,), self.scalex_sampler, same_on_batch)
        scaley = _adapted_rsampling((batch_size,), self.scaley_sampler, same_on_batch)
        scalez = _adapted_rsampling((batch_size,), self.scalez_sampler, same_on_batch)
        scale = torch.stack([scalex, scaley, scalez], dim=1)

        if self.one_dim:
            # A scale of 1.0 leaves an axis at full resolution.
            scale = _keep_one_axis(scale, _choose_axis(batch_size, scale.device, same_on_batch), 1.0)

        return {"scale": torch.as_tensor(scale, device=_device, dtype=_dtype)}


# Acquisition transforms
@register(
    aug_id=AugId.ACQ,
    backend=Backend.GPU,
    group=AugType.GE,
)
class RandomAcqTransformGPU(ImageOnlyTransform):
    """
    Randomly lower acquisition along one axes only.
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        same_on_batch: bool = False,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = {"resample": "trilinear"}
        self.apply_to_channel = apply_to_channel
        # one_dim is fixed rather than exposed: this class *is* the single-axis case and
        # RandomLowResTransformGPU the isotropic one. Configurable, either key could do both.
        self._param_generator = ScaleGenerator3D(scale=scale, one_dim=True)

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # input shape: (B, C, D, H, W)
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Expected input to be a Tensor. Got {type(input)}")

        batch_size, C, D, H, W = input.shape

        # params expected to contain 'scale' as a tensor of shape (B, 3)
        if params is None or "scale" not in params:
            raise ValueError("params must contain 'scale' tensor")

        scales = params["scale"]  # shape [B, 3]

        resample = self.flags.get("resample", "trilinear")

        # Define interpolation modes
        interp_down = resample
        interp_up = resample

        # One `.tolist()` instead of three device reads per sample, and one `F.interpolate`
        # per distinct target size -- see the sibling in RandomLowResTransformGPU.
        scale_rows = scales.tolist()
        align = False if "linear" in interp_down else None
        channels = list(self.apply_to_channel)

        groups: dict[tuple[int, int, int], list[int]] = {}
        for b in range(batch_size):
            sx, sy, sz = scale_rows[b]
            size = (max(1, round(sz * D)), max(1, round(sy * H)), max(1, round(sx * W)))
            groups.setdefault(size, []).append(b)

        # Only the selected channels are rewritten, so the untouched ones have to be
        # carried over -- hence a clone rather than an empty tensor.
        out = input.clone()
        for size, members in groups.items():
            # A one-member group goes through views, as in RandomLowResTransformGPU above.
            if len(members) == 1:
                b = members[0]
                for c in channels:
                    chunk = F.interpolate(input[b, c][None, None], size=size, mode=interp_down, align_corners=align)
                    out[b, c] = F.interpolate(chunk, size=(D, H, W), mode=interp_up, align_corners=align)[0, 0]
                continue
            index = torch.as_tensor(members, device=input.device)
            chunk = input.index_select(0, index).index_select(1, torch.as_tensor(channels, device=input.device))
            chunk = F.interpolate(chunk, size=size, mode=interp_down, align_corners=align)
            chunk = F.interpolate(chunk, size=(D, H, W), mode=interp_up, align_corners=align)
            for position, c in enumerate(channels):
                out[index, c] = chunk[:, position]

        return out


# Flip transforms
@register(
    aug_id=AugId.FLIP,
    backend=Backend.GPU,
    group=AugType.GEO,
)
class RandomFlipTransformGPU(RigidAffineAugmentationBase3D):
    """
    Apply low resolution simulation to 3D volumes (5D tensor).
    """

    def __init__(
        self,
        # Both forms are accepted and normalised below; the annotation said `int`
        # while the default was a list and every caller passes a list.
        flip_axis: Union[int, Sequence[int]] = (0,),
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        # normalize flip_axis into a list of ints
        if isinstance(flip_axis, int):
            self.flip_axis = [flip_axis]
        else:
            self.flip_axis = list(flip_axis)

        # generator creates per-batch flip flags for axes (z, y, x)
        self._param_generator = FlipGenerator3D(flip_axis=self.flip_axis)

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # input shape: (B, C, D, H, W)
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Expected input to be a Tensor. Got {type(input)}")

        batch_size, C, D, H, W = input.shape

        # params["flip"] is [B, 3] of 0/1 flags over (z, y, x) from FlipGenerator3D.
        # Reading it is what makes this transform random: the loop below recomputed the
        # same `flip_axis`-derived list for every b and ignored the sampled flags, so every
        # call flipped all configured axes identically -- three seeded calls gave
        # byte-identical output, and the generator's "at least one axis" guarantee was dead.
        flips = params.get("flip")

        # `bool(flips[b, axis])` read one device element at a time, three synchronisations
        # per sample. One `.tolist()` brings the whole [B, 3] flag table over at once.
        flip_rows = None if flips is None else flips.tolist()

        out = input.clone()
        # For each batch element, build list of spatial dims to flip. `input[b]` is
        # [C, D, H, W], so spatial axis i sits at dim 1 + i.
        for b in range(batch_size):
            if flip_rows is None:
                # No sampled flags (a caller invoking apply_transform directly): fall
                # back to flipping every configured axis.
                flip_dims = [1 + axis for axis in range(3) if axis in self.flip_axis]
            else:
                flip_dims = [1 + axis for axis in range(3) if axis in self.flip_axis and flip_rows[b][axis]]

            if len(flip_dims) > 0:
                out[b] = torch.flip(input[b], dims=tuple(flip_dims))

        return out

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.

        """
        output = self.apply_transform(input, params, flags, transform)
        return output


class FlipGenerator3D(RandomGeneratorBase):
    """
    Generate per-batch flip flags for 3 axes (z, y, x).

    Returns a dict with key "flip" and value tensor of shape (B, 3) with 0/1 values.
    Ensures at least one axis is flipped per batch element.
    """

    def __init__(self, flip_axis):
        super().__init__()
        # flip_axis is a list of allowed axes (subset of [0,1,2]).
        if isinstance(flip_axis, int):
            self.flip_axis = [flip_axis]
        else:
            self.flip_axis = list(flip_axis)

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        # use uniform samplers per axis and threshold at 0.5
        # list[Any], not list[UniformDistribution]: kornia's _extract_device_dtype
        # takes a heterogeneous list, and list is invariant.
        self._samplers: list[Any] = [UniformDistribution(0.0, 1.0, validate_args=False) for _ in range(3)]

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]

        _device, _dtype = _extract_device_dtype(self._samplers)

        samples = []
        for s in self._samplers:
            r = _adapted_rsampling((batch_size,), s, same_on_batch)
            samples.append(r)

        flips = torch.stack(samples, dim=1).to(device=_device, dtype=_dtype)
        flips = (flips > 0.5).to(torch.int8)

        # At least one *allowed* axis flipped per batch element. The zero-test has to look
        # at self.flip_axis, not at all three columns: `flips` is sampled over every axis
        # but `apply_transform` only acts on the allowed ones, so a 1 drawn on a disallowed
        # axis satisfied the test while nothing was flipped. With the default flip_axis=(0,)
        # that made RandomFlipTransformGPU(p=1.0) a no-op on 35% of draws.
        if len(self.flip_axis) == 0:
            return {"flip": flips}
        allowed = torch.as_tensor(self.flip_axis, device=flips.device, dtype=torch.long)
        # Which samples drew nothing, and the rescue axis for each, decided for the whole
        # batch at once. The loop this replaces read `flips[b, allowed].sum()` back to the
        # host per sample, then drew a separate scalar for the ones that needed it.
        empty = flips.index_select(1, allowed).sum(dim=1) == 0
        rescue = allowed[torch.randint(low=0, high=len(self.flip_axis), size=(batch_size,), device=flips.device)]
        flips.scatter_(
            1,
            rescue.view(-1, 1),
            torch.where(empty, torch.ones_like(empty, dtype=flips.dtype), flips.gather(1, rescue.view(-1, 1)).squeeze(1)).view(-1, 1),
        )

        return {"flip": flips}


# Crop transform
@register(
    aug_id=AugId.CROP,
    backend=Backend.GPU,
    group=AugType.GEO,
)
class RandomCropTransformGPU(RigidAffineAugmentationBase3D):
    """Restrict the field of view to a random sub-box, filling the rest with `pad_value`.

    The kept box stays at the coordinates it was read from -- this occludes everything
    outside it rather than translating the contents -- so image and mask stay aligned
    voxel for voxel.

    `pad_value` carries more weight than it looks. The volumes reaching this transform
    are z-scored, so 0.0 is the dataset mean -- water, not air -- and a zero fill leaves
    a flat soft-tissue slab where the field of view ended. Everything downstream then
    treats that slab as tissue: PALETTE's foreground test (`> dark_threshold` on the
    min-max normalised volume) keeps it and paints per-cluster texture into it, and the
    edge filters render the one-voxel step at the box faces as the dominant structure in
    the image. `"min"`, the default, fills with the volume's own minimum instead, which
    is what air already is, so the discarded region stays background to every later
    transform. Pass `pad_value=0.0` for the old behaviour.
    """

    def __init__(
        self,
        crop: tuple[float, float] = (1.0, 1.0),
        # A (low, high) range like `crop`, not a per-axis triple: CropGenerator3D
        # feeds it to _tuple_range_reader(..., 3, ...), which broadcasts the range
        # across all three axes. The annotation said triple, the default was a pair.
        pos: tuple[float, float] = (0.0, 1.0),  # Fraction of the pos
        pad_value: float | str = "min",  # "min" = this volume's own minimum, else a literal
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if not (pad_value == "min" or (isinstance(pad_value, (int, float)) and not isinstance(pad_value, bool))):
            raise ValueError(f'pad_value must be "min" or a number. Got {pad_value!r}.')
        self.pad_value = pad_value
        self._param_generator = CropGenerator3D(crop=crop, pos=pos)

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def _fill_values(self, batch: Tensor, flags: dict[str, Any]) -> Tensor:
        """What goes outside the kept box, as a `[B, 1, 1, 1, 1]` per-sample value.

        A mask is padded with 0 -- background -- whatever `pad_value` says: a label
        number is not an intensity, and an argmax over a one-hot volume still has to
        resolve to the background channel out there. The mask pass identifies itself
        through `flags["data_keys"]`, which `MaskSequentialOpsCustom` sets before calling
        `transform_masks` (see gpu/base.py).

        Per-sample and on the device: `float(volume.amin())` once per sample was a
        host synchronisation per sample, for a number that never leaves the GPU.
        """
        shape = (batch.shape[0], *([1] * (batch.dim() - 1)))
        if DataKey.MASK in (flags or {}).get("data_keys", ()):
            return torch.zeros(shape, device=batch.device, dtype=batch.dtype)
        if self.pad_value == "min":
            return batch.amin(dim=tuple(range(1, batch.dim())), keepdim=True)
        return torch.full(shape, float(self.pad_value), device=batch.device, dtype=batch.dtype)

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # input shape: (B, C, D, H, W)
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Expected input to be a Tensor. Got {type(input)}")

        batch_size, C, D, H, W = input.shape

        # params expected to contain 'crop' as a tensor of shape (B, 3)
        if params is None or "crop" not in params or "pos" not in params:
            raise ValueError("params must contain 'crop' and 'pos' tensors")

        crops = params["crop"]  # shape [B, 3]
        pos = params["pos"]  # shape [B, 3]

        # One transfer for the whole batch's draw, then the identical box arithmetic in
        # Python. This read six scalars off the device per sample -- each `float(...)` on a
        # device tensor blocks the host until the queue drains -- and the arithmetic is
        # cheap enough that moving it onto the GPU costs more in launches than it saves.
        draws = torch.stack([crops, pos], dim=1).tolist()  # [B][2][3], each (x, y, z)

        # The discarded region's fill value stays a device tensor -- the class docstring says
        # why it is not 0 -- so the `amin` behind it is one batched reduction, not a host read.
        fills = self._fill_values(input, flags)

        out = torch.empty_like(input)
        for b in range(batch_size):
            (cx, cy, cz), (px, py, pz) = draws[b]

            # interpret crop as fraction of upsampled size to keep
            crop_D = max(1, round(cz * D))
            crop_H = max(1, round(cy * H))
            crop_W = max(1, round(cx * W))

            # top-left-front corner of the box centred on (pz, py, px), clamped inside
            z1 = max(0, min(round(pz * D - crop_D / 2.0), max(0, D - crop_D)))
            y1 = max(0, min(round(py * H - crop_H / 2.0), max(0, H - crop_H)))
            x1 = max(0, min(round(px * W - crop_W / 2.0), max(0, W - crop_W)))

            # Fill, then copy the kept box back over it. The old code built a per-sample
            # canvas and copied that into an `input.clone()`, writing the batch three times.
            out[b] = fills[b]
            out[b][:, z1 : z1 + crop_D, y1 : y1 + crop_H, x1 : x1 + crop_W] = input[b][
                :, z1 : z1 + crop_D, y1 : y1 + crop_H, x1 : x1 + crop_W
            ]

        return out

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are no transformation applied."""
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed.

        Note:
            Convert "resample" arguments to "nearest" by default.

        """
        output = self.apply_transform(input, params, flags, transform)
        return output


class CropGenerator3D(RandomGeneratorBase):
    def __init__(self, crop: tuple[float, float], pos: tuple[float, float], one_dim: bool = False) -> None:
        super().__init__()
        self.crop = crop
        self.pos = pos  # Position of the crop box center, as a fraction of the image dimensions (e.g. 0.5 for centered)
        self.one_dim = one_dim

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        crop = _tuple_range_reader(self.crop, 3, device, dtype)
        self.cropx_sampler = UniformDistribution(crop[0, 0], crop[0, 1], validate_args=False)
        self.cropy_sampler = UniformDistribution(crop[1, 0], crop[1, 1], validate_args=False)
        self.cropz_sampler = UniformDistribution(crop[2, 0], crop[2, 1], validate_args=False)

        pos = _tuple_range_reader(self.pos, 3, device, dtype)
        self.posx_sampler = UniformDistribution(pos[0, 0], pos[0, 1], validate_args=False)
        self.posy_sampler = UniformDistribution(pos[1, 0], pos[1, 1], validate_args=False)
        self.posz_sampler = UniformDistribution(pos[2, 0], pos[2, 1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, torch.Tensor]:
        batch_size = batch_shape[0]

        _device, _dtype = _extract_device_dtype(
            [self.cropx_sampler, self.cropy_sampler, self.cropz_sampler, self.posx_sampler, self.posy_sampler, self.posz_sampler]
        )

        cropx = _adapted_rsampling((batch_size,), self.cropx_sampler, same_on_batch)
        cropy = _adapted_rsampling((batch_size,), self.cropy_sampler, same_on_batch)
        cropz = _adapted_rsampling((batch_size,), self.cropz_sampler, same_on_batch)
        crop = torch.stack([cropx, cropy, cropz], dim=1)

        posx = _adapted_rsampling((batch_size,), self.posx_sampler, same_on_batch)
        posy = _adapted_rsampling((batch_size,), self.posy_sampler, same_on_batch)
        posz = _adapted_rsampling((batch_size,), self.posz_sampler, same_on_batch)
        pos = torch.stack([posx, posy, posz], dim=1)

        if self.one_dim:
            # One axis for both: `make_samplers` drew a separate `dim` for crop and for pos,
            # so the crop could be taken along one axis and placed along another.
            keep = _choose_axis(batch_size, crop.device, same_on_batch)
            # A crop fraction of 1.0 keeps the whole axis, but the *position* is the crop
            # centre as a fraction of the axis, so its neutral value is 0.5. Copying the
            # crop's 1.0 onto it put the box centre on the far edge and left the crop
            # flush against it after clamping.
            crop = _keep_one_axis(crop, keep, 1.0)
            pos = _keep_one_axis(pos, keep, 0.5)

        return {"crop": torch.as_tensor(crop, device=_device, dtype=_dtype), "pos": torch.as_tensor(pos, device=_device, dtype=_dtype)}
