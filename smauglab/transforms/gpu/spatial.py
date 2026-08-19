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
    order=20,
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
    order=200,
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
        # `flags["data_keys"]` raised KeyError for every other caller -- calling this
        # transform standalone, or from inside RandomChooseXTransformsGPU, which passes
        # the transform's own `flags`. Defaulting to IMAGE is what those callers mean.
        data_keys = flags.get("data_keys") or [DataKey.INPUT]
        if data_keys[0] in (DataKey.INPUT, DataKey.IMAGE):
            resample = "trilinear"
        elif data_keys[0] is DataKey.MASK:
            resample = "nearest"
        else:
            raise ValueError(f"Unsupported data key {data_keys[0]} for RandomLowResTransformGPU. Expected IMAGE or MASK.")

        # Define interpolation modes
        interp_down = resample
        interp_up = resample

        # Process per-channel and per-batch element
        out = input.clone()

        for b in range(batch_size):
            x = input[b]  # [C, D, H, W]

            sx, sy, sz = scales[b]
            # compute downsampled size
            down_D = max(1, round(float(sz) * D))
            down_H = max(1, round(float(sy) * H))
            down_W = max(1, round(float(sx) * W))

            # downsample
            x_down = F.interpolate(
                x.unsqueeze(0),
                size=(down_D, down_H, down_W),
                mode=interp_down,
                align_corners=False if "linear" in interp_down else None,
            )

            # upsample back to original resolution (keep as 4D tensor [1,1,D,H,W])
            x_up = F.interpolate(
                x_down,
                size=(D, H, W),
                mode=interp_up,
                align_corners=False if "linear" in interp_up else None,
            ).squeeze(0)  # [C, D, H, W]

            out[b] = x_up

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
    order=210,
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
        # one_dim is fixed rather than exposed: this class *is* the single-axis case,
        # and RandomLowResTransformGPU is the isotropic one. Leaving it configurable
        # meant two config keys could each produce either behaviour.
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

        # Process per-channel and per-batch element
        out = input.clone()
        for b in range(batch_size):
            # start from the original per-sample tensor so we only overwrite selected channels
            canvas = input[b].clone()

            for c in self.apply_to_channel:
                x = input[b, c]  # [D, H, W]

                sx, sy, sz = scales[b]
                # compute downsampled size
                down_D = max(1, round(float(sz) * D))
                down_H = max(1, round(float(sy) * H))
                down_W = max(1, round(float(sx) * W))

                # downsample
                x_down = F.interpolate(
                    x.unsqueeze(0).unsqueeze(0),
                    size=(down_D, down_H, down_W),
                    mode=interp_down,
                    align_corners=False if "linear" in interp_down else None,
                )

                # upsample back to original resolution
                x_up = (
                    F.interpolate(
                        x_down,
                        size=(D, H, W),
                        mode=interp_up,
                        align_corners=False if "linear" in interp_up else None,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )  # [D, H, W]

                # place patch back into the canvas for the correct channel only
                canvas[c] = x_up

            out[b] = canvas

        return out


# Flip transforms
@register(
    aug_id=AugId.FLIP,
    backend=Backend.GPU,
    group=AugType.GEO,
    order=10,
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

        # params["flip"] is [B, 3] of 0/1 flags over (z, y, x), produced by
        # FlipGenerator3D. Reading it is what makes this transform random: the loop
        # below used to recompute the same `flip_axis`-derived list for every b and
        # ignore the sampled flags entirely, so every call flipped all configured axes
        # identically -- three seeded calls gave byte-identical output, and the
        # generator (including its "at least one axis" guarantee) was dead code.
        flips = params.get("flip")

        out = input.clone()
        # For each batch element, build list of spatial dims to flip. `input[b]` is
        # [C, D, H, W], so spatial axis i sits at dim 1 + i.
        for b in range(batch_size):
            if flips is None:
                # No sampled flags (a caller invoking apply_transform directly): fall
                # back to flipping every configured axis.
                flip_dims = [1 + axis for axis in range(3) if axis in self.flip_axis]
            else:
                flip_dims = [1 + axis for axis in range(3) if axis in self.flip_axis and bool(flips[b, axis])]

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

        # ensure at least one flip per batch element (choose randomly among allowed axes)
        for b in range(batch_size):
            if flips[b].sum() == 0:
                # pick one allowed axis at random
                if len(self.flip_axis) == 0:
                    # nothing to flip
                    continue
                choice = int(torch.randint(low=0, high=len(self.flip_axis), size=(1,)).item())
                axis = int(self.flip_axis[choice])
                flips[b, axis] = 1

        return {"flip": flips}


# Crop transform
@register(
    aug_id=AugId.CROP,
    backend=Backend.GPU,
    group=AugType.GEO,
    order=220,
)
class RandomCropTransformGPU(RigidAffineAugmentationBase3D):
    """
    Apply low resolution simulation to 3D volumes (5D tensor).
    """

    def __init__(
        self,
        crop: tuple[float, float] = (1.0, 1.0),
        # A (low, high) range like `crop`, not a per-axis triple: CropGenerator3D
        # feeds it to _tuple_range_reader(..., 3, ...), which broadcasts the range
        # across all three axes. The annotation said triple, the default was a pair.
        pos: tuple[float, float] = (0.0, 1.0),  # Fraction of the pos
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = CropGenerator3D(crop=crop, pos=pos)

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

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

        # Process per-channel and per-batch element
        out = input.clone()

        for b in range(batch_size):
            x = input[b]  # [C, D, H, W]

            # determine crop fraction and crop size on the image
            cx, cy, cz = crops[b]
            # interpret crop as fraction of upsampled size to keep
            crop_D = max(1, round(float(cz) * D))
            crop_H = max(1, round(float(cy) * H))
            crop_W = max(1, round(float(cx) * W))

            # determine pos fraction of the image
            px, py, pz = pos[b]

            # center position
            center_z = float(pz) * D
            center_y = float(py) * H
            center_x = float(px) * W

            # choose top-left-front corner
            start_z = round(center_z - crop_D / 2.0)
            start_y = round(center_y - crop_H / 2.0)
            start_x = round(center_x - crop_W / 2.0)

            # clamp to valid limits
            max_z = max(0, D - crop_D)
            max_y = max(0, H - crop_H)
            max_x = max(0, W - crop_W)

            z1 = max(0, min(start_z, max_z))
            y1 = max(0, min(start_y, max_y))
            x1 = max(0, min(start_x, max_x))

            z2 = z1 + crop_D
            y2 = y1 + crop_H
            x2 = x1 + crop_W

            patch = x[:, z1:z2, y1:y2, x1:x2]

            # place patch back into a full-resolution canvas of the original size (zeros elsewhere)
            canvas = torch.zeros((C, D, H, W), dtype=patch.dtype, device=patch.device)
            canvas[:, z1:z2, y1:y2, x1:x2] = patch

            out[b] = canvas

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
            # One axis for both: `make_samplers` drew a separate `dim` for crop and for
            # pos, so the crop could be taken along one axis while the position that
            # placed it was randomised along another.
            keep = _choose_axis(batch_size, crop.device, same_on_batch)
            # A crop fraction of 1.0 keeps the whole axis. The *position*, though, is
            # the crop centre as a fraction of the axis, so its neutral value is 0.5
            # (centred) -- the previous code copied the crop's 1.0 onto it, which put
            # the box centre on the far edge and left the crop flush against it after
            # clamping.
            crop = _keep_one_axis(crop, keep, 1.0)
            pos = _keep_one_axis(pos, keep, 0.5)

        return {"crop": torch.as_tensor(crop, device=_device, dtype=_dtype), "pos": torch.as_tensor(pos, device=_device, dtype=_dtype)}
