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
    from kornia.core.utils import _extract_device_dtype

from smauglab.transforms.gpu.base import ImageOnlyTransform


# Affine transform
class RandomAffine3DCustom(RigidAffineAugmentationBase3D):
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
        ],
        translate: Union[Tensor, tuple[float, float, float]] | None = None,
        scale: Union[Tensor, tuple[float, float], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None = None,
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
        ] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 0.5,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
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
        resample_method: Resample | None
        if "resample" in flags:
            resample_method = flags["resample"]
            flags["resample"] = Resample.get("nearest")
        output = self.apply_transform(input, params, flags, transform)
        if resample_method is not None:
            flags["resample"] = resample_method
        return output


# Low resolution transform
class RandomLowResTransformGPU(RigidAffineAugmentationBase3D):
    """
    Apply low resolution simulation to 3D volumes (5D tensor).
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
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

        if flags["data_keys"][0] is DataKey.IMAGE:
            resample = "trilinear"
        elif flags["data_keys"][0] is DataKey.MASK:
            resample = "nearest"
        else:
            raise ValueError(f"Unsupported data key {flags['data_keys'][0]} for RandomLowResTransformGPU. Expected IMAGE or MASK.")

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


class ScaleGenerator3D(RandomGeneratorBase):
    def __init__(self, scale: tuple[float, float], one_dim: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.one_dim = one_dim

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        scale = _tuple_range_reader(self.scale, 3, device, dtype)
        if self.one_dim:
            # Pick a random dimension to apply scaling
            dim = torch.randint(0, 3, (1,)).item()
            for i in range(3):
                if i != dim:
                    scale[i, 0] = 1.0
                    scale[i, 1] = 1.0
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

        return {"scale": torch.as_tensor(scale, device=_device, dtype=_dtype)}


# Acquisition transforms
class RandomAcqTransformGPU(ImageOnlyTransform):
    """
    Randomly lower acquisition along one axes only.
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.3, 1.0),
        one_dim: bool = False,
        same_on_batch: bool = False,
        apply_to_channel: list[int] | None = None,  # Apply to first channel by default
        p: float = 1.0,
        keepdim: bool = True,
        **kwargs,
    ) -> None:
        if apply_to_channel is None:
            apply_to_channel = [0]
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.flags = {"resample": "trilinear"}
        self.apply_to_channel = apply_to_channel
        self._param_generator = ScaleGenerator3D(scale=scale, one_dim=one_dim)

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
class RandomFlipTransformGPU(RigidAffineAugmentationBase3D):
    """
    Apply low resolution simulation to 3D volumes (5D tensor).
    """

    def __init__(
        self,
        flip_axis: int = [0, 1, 2],
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
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

        # Expect params to contain 'flip' tensor of shape [B, 3] with 0/1 values
        # flips = None
        # if params is not None and 'flip' in params:
        #    flips = params['flip']

        out = input.clone()
        # For each batch element, build list of spatial dims to flip (D,H,W -> dims 2,3,4)
        for b in range(batch_size):
            # fb expected as length-3 tensor for (z,y,x)
            flip_dims = [1 + axis for axis in range(3) if axis in self.flip_axis]

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
        self._samplers = [UniformDistribution(0.0, 1.0, validate_args=False) for _ in range(3)]

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
class RandomCropTransformGPU(RigidAffineAugmentationBase3D):
    """
    Apply low resolution simulation to 3D volumes (5D tensor).
    """

    def __init__(
        self,
        crop: tuple[float, float] = (1.0, 1.0),
        pos: tuple[float, float, float] = (0.5, 1),  # Fraction of the pos
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
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
        if self.one_dim:
            # Pick a random dimension to apply cropping
            dim = torch.randint(0, 3, (1,)).item()
            for i in range(3):
                if i != dim:
                    crop[i, 0] = 1.0
                    crop[i, 1] = 1.0
        self.cropx_sampler = UniformDistribution(crop[0, 0], crop[0, 1], validate_args=False)
        self.cropy_sampler = UniformDistribution(crop[1, 0], crop[1, 1], validate_args=False)
        self.cropz_sampler = UniformDistribution(crop[2, 0], crop[2, 1], validate_args=False)

        pos = _tuple_range_reader(self.pos, 3, device, dtype)
        if self.one_dim:
            # Pick a random dimension to apply cropping
            dim = torch.randint(0, 3, (1,)).item()
            for i in range(3):
                if i != dim:
                    pos[i, 0] = 1.0
                    pos[i, 1] = 1.0
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

        return {"crop": torch.as_tensor(crop, device=_device, dtype=_dtype), "pos": torch.as_tensor(pos, device=_device, dtype=_dtype)}
