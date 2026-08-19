import math
import random
from collections.abc import Callable, Sequence
from typing import Any, Union

import torch
import torchvision.transforms._functional_tensor as F_t
from torch import Tensor
from torch.nn import functional as F

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform


def _choose_region_mode(p_in: float, p_out: float, seg_mask: torch.Tensor | None) -> str:  # noqa: ARG001 -- seg_mask kept for signature symmetry with _apply_region_mode
    """Sample where to apply the transform: 'in', 'out', or 'all'.

    - p_in, p_out are probabilities in [0,1].
    - If both probs are 0, or both fire at once, return 'all'.
    - seg_mask is accepted but unused here; _apply_region_mode treats a None
      mask as 'all' regardless of the mode chosen.
    """
    p_in = float(max(0.0, min(1.0, p_in)))
    p_out = float(max(0.0, min(1.0, p_out)))
    in_bool = torch.rand(()) < p_in
    out_bool = torch.rand(()) < p_out
    if in_bool and not out_bool:
        return "in"
    if out_bool and not in_bool:
        return "out"
    return "all"


def _apply_region_mode(
    orig: torch.Tensor,
    transformed: torch.Tensor,
    seg_mask: torch.Tensor | None,
    mode: str,
    normalize: bool = False,
    mix_in_out: bool = False,
) -> torch.Tensor:
    """Blend transformed with orig based on region selection mode.

    - mode 'all': return transformed
    - mode 'in': apply transform inside seg, keep orig outside
    - mode 'out': apply  transform outside seg, keep orig inside

    mix_in_out: if True, randomly apply transform to some of the segmentation, not all.
    """
    if seg_mask is None or mode == "all":
        return transformed

    # Rescale transformed based on min max orig
    # Needed due to the important change in the image
    if orig.dim() == 4:
        if normalize:
            orig_min = torch.amin(orig, dim=tuple(range(1, orig.dim())), keepdim=True)
            orig_max = torch.amax(orig, dim=tuple(range(1, orig.dim())), keepdim=True)
            transformed_min = torch.amin(transformed, dim=tuple(range(1, transformed.dim())), keepdim=True)
            transformed_max = torch.amax(transformed, dim=tuple(range(1, transformed.dim())), keepdim=True)
            transformed = (transformed - transformed_min) / (transformed_max - transformed_min + 1e-8) * (orig_max - orig_min) + orig_min

        m = seg_mask.to(transformed.dtype).clone()
        if mix_in_out:
            for i in range(seg_mask.shape[0]):
                # Create a tensor with random one and zero

                o = torch.randint(0, 2, (seg_mask.shape[1],), device=seg_mask.device, dtype=seg_mask.dtype)
                m[i] = m[i] * o.view(-1, 1, 1, 1)  # Broadcasting o to match the dimensions of m

        m = torch.argmax(m, dim=1) > 0
        m = m.to(transformed.dtype)
        if mode == "out":
            m = 1.0 - m

    elif orig.dim() == 3:
        if normalize:
            orig_min = torch.amin(orig)
            orig_max = torch.amax(orig)
            transformed_min = torch.amin(transformed)
            transformed_max = torch.amax(transformed)
            transformed = (transformed - transformed_min) / (transformed_max - transformed_min + 1e-8) * (orig_max - orig_min) + orig_min

        m = seg_mask.to(transformed.dtype).clone()
        if mix_in_out:
            # Create a tensor with random one and zero
            o = torch.randint(0, 2, (seg_mask.shape[0],), device=seg_mask.device, dtype=seg_mask.dtype)
            m = m * o.view(-1, 1, 1, 1)  # Broadcasting o to match the dimensions of m
        m = torch.argmax(m, dim=0) > 0
        m = m.to(transformed.dtype)
        if mode == "out":
            m = 1.0 - m

    else:
        raise ValueError(f"Only 4D and 3D images are supported. Got {orig.dim()}D.")

    return m * transformed + (1.0 - m) * orig


## Convolution transform
class _RandomConvBaseGPU(ImageOnlyTransform):
    """Apply convolution to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.
    Based on https://docs.pytorch.org/vision/0.9/transforms.html#torchvision.transforms.GaussianBlur

    Args:
        kernel_type (str): One of 'Laplace', 'Scharr', 'GaussianBlur', 'UnsharpMask', 'RandConv'.
        apply_to_channel (list of int): Channel indices to convolve. Default is [0].
        absolute (bool): If True, take the absolute value of the result. Scharr only.
        sigma (float): Gaussian width. GaussianBlur and UnsharpMask only.
        unsharp_amount (float): Strength of the unsharp mask. UnsharpMask only.
        kernel_sizes (list of int): Multi-scale kernel sizes to draw from. RandConv only.
        mix_prob (float): Probability of blending the result back with the original.
        retain_stats (bool): If True, restore the original mean and std afterwards.

    Returns:
        Tensor: Convolved version of the input image.

    """

    def __init__(
        self,
        kernel_type: str = "Laplace",
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        same_on_batch: bool = False,
        retain_stats: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        # Kernel-specific. These used to be read out of **kwargs, which meant they
        # were invisible to `inspect.signature` and a typo in a config silently
        # selected the default instead. Defaults here are the historical
        # kwargs.get() ones, so behaviour is unchanged.
        absolute: bool = False,
        sigma: float = 1.0,
        unsharp_amount: float = 1.0,
        kernel_sizes: Sequence[int] = (1, 3, 5, 7),
        mix_prob: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if kernel_type not in ["Laplace", "Scharr", "GaussianBlur", "UnsharpMask", "RandConv"]:
            raise NotImplementedError('Currently only "Laplace", "Scharr", "GaussianBlur", "UnsharpMask" and "RandConv" are supported.')
        else:
            self.kernel_type = kernel_type
        self.apply_to_channel = apply_to_channel
        self.absolute = absolute
        self.sigma = sigma
        self.retain_stats = retain_stats
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out
        self.unsharp_amount = unsharp_amount
        self.kernel_sizes = kernel_sizes
        self.mix_prob = mix_prob

    def get_kernel(self, device: torch.device) -> Union[Tensor, list[Tensor]]:
        # Scharr is the odd one out: it returns the three directional kernels as a
        # list, which apply_transform convolves separately and sums. Every other
        # kernel type returns a single tensor.
        kernel: Union[Tensor, list[Tensor]]
        if self.kernel_type == "Laplace":
            kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32, device=device)
            kernel[1, 1, 1] = 26.0
        elif self.kernel_type == "Scharr":
            kernel_x = torch.tensor(
                [
                    [[9, 0, -9], [30, 0, -30], [9, 0, -9]],
                    [[30, 0, -30], [100, 0, -100], [30, 0, -30]],
                    [[9, 0, -9], [30, 0, -30], [9, 0, -9]],
                ],
                dtype=torch.float32,
                device=device,
            )

            kernel_y = torch.tensor(
                [
                    [[9, 30, 9], [0, 0, 0], [-9, -30, -9]],
                    [[30, 100, 30], [0, 0, 0], [-30, -100, -30]],
                    [[9, 30, 9], [0, 0, 0], [-9, -30, -9]],
                ],
                dtype=torch.float32,
                device=device,
            )

            kernel_z = torch.tensor(
                [
                    [[9, 30, 9], [30, 100, 30], [9, 30, 9]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]],
                ],
                dtype=torch.float32,
                device=device,
            )
            kernel = [kernel_x, kernel_y, kernel_z]
        elif self.kernel_type == "GaussianBlur":
            sigma = torch.rand(3, device=device) * self.sigma
            kernel_size = 3
            kernel = get_gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        elif self.kernel_type == "UnsharpMask":
            # For unsharp masking we use a Gaussian blur kernel; amount is applied in apply_transform.
            sigma = torch.rand(3, device=device) * self.sigma
            kernel_size = 3
            kernel = get_gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        elif self.kernel_type == "RandConv":
            # choose random odd kernel size e.g. [1,3,5,7]
            k = int(random.choice(self.kernel_sizes))  # define kernel_sizes in __init__

            std = 1.0 / math.sqrt(k * k)
            kernel = torch.randn((k, k, k), device=device) * std  # for 3D
        else:
            raise NotImplementedError("Kernel type not implemented.")
        return kernel

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # Initialize kernel
        kernel = self.get_kernel(device=input.device)

        # Load segmentation
        seg_mask = params.get("seg")

        # Apply convolution
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()

            if self.retain_stats:
                reduce_dims = tuple(range(1, channel_data.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = channel_data.mean(dim=reduce_dims)
                orig_stds = channel_data.std(dim=reduce_dims)

            # The asserts below restate what get_kernel guarantees per kernel_type:
            # only Scharr yields a list, and only its branch iterates.
            if self.kernel_type in ["Laplace", "GaussianBlur"]:
                assert isinstance(kernel, Tensor)
                x = apply_convolution(channel_data, kernel, dim=3)
            elif self.kernel_type == "UnsharpMask":
                # blur selected channel, compute mask and add scaled mask back Isharp​=I+α(I−G​∗I)
                assert isinstance(kernel, Tensor)
                blurred = apply_convolution(channel_data, kernel, dim=3)
                mask = channel_data - blurred
                unsharp_amount = torch.rand(1, device=input.device) * self.unsharp_amount
                x = channel_data + unsharp_amount * mask
            elif self.kernel_type == "Scharr":
                tot_ = torch.zeros_like(channel_data, device=input.device)
                for k in kernel:
                    if self.absolute:
                        tot_ += torch.abs(apply_convolution(channel_data, k, dim=3))
                    else:
                        tot_ += apply_convolution(channel_data, k, dim=3)
                x = tot_
            elif self.kernel_type == "RandConv":
                # RandConv kernels are per-sample, per-call
                out = []
                for b in range(channel_data.shape[0]):
                    kernel = self.get_kernel(device=input.device)
                    assert isinstance(kernel, Tensor)

                    conv = apply_convolution(channel_data[b : b + 1], kernel, dim=3).squeeze(0)

                    out.append(conv)

                x = torch.stack(out, dim=0)

            # Mix with original based on mix_prob
            if torch.rand(1).item() < self.mix_prob:
                alpha = torch.rand(1, device=input.device)
                x = alpha * orig + (1 - alpha) * x

            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, x.dim()))
                new_mean = x.mean(dim=reduce_dims)  # [N]
                new_std = x.std(dim=reduce_dims)  # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [x.shape[0]] + [1] * (x.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                x = (x - nm) / (ns + eps) * os + om

            # Apply region selection
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__} with kernel={self.kernel_type}", flush=True)
                continue
            input[:, c] = x

        return input


# One class per convolution kernel.
#
# These used to be a single `kernel_type=` argument on the base, which meant four
# different augmentations shared one config key and every config had to repeat the
# kernel name redundantly. A class each keeps the config key 1:1 with the class,
# lets each expose only the parameters its kernel actually reads, and makes the
# CPU/GPU coverage matrix able to tell them apart.
#
# Defaults below are the values the old `_build_transforms` ladder passed for that
# kernel, NOT the base class defaults -- that is what keeps behaviour identical once
# the ladder is gone.


@register(
    aug_id=AugId.LAPLACE,
    backend=Backend.GPU,
    group=AugType.TA,
    order=115,
)
class RandomLaplaceGPU(_RandomConvBaseGPU):
    """Laplacian edge enhancement."""

    def __init__(
        self,
        absolute: bool = False,
        mix_prob: float = 0.0,
        apply_to_channel: Sequence[int] = (0,),
        same_on_batch: bool = False,
        retain_stats: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            kernel_type="Laplace",
            absolute=absolute,
            mix_prob=mix_prob,
            apply_to_channel=apply_to_channel,
            same_on_batch=same_on_batch,
            retain_stats=retain_stats,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


@register(
    aug_id=AugId.SCHARR,
    backend=Backend.GPU,
    group=AugType.TA,
    order=90,
)
class RandomScharrGPU(_RandomConvBaseGPU):
    """Scharr gradient-magnitude edge filter."""

    def __init__(
        self,
        absolute: bool = True,
        retain_stats: bool = True,
        mix_prob: float = 0.0,
        apply_to_channel: Sequence[int] = (0,),
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            kernel_type="Scharr",
            absolute=absolute,
            retain_stats=retain_stats,
            mix_prob=mix_prob,
            apply_to_channel=apply_to_channel,
            same_on_batch=same_on_batch,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


@register(
    aug_id=AugId.GAUSSIAN_BLUR,
    backend=Backend.GPU,
    group=AugType.GE,
    order=140,
)
class RandomGaussianBlurGPU(_RandomConvBaseGPU):
    """Gaussian blur via separable convolution."""

    def __init__(
        self,
        sigma: float = 1.0,
        apply_to_channel: Sequence[int] = (0,),
        same_on_batch: bool = False,
        retain_stats: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        mix_prob: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            kernel_type="GaussianBlur",
            sigma=sigma,
            apply_to_channel=apply_to_channel,
            same_on_batch=same_on_batch,
            retain_stats=retain_stats,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            mix_prob=mix_prob,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


@register(
    aug_id=AugId.UNSHARP_MASK,
    backend=Backend.GPU,
    group=AugType.TA,
    order=100,
)
class RandomUnsharpMaskGPU(_RandomConvBaseGPU):
    """Unsharp masking: sharpen by subtracting a blurred copy."""

    def __init__(
        self,
        sigma: float = 1.0,
        unsharp_amount: float = 1.5,
        mix_prob: float = 0.0,
        apply_to_channel: Sequence[int] = (0,),
        same_on_batch: bool = False,
        retain_stats: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            kernel_type="UnsharpMask",
            sigma=sigma,
            unsharp_amount=unsharp_amount,
            mix_prob=mix_prob,
            apply_to_channel=apply_to_channel,
            same_on_batch=same_on_batch,
            retain_stats=retain_stats,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


@register(
    aug_id=AugId.RAND_CONV,
    backend=Backend.GPU,
    group=AugType.TA,
    order=110,
)
class RandomRandConvGPU(_RandomConvBaseGPU):
    """RandConv: convolution with a randomly drawn multi-scale kernel."""

    def __init__(
        self,
        kernel_sizes: Sequence[int] = (1, 3, 5, 7),
        mix_prob: float = 0.0,
        apply_to_channel: Sequence[int] = (0,),
        same_on_batch: bool = False,
        retain_stats: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            kernel_type="RandConv",
            kernel_sizes=kernel_sizes,
            mix_prob=mix_prob,
            apply_to_channel=apply_to_channel,
            same_on_batch=same_on_batch,
            retain_stats=retain_stats,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


def apply_convolution(img: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Based on https://github.com/pytorch/vision/blob/e3b5d3a8bf5e8636462fd8bce9897bccc690b2a0/torchvision/transforms/_functional_tensor.py#L746
    """
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    if dim == 2:
        kernel = kernel.expand(img.shape[-(1 + dim)], 1, kernel.shape[0], kernel.shape[1])
        padding = [kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2]
    elif dim == 3:
        kernel = kernel.expand(img.shape[-(1 + dim)], 1, kernel.shape[0], kernel.shape[1], kernel.shape[2])
        padding = [
            kernel.shape[2] // 2,
            kernel.shape[2] // 2,
            kernel.shape[3] // 2,
            kernel.shape[3] // 2,
            kernel.shape[4] // 2,
            kernel.shape[4] // 2,
        ]
    else:
        raise ValueError(f"Only 2D and 3D convolution are supported. Got {dim}D.")

    img, need_cast, need_squeeze, out_dtype = F_t._cast_squeeze_in(img, [kernel.dtype])

    # padding = (left, right, top, bottom)
    img = F.pad(img, padding, mode="reflect")
    if dim == 2:  # noqa: SIM108 -- the 2d/3d split reads better spelled out than as a ternary
        img = F.conv2d(img, kernel, groups=img.shape[-(1 + dim)])
    else:  # dim == 3
        img = F.conv3d(img, kernel, groups=img.shape[-(1 + dim)])

    img = F_t._cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def get_gaussian_kernel1d(kernel_size: int, sigma: Union[float, Tensor], dtype: torch.dtype, device: torch.device) -> Tensor:
    """Create a 1D Gaussian kernel."""

    x = torch.arange(kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def get_gaussian_kernel3d(kernel_size: int, sigma: Union[float, Tensor], dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Create a 3D Gaussian kernel by multiplying 1D kernels along each axis.
    Args:
        kernel_size (int)
        sigma (float or tuple of three floats): Standard deviation of the Gaussian kernel.
    """
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor([sigma, sigma, sigma], device=device)
    elif isinstance(sigma, torch.Tensor):
        assert sigma.shape == (3,), "Sigma must be a float or a tensor of three floats."
    else:
        raise TypeError("Sigma must be a float or a tensor of three floats.")

    gz = get_gaussian_kernel1d(kernel_size, sigma[0], dtype, device)
    gy = get_gaussian_kernel1d(kernel_size, sigma[1], dtype, device)
    gx = get_gaussian_kernel1d(kernel_size, sigma[2], dtype, device)

    # Outer product using broadcasting
    kernel = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

    # Normalize
    kernel /= kernel.sum()

    return kernel


## Noise transform
@register(
    aug_id=AugId.GAUSSIAN_NOISE,
    backend=Backend.GPU,
    group=AugType.GE,
    order=130,
)
class RandomGaussianNoiseGPU(ImageOnlyTransform):
    """Add random Gaussian noise to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        mean (float): Mean of the Gaussian noise. Default is 0.0.
        std (float): Standard deviation of the Gaussian noise. Default is 0.1.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with added Gaussian noise.
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.apply_to_channel = apply_to_channel
        self.mean = mean
        self.std = std
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # Generate Gaussian noise with the same shape as input
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            if self.same_on_batch:
                std = torch.rand(1, device=input.device, dtype=input.dtype) * self.std
                noise = torch.randn_like(input[:, c], device=input.device, dtype=input.dtype)
                noise = noise * std + self.mean
            else:
                std = torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * self.std
                noise = torch.randn_like(input[:, c], device=input.device, dtype=input.dtype)
                for i in range(input.shape[0]):
                    noise[i] = noise[i] * std[i] + self.mean

            orig = input[:, c]
            x = orig + noise
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = x

        return input


## Multiplicative brightness transform
@register(
    aug_id=AugId.BRIGHTNESS,
    backend=Backend.GPU,
    group=AugType.GE,
    order=150,
)
class RandomBrightnessGPU(ImageOnlyTransform):
    """Apply random brightness adjustment to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        brightness_range (tuple of float): Range of brightness multipliers. Default is (0.9, 1.1).
        apply_to_channel (list of int): List of channel indices to apply the brightness adjustment to. Default is [0].
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.5, 1.5),
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.brightness_range = brightness_range
        self.apply_to_channel = apply_to_channel
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Apply brightness adjustment
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()
            if self.same_on_batch:
                factor = (
                    torch.rand(1, device=input.device, dtype=input.dtype) * (self.brightness_range[1] - self.brightness_range[0])
                    + self.brightness_range[0]
                )
                x = channel_data * factor
            else:
                factor = (
                    torch.rand(input.shape[0], device=input.device, dtype=input.dtype)
                    * (self.brightness_range[1] - self.brightness_range[0])
                    + self.brightness_range[0]
                )
                x = channel_data.clone()
                for i in range(input.shape[0]):
                    x[i] = x[i] * factor[i]
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = x

        return input


## Gamma transform
class _RandomGammaBaseGPU(ImageOnlyTransform):
    """Apply random gamma adjustment to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        gamma_range (tuple of float): Range of gamma multipliers. Default is (0.7, 1.5).
        invert_image (bool): If True, invert the image before and after gamma adjustment. Default is False.
        apply_to_channel (list of int): List of channel indices to apply the gamma adjustment to. Default is [0].
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after gamma adjustment. Default is False.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        invert_image: bool = False,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.retain_stats = retain_stats
        self.apply_to_channel = apply_to_channel
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Apply gamma transform
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            # [N, ...spatial...]
            channel_data = -input[:, c] if self.invert_image else input[:, c]
            orig_full = input[:, c].clone()

            if self.retain_stats:
                reduce_dims = tuple(range(1, channel_data.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = channel_data.mean(dim=reduce_dims)
                orig_stds = channel_data.std(dim=reduce_dims)

            if self.same_on_batch:
                gamma = (
                    torch.rand(1, device=input.device, dtype=input.dtype) * (self.gamma_range[1] - self.gamma_range[0])
                    + self.gamma_range[0]
                )
            else:
                gamma = (
                    torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * (self.gamma_range[1] - self.gamma_range[0])
                    + self.gamma_range[0]
                )

            # Compute min and range per batch element for the current channel
            # Flatten spatial dimensions to compute min/max per batch element
            batch_size = channel_data.shape[0]
            flat_data = channel_data.view(batch_size, -1)  # [N, spatial_flattened]
            minm = flat_data.min(dim=1, keepdim=self.keepdim)[0]  # [N, 1]
            maxm = flat_data.max(dim=1, keepdim=self.keepdim)[0]  # [N, 1]
            rnge = maxm - minm

            # Reshape min, max, range to broadcast over spatial dims: [N, 1] -> [N, 1, 1, ...]
            reshape_dims = [batch_size] + [1] * (channel_data.dim() - 1)
            minm = minm.view(reshape_dims)
            rnge = rnge.view(reshape_dims)

            # Reshape gamma to broadcast properly: [N] -> [N, 1, 1, ...]
            if not self.same_on_batch:
                gamma = gamma.view(reshape_dims)

            # Apply gamma transform per batch element
            channel_data = torch.pow(((channel_data - minm) / (rnge + 1e-8)), gamma) * rnge + minm

            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, channel_data.dim()))
                new_mean = channel_data.mean(dim=reduce_dims)  # [N]
                new_std = channel_data.std(dim=reduce_dims)  # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [channel_data.shape[0]] + [1] * (channel_data.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                channel_data = (channel_data - nm) / (ns + eps) * os + om

            if self.invert_image:
                channel_data = -channel_data
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                channel_data = _apply_region_mode(orig_full, channel_data, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(channel_data).any() or torch.isinf(channel_data).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = channel_data

        return input


# Gamma, split so that "gamma" and "inverted gamma" are two config keys rather than
# one key plus an `invert_image` flag. Neither leaf exposes the flag, so a config
# cannot express the same augmentation two ways.


@register(
    aug_id=AugId.GAMMA,
    backend=Backend.GPU,
    group=AugType.GE,
    order=160,
)
class RandomGammaGPU(_RandomGammaBaseGPU):
    """Random gamma adjustment."""

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        apply_to_channel: Sequence[int] = (0,),
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            gamma_range=gamma_range,
            invert_image=False,
            apply_to_channel=apply_to_channel,
            retain_stats=retain_stats,
            same_on_batch=same_on_batch,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


@register(
    aug_id=AugId.INV_GAMMA,
    backend=Backend.GPU,
    group=AugType.GE,
    order=170,
)
class RandomInvGammaGPU(_RandomGammaBaseGPU):
    """Random gamma adjustment applied to the inverted image."""

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        apply_to_channel: Sequence[int] = (0,),
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            gamma_range=gamma_range,
            invert_image=True,
            apply_to_channel=apply_to_channel,
            retain_stats=retain_stats,
            same_on_batch=same_on_batch,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


## nnunetv2 contrast transform
@register(
    aug_id=AugId.CONTRAST,
    backend=Backend.GPU,
    group=AugType.GE,
    order=180,
)
class RandomContrastGPU(ImageOnlyTransform):
    """Apply random gamma adjustment to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        contrast_range (tuple of float): Range of gamma multipliers. Default is (0.9, 1.1).
        apply_to_channel (list of int): List of channel indices to apply the gamma adjustment to. Default is [0].
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after gamma adjustment. Default is False.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        contrast_range: tuple[float, float] = (0.75, 1.25),
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.contrast_range = contrast_range
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Apply brightness adjustment
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()
            if self.retain_stats:
                reduce_dims = tuple(range(1, channel_data.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = channel_data.mean(dim=reduce_dims)
                orig_stds = channel_data.std(dim=reduce_dims)

            if self.same_on_batch:
                factor = (
                    torch.rand(1, device=input.device, dtype=input.dtype) * (self.contrast_range[1] - self.contrast_range[0])
                    + self.contrast_range[0]
                )
                x = channel_data.clone()
                for i in range(input.shape[0]):
                    mean = x[i].mean()
                    x[i] = (x[i] - mean) * factor + mean
            else:
                factor = (
                    torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * (self.contrast_range[1] - self.contrast_range[0])
                    + self.contrast_range[0]
                )
                x = channel_data.clone()
                for i in range(input.shape[0]):
                    mean = x[i].mean()
                    x[i] = (x[i] - mean) * factor[i] + mean

            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, x.dim()))
                new_mean = x.mean(dim=reduce_dims)  # [N]
                new_std = x.std(dim=reduce_dims)  # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [x.shape[0]] + [1] * (x.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                x = (x - nm) / (ns + eps) * os + om
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = x

        return input


## Function transform
class _RandomFunctionBaseGPU(ImageOnlyTransform):
    """Apply function to the image based on probability.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        func (callable): Random function to apply. Default is a gamma adjustment function.
        apply_to_channel (list of int): List of channel indices to apply the function to. Default is [0].
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after gamma adjustment. Default is False.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        func: Callable[[Tensor], Tensor] = lambda x: x**2,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.func = func
        self.retain_stats = retain_stats
        self.apply_to_channel = apply_to_channel
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Apply function transform
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            x = input[:, c]  # shape [N, ...spatial...]
            orig = x.clone()
            if self.retain_stats:
                reduce_dims = tuple(range(1, x.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = x.mean(dim=reduce_dims)
                orig_stds = x.std(dim=reduce_dims)

            # Normalize to make values >=0
            x = (x - x.min()) / (x.max() - x.min() + 0.00001)

            # Apply function
            x = self.func(x)

            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, x.dim()))
                new_mean = x.mean(dim=reduce_dims)  # [N]
                new_std = x.std(dim=reduce_dims)  # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [x.shape[0]] + [1] * (x.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                x = (x - nm) / (ns + eps) * os + om
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = x

        return input


# One class per elementwise function.
#
# `func` was a callable parameter, which no JSON config could ever express -- the old
# ladder worked around that by expanding a single "FunctionTransform" block into five
# transforms from a hardcoded lambda list. A class each makes every one addressable
# from a config, and removes the un-serialisable parameter entirely.
#
# Written out longhand rather than as torch.log1p / torch.sigmoid on purpose: those
# differ from the originals in the last ulp, which is enough to move the seeded
# determinism hashes and invalidate every published experiment.


def _log1p(x: Tensor) -> Tensor:
    return torch.log(1 + x)


def _sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-x))


class _RandomNamedFunctionGPU(_RandomFunctionBaseGPU):
    """Shared constructor for the fixed-function leaves. Not registered itself."""

    #: Set by each leaf; `func` is therefore absent from the config surface.
    function: staticmethod

    def __init__(
        self,
        apply_to_channel: Sequence[int] = (0,),
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            func=type(self).function,
            apply_to_channel=apply_to_channel,
            retain_stats=retain_stats,
            same_on_batch=same_on_batch,
            in_seg=in_seg,
            out_seg=out_seg,
            mix_in_out=mix_in_out,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )


@register(
    aug_id=AugId.FUNC_LOG1P,
    backend=Backend.GPU,
    group=AugType.TA,
    order=190,
)
class RandomLog1pGPU(_RandomNamedFunctionGPU):
    """Apply log(1 + x)."""

    function = staticmethod(_log1p)


@register(
    aug_id=AugId.FUNC_SQRT,
    backend=Backend.GPU,
    group=AugType.TA,
    order=191,
)
class RandomSqrtGPU(_RandomNamedFunctionGPU):
    """Apply sqrt(x)."""

    function = staticmethod(torch.sqrt)


@register(
    aug_id=AugId.FUNC_SIN,
    backend=Backend.GPU,
    group=AugType.TA,
    order=192,
)
class RandomSinGPU(_RandomNamedFunctionGPU):
    """Apply sin(x)."""

    function = staticmethod(torch.sin)


@register(
    aug_id=AugId.FUNC_EXP,
    backend=Backend.GPU,
    group=AugType.TA,
    order=193,
)
class RandomExpGPU(_RandomNamedFunctionGPU):
    """Apply exp(x)."""

    function = staticmethod(torch.exp)


@register(
    aug_id=AugId.FUNC_SIGMOID,
    backend=Backend.GPU,
    group=AugType.TA,
    order=194,
)
class RandomSigmoidGPU(_RandomNamedFunctionGPU):
    """Apply the logistic sigmoid 1 / (1 + exp(-x))."""

    function = staticmethod(_sigmoid)


## Inverse transform
@register(
    aug_id=AugId.INVERSE,
    backend=Backend.GPU,
    group=AugType.TA,
    order=60,
)
class RandomInverseGPU(ImageOnlyTransform):
    """Inverse image based on probability.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        apply_to_channel (list of int): List of channel indices to apply the brightness adjustment to. Default is [0].
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        mix_prob: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_prob = mix_prob
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Inverse image
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            for i in range(input.shape[0]):
                x = input[i, c]  # shape [...spatial...]
                orig = x.clone()
                if self.retain_stats:
                    orig_means = x.mean()
                    orig_stds = x.std()
                max_val = x.max()
                x = max_val - x

                if self.retain_stats:
                    # Adjust mean and std to match original
                    eps = 1e-8
                    new_mean = x.mean()  # scalar
                    new_std = x.std()  # scalar
                    x = (x - new_mean) / (new_std + eps) * orig_stds + orig_means

                # Mix with original based on mix_prob
                if torch.rand(1).item() < self.mix_prob:
                    alpha = torch.rand(1, device=input.device)
                    x = alpha * orig + (1 - alpha) * x

                if seg_mask is not None:
                    region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask[i])
                    x = _apply_region_mode(orig, x, seg_mask[i], region_mode, mix_in_out=self.mix_in_out)
                # Final safety: check if nan/inf appeared
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning nan: {self.__class__.__name__}", flush=True)
                    continue
                input[i, c] = x

        return input


## Histogram transform
@register(
    aug_id=AugId.HISTOGRAM_EQUAL,
    backend=Backend.GPU,
    group=AugType.TA,
    order=70,
)
class RandomHistogramEqualizationGPU(ImageOnlyTransform):
    """Apply histogram equalization transformation to the image based on probability.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        apply_to_channel (list of int): List of channel indices to apply the histogram equalization to. Default is [0].
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after histogram equalization. Default is False.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        mix_prob: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.retain_stats = retain_stats
        self.apply_to_channel = apply_to_channel
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out
        self.mix_prob = mix_prob

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Apply histogram equalization transform
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # shape [N, ...spatial...]
            orig = channel_data.clone()

            if self.retain_stats:
                reduce_dims = tuple(range(1, channel_data.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = channel_data.mean(dim=reduce_dims)
                orig_stds = channel_data.std(dim=reduce_dims)

            # Process each batch element independently
            batch_size = channel_data.shape[0]
            for b in range(batch_size):
                img_b = channel_data[b]  # Single image from batch [...spatial...]

                img_min, img_max = img_b.min(), img_b.max()

                # Flatten the image and compute the histogram
                img_flattened = img_b.flatten().to(torch.float32)
                hist = torch.histc(img_flattened, bins=256, min=img_min.item(), max=img_max.item())

                # Compute the normalized cumulative distribution function (CDF)
                cdf = hist.cumsum(dim=0)
                cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else cdf.min()
                cdf = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-10)  # Normalize to [0,1]
                cdf = cdf * (img_max - img_min) + img_min  # Scale back to image range

                # Compute bin edges and indices
                bin_width = (img_max - img_min) / 256
                indices = ((img_flattened - img_min) / (bin_width + 1e-10)).long()
                indices = torch.clamp(indices, 0, 255)

                # Perform histogram equalization
                img_eq = cdf[indices]
                channel_data[b] = img_eq.reshape(img_b.shape)

                # Mix with original based on mix_prob
                if torch.rand(1).item() < self.mix_prob:
                    alpha = torch.rand(1, device=input.device)
                    channel_data[b] = alpha * orig[b] + (1 - alpha) * channel_data[b]

            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, channel_data.dim()))
                new_mean = channel_data.mean(dim=reduce_dims)  # [N]
                new_std = channel_data.std(dim=reduce_dims)  # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [channel_data.shape[0]] + [1] * (channel_data.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                channel_data = (channel_data - nm) / (ns + eps) * os + om

            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                channel_data = _apply_region_mode(orig, channel_data, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(channel_data).any() or torch.isinf(channel_data).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = channel_data

        return input


@register(
    aug_id=AugId.BIAS_FIELD,
    backend=Backend.GPU,
    group=AugType.TA,
    order=230,
)
class RandomBiasFieldGPU(ImageOnlyTransform):
    """Apply a smooth multiplicative bias field to selected channels.

    The bias field simulates low-frequency intensity inhomogeneity (MRI bias field).
    It is constructed as the exponential of a polynomial combination of the
    spatial coordinates (x, y, z) up to a given order with random coefficients.

    Supports 2D (N, C, H, W) and 3D (N, C, D, H, W) tensors.

    Args:
        coefficients (float | tuple[float, float]): If float c, coefficients sampled
            uniformly from (-c, c). If tuple (a, b) coefficients sampled from (a, b).
        order (int): Polynomial order (>=0).
        apply_to_channel (list[int]): Channels to which the bias field is applied.
        invert (bool): If True, uses inverse bias field (1 / field).
        retain_stats (bool): If True, restores original per-sample mean and std for affected channels.
        same_on_batch (bool): If True, uses the same sampled coefficients for all batch elements.
        p (float): Application probability.
        keepdim (bool): Keep input dimensions flag (passed to base).
    """

    def __init__(
        self,
        coefficients: Union[float, tuple[float, float]] = 0.5,
        order: int = 3,
        apply_to_channel: Sequence[int] = (0,),
        invert: bool = False,
        retain_stats: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if isinstance(coefficients, (int, float)):
            self.coeff_range = (-float(coefficients), float(coefficients))
        elif isinstance(coefficients, (tuple, list)) and len(coefficients) == 2:
            self.coeff_range = (float(coefficients[0]), float(coefficients[1]))
        else:
            raise TypeError("coefficients must be float or (min, max) tuple")
        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a non-negative int")
        self.order = order
        self.apply_to_channel = apply_to_channel
        self.invert = invert
        self.retain_stats = retain_stats
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    def _num_coeffs(self, dim: int) -> int:
        # Count coefficients generated by nested loops matching TorchIO logic.
        count = 0
        if dim == 3:
            for xo in range(self.order + 1):
                for yo in range(self.order + 1 - xo):
                    for _zo in range(self.order + 1 - (xo + yo)):
                        count += 1
        elif dim == 2:
            for xo in range(self.order + 1):
                for _yo in range(self.order + 1 - xo):
                    count += 1
        else:
            raise ValueError("Only 2D or 3D spatial dims supported for bias field")
        return count

    def _sample_coeffs(self, batch_size: int, device: torch.device, dtype: torch.dtype, dim: int) -> torch.Tensor:
        n = self._num_coeffs(dim)
        low, high = self.coeff_range
        if self.same_on_batch:
            coeff = torch.empty(n, 1, device=device, dtype=dtype).uniform_(low, high)
            coeff = coeff.expand(n, batch_size)
        else:
            coeff = torch.empty(n, batch_size, device=device, dtype=dtype).uniform_(low, high)
        return coeff  # shape (n_coeffs, B)

    def _make_grids(self, spatial_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        # Create coordinate grids normalized to [-1, 1]
        if len(spatial_shape) == 2:
            h, w = spatial_shape
            ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
            xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
            y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
            return [x_grid, y_grid]
        elif len(spatial_shape) == 3:
            d, h, w = spatial_shape
            zs = torch.linspace(-1, 1, d, device=device, dtype=dtype)
            ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
            xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
            z_grid, y_grid, x_grid = torch.meshgrid(zs, ys, xs, indexing="ij")
            return [x_grid, y_grid, z_grid]
        else:
            raise ValueError("Spatial dims must be 2 or 3 for bias field")

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        # input: (N, C, [D,] H, W)
        if input.dim() not in (4, 5):
            raise ValueError("Expected 4D or 5D tensor (N,C,...) for RandomBiasFieldGPU")
        batch_size = input.shape[0]
        spatial = input.shape[2:]
        dim = len(spatial)
        device = input.device
        dtype = input.dtype

        coeffs = self._sample_coeffs(batch_size, device, dtype, dim)  # (n_coeffs, B)
        grids = self._make_grids(spatial, device, dtype)
        seg_mask = params.get("seg")

        # Initialize bias map per batch element
        bias_map = torch.zeros((batch_size, *spatial), device=device, dtype=dtype)

        idx = 0
        if dim == 3:
            xg, yg, zg = grids  # each shape (D,H,W)
            for xo in range(self.order + 1):
                x_term = xg.pow(xo) if xo > 0 else 1.0
                for yo in range(self.order + 1 - xo):
                    y_term = yg.pow(yo) if yo > 0 else 1.0
                    for zo in range(self.order + 1 - (xo + yo)):
                        z_term = zg.pow(zo) if zo > 0 else 1.0
                        # term shape (D,H,W)
                        term = x_term * y_term * z_term
                        # Add coefficient * term for each batch element
                        bias_map += coeffs[idx].view(-1, *([1] * dim)) * term  # broadcast over spatial
                        idx += 1
        else:  # dim == 2
            xg, yg = grids  # (H,W)
            for xo in range(self.order + 1):
                x_term = xg.pow(xo) if xo > 0 else 1.0
                for yo in range(self.order + 1 - xo):
                    y_term = yg.pow(yo) if yo > 0 else 1.0
                    term = x_term * y_term  # (H,W)
                    bias_map += coeffs[idx].view(-1, *([1] * dim)) * term
                    idx += 1

        # Exponential to ensure positive field
        bias_field = torch.exp(bias_map)  # (N, *spatial)
        if self.invert:
            bias_field = 1.0 / (bias_field + 1e-8)

        # Apply to channels
        for c in self.apply_to_channel:
            if c < 0 or c >= input.shape[1]:
                continue  # skip invalid channel index
            channel = input[:, c]
            orig = channel.clone()
            if self.retain_stats:
                reduce_dims = tuple(range(1, channel.dim()))
                orig_mean = channel.mean(dim=reduce_dims)
                orig_std = channel.std(dim=reduce_dims)
            channel = channel * bias_field
            if self.retain_stats:
                eps = 1e-8
                new_mean = channel.mean(dim=reduce_dims)
                new_std = channel.std(dim=reduce_dims)
                # reshape stats for broadcasting
                shape = [channel.shape[0]] + [1] * (channel.dim() - 1)
                om = orig_mean.view(shape)
                os = orig_std.view(shape)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                channel = (channel - nm) / (ns + eps) * os + om
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                channel = _apply_region_mode(orig, channel, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(channel).any() or torch.isinf(channel).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = channel

        return input


# Random clamping transform
@register(
    aug_id=AugId.CLAMP,
    backend=Backend.GPU,
    group=AugType.GE,
    order=120,
)
class RandomClampGPU(ImageOnlyTransform):
    """Apply random gamma adjustment to image.
    If the image is torch Tensor, it is expected to have [N, C, X, Y] or [N, C, X, Y, Z] shape.

    Args:
        max_clamp_amount (float): Amount to clamp the image values (0 < min_clamp < max_clamp_amount and 1 - max_clamp_amount < max_clamp < 1). Default is 0.2.
        apply_to_channel (list of int): List of channel indices to apply the gamma adjustment to. Default is [0].
        retain_stats (bool): If True, retain the original mean and standard deviation of the image after gamma adjustment. Default is False.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        p (float): Probability of applying the transform. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is False.

    Returns:
        Tensor: Image with adjusted brightness.
    """

    def __init__(
        self,
        max_clamp_amount: float = 0.0,
        apply_to_channel: Sequence[int] = (0,),  # Apply to first channel by default
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.max_clamp_amount = max_clamp_amount
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:

        # Apply clamping
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()
            if self.retain_stats:
                reduce_dims = tuple(range(1, channel_data.dim()))
                # store per-sample mean/std (shape [N])
                orig_means = channel_data.mean(dim=reduce_dims)
                orig_stds = channel_data.std(dim=reduce_dims)

            if self.same_on_batch:
                min_percentile = torch.rand(1, device=input.device, dtype=input.dtype) * self.max_clamp_amount
                max_percentile = 1.0 - (torch.rand(1, device=input.device, dtype=input.dtype) * self.max_clamp_amount)
                x = channel_data.clone()
                for i in range(input.shape[0]):
                    min_val = torch.quantile(x[i].flatten(), min_percentile)
                    max_val = torch.quantile(x[i].flatten(), max_percentile)
                    x[i] = torch.clamp(x[i], min_val, max_val)
            else:
                x = channel_data.clone()
                for i in range(input.shape[0]):
                    min_percentile = torch.rand(1, device=input.device, dtype=input.dtype) * self.max_clamp_amount
                    max_percentile = 1.0 - (torch.rand(1, device=input.device, dtype=input.dtype) * self.max_clamp_amount)
                    min_val = torch.quantile(x[i].flatten(), min_percentile)
                    max_val = torch.quantile(x[i].flatten(), max_percentile)
                    x[i] = torch.clamp(x[i], min_val, max_val)

            if self.retain_stats:
                # Adjust mean and std to match original
                eps = 1e-8
                reduce_dims = tuple(range(1, x.dim()))
                new_mean = x.mean(dim=reduce_dims)  # [N]
                new_std = x.std(dim=reduce_dims)  # [N]
                # reshape stats to broadcast over spatial dims: [N,1,1,...]
                shape = [x.shape[0]] + [1] * (x.dim() - 1)
                nm = new_mean.view(shape)
                ns = new_std.view(shape)
                om = orig_means.view(shape)
                os = orig_stds.view(shape)
                x = (x - nm) / (ns + eps) * os + om
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=self.mix_in_out)
            # Final safety: check if nan/inf appeared
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = x

        return input


@register(
    aug_id=AugId.ZSCORE,
    backend=Backend.GPU,
    group=AugType.GE,
    order=240,
)
class ZscoreNormalizationGPU(ImageOnlyTransform):
    """Apply z-score normalization to selected channels.

    Args:
        apply_to_channel (list[int]): Channels to which the normalization is applied.
        p (float): Application probability.
        keepdim (bool): Keep input dimensions flag (passed to base).
    """

    def __init__(
        self,
        apply_to_channel: Sequence[int] = (0,),
        keepdim: bool = True,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=False, keepdim=keepdim)
        self.apply_to_channel = apply_to_channel
        self.in_seg = in_seg
        self.out_seg = out_seg

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        # input: (N, C, [D,] H, W)
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            if c < 0 or c >= input.shape[1]:
                continue  # skip invalid channel index
            channel = input[:, c]
            orig = channel.clone()
            reduce_dims = tuple(range(1, channel.dim()))
            mean = channel.mean(dim=reduce_dims, keepdim=True)
            # use unbiased=False for stability, and clamp std to avoid division by ~0
            std = channel.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-8)
            channel = (channel - mean) / std
            if seg_mask is not None:
                region_mode = _choose_region_mode(self.in_seg, self.out_seg, seg_mask)
                channel = _apply_region_mode(orig, channel, seg_mask, region_mode)
            # Final safety: check if nan/inf appeared
            if torch.isnan(channel).any() or torch.isinf(channel).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = channel

        return input
