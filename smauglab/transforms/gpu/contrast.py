import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, Union

import torch
import torchvision.transforms._functional_tensor as F_t
from torch import Tensor
from torch.nn import functional as F

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform
from smauglab.transforms.kernels import depthwise_conv3d, gaussian_kernel3d, laplace_kernel, scharr_kernels, stacked_scharr_kernels
from smauglab.transforms.rng import shared_choice


def _choose_region_mode(
    p_in: float,
    p_out: float,
    seg_mask: torch.Tensor | None,  # noqa: ARG001
) -> str:
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


def _channel_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample mean and std of a `[N, ...spatial]` channel, each shaped `[N]`."""
    reduce_dims = tuple(range(1, x.dim()))
    return x.mean(dim=reduce_dims), x.std(dim=reduce_dims)


def _restore_stats(x: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rescale `x` so its per-sample mean and std match `stats` again.

    What `retain_stats=True` means, written out identically in nine `apply_transform`
    methods before this.
    """
    orig_means, orig_stds = stats
    eps = 1e-8
    reduce_dims = tuple(range(1, x.dim()))
    # broadcast the stats over the spatial dims: [N, 1, 1, ...]
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    new_mean = x.mean(dim=reduce_dims).view(shape)
    new_std = x.std(dim=reduce_dims).view(shape)
    return (x - new_mean) / (new_std + eps) * orig_stds.view(shape) + orig_means.view(shape)


def _mix_with_original(orig: torch.Tensor, x: torch.Tensor, mix_prob: float) -> torch.Tensor:
    """Blend each sample back towards the original with probability `mix_prob`.

    Per sample: an independent draw and an independent alpha for every element of
    the batch, which is what "probability of blending the result back" means and
    what the sibling transforms that write this inline already do.
    """
    if mix_prob <= 0.0:
        return x
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    mix = torch.rand(x.shape[0], device=x.device).view(shape) < mix_prob
    alpha = torch.rand(x.shape[0], device=x.device).view(shape)
    return torch.where(mix, alpha * orig + (1 - alpha) * x, x)


def _check_channel(transform: torch.nn.Module, channel: int, channels: int) -> None:
    """Reject an `apply_to_channel` entry the input does not have.

    Two transforms used to `continue` past an out-of-range index. Every other one
    indexes `input[:, c]` straight away and raises IndexError, so a config typo --
    `apply_to_channel: [1]` on single-channel data is the easy one -- turned those
    two into a silent no-op while the rest of the pipeline failed loudly. Failing
    loudly is the useful half of that pair, and it names the config key.
    """
    if channel < 0 or channel >= channels:
        raise IndexError(
            f"{type(transform).__name__}: apply_to_channel {channel} is out of range for a "
            f"{channels}-channel input. Channels are 0-based, so valid entries are 0..{channels - 1}."
        )


class _RegionSelecting(Protocol):
    """What `_select_and_check` needs off the transform it is handed.

    A Protocol rather than `ImageOnlyTransform`, because these are plain attributes on
    an nn.Module: reading them through the base class resolves via `Module.__getattr__`,
    which is typed as returning `Tensor | Module`. Naming them here is both what makes
    that type-check and a statement of the helper's actual requirement.

    All three are attributes rather than constructor parameters as far as this
    Protocol is concerned; the registry derives a transform's config surface from its
    __init__ signature, so declaring one here does not make it settable in a config.
    """

    in_seg: float
    out_seg: float
    mix_in_out: bool


def _select_and_check(
    transform: _RegionSelecting,
    orig: torch.Tensor,
    x: torch.Tensor,
    seg_mask: torch.Tensor | None,
    note: str = "",
) -> torch.Tensor | None:
    """Apply region selection, then reject the result if it went non-finite.

    Returns None when the channel should be left as it was -- the "Final safety" check
    that closed all nine of these loops identically.

    Takes the transform rather than its three region attributes: every call site read
    exactly `self.in_seg`, `self.out_seg` and `self.mix_in_out`, and spelling them out
    made the call longer than the code it replaces.
    """
    if seg_mask is not None:
        region_mode = _choose_region_mode(transform.in_seg, transform.out_seg, seg_mask)
        x = _apply_region_mode(orig, x, seg_mask, region_mode, mix_in_out=transform.mix_in_out)
    # One `isfinite(...).all()` rather than `isnan().any() or isinf().any()`: each of
    # those reads a reduction back to the host, and the `or` makes the second one
    # unavoidable on the common (finite) path. This guard runs for every channel of
    # every region-selecting transform.
    if not bool(torch.isfinite(x).all()):
        print(f"Warning nan: {type(transform).__name__}{note}", flush=True)
        return None
    return x


def _foreground(mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Which voxels the segmentation covers, reducing over the class axis.

    This used to be `torch.argmax(mask, dim) > 0`, which is only "is anything labelled
    here" if class 0 is background -- and it is not:

    * For a single-channel `[B, 1, D, H, W]` mask (an ordinary nnU-Net target, and what
      the tests build) `argmax` over a length-1 axis is always 0, so the result was
      **all False**. `in_seg` then applied the transform nowhere and `out_seg` applied
      it everywhere: the two knobs did nothing and the opposite of nothing.
    * For a one-hot mask, this repository's convention (`collapse_onehot_to_index` in
      gpu/fromSeg.py) is that channel `c` encodes label `c + 1` with background
      implicit, so `argmax == 0` is a real foreground class and was being dropped.

    `amax > 0` asks the question that was meant, and matches what
    `collapse_onehot_to_index` already does with `seg_raw.any(dim=1)`.

    That still leaves one layout it cannot read off the values alone: a one-hot
    that *includes* a background channel, which is what `seg_region_masks` emits
    for a single-channel label map and what `unit_tests/test_seg_layout.py`
    builds. Every voxel then has some channel set, `amax > 0` is True everywhere,
    and `in_seg` applies the transform to the whole patch while `out_seg` applies
    it nowhere -- the exact failure this function was written to fix, in the one
    layout it was not checked against.

    A background-inclusive one-hot is recognisable: it is multi-channel and leaves
    no voxel unset, because the background channel covers whatever the foreground
    channels do not. A background-implicit one-hot always has all-zero background
    voxels unless the patch is labelled edge to edge, which a real segmentation
    patch is not. So full coverage plus more than one channel means channel 0 is
    background, and the reduction skips it.
    """
    covered = mask.amax(dim=dim) > 0
    if mask.shape[dim] == 1:
        return covered
    # `torch.where` rather than `if bool(...)`: reading that test back to Python blocks
    # the host on everything queued behind it, and this runs once per region-selecting
    # transform per batch. Both branches are a reduction over the class axis, so
    # computing the unused one costs a fraction of what the stall does.
    without_background = mask.narrow(dim, 1, mask.shape[dim] - 1).amax(dim=dim) > 0
    return torch.where(covered.all(), without_background, covered)


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

        # No `.clone()`: nothing below writes into `m`, and on a [B, C, D, H, W] mask
        # that copy is the size of the batch.
        m = seg_mask.to(transformed.dtype)
        if mix_in_out:
            # One keep/drop draw per (sample, class), in a single call rather than one
            # `randint` per sample: the same draws in the same order, one kernel.
            keep = torch.randint(0, 2, seg_mask.shape[:2], device=seg_mask.device, dtype=m.dtype)
            m = m * keep.view(*seg_mask.shape[:2], *([1] * (m.dim() - 2)))

        m = _foreground(m, dim=1)
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

        m = seg_mask.to(transformed.dtype)
        if mix_in_out:
            # Create a tensor with random one and zero
            o = torch.randint(0, 2, (seg_mask.shape[0],), device=seg_mask.device, dtype=m.dtype)
            m = m * o.view(-1, 1, 1, 1)  # Broadcasting o to match the dimensions of m
        m = _foreground(m, dim=0)
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
            kernel = laplace_kernel(3, device=device)
        elif self.kernel_type == "Scharr":
            kernel = scharr_kernels(3, device=device)
        elif self.kernel_type == "GaussianBlur":
            sigma = torch.rand(3, device=device) * self.sigma
            kernel_size = 3
            kernel = gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        elif self.kernel_type == "UnsharpMask":
            # For unsharp masking we use a Gaussian blur kernel; amount is applied in apply_transform.
            sigma = torch.rand(3, device=device) * self.sigma
            kernel_size = 3
            kernel = gaussian_kernel3d(kernel_size, sigma, torch.float32, device)
        elif self.kernel_type == "RandConv":
            # choose random odd kernel size e.g. [1,3,5,7]
            k = int(shared_choice(self.kernel_sizes))  # define kernel_sizes in __init__

            # 1/sqrt(k**3), not 1/sqrt(k*k): the kernel has k**3 taps, so unit
            # output variance needs per-tap variance 1/k**3. The 2-D formula left
            # a gain of sqrt(k) -- measured output std 0.85 / 1.59 / 2.11 / 2.43
            # for k = 1 / 3 / 5 / 7 on unit-variance input -- which made the
            # augmentation's strength a function of the randomly drawn kernel
            # size. Nothing corrected it downstream: RandomRandConvGPU defaults
            # retain_stats to False.
            std = 1.0 / math.sqrt(k**3)
            kernel = torch.randn((k, k, k), device=device) * std
        else:
            raise NotImplementedError("Kernel type not implemented.")
        return kernel

    def _per_sample_kernels(self, batch_size: int, device: torch.device) -> list[Tensor]:
        """One freshly drawn kernel per sample, for the `same_on_batch=False` paths."""
        kernels = []
        for _ in range(batch_size):
            drawn = self.get_kernel(device=device)
            assert isinstance(drawn, Tensor)
            kernels.append(drawn)
        return kernels

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # Initialize kernel
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        kernel = self.get_kernel(device=input.device)

        # Load segmentation
        seg_mask = params.get("seg")

        # Apply convolution
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()

            if self.retain_stats:
                stats = _channel_stats(channel_data)

            # The asserts below restate what get_kernel guarantees per kernel_type:
            # only Scharr yields a list, and only its branch iterates.
            if self.kernel_type == "Laplace":
                assert isinstance(kernel, Tensor)
                x = apply_convolution(channel_data, kernel, dim=3)
            elif self.kernel_type == "GaussianBlur":
                # The sigma is drawn in get_kernel, so sharing the kernel shares the
                # sigma. With same_on_batch off, every sample gets its own -- which
                # is what the flag asks for and what it silently did not do.
                if self.same_on_batch:
                    assert isinstance(kernel, Tensor)
                    x = apply_convolution(channel_data, kernel, dim=3)
                else:
                    x = apply_convolution_per_sample(channel_data, self._per_sample_kernels(channel_data.shape[0], input.device))
            elif self.kernel_type == "UnsharpMask":
                # blur selected channel, compute mask and add scaled mask back Isharp​=I+α(I−G​∗I)
                if self.same_on_batch:
                    assert isinstance(kernel, Tensor)
                    blurred = apply_convolution(channel_data, kernel, dim=3)
                    unsharp_amount = torch.rand(1, device=input.device) * self.unsharp_amount
                else:
                    blurred = apply_convolution_per_sample(channel_data, self._per_sample_kernels(channel_data.shape[0], input.device))
                    amount_shape = [channel_data.shape[0]] + [1] * (channel_data.dim() - 1)
                    unsharp_amount = torch.rand(channel_data.shape[0], device=input.device).view(amount_shape) * self.unsharp_amount
                mask = channel_data - blurred
                x = channel_data + unsharp_amount * mask
            elif self.kernel_type == "Scharr":
                # One convolution with three filters, not three convolutions summed:
                # the three directional kernels differ only in their weights, so they
                # fit in the output-channel axis of a single grouped conv. Same
                # arithmetic, one pad and one kernel launch instead of three each.
                weight = stacked_scharr_kernels(3, input.device, channel_data.dtype)
                padded = F.pad(channel_data.unsqueeze(1), [1] * 6, mode="reflect")
                directional = depthwise_conv3d(padded, weight, filters_per_plane=weight.shape[0])
                # `vector_norm(ord=1)` is abs-then-sum in one pass; `.abs().sum()` writes a
                # second [N, 3, D, H, W] tensor out to memory and reads it straight back.
                x = torch.linalg.vector_norm(directional, ord=1, dim=1) if self.absolute else directional.sum(dim=1)
            elif self.kernel_type == "RandConv":
                # One kernel for the batch under same_on_batch, a fresh one per
                # sample otherwise. This used to draw per sample unconditionally,
                # so `same_on_batch=True` -- which the sequential forces onto every
                # child -- did nothing here while it held for every sibling.
                if self.same_on_batch:
                    assert isinstance(kernel, Tensor)
                    x = apply_convolution(channel_data, kernel, dim=3)
                else:
                    x = apply_convolution_per_sample(channel_data, self._per_sample_kernels(channel_data.shape[0], input.device))

            # Mix with original based on mix_prob, per sample.
            #
            # This draw used to sit outside any loop, so one coin flip and one alpha
            # decided the whole batch -- unlike RandomInverseGPU and
            # RandomHistogramEqualizationGPU, which run the identical three lines
            # inside their per-sample loop. `mix_prob` is documented as "probability
            # of blending the result back with the original", which is a per-sample
            # statement.
            x = _mix_with_original(orig, x, self.mix_prob)

            if self.retain_stats:
                x = _restore_stats(x, stats)

            # Apply region selection
            checked = _select_and_check(self, orig, x, seg_mask, f" with kernel={self.kernel_type}")
            if checked is None:
                continue
            input[:, c] = checked

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
        # Via `depthwise_conv3d` rather than `F.conv3d` directly. Every row of
        # `kernel` is the same filter -- it came from an `expand` above -- so one row
        # is handed over and the helper broadcasts it back across the planes, which
        # is also what lets it make `groups > 1` out of a single-plane call. The
        # bucketed pipelines hand every transform one sample at a time, and that is
        # precisely the case cuDNN serves 35x slower than the depthwise kernel.
        leading = img.shape[:-3]
        planes = int(torch.tensor(leading).prod()) if leading else 1
        out = depthwise_conv3d(img.reshape(planes, 1, *img.shape[-3:]), kernel[:1, :1])
        img = out.reshape(*leading, *out.shape[-3:])

    img = F_t._cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def apply_convolution_per_sample(channel_data: torch.Tensor, kernels: Sequence[Tensor]) -> torch.Tensor:
    """Convolve `[N, D, H, W]` with one 3-D kernel per sample, in a single conv.

    The alternative -- and what every caller here used to do -- is a Python loop
    calling `apply_convolution` on `channel_data[b : b + 1]`. That is a
    single-channel `F.conv3d`, which cuDNN serves from its implicit-GEMM path: 4.3 ms
    per sample for a 3x3x3 kernel over a 128^3 patch on an A40, against 0.12 ms for
    the whole batch here. See `kernels.depthwise_conv3d` for why the shape matters.

    Kernels of different sizes are zero-padded up to the widest one. That is exact,
    not an approximation: a zero tap contributes nothing, and `reflect` padding gives
    the same value at a given offset however wide the pad is, so the extra taps read
    real (if irrelevant) voxels and multiply them by zero. `RandConv` draws its kernel
    size per sample, so without this the batch could not be convolved in one call.
    """
    sizes = [int(k.shape[-1]) for k in kernels]
    widest = max(sizes)
    stacked = []
    for kernel, size in zip(kernels, sizes):
        if size == widest:
            stacked.append(kernel)
        else:
            lo = (widest - size) // 2
            padded = torch.zeros((widest, widest, widest), device=kernel.device, dtype=kernel.dtype)
            padded[lo : lo + size, lo : lo + size, lo : lo + size] = kernel
            stacked.append(padded)
    weight = torch.stack(stacked).unsqueeze(1)  # [N, 1, k, k, k]

    pad = widest // 2
    padded_img = F.pad(channel_data.unsqueeze(1), [pad] * 6, mode="reflect")
    return depthwise_conv3d(padded_img, weight).squeeze(1)


## Noise transform
@register(
    aug_id=AugId.GAUSSIAN_NOISE,
    backend=Backend.GPU,
    group=AugType.GE,
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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            if self.same_on_batch:
                std = torch.rand(1, device=input.device, dtype=input.dtype) * self.std
                noise = torch.randn_like(input[:, c], device=input.device, dtype=input.dtype)
                noise = noise * std + self.mean
            else:
                std = torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * self.std
                noise = torch.randn_like(input[:, c], device=input.device, dtype=input.dtype)
                # Broadcast the per-sample std instead of writing one row at a time:
                # same draws, same arithmetic, one kernel instead of N.
                noise = noise * std.view(-1, *([1] * (noise.dim() - 1))) + self.mean

            orig = input[:, c]
            x = orig + noise
            checked = _select_and_check(self, orig, x, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input


## Multiplicative brightness transform
@register(
    aug_id=AugId.BRIGHTNESS,
    backend=Backend.GPU,
    group=AugType.GE,
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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
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
                x = channel_data * factor.view(-1, *([1] * (channel_data.dim() - 1)))
            checked = _select_and_check(self, orig, x, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            # [N, ...spatial...]
            channel_data = -input[:, c] if self.invert_image else input[:, c]
            orig_full = input[:, c].clone()

            if self.retain_stats:
                stats = _channel_stats(channel_data)

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
                channel_data = _restore_stats(channel_data, stats)

            if self.invert_image:
                channel_data = -channel_data
            checked = _select_and_check(self, orig_full, channel_data, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input


# Gamma, split so that "gamma" and "inverted gamma" are two config keys rather than
# one key plus an `invert_image` flag. Neither leaf exposes the flag, so a config
# cannot express the same augmentation two ways.


@register(
    aug_id=AugId.GAMMA,
    backend=Backend.GPU,
    group=AugType.GE,
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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()
            if self.retain_stats:
                stats = _channel_stats(channel_data)

            if self.same_on_batch:
                factor = (
                    torch.rand(1, device=input.device, dtype=input.dtype) * (self.contrast_range[1] - self.contrast_range[0])
                    + self.contrast_range[0]
                )
                mean = channel_data.mean(dim=tuple(range(1, channel_data.dim())), keepdim=True)
                x = (channel_data - mean) * factor + mean
            else:
                factor = (
                    torch.rand(input.shape[0], device=input.device, dtype=input.dtype) * (self.contrast_range[1] - self.contrast_range[0])
                    + self.contrast_range[0]
                )
                mean = channel_data.mean(dim=tuple(range(1, channel_data.dim())), keepdim=True)
                x = (channel_data - mean) * factor.view(-1, *([1] * (channel_data.dim() - 1))) + mean

            if self.retain_stats:
                x = _restore_stats(x, stats)
            checked = _select_and_check(self, orig, x, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            x = input[:, c]  # shape [N, ...spatial...]
            orig = x.clone()
            if self.retain_stats:
                stats = _channel_stats(x)

            # Normalize to make values >=0, per sample.
            #
            # This used to be a bare `x.min()` / `x.max()`, which reduces over the whole
            # [N, ...] slab: an image's augmentation then depended on which other images
            # happened to share its batch, so the same volume augmented twice in
            # different batches came out differently. Every other transform in this file
            # reduces over `dim=reduce_dims` per sample.
            keep_dims = tuple(range(1, x.dim()))
            x_min = x.amin(dim=keep_dims, keepdim=True)
            x_max = x.amax(dim=keep_dims, keepdim=True)
            x = (x - x_min) / (x_max - x_min + 0.00001)

            # Apply function
            x = self.func(x)

            if self.retain_stats:
                x = _restore_stats(x, stats)
            checked = _select_and_check(self, orig, x, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

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
)
class RandomLog1pGPU(_RandomNamedFunctionGPU):
    """Apply log(1 + x)."""

    function = staticmethod(_log1p)


@register(
    aug_id=AugId.FUNC_SQRT,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomSqrtGPU(_RandomNamedFunctionGPU):
    """Apply sqrt(x)."""

    function = staticmethod(torch.sqrt)


@register(
    aug_id=AugId.FUNC_SIN,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomSinGPU(_RandomNamedFunctionGPU):
    """Apply sin(x)."""

    function = staticmethod(torch.sin)


@register(
    aug_id=AugId.FUNC_EXP,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomExpGPU(_RandomNamedFunctionGPU):
    """Apply exp(x)."""

    function = staticmethod(torch.exp)


@register(
    aug_id=AugId.FUNC_SIGMOID,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomSigmoidGPU(_RandomNamedFunctionGPU):
    """Apply the logistic sigmoid 1 / (1 + exp(-x))."""

    function = staticmethod(_sigmoid)


## Inverse transform
@register(
    aug_id=AugId.INVERSE,
    backend=Backend.GPU,
    group=AugType.TA,
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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            # The per-sample reductions -- max, and the two retain_stats passes -- are
            # the expensive part and they are independent across samples, so they run
            # once for the whole channel. What stays in the loop is the mix draw and
            # the region selection, which both have to keep their per-sample identity:
            # `_choose_region_mode` picks `in` / `out` / `all` for each sample
            # separately, and hoisting it would silently make that one decision for the
            # batch. Keeping the draws in the loop also keeps the RNG stream unchanged.
            channel = input[:, c]
            reduce_dims = tuple(range(1, channel.dim()))
            keep_shape = (-1, *([1] * (channel.dim() - 1)))
            inverted = channel.amax(dim=reduce_dims, keepdim=True) - channel

            if self.retain_stats:
                eps = 1e-8
                orig_means = channel.mean(dim=reduce_dims).view(keep_shape)
                orig_stds = channel.std(dim=reduce_dims).view(keep_shape)
                new_mean = inverted.mean(dim=reduce_dims).view(keep_shape)
                new_std = inverted.std(dim=reduce_dims).view(keep_shape)
                inverted = (inverted - new_mean) / (new_std + eps) * orig_stds + orig_means

            for i in range(input.shape[0]):
                orig = channel[i]
                x = inverted[i]

                # Mix with original based on mix_prob
                if torch.rand(1).item() < self.mix_prob:
                    alpha = torch.rand(1, device=input.device)
                    x = alpha * orig + (1 - alpha) * x

                checked = _select_and_check(self, orig, x, None if seg_mask is None else seg_mask[i])
                if checked is None:
                    continue
                input[i, c] = checked

        return input


## Histogram transform
#: Bin count for the histogram-equalisation CDF. Was a literal, repeated four times
#: across the index arithmetic and the clamp that has to agree with it.
_HIST_BINS = 256


@register(
    aug_id=AugId.HISTOGRAM_EQUAL,
    backend=Backend.GPU,
    group=AugType.TA,
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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            # `.clone()`, not the bare `input[:, c]` view this used to take: the loop
            # below assigns into `channel_data[b]`, which through a view writes straight
            # into `input`. The NaN guard at the bottom would then `continue` over
            # values that were already in the batch -- the guard skipped nothing.
            channel_data = input[:, c].clone()  # shape [N, ...spatial...]
            orig = channel_data.clone()

            if self.retain_stats:
                stats = _channel_stats(channel_data)

            # Equalise the whole batch at once.
            #
            # This was a per-sample loop around `torch.histc`, which needs its range as
            # Python floats -- so every sample paid two `.item()` calls, and each of
            # those blocks the host until the queue drains. The histogram is built here
            # by scattering the bin indices the lookup below already needs, which is the
            # same binning `histc(bins=256, min, max)` performs and removes the stalls
            # along with the loop.
            batch_size = channel_data.shape[0]
            flat = channel_data.reshape(batch_size, -1).to(torch.float32)
            img_min = flat.amin(dim=1, keepdim=True)
            img_max = flat.amax(dim=1, keepdim=True)

            bin_width = (img_max - img_min) / _HIST_BINS
            indices = ((flat - img_min) / (bin_width + 1e-10)).long().clamp_(0, _HIST_BINS - 1)

            # `bincount` over row-offset indices, not `scatter_add_`: with only 256 bins
            # a scatter has every thread in the block contending for the same handful
            # of addresses, and measured 0.97 ms against bincount's 0.36 ms for a
            # 2 x 128^3 batch. Counts are integers, so neither is approximate.
            offset = indices + torch.arange(batch_size, device=indices.device).view(-1, 1) * _HIST_BINS
            hist = torch.bincount(offset.reshape(-1), minlength=batch_size * _HIST_BINS).view(batch_size, _HIST_BINS)

            cdf = hist.cumsum(dim=1).to(flat.dtype)
            # The smallest non-zero entry per sample. `cdf[cdf > 0].min()` cannot be
            # written per row, so the empty bins are masked to +inf and reduced; a
            # sample with no non-zero bin at all (an empty volume) falls back to the
            # plain minimum, as the scalar version did.
            positive = torch.where(cdf > 0, cdf, torch.full_like(cdf, float("inf")))
            cdf_min = positive.amin(dim=1, keepdim=True)
            cdf_min = torch.where(torch.isinf(cdf_min), cdf.amin(dim=1, keepdim=True), cdf_min)
            cdf = (cdf - cdf_min) / (cdf[:, -1:] - cdf_min + 1e-10)  # Normalize to [0,1]
            cdf = cdf * (img_max - img_min) + img_min  # Scale back to image range

            channel_data = cdf.gather(1, indices).reshape(channel_data.shape).to(channel_data.dtype)

            # Mix with original based on mix_prob, per sample. The draws stay in a loop
            # so the RNG stream is the one the scalar version produced.
            for b in range(batch_size):
                if torch.rand(1).item() < self.mix_prob:
                    alpha = torch.rand(1, device=input.device)
                    channel_data[b] = alpha * orig[b] + (1 - alpha) * channel_data[b]

            if self.retain_stats:
                channel_data = _restore_stats(channel_data, stats)

            checked = _select_and_check(self, orig, channel_data, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input


@functools.cache
def _polynomial_slot_map(order: int, dim: int, device: torch.device) -> Tensor:
    """Flat coefficient index for every monomial, or -1 where the order forbids it.

    Shaped `(order+1,) * dim`, indexed by the per-axis exponents in the same
    `x, y[, z]` order `RandomBiasFieldGPU._num_coeffs` counts them.
    """
    size = order + 1
    slots = torch.full((size,) * dim, -1, dtype=torch.long)
    index = 0
    if dim == 3:
        for xo in range(size):
            for yo in range(size - xo):
                for zo in range(size - (xo + yo)):
                    slots[xo, yo, zo] = index
                    index += 1
    elif dim == 2:
        for xo in range(size):
            for yo in range(size - xo):
                slots[xo, yo] = index
                index += 1
    else:
        raise ValueError("Only 2D or 3D spatial dims supported for bias field")
    return slots.to(device)


@functools.cache
def _axis_power_tables(order: int, spatial: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
    """`[order+1, axis_length]` tables of `coordinate ** exponent`, one per axis.

    Returned innermost-axis-first (x, y[, z]) to match the exponent order of
    `_polynomial_slot_map`. Cached: the tables depend only on the patch shape, which
    is fixed for a training run, while the coefficients are redrawn every call.
    """
    exponents = torch.arange(order + 1, device=device, dtype=dtype).view(-1, 1)
    axes = []
    for length in reversed(spatial):  # spatial is (D, H, W); x runs along W
        coordinate = torch.linspace(-1, 1, length, device=device, dtype=dtype).view(1, -1)
        axes.append(coordinate.pow(exponents))
    return tuple(axes)


@register(
    aug_id=AugId.BIAS_FIELD,
    backend=Backend.GPU,
    group=AugType.TA,
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

    def _coefficient_cube(self, coeffs: torch.Tensor, dim: int) -> torch.Tensor:
        """Scatter the flat `(n_coeffs, B)` draw into a dense `(B, order+1, ...)` cube.

        The flat order is the nested `xo / yo / zo` walk the term loop used, so the
        coefficient a given monomial gets is unchanged; the cube is zero wherever
        `xo + yo + zo > order`, which is exactly the monomials that walk skipped.
        """
        order = self.order
        slot = _polynomial_slot_map(order, dim, coeffs.device)
        used = slot >= 0
        cube = torch.zeros((coeffs.shape[1], *slot.shape), device=coeffs.device, dtype=coeffs.dtype)
        cube[:, used] = coeffs[slot[used]].transpose(0, 1)
        return cube

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        # input: (N, C, [D,] H, W)
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        if input.dim() not in (4, 5):
            raise ValueError("Expected 4D or 5D tensor (N,C,...) for RandomBiasFieldGPU")
        batch_size = input.shape[0]
        spatial = input.shape[2:]
        dim = len(spatial)
        device = input.device
        dtype = input.dtype

        coeffs = self._sample_coeffs(batch_size, device, dtype, dim)  # (n_coeffs, B)
        seg_mask = params.get("seg")

        # Evaluate the polynomial as a separable contraction rather than a term loop.
        #
        # Every monomial is a product of three one-dimensional powers, so the sum over
        # monomials factorises: contract the coefficient cube against the axis power
        # tables one axis at a time and the full-volume work collapses to the last
        # step, a single matrix multiply. The loop this replaces materialised one
        # [B, D, H, W] temporary per monomial -- twenty of them at the default order 3,
        # each read and written in full -- to accumulate the same sum.
        cube = self._coefficient_cube(coeffs, dim)
        powers = _axis_power_tables(self.order, spatial, device, dtype)
        if dim == 3:
            x_pow, y_pow, z_pow = powers  # each (order+1, axis_length)
            partial = torch.einsum("bxyz,zd->bxyd", cube, z_pow)
            partial = torch.einsum("bxyd,yh->bxdh", partial, y_pow)
            bias_map = torch.einsum("bxdh,xw->bdhw", partial, x_pow)
        else:  # dim == 2
            x_pow, y_pow = powers
            partial = torch.einsum("bxy,yh->bxh", cube, y_pow)
            bias_map = torch.einsum("bxh,xw->bhw", partial, x_pow)

        # Exponential to ensure positive field
        bias_field = torch.exp(bias_map)  # (N, *spatial)
        if self.invert:
            bias_field = 1.0 / (bias_field + 1e-8)

        # Apply to channels
        for c in self.apply_to_channel:
            _check_channel(self, c, input.shape[1])
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
            checked = _select_and_check(self, orig, channel, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input


def _batched_quantiles(channel_data: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample `torch.quantile` at a per-sample probability, in one sort.

    `torch.quantile` applies every `q` it is given to every row, so a per-sample
    probability forced a Python loop with two calls per sample -- and each call sorts
    the row again. Here the rows are sorted once and both quantiles are read off by
    index, with `quantile`'s own linear interpolation
    (`lower + frac * (upper - lower)`), so the values are the ones it would return.

    The default `max_clamp_amount=0` asks for the 0th and 100th percentile, which is
    the volume's own range and makes the clamp a no-op; that case skips the sort.
    """
    batch_size = channel_data.shape[0]
    flat = channel_data.reshape(batch_size, -1)
    if bool((lower == 0).all()) and bool((upper == 1).all()):
        return flat.amin(dim=1), flat.amax(dim=1)

    ordered, _ = torch.sort(flat, dim=1)
    last = ordered.shape[1] - 1

    def pick(prob: torch.Tensor) -> torch.Tensor:
        position = prob.clamp(0.0, 1.0).to(ordered.dtype) * last
        low_idx = position.floor().long().clamp(0, last)
        high_idx = position.ceil().long().clamp(0, last)
        frac = (position - low_idx.to(position.dtype)).unsqueeze(1)
        low_val = ordered.gather(1, low_idx.unsqueeze(1))
        high_val = ordered.gather(1, high_idx.unsqueeze(1))
        return (low_val + frac * (high_val - low_val)).squeeze(1)

    return pick(lower), pick(upper)


# Random clamping transform
@register(
    aug_id=AugId.CLAMP,
    backend=Backend.GPU,
    group=AugType.GE,
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
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            channel_data = input[:, c]  # [N, ...spatial...]
            orig = channel_data.clone()
            if self.retain_stats:
                stats = _channel_stats(channel_data)

            batch_size = input.shape[0]
            if self.same_on_batch:
                min_percentile = (torch.rand(1, device=input.device, dtype=input.dtype) * self.max_clamp_amount).expand(batch_size)
                max_percentile = (1.0 - (torch.rand(1, device=input.device, dtype=input.dtype) * self.max_clamp_amount)).expand(batch_size)
            else:
                # [N, 2] rather than 2N scalar draws: the row-major fill order is the
                # same (min, max) per sample the loop produced.
                draws = torch.rand(batch_size, 2, device=input.device, dtype=input.dtype) * self.max_clamp_amount
                min_percentile = draws[:, 0]
                max_percentile = 1.0 - draws[:, 1]

            lo, hi = _batched_quantiles(channel_data, min_percentile, max_percentile)
            bounds_shape = (-1, *([1] * (channel_data.dim() - 1)))
            x = torch.clamp(channel_data, lo.view(bounds_shape), hi.view(bounds_shape))

            if self.retain_stats:
                x = _restore_stats(x, stats)
            checked = _select_and_check(self, orig, x, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input


@register(
    aug_id=AugId.ZSCORE,
    backend=Backend.GPU,
    group=AugType.GE,
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
        # Not a constructor parameter: this transform has no mix_in_out knob, and the
        # registry derives a config's accepted keys from __init__, so adding one there
        # would invent a setting. Set here so the region-selection helper's contract
        # holds for every transform that uses it.
        self.mix_in_out = False

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        # input: (N, C, [D,] H, W)
        # A clone, not the caller's tensor: this method writes channels back with
        # `input[:, c] = ...`, and kornia hands the caller's own tensor straight
        # through when every sample applies. Every transform in gpu/spatial.py and
        # the palette/domain-transfer transforms already clone; these did not, so
        # `batch["data"]` was destroyed under any caller holding a reference.
        input = input.clone()
        seg_mask = params.get("seg")
        for c in self.apply_to_channel:
            _check_channel(self, c, input.shape[1])
            channel = input[:, c]
            orig = channel.clone()
            reduce_dims = tuple(range(1, channel.dim()))
            mean = channel.mean(dim=reduce_dims, keepdim=True)
            # use unbiased=False for stability, and clamp std to avoid division by ~0
            std = channel.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(1e-8)
            # The clamp stops the NaN but not the nonsense: the mean is still *computed*,
            # so a constant volume leaves floating-point residue behind, and dividing that
            # by 1e-8 scales it by a hundred million. A constant -2.709 patch -- clipped
            # CT air -- came back as a constant -1.0, an O(1) value decided purely by
            # rounding. There is no z-score of a volume with no variance; leave it alone.
            #
            # Degeneracy is tested on the range rather than the standard deviation
            # because `amax - amin` is exactly zero for a constant tensor while a summed
            # std is not: the mean of N identical floats does not round back to the value
            # itself, so `std == 0` misses precisely the patches this is here for.
            flat = channel.reshape(channel.shape[0], -1)
            spread = (flat.amax(dim=1) - flat.amin(dim=1)).view(-1, *([1] * (channel.dim() - 1)))
            channel = torch.where(spread == 0, channel, (channel - mean) / std)
            # No mix_in_out here: z-scoring is applied whole, never to a random subset
            # of the mask channels.
            checked = _select_and_check(self, orig, channel, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input
