"""Whole-volume intensity standardizations, and the schemes nnU-Net only has offline.

`ZscoreNormalizationGPU` in `gpu/contrast.py` was the only normalization reachable from
a GPU config. Every other scheme a medical imaging pipeline uses -- min-max, the CT-style
percentile clip, a robust median/IQR scaling, divide-by-max, divide-by-mean -- existed
only inside nnU-Net's preprocessing
(`nnunetv2/preprocessing/normalization/default_normalization_schemes.py`), which runs once
per dataset, long before a patch reaches a training step.

Putting them here makes the *choice of scheme* an augmentation: the builder puts this
group in a bucket that picks one member per draw (see `AugType.NORM` and
`build._normalization_stage`), so the network cannot come to depend on one fixed intensity
standardization. Two consequences of running at augmentation time rather than in
preprocessing are worth knowing, because they shape the parameters below:

* The patch has already been normalized once, by preprocessing. These schemes therefore
  act on standardized intensities, not on HU. A CT window in HU means nothing here unless
  its bounds are given explicitly, which is why `PercentileNormalizationGPU` takes both an
  explicit `clip_range` (nnU-Net parity, for a config that knows its dataset) and
  per-sample `percentiles` (the useful augmentation).
* The global mean of z-scored data is ~0, so dividing by it amplifies rounding noise by
  1/mean. Hence `min_scale` -- a sample whose scale is negligible against its own range is
  left alone -- and `stats_from="nonzero"` as `MeanScaleNormalizationGPU`'s default.

Every scheme is a per-sample affine map with an optional clip in front of it, so the base
class owns the clone, the channel guard, the statistics mask, both degeneracy guards and
the region selection, and a leaf supplies only `_clip_bounds` and `_statistics`. That is
the same split `contrast.py` already uses for its gamma, convolution and function
families.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform, segmentation_from

# Private to contrast.py, imported rather than copied: these four are the house rules for
# an intensity transform (clone, reject an out-of-range channel, apply the in_seg/out_seg
# region selection, refuse a non-finite result) and a second copy would be a second
# behaviour. The dependency is one-way -- contrast.py needs nothing from here.
from smauglab.transforms.gpu.contrast import _check_channel, _foreground, _select_and_check

#: Which voxels a scheme reads its statistics from. `"seg"` is the labelled anatomy, via
#: the mask the pipeline injects; `"nonzero"` is everything the preprocessing did not zero
#: out, which is the closest thing here to nnU-Net's `use_mask_for_norm`.
STATS_SOURCES = ("all", "seg", "nonzero")

#: Floor on a divisor's magnitude. Matches nnU-Net's `max(std, 1e-8)` so the parity tests
#: compare the same arithmetic, not an approximation of it.
_EPS = 1e-8

#: IQR and MAD of a standard normal. Dividing by these is what makes the robust scaling
#: comparable to a standard deviation instead of ~1.35x or ~0.67x of one.
_NORMAL_IQR = 1.34898
_NORMAL_MAD = 0.67449


def _broadcast(values: Tensor, like: Tensor) -> Tensor:
    """Reshape a per-sample `[N]` tensor to broadcast over `like`'s spatial dims."""
    return values.view(-1, *([1] * (like.dim() - 1)))


def _resolve_mask(transform: _NormalizationBaseGPU, channel: Tensor, seg_mask: Tensor | None) -> Tensor | None:
    """The voxels this transform's statistics come from, or None for all of them.

    A sample whose mask selects nothing has no statistics to read, and letting that
    through would produce inf, trip the non-finite guard and print a warning for what is
    just an unlabelled patch. Those rows fall back to the whole volume instead. The
    fallback is unconditional rather than behind an `if`: the test is a device tensor and
    reading it back to the host would stall the stream on every batch.
    """
    if transform.stats_from == "all":
        return None
    if transform.stats_from == "nonzero":
        mask = channel.abs() > transform.nonzero_threshold
    else:  # "seg"
        if seg_mask is None:
            return None
        mask = _foreground(seg_mask, 1)
    reduce_dims = tuple(range(1, mask.dim()))
    return mask | _broadcast(~mask.any(dim=reduce_dims), mask)


def _masked_extrema(channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Per-sample min and max over the masked voxels, as `[N]` tensors."""
    flat = channel.reshape(channel.shape[0], -1)
    if mask is None:
        return flat.amin(dim=1), flat.amax(dim=1)
    flat_mask = mask.reshape(channel.shape[0], -1)
    return (
        flat.masked_fill(~flat_mask, float("inf")).amin(dim=1),
        flat.masked_fill(~flat_mask, float("-inf")).amax(dim=1),
    )


def _masked_moments(channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Per-sample mean and population std over the masked voxels, as `[N]` tensors.

    `unbiased=False`, because nnU-Net normalizes with `np.std` (ddof=0) and so does
    `ZscoreNormalizationGPU`; an unbiased estimate here would make the parity tests fail
    on a difference that is not a bug.
    """
    reduce_dims = tuple(range(1, channel.dim()))
    if mask is None:
        return channel.mean(dim=reduce_dims), channel.std(dim=reduce_dims, unbiased=False)
    weights = mask.to(channel.dtype)
    count = weights.sum(dim=reduce_dims).clamp_min(1.0)
    mean = (channel * weights).sum(dim=reduce_dims) / count
    deviation = channel - _broadcast(mean, channel)
    variance = (deviation.pow(2) * weights).sum(dim=reduce_dims) / count
    return mean, variance.sqrt()


def _masked_quantiles(channel: Tensor, mask: Tensor | None, probabilities: Sequence[float]) -> list[Tensor]:
    """Per-sample quantiles over the masked voxels, in one batched sort.

    The other half of the problem `contrast._batched_quantiles` solves: there the
    probability varies per sample over the whole volume, here the probabilities are shared
    and the *population* varies per sample. Out-of-mask voxels are pushed to +inf so they
    sort to the end, and each row is then indexed by its own mask count, with
    `torch.quantile`'s linear interpolation (`low + frac * (high - low)`) so the values are
    the ones it would return. No per-sample Python loop and no host read.
    """
    batch = channel.shape[0]
    flat = channel.reshape(batch, -1)
    if mask is None:
        ordered, _ = torch.sort(flat, dim=1)
        counts = torch.full((batch,), flat.shape[1], device=channel.device, dtype=torch.long)
    else:
        flat_mask = mask.reshape(batch, -1)
        ordered, _ = torch.sort(flat.masked_fill(~flat_mask, float("inf")), dim=1)
        counts = flat_mask.sum(dim=1)
    last = (counts - 1).clamp_min(0)

    picked = []
    for probability in probabilities:
        position = last.to(ordered.dtype) * float(probability)
        low_idx = position.floor().long()
        high_idx = position.ceil().long()
        frac = (position - low_idx.to(position.dtype)).unsqueeze(1)
        low_val = ordered.gather(1, low_idx.unsqueeze(1))
        high_val = ordered.gather(1, high_idx.unsqueeze(1))
        picked.append((low_val + frac * (high_val - low_val)).squeeze(1))
    return picked


class _NormalizationBaseGPU(ImageOnlyTransform):
    """Clone, mask, clip, affine-map, guard, write back -- everything but the scheme.

    A leaf implements `_statistics` (and `_clip_bounds` if it clips), each returning
    per-sample `[N]` tensors; the base broadcasts them. `same_on_batch` is absent for the
    same reason it is absent from `ZscoreNormalizationGPU`: a normalization draws no random
    parameters, so there is nothing for the flag to hold the same.

    Args:
        apply_to_channel: Channels to normalize.
        stats_from: Which voxels the statistics come from; one of `STATS_SOURCES`.
        nonzero_threshold: `|x|` above which a voxel counts when `stats_from="nonzero"`.
        min_scale: Leave a sample alone when its divisor is below this fraction of the
            sample's standard deviation. The guard that makes dividing by a near-zero mean
            safe.
        in_seg: Probability of applying the result inside the segmentation only.
        out_seg: Probability of applying it outside the segmentation only.
        p: Application probability.
        p_batch: Batch-wise application probability.
        keepdim: Keep the input's dimensions (passed to the base).
    """

    def __init__(
        self,
        apply_to_channel: Sequence[int] = (0,),
        stats_from: str = "all",
        nonzero_threshold: float = 1e-6,
        min_scale: float = 1e-3,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=False, keepdim=keepdim)
        if stats_from not in STATS_SOURCES:
            raise ValueError(f"{type(self).__name__}: stats_from must be one of {STATS_SOURCES}, got {stats_from!r}.")
        self.apply_to_channel = apply_to_channel
        self.stats_from = stats_from
        self.nonzero_threshold = float(nonzero_threshold)
        self.min_scale = float(min_scale)
        self.in_seg = in_seg
        self.out_seg = out_seg
        # Not a constructor parameter: these schemes have no mix_in_out knob and the
        # registry derives a config's accepted keys from __init__, so adding one there
        # would invent a setting. Set here so the region-selection helper's contract holds.
        self.mix_in_out = False

    def _clip_bounds(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor] | None:
        """Per-sample `(lo, hi)` to clamp to before the affine map, or None for no clip."""
        return None

    def _statistics(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        """Per-sample `(center, scale, offset)` for `y = (x - center) / scale + offset`."""
        raise NotImplementedError

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        # input: (N, C, [D,] H, W)
        # Clone: kornia passes the caller's tensor through; this writes channels back.
        input = input.clone()
        seg_mask = segmentation_from(params)
        for c in self.apply_to_channel:
            _check_channel(self, c, input.shape[1])
            channel = input[:, c]
            orig = channel.clone()
            mask = _resolve_mask(self, channel, seg_mask)

            flat = channel.reshape(channel.shape[0], -1)
            spread = _broadcast(flat.amax(dim=1) - flat.amin(dim=1), channel)
            deviation = _broadcast(_masked_moments(channel, mask)[1], channel)

            bounds = self._clip_bounds(channel, mask)
            clipped = channel if bounds is None else torch.clamp(channel, _broadcast(bounds[0], channel), _broadcast(bounds[1], channel))
            center, scale, offset = (_broadcast(t, channel) for t in self._statistics(clipped, mask))

            # Two guards, both "leave this sample as it is":
            #
            # * No range at all. There is no normalization of a volume with no variance,
            #   and the clamp below would turn floating-point residue into an O(1) value
            #   decided by rounding -- `ZscoreNormalizationGPU` returned a constant -1.0
            #   for a constant -2.709 CT-air patch before it grew the same guard. Tested on
            #   the range rather than the scale: `amax - amin` is exactly zero for a
            #   constant tensor while a summed standard deviation is not.
            # * A divisor negligible against the sample's own standard deviation. This is
            #   what a mean-scaling scheme hits on already-z-scored data, whose mean is ~0:
            #   the result is finite, so nothing downstream rejects it, and it is still
            #   noise times 1e8.
            #
            # The second guard is measured against the standard deviation and not against
            # `spread`, which was the first thing tried and is wrong for the robust
            # schemes: one 1e4 voxel puts the range four orders of magnitude above the IQR,
            # so a range-relative floor declined to run on exactly the volume
            # `RobustZscoreNormalizationGPU` exists for. The standard deviation moves with
            # the outlier (240 rather than 2.9 on that volume) and so stays in proportion
            # to any scale a scheme might derive from the same data.
            applies = (spread > 0) & (scale.abs() >= self.min_scale * deviation)
            # Sign-preserving floor, not `clamp_min`: a negative scale is meaningful
            # (dividing by a negative mean inverts the volume, which is what the scheme
            # says), and `clamp_min(-0.5, 1e-8)` would return 1e-8 and flip it.
            safe = torch.where(scale.abs() < _EPS, torch.full_like(scale, _EPS), scale)
            normalized = torch.where(applies, (clipped - center) / safe + offset, channel)

            checked = _select_and_check(self, orig, normalized, seg_mask)
            if checked is None:
                continue
            input[:, c] = checked

        return input


@register(
    aug_id=AugId.NORM_MINMAX,
    backend=Backend.GPU,
    group=AugType.NORM,
)
class MinMaxNormalizationGPU(_NormalizationBaseGPU):
    """Rescale each sample's intensity range onto a fixed interval.

    At its defaults this is nnU-Net's `RescaleTo01Normalization`, including its floored
    denominator. `clip_percentiles` clips to a percentile window first, which is what makes
    the result robust to a single hot voxel -- and a different augmentation, since the
    voxels outside the window are flattened onto the interval's ends.

    Args:
        out_range: Interval to map onto. Must be non-empty.
        clip_percentiles: `[lower, upper]` in [0, 1] to clip to first, or None for the
            sample's own min and max.
    """

    def __init__(
        self,
        out_range: Sequence[float] = (0.0, 1.0),
        clip_percentiles: Sequence[float] | None = None,
        apply_to_channel: Sequence[int] = (0,),
        stats_from: str = "all",
        nonzero_threshold: float = 1e-6,
        min_scale: float = 1e-3,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            apply_to_channel=apply_to_channel,
            stats_from=stats_from,
            nonzero_threshold=nonzero_threshold,
            min_scale=min_scale,
            in_seg=in_seg,
            out_seg=out_seg,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )
        low, high = float(out_range[0]), float(out_range[1])
        if high <= low:
            raise ValueError(f"MinMaxNormalizationGPU: out_range must be increasing, got {tuple(out_range)!r}.")
        self.out_range = (low, high)
        self.clip_percentiles = None if clip_percentiles is None else (float(clip_percentiles[0]), float(clip_percentiles[1]))

    def _clip_bounds(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor] | None:
        if self.clip_percentiles is None:
            return None
        low, high = _masked_quantiles(channel, mask, self.clip_percentiles)
        return low, high

    def _statistics(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        low, high = _masked_extrema(channel, mask)
        out_low, out_high = self.out_range
        return low, (high - low) / (out_high - out_low), torch.full_like(low, out_low)


@register(
    aug_id=AugId.NORM_PERCENTILE,
    backend=Backend.GPU,
    group=AugType.NORM,
)
class PercentileNormalizationGPU(_NormalizationBaseGPU):
    """Clip to a percentile window, then standardize. nnU-Net's CT scheme, generalized.

    With `clip_range`, `mean` and `std` supplied this is exactly
    `CTNormalization`: nnU-Net reads those three from the dataset's intensity properties,
    which nothing hands a transform at training time, so a config that knows them states
    them. Left as None they are measured per sample -- the same scheme, estimated from the
    patch, which is the form that works as an augmentation.

    Args:
        percentiles: `[lower, upper]` in [0, 1] to clip to when `clip_range` is None.
        clip_range: Explicit `[lower, upper]` intensities to clip to, overriding
            `percentiles`.
        mean: Center to subtract, or None to use the clipped sample's own mean.
        std: Divisor, or None to use the clipped sample's own standard deviation.
    """

    def __init__(
        self,
        percentiles: Sequence[float] = (0.005, 0.995),
        clip_range: Sequence[float] | None = None,
        mean: float | None = None,
        std: float | None = None,
        apply_to_channel: Sequence[int] = (0,),
        stats_from: str = "all",
        nonzero_threshold: float = 1e-6,
        min_scale: float = 1e-3,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            apply_to_channel=apply_to_channel,
            stats_from=stats_from,
            nonzero_threshold=nonzero_threshold,
            min_scale=min_scale,
            in_seg=in_seg,
            out_seg=out_seg,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )
        self.percentiles = (float(percentiles[0]), float(percentiles[1]))
        self.clip_range = None if clip_range is None else (float(clip_range[0]), float(clip_range[1]))
        self.mean = None if mean is None else float(mean)
        self.std = None if std is None else float(std)

    def _clip_bounds(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor] | None:
        if self.clip_range is not None:
            batch = (channel.shape[0],)
            return (
                channel.new_full(batch, self.clip_range[0]),
                channel.new_full(batch, self.clip_range[1]),
            )
        low, high = _masked_quantiles(channel, mask, self.percentiles)
        return low, high

    def _statistics(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        center, scale = _masked_moments(channel, mask)
        if self.mean is not None:
            center = torch.full_like(center, self.mean)
        if self.std is not None:
            scale = torch.full_like(scale, self.std)
        return center, scale, torch.zeros_like(center)


@register(
    aug_id=AugId.NORM_ROBUST_ZSCORE,
    backend=Backend.GPU,
    group=AugType.NORM,
)
class RobustZscoreNormalizationGPU(_NormalizationBaseGPU):
    """Z-score on robust statistics: the median, scaled by the IQR or the MAD.

    The scheme to reach for when the patch has outliers a mean and standard deviation
    would chase -- metal artefacts, a bright rim, a fat saturation band. `unit_variance`
    divides the spread by its value for a standard normal, so the output is on the same
    scale as an ordinary z-score rather than ~1.35x or ~0.67x of it.

    Args:
        percentiles: `[lower, upper]` defining the IQR. Ignored when `spread="mad"`.
        spread: `"iqr"` or `"mad"`.
        unit_variance: Rescale the spread so Gaussian input lands at unit variance.
    """

    SPREADS = ("iqr", "mad")

    def __init__(
        self,
        percentiles: Sequence[float] = (0.25, 0.75),
        spread: str = "iqr",
        unit_variance: bool = True,
        apply_to_channel: Sequence[int] = (0,),
        stats_from: str = "all",
        nonzero_threshold: float = 1e-6,
        min_scale: float = 1e-3,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            apply_to_channel=apply_to_channel,
            stats_from=stats_from,
            nonzero_threshold=nonzero_threshold,
            min_scale=min_scale,
            in_seg=in_seg,
            out_seg=out_seg,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )
        if spread not in self.SPREADS:
            raise ValueError(f"RobustZscoreNormalizationGPU: spread must be one of {self.SPREADS}, got {spread!r}.")
        self.percentiles = (float(percentiles[0]), float(percentiles[1]))
        self.spread = spread
        self.unit_variance = bool(unit_variance)

    def _statistics(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        (median,) = _masked_quantiles(channel, mask, (0.5,))
        if self.spread == "iqr":
            low, high = _masked_quantiles(channel, mask, self.percentiles)
            scale = high - low
            if self.unit_variance:
                scale = scale / _NORMAL_IQR
        else:
            deviation = (channel - _broadcast(median, channel)).abs()
            (scale,) = _masked_quantiles(deviation, mask, (0.5,))
            if self.unit_variance:
                scale = scale / _NORMAL_MAD
        return median, scale, torch.zeros_like(median)


@register(
    aug_id=AugId.NORM_MAX_SCALE,
    backend=Backend.GPU,
    group=AugType.NORM,
)
class MaxScaleNormalizationGPU(_NormalizationBaseGPU):
    """Divide by the sample's maximum, leaving zero where it is.

    The MRI convention that a z-score does not cover: it fixes the scale without moving
    the origin, so a background that preprocessing set to zero stays at zero. `percentile`
    uses a high quantile instead of the maximum, which keeps one hot voxel from deciding
    the scale for the whole volume -- the values above it then exceed the nominal 1.

    Args:
        use_abs: Scale by `max|x|` rather than by `max x`, so negative data (a z-scored
            patch) does not get a scale smaller than its own amplitude.
        percentile: Quantile in [0, 1] to use in place of the maximum, or None.
    """

    def __init__(
        self,
        use_abs: bool = True,
        percentile: float | None = None,
        apply_to_channel: Sequence[int] = (0,),
        stats_from: str = "all",
        nonzero_threshold: float = 1e-6,
        min_scale: float = 1e-3,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            apply_to_channel=apply_to_channel,
            stats_from=stats_from,
            nonzero_threshold=nonzero_threshold,
            min_scale=min_scale,
            in_seg=in_seg,
            out_seg=out_seg,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )
        self.use_abs = bool(use_abs)
        self.percentile = None if percentile is None else float(percentile)

    def _statistics(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        source = channel.abs() if self.use_abs else channel
        if self.percentile is None:
            _, scale = _masked_extrema(source, mask)
        else:
            (scale,) = _masked_quantiles(source, mask, (self.percentile,))
        return torch.zeros_like(scale), scale, torch.zeros_like(scale)


@register(
    aug_id=AugId.NORM_MEAN_SCALE,
    backend=Backend.GPU,
    group=AugType.NORM,
)
class MeanScaleNormalizationGPU(_NormalizationBaseGPU):
    """Divide by the sample's mean, leaving zero where it is.

    The other scale-only MRI scheme, and the one that needs its statistics masked: the
    mean of a z-scored volume is ~0 over every voxel, so `stats_from` defaults to
    `"nonzero"` here. Where the mean is still negligible against the sample's range the
    base class leaves that sample alone rather than dividing by it; see `min_scale`.
    """

    def __init__(
        self,
        apply_to_channel: Sequence[int] = (0,),
        stats_from: str = "nonzero",
        nonzero_threshold: float = 1e-6,
        min_scale: float = 1e-3,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(
            apply_to_channel=apply_to_channel,
            stats_from=stats_from,
            nonzero_threshold=nonzero_threshold,
            min_scale=min_scale,
            in_seg=in_seg,
            out_seg=out_seg,
            p=p,
            p_batch=p_batch,
            keepdim=keepdim,
        )

    def _statistics(self, channel: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        mean, _ = _masked_moments(channel, mask)
        return torch.zeros_like(mean), mean, torch.zeros_like(mean)
