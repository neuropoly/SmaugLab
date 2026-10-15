"""Snapshot the image's intensity profile before the augmentations, restore it after.

The per-transform `retain_stats` flag does this one transform at a time: ~15 GPU
transforms each measure their channel's mean and standard deviation, do their work, and
rescale the result back. Three things are wrong with that. It restores once per
transform, so a chain of them pays for N restores to end up where the last one puts it.
It knows only two statistics, so a transform that manufactures extreme voxels
(`RandomInverseGPU`, `RandomScharrGPU`, `RandomRandConvGPU`) drags the standard deviation
and the restore with it. And it restores the input's statistics *exactly*, which teaches
the network that its input scale is reliable -- the opposite of what a domain-generalising
augmentation pipeline is for.

This module is the pipeline-level replacement: one snapshot after the geometry, one
restore after the augmentations, four statistics instead of two, and a jitter on the
targets so the profile the network sees is never quite the one the patch arrived with.
`smauglab.transforms.build` emits the pair around the stack; see `build_gpu_pipeline`.

**Two of four.** The snapshot is the per-sample, per-channel mean, standard deviation,
1st and 99th percentile. All four cannot be restored at once: an affine map `y = ax + b`
has two degrees of freedom, and four targets over-determine it. So each pass *draws two*
of the four and solves for the map that puts exactly those on target. That turns the
over-determination into a feature -- the drawn pair is itself an augmentation, and the
six pairs cover the whole range from "keep the moments" (mean, std -- what `retain_stats`
did) to "keep the robust dynamic range" (p1, p99 -- immune to the outliers that make the
moments a bad target after an edge filter).

Writing each statistic as a linear functional of the map, `t = a * u + b * c` with `c = 1`
for the three that move with an offset and `c = 0` for the standard deviation, which is
offset-invariant, makes all six pairs one 2x2 solve:

    det = u1 * c2 - u2 * c1
    a   = (t1 * c2 - t2 * c1) / det
    b   = t1 - a * u1   (or t2 - a * u2 when the first drawn statistic is the std)

`det` is a difference of intensities -- a span between two anchors, or the standard
deviation itself -- so a sample whose span has collapsed (a constant patch, or
`p1 == p99` on a nearly empty mask) is left exactly as it is, the same decision
`ZscoreNormalizationGPU` and the normalization schemes make for the same reason.

**What spends random numbers.** With no `pipeline.retain_stats` block nothing here is
built and the RNG stream is untouched. With one, the restore spends a fixed five uniforms
per channel -- pair, std scale, two percentile scales, gate -- whichever pair comes out
and whatever the ranges are, so two runs that differ only in the configured jitter stay
comparable at a fixed seed. That is invariant 2 of `gpu/regions.py`: the number of draws
is a function of the configuration, never of the data.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from smauglab.transforms.gpu.base import ImageOnlyTransform, segmentation_from
from smauglab.transforms.gpu.contrast import _check_channel
from smauglab.transforms.gpu.mixing import normalise_blend_range
from smauglab.transforms.gpu.normalization import STATS_SOURCES, _broadcast, _masked_moments, _masked_quantiles, _resolve_mask

#: The four statistics, in the one canonical order. The index into this tuple is what the
#: drawn pairs, the kinds below and the `[4, N]` stacks all agree on.
STAT_NAMES = ("mean", "std", "p1", "p99")

#: Each statistic's coefficient on the affine map's offset: 1 for the three that move with
#: it, 0 for the standard deviation, which does not. This is the whole reason one solver
#: serves all six pairs.
_STAT_OFFSET_COEFFICIENTS = (1.0, 0.0, 1.0, 1.0)

#: Default percentile pair, as percentages. The 1st and 99th are what the request asked
#: for and are far enough from the extremes to be a robust span on a patch of any size.
DEFAULT_PERCENTILES = (1.0, 99.0)

#: Default jitter: none. An absent `pipeline.retain_stats` block is off, and a present one
#: that names no scale restores the snapshot exactly -- the pipeline-level equivalent of
#: `retain_stats: true`, which is the behaviour a config inherits if it asks for nothing.
DEFAULT_SCALE = (1.0, 1.0)

#: A span below this fraction of the sample's own range makes the solve meaningless. Same
#: value and same intent as `min_scale` in gpu/normalization.py.
_DET_FLOOR = 1e-3

#: Sign-preserving floor on the determinant, so the division is finite even for the rows
#: the guard is about to discard anyway.
_EPS = 1e-8


@dataclass(frozen=True)
class StatsSpec:
    """A parsed `pipeline.retain_stats` block.

    Built by `parse_spec` from the config, and frozen because both leaves of one bracket
    read the same instance: a restore that disagreed with its snapshot about which
    percentiles were measured would silently restore the wrong thing.
    """

    stats: tuple[str, ...] = STAT_NAMES
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES
    std_scale: tuple[float, float] = DEFAULT_SCALE
    percentile_scale: tuple[float, float] = DEFAULT_SCALE
    stats_from: str = "all"
    nonzero_threshold: float = 1e-6
    p: float = 1.0
    #: Which channels the bracket covers. None -- the default -- is every channel, which
    #: is what makes this a property of the image rather than of one transform's
    #: `apply_to_channel`. A multi-sequence input gets an independent draw per channel.
    apply_to_channel: tuple[int, ...] | None = None

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Index pairs into `STAT_NAMES`: the table the per-batch draw indexes into."""
        return tuple(itertools.combinations((STAT_NAMES.index(n) for n in self.stats), 2))

    @property
    def probabilities(self) -> tuple[float, float]:
        """The percentile pair as quantile probabilities, for `_masked_quantiles`."""
        return (self.percentiles[0] / 100.0, self.percentiles[1] / 100.0)


#: Keys a `pipeline.retain_stats` block may hold. Checked by `smauglab/config.py` so a
#: typo is reported with the rest of a config's problems rather than at build time.
SPEC_KEYS = ("stats", "percentiles", "std_scale", "percentile_scale", "stats_from", "nonzero_threshold", "p", "apply_to_channel")

_WHERE = "pipeline.retain_stats"


def parse_spec(options: dict[str, Any] | None) -> StatsSpec | None:
    """A validated `StatsSpec`, or None when the feature is off.

    Off means the block is *absent*, and then `build_gpu_pipeline` emits no leaves at all
    -- not a pair of no-ops. The difference is visible in the RNG stream. A block that is
    present but empty is on, with every default, which restores the drawn pair exactly;
    that is the one reading of `"retain_stats": {}` an author could mean.

    Value checks live here rather than in `smauglab/config.py`, which only ever asks
    whether a key is accepted; that is where every other parameter in this package is
    checked too. A pipeline is built before the first batch, so a bad range still fails
    long before training.
    """
    if options is None:
        return None

    names = options.get("stats", STAT_NAMES)
    if isinstance(names, str) or not isinstance(names, Sequence):
        # ValueError, not the TypeError TRY004 asks for: every rejection in this parser is
        # one, and a caller should only have to catch one thing for a bad block.
        raise ValueError(f"{_WHERE}.stats must be a list of names from {', '.join(STAT_NAMES)}, got {names!r}")  # noqa: TRY004
    unknown = [n for n in names if n not in STAT_NAMES]
    if unknown:
        raise ValueError(f"{_WHERE}.stats: unknown name(s) {', '.join(map(repr, unknown))}. Accepted: {', '.join(STAT_NAMES)}.")
    # Canonicalised, not taken in the order written: the pair table is indexed by a draw,
    # so the order of the keys in a config would otherwise decide which pair a seed picks.
    stats = tuple(n for n in STAT_NAMES if n in set(names))
    if len(stats) < 2:
        raise ValueError(f"{_WHERE}.stats needs at least two statistics to determine an affine map, got {list(names)!r}")

    percentiles = options.get("percentiles", DEFAULT_PERCENTILES)
    try:
        low, high = (float(v) for v in percentiles)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_WHERE}.percentiles must be a [low, high] pair in percent, got {percentiles!r}") from exc
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"{_WHERE}.percentiles must satisfy 0 <= low < high <= 100, got [{low}, {high}]")

    stats_from = options.get("stats_from", "all")
    if stats_from not in STATS_SOURCES:
        raise ValueError(f"{_WHERE}.stats_from must be one of {STATS_SOURCES}, got {stats_from!r}.")

    probability = options.get("p", 1.0)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError(f"{_WHERE}.p must be a number in [0, 1], got {probability!r}")

    channels = options.get("apply_to_channel")
    if channels is not None:
        if isinstance(channels, (str, bool)) or not isinstance(channels, Sequence) or not channels:
            raise ValueError(f"{_WHERE}.apply_to_channel must be a non-empty list of channel indices, got {channels!r}")
        if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in channels):
            raise ValueError(f"{_WHERE}.apply_to_channel must hold non-negative integers, got {list(channels)!r}")
        channels = tuple(int(c) for c in channels)

    threshold = float(options.get("nonzero_threshold", 1e-6))
    if threshold <= 0.0:
        raise ValueError(f"{_WHERE}.nonzero_threshold must be positive, got {threshold}")

    hint = "It scales the restored target, so 1 restores the snapshot exactly."
    return StatsSpec(
        stats=stats,
        percentiles=(low, high),
        std_scale=normalise_blend_range(
            options.get("std_scale", DEFAULT_SCALE), key="std_scale", owner=_WHERE, hint=hint, bounds=(0.0, None)
        ),
        percentile_scale=normalise_blend_range(
            options.get("percentile_scale", DEFAULT_SCALE), key="percentile_scale", owner=_WHERE, hint=hint, bounds=(0.0, None)
        ),
        stats_from=stats_from,
        nonzero_threshold=threshold,
        p=float(probability),
        apply_to_channel=channels,
    )


def spec_problems(options: Any) -> list[str]:
    """Every problem in a `pipeline.retain_stats` block, for cumulative config reporting."""
    if options is None:
        return []
    if not isinstance(options, dict):
        return [f"{_WHERE}: expected an object"]
    problems = [f"{_WHERE}: unknown key {key!r}. Accepted: {', '.join(SPEC_KEYS)}." for key in options if key not in SPEC_KEYS]
    try:
        parse_spec(options)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


@dataclass
class _ChannelSnapshot:
    """One channel's four statistics, and the voxels they were read from."""

    #: `[4, N]`, in `STAT_NAMES` order, always float32.
    values: Tensor
    #: The resolved statistics mask, or None for "all voxels". Kept rather than recomputed
    #: at restore time so both ends read the *same* voxel set: with `stats_from="nonzero"`
    #: the augmentations themselves move which voxels pass the threshold, and a restore
    #: measured over a different population is not a restore.
    mask: Tensor | None


class StatsSlot:
    """The handoff between the two leaves of one bracket.

    A plain object, not a buffer and not a module: the snapshot is per batch, lives for
    one forward pass, and has no business being moved by `.to(device)` or saved in a
    state dict. One instance per pipeline, held by both leaves.
    """

    __slots__ = ("channels",)

    def __init__(self) -> None:
        self.channels: dict[int, _ChannelSnapshot] | None = None


def measure(channel: Tensor, mask: Tensor | None, probabilities: tuple[float, float]) -> Tensor:
    """The four statistics of a `[N, *spatial]` channel, stacked `[4, N]` and float32.

    float32 whatever the input is: the GPU pipeline runs inside `autocast`
    (`nnUNetTrainerDAExt.train_step`), and a half-precision sum over a 128^3 patch loses
    enough of the mean to move the restored level visibly. Same reason the region moments
    are pinned to fp32.
    """
    work = channel.float()
    mean, std = _masked_moments(work, mask)
    low, high = _masked_quantiles(work, mask, probabilities)
    return torch.stack((mean, std, low, high))


def _lerp(bounds: tuple[float, float], draw: Tensor) -> Tensor:
    """A uniform draw mapped onto `[lo, hi]`. Exact at `lo == hi`, and always consumed."""
    low, high = bounds
    return draw * (high - low) + low


def _targets(snapshot: Tensor, spec: StatsSpec, std_draw: Tensor, low_draw: Tensor, high_draw: Tensor) -> Tensor:
    """The four jittered targets, `[4, N]`.

    The mean is never jittered: it is the one statistic every pair can anchor on, and
    shifting it as well would make the drawn pair decide the output's level as well as its
    scale -- two unrelated augmentations behind one knob. The percentiles are jittered on
    their *distance from the mean*, independently, so the span can widen, narrow or go
    asymmetric while staying a span that contains the mean.
    """
    mean, std, low, high = snapshot
    return torch.stack(
        (
            mean,
            std * _lerp(spec.std_scale, std_draw),
            mean - (mean - low) * _lerp(spec.percentile_scale, low_draw),
            mean + (high - mean) * _lerp(spec.percentile_scale, high_draw),
        )
    )


def _solve(sources: Tensor, targets: Tensor, pairs: Tensor, spread: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """`(alpha, beta, valid)` for the drawn pair of each sample. All `[N]`.

    `sources` and `targets` are `[4, N]`; `pairs` is `[N, 2]` of indices into them.
    """
    kinds = torch.tensor(_STAT_OFFSET_COEFFICIENTS, device=sources.device, dtype=sources.dtype)
    rows = torch.arange(sources.shape[1], device=sources.device)
    first, second = pairs[:, 0], pairs[:, 1]

    u1, u2 = sources[first, rows], sources[second, rows]
    t1, t2 = targets[first, rows], targets[second, rows]
    c1, c2 = kinds[first], kinds[second]

    determinant = u1 * c2 - u2 * c1
    # Sign-preserving, as in gpu/normalization.py: a negative determinant is ordinary (the
    # p1/p99 pair always has one) and clamping its magnitude must not flip it.
    safe = torch.where(determinant.abs() < _EPS, torch.full_like(determinant, _EPS).copysign(determinant), determinant)
    alpha = (t1 * c2 - t2 * c1) / safe
    # Whichever of the two carries the offset; at most one of them can be the std.
    beta = torch.where(c1 != 0, t1 - alpha * u1, t2 - alpha * u2)

    valid = (spread > 0) & (determinant.abs() >= _DET_FLOOR * spread) & torch.isfinite(alpha) & torch.isfinite(beta)
    return alpha, beta, valid


class StatsSnapshotGPU(ImageOnlyTransform):
    """Records the intensity profile the augmentations are about to change. Not an augmentation.

    Returns its input untouched -- it only reads -- and carries no probability worth
    drawing: `p=1.0`, because a snapshot that sometimes did not happen would leave the
    restore at the other end of the pipeline with nothing to restore.

    Pipeline structure rather than a registered augmentation, like the 2D/3D converters
    `build_cpu_pipeline` emits around `SpatialTransform`: its position is not a choice a
    config makes, it is implied by where the geometry ends.
    """

    spec: StatsSpec
    slot: StatsSlot
    stats_from: str
    nonzero_threshold: float

    def __init__(self, spec: StatsSpec, slot: StatsSlot, keepdim: bool = True) -> None:
        super().__init__(p=1.0, p_batch=1.0, same_on_batch=False, keepdim=keepdim)
        self.spec = spec
        self.slot = slot
        # Read by `_resolve_mask`; see the `_MaskSource` protocol in gpu/normalization.py.
        self.stats_from = spec.stats_from
        self.nonzero_threshold = spec.nonzero_threshold

    def channels(self, total: int) -> tuple[int, ...]:
        """Which channels this bracket covers, with every index checked against the input."""
        selected = self.spec.apply_to_channel if self.spec.apply_to_channel is not None else tuple(range(total))
        for channel in selected:
            _check_channel(self, channel, total)
        return selected

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        seg_mask = segmentation_from(params)
        snapshot: dict[int, _ChannelSnapshot] = {}
        for channel in self.channels(input.shape[1]):
            data = input[:, channel]
            mask = _resolve_mask(self, data, seg_mask)
            snapshot[channel] = _ChannelSnapshot(values=measure(data, mask, self.spec.probabilities), mask=mask)
        self.slot.channels = snapshot
        return input


class StatsRestoreGPU(ImageOnlyTransform):
    """Puts two of the four snapshotted statistics back, jittered. Not an augmentation.

    `p=1.0` on the transform; the per-sample probability the config asks for is
    `StatsSpec.p` and is spent here, so that a sample which is not restored still costs
    the same draws as one that is.
    """

    spec: StatsSpec
    slot: StatsSlot

    def __init__(self, spec: StatsSpec, slot: StatsSlot, keepdim: bool = True) -> None:
        super().__init__(p=1.0, p_batch=1.0, same_on_batch=False, keepdim=keepdim)
        self.spec = spec
        self.slot = slot
        self._pairs = torch.tensor(spec.pairs, dtype=torch.long)

    def _draw(self, rows: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pair, three jitter uniforms and the gate -- five draws, unconditionally.

        `torch.rand` scaled into the table rather than `torch.randint`, which is the
        fixed-width trick `gpu/regions.py` uses for the same purpose: one uniform per row
        whatever the pool's size, so adding a statistic to `stats` does not reshape the
        stream. No `dtype=`, following gpu/mixing.py, so a half-precision batch draws from
        the same stream as a single-precision one.
        """
        count = self._pairs.shape[0]
        drawn = (torch.rand(rows, device=device) * count).long().clamp_max_(count - 1)
        return (
            drawn,
            torch.rand(rows, device=device),
            torch.rand(rows, device=device),
            torch.rand(rows, device=device),
            torch.rand(rows, device=device),
        )

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        taken = self.slot.channels
        self.slot.channels = None  # consumed: a second pass must not read this one's stats
        if taken is None:
            raise RuntimeError(
                "StatsRestoreGPU ran without a snapshot. The two are emitted as a pair by "
                "smauglab.transforms.build.build_gpu_pipeline and the snapshot has to run first."
            )

        batch = input.shape[0]
        # One draw shared by the batch under `same_on_batch`, which the container forces
        # onto every child at the shipped default -- the same way a bucket takes one
        # decision for the whole batch. See SmaugConfig.same_on_batch.
        rows = 1 if self.same_on_batch else batch
        out = input.clone()

        for channel, recorded in taken.items():
            if recorded.values.shape[1] != batch:
                raise RuntimeError(
                    f"StatsRestoreGPU got {batch} samples but the snapshot holds {recorded.values.shape[1]}. "
                    "Nothing between the two may change the batch size."
                )
            data = out[:, channel]
            work = data.float()
            sources = measure(work, recorded.mask, self.spec.probabilities)
            pair, std_draw, low_draw, high_draw, gate = self._draw(rows, input.device)
            targets = _targets(recorded.values, self.spec, std_draw, low_draw, high_draw)

            flat = work.reshape(batch, -1)
            spread = flat.amax(dim=1) - flat.amin(dim=1)
            pairs = self._pairs.to(input.device)[pair].expand(batch, 2)
            alpha, beta, valid = _solve(sources, targets, pairs, spread)
            applies = valid & (gate < self.spec.p).expand(batch)

            restored = work * _broadcast(alpha, work) + _broadcast(beta, work)
            # Per row, not per batch: one degenerate sample must not cost the others their
            # restore. The guard is the same one `_select_and_check` applies globally.
            finite = torch.isfinite(restored).reshape(batch, -1).all(dim=1)
            out[:, channel] = torch.where(_broadcast(applies & finite, work), restored, work).to(data.dtype)

        return out


def make_bracket(spec: StatsSpec) -> tuple[StatsSnapshotGPU, StatsRestoreGPU]:
    """The two leaves of one bracket, already sharing a slot."""
    slot = StatsSlot()
    return StatsSnapshotGPU(spec, slot), StatsRestoreGPU(spec, slot)
