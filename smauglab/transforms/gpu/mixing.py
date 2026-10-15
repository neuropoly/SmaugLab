"""Blending a transform's output back towards the image it was handed.

Six of the GPU transfer augmentations already did this, under the name `mix_prob`,
from three open-coded copies of the same four lines -- and with the blend weight drawn
from a hardcoded ``U(0, 1)``, so the only expressible blends were "none" and "anything
from the identity to full strength". `mix_alpha` is that range made configurable, and
this module is the one implementation the whole group shares.

`alpha` is the weight on the **original**: 0.0 is the transform's output untouched, 1.0
is the image it was given. That is the direction the open-coded copies used, so the
default range of ``(0.0, 1.0)`` reproduces their draw exactly -- ``draw * 1.0 + 0.0`` is
exact in IEEE-754, and `unit_tests/test_mix_prob.py` is the check that it stayed that
way.

The blend belongs **before** a transform's `retain_stats` restore, which is where those
copies already sat. `retain_stats` promises that the output carries the input's
per-sample mean and std, and blending after the restore would quietly break that
promise: the convex combination of two images with the same moments has the same mean
but a smaller standard deviation. `mix_prob` is above zero in 29 of the 33 shipped
configs, on transforms that mostly ask for `retain_stats` as well, so that is not a
hypothetical.

Three entry points, because the transforms come in three shapes:

* :func:`mix_with_original` for a whole channel, ``[N, *spatial]``, drawing one gate and
  one alpha per sample in two vectorised calls.
* :func:`blend_one` for the transforms that assemble their result one sample at a time
  inside a Python loop, where a vectorised draw has nothing to vectorise over.
* :func:`sample_mix` with :func:`blend_at` for the loops that *also draw as they go*.
  Blending inside one of those would interleave two draws per sample into the transform's
  own stream, so switching mixing on would change what every later sample did -- and
  `mix_alpha = 0` would stop being a no-op. Drawing up front keeps the blend out of the
  way, which is why `gpu/fromSeg.py` already hoists its `in_seg` and dilation draws.

The two draw differently *on purpose*: each reproduces the draw order of the copy it
replaces, so a transform that already had `mix_prob` set above zero keeps its seeded
output. The one thing that did change is what happens at `mix_prob = 0`. The two
per-sample copies spent a draw per sample to decide something `mix_prob` had already
settled --

    if torch.rand(1).item() < self.mix_prob:   # drawn even when mix_prob is 0.0

-- so `RandomInverseGPU` and `RandomHistogramEqualizationGPU` shifted the stream for
every transform that ran after them, whether or not mixing was on. Both helpers return
before drawing anything now, as the vectorised copy always did. Measured consequence: 15
of the 28 runnable shipped configs produce a different seeded sequence than the previous
release, same distribution; every transform in isolation is bit-identical except
`RandomInverseGPU` with `mix_prob > 0`, whose blend also moved. The property that buys
is the one this feature needs -- a `mix_prob` or `mix_alpha` sweep can be compared
against its own `mix_prob = 0` baseline at a fixed seed, which it could not be before.
"""

from collections.abc import Sequence
from typing import Protocol, Union

import torch
from torch import Tensor

#: A fixed alpha, or a ``[lo, hi]`` range drawn from per sample. Mirrors `BlendSpec`
#: in gpu/palette/overlay.py, which is the same idea for a different blend.
MixAlphaSpec = Union[float, Sequence[float]]

#: What `mix_alpha` defaults to everywhere: the hardcoded draw this replaced.
DEFAULT_MIX_ALPHA = (0.0, 1.0)


class Mixing(Protocol):
    """What the helpers below need off the transform they are handed.

    A Protocol rather than `ImageOnlyTransform`, for the same reason `_RegionSelecting`
    in gpu/contrast.py is one: these are plain attributes on an `nn.Module`, so reading
    them through the base class resolves via `Module.__getattr__`, which is typed as
    returning `Tensor | Module` -- and a `Tensor` is not a float. Naming them here is
    both what makes the arithmetic below type-check and a statement of the requirement.
    """

    mix_prob: float
    mix_alpha: tuple[float, float]


#: Appended to a `mix_alpha` bound error, because the direction is the one thing about
#: this parameter nobody guesses right.
_MIX_ALPHA_HINT = "It is the weight on the original input: 0 is the transform's output, 1 is the input untouched."


def normalise_blend_range(spec: MixAlphaSpec, key: str = "mix_alpha", owner: str = "", hint: str = _MIX_ALPHA_HINT) -> tuple[float, float]:
    """A scalar or ``[lo, hi]`` as a validated pair inside ``[0, 1]``.

    Checked here rather than in `smauglab/config.py`, which only ever asks whether a key
    is accepted, never what its value is -- every value check in this package happens in
    a constructor. A pipeline is built before the first batch, so a bad range still fails
    long before training.

    `key` and `owner` are for the message alone. `AnatomicalLabelOverlay.blend_strength`
    is a different blend with the same shape of argument, and shares this.
    """
    where = f"{owner}: " if owner else ""
    if isinstance(spec, bool):  # an int subtype, and never a blend weight anyone meant
        # ValueError, not the TypeError TRY004 asks for: every other rejection here is a
        # ValueError, and a caller should only have to catch one thing for a bad range.
        raise ValueError(f"{where}{key} must be a number or a [lo, hi] pair, got {spec!r}")  # noqa: TRY004
    if isinstance(spec, (int, float)):
        lo = hi = float(spec)
    else:
        try:
            values = [float(v) for v in spec]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}{key} must be a number or a [lo, hi] pair, got {spec!r}") from exc
        if len(values) != 2:
            raise ValueError(f"{where}{key} must be a number or a [lo, hi] pair, got {len(values)} values: {spec!r}")
        lo, hi = values
    if lo > hi:
        raise ValueError(f"{where}{key} range must be non-decreasing, got [{lo}, {hi}]")
    if lo < 0.0 or hi > 1.0:
        suffix = f" {hint}" if hint else ""
        raise ValueError(f"{where}{key} must lie inside [0, 1], got [{lo}, {hi}].{suffix}")
    return (lo, hi)


def configure_mix(transform: Mixing, mix_prob: float, mix_alpha: MixAlphaSpec) -> None:
    """Validate and assign the two parameters. One line per constructor.

    `ImageOnlyTransform` declares both attributes, which is what makes `transform`
    satisfy `Mixing` here and at every read site -- see the note on that class.
    """
    transform.mix_prob = float(mix_prob)
    transform.mix_alpha = normalise_blend_range(mix_alpha, owner=type(transform).__name__)


def mix_with_original(transform: Mixing, orig: Tensor, x: Tensor) -> Tensor:
    """Blend each sample of a channel back towards `orig`, gated by `mix_prob`.

    `orig` and `x` are ``[N, *spatial]`` -- a whole channel, batch axis first. One
    independent gate and one independent alpha per sample, which is what "probability of
    blending the result back" means element-wise over a batch.

    Returns `x` untouched, **having drawn nothing**, when `mix_prob` is zero. That matters
    beyond the saving: a transform at `mix_prob=0` has to leave the RNG stream exactly
    where it found it, or every test that compares a mixed run against an unmixed one at
    the same seed would be comparing two different streams.
    """
    if transform.mix_prob <= 0.0:
        return x
    lo, hi = transform.mix_alpha
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    # No `dtype=`: the draw this replaced passed none either, so it follows torch's
    # default rather than the image's -- which keeps a half-precision batch on the same
    # stream as a single-precision one.
    mix = torch.rand(x.shape[0], device=x.device).view(shape) < transform.mix_prob
    # Drawn even when lo == hi, where the multiply is exactly zero and the result exactly
    # `lo`. Skipping it would make the RNG stream a function of `mix_alpha`, so two runs
    # that differ only in the range would not be comparable -- which is the one thing a
    # configurable range exists to allow.
    alpha = torch.rand(x.shape[0], device=x.device).view(shape) * (hi - lo) + lo
    return torch.where(mix, alpha * orig + (1 - alpha) * x, x)


def sample_mix(transform: Mixing, batch: int, device: torch.device) -> tuple[list[bool], Tensor] | None:
    """Every sample's gate and alpha up front, or None when mixing is off.

    The gates come back as a Python list and the alphas stay on the device: reading a
    device scalar inside a loop stalls the host until the queue drains, so the one
    `.tolist()` here replaces one stall per sample. `gpu/fromSeg.py` hoists its other
    per-sample draws the same way and for the same reason.
    """
    if transform.mix_prob <= 0.0:
        return None
    lo, hi = transform.mix_alpha
    gates = (torch.rand(batch, device=device) < transform.mix_prob).tolist()
    alphas = torch.rand(batch, device=device) * (hi - lo) + lo
    return gates, alphas


def blend_at(drawn: tuple[list[bool], Tensor] | None, index: int, orig: Tensor, x: Tensor) -> Tensor:
    """One sample's blend from a :func:`sample_mix` draw. A no-op when mixing is off."""
    if drawn is None:
        return x
    gates, alphas = drawn
    if not gates[index]:
        return x
    alpha = alphas[index]
    return alpha * orig + (1 - alpha) * x


def blend_one(transform: Mixing, orig: Tensor, x: Tensor) -> Tensor:
    """The same blend for one sample, out of a Python loop.

    For the transforms that build their result per sample -- a histogram CDF, a
    per-region redistribution, a per-sample LUT -- where `orig` and `x` are a single
    ``[*spatial]`` volume and there is no batch axis to vectorise over.

    The gate comes off the default generator via `torch.rand(1).item()` and the alpha off
    `x`'s device, which is the draw order of the inline copies this replaces; the alpha is
    drawn only when the gate fires, as they did. A CPU scalar costs no device
    synchronisation, so the loop is not paying for the `.item()`.
    """
    if transform.mix_prob <= 0.0:
        return x
    if torch.rand(1).item() >= transform.mix_prob:
        return x
    lo, hi = transform.mix_alpha
    alpha = torch.rand(1, device=x.device) * (hi - lo) + lo
    return alpha * orig + (1 - alpha) * x
