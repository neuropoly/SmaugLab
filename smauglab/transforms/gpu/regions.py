"""Where a transform applies, and how sharply it stops applying.

Three parameters predate this module -- `in_seg`, `out_seg` and `mix_in_out` -- and
between them they expressed about a third of what they looked like they expressed:

* `in_seg` and `out_seg` were two independent uniforms, and *both fired* fell through
  to the same ``"all"`` as *neither fired*. "A mixture of foreground and background"
  was not expressible, and `in_seg=1, out_seg=1` was indistinguishable from
  `in_seg=0, out_seg=0`.
* `mix_in_out` keyed its keep/drop draw on `seg_mask.shape[:2]`, the **channel** axis.
  For the single-channel integer label map nnU-Net's trainer passes, that is `(B, 1)`
  -- one draw per sample, deleting the whole mask half the time. So ``"in"`` applied
  the transform nowhere and ``"out"`` applied it everywhere, and the thirteen shipped
  configs that set `mix_in_out: true` were running a coin flip rather than a label mix
  in every nnU-Net run. It only ever meant "per label" on a one-hot mask.
* The background was one blob. `out_seg` took all of it or none of it.
* The composite was `m * transformed + (1 - m) * orig` with a **binary** `m`, so every
  region-selected augmentation left a one-voxel discontinuity sitting exactly on the
  label boundary -- an edge perfectly correlated with the segmentation target, which is
  the one thing an augmentation must not hand the network.

The replacement separates four questions that were tangled together:

================  ============================================================
`in_seg`,         **where**: foreground labels, background, a mixture of both,
`out_seg`         or the whole patch. Both fired now means ``"both"``.
`seg_unit`,       **at what granularity**: the whole side as one piece, one unit
`seg_unit_prob`   per label value, or one unit per connected component.
`bg_split`,       **how finely the background is divided**: one blob, or EM
`bg_em_clusters`  intensity clusters, SynthSeg-style (Billot et al. 2023, 5.4).
`seg_feather`     **how sharp the boundary is**: a hard step, or a Gaussian ramp.
================  ============================================================

Everything resolves to a single integer **unit id map** (:class:`RegionUnits`) and one
gather off it. Nothing materialises per-unit volumes: a noisy mask under
``seg_unit="component"`` can hold hundreds of components, and `[2, 300, 128, 128, 128]`
of bool is 1.2 GB.

Three properties this module goes out of its way to keep, each of which the code it
replaces broke:

**A knob that is off spends no random numbers.** `_choose_region_mode` drew its two
uniforms whenever a mask was attached, including at `in_seg = out_seg = 0`, so an
unselected run and a selected one at the same seed were never comparable. That is the
defect `gpu/mixing.py` removed for `mix_prob`, and :func:`choose_region_modes` returns
before drawing anything for the same reason.

**The number of uniforms a draw spends is a function of the configuration, never of the
data.** A per-unit keep draw would spend one uniform per label *present*, so a patch
with four labels and one with five would leave the stream in different places and every
transform after this one would diverge on the data. :data:`UNIT_DRAW_WIDTH` fixes the
width per decomposition kind instead.

**The decomposition is computed once per pipeline pass, not once per transform.** See
:func:`region_units`. Connected components and EM each cost tens of milliseconds on a
128^3 patch; a pipeline runs twenty-odd region-selecting transforms over one batch.

One consequence worth stating plainly rather than leaving to be discovered:
``bg_split="em"`` is the only decomposition that reads the image, and the image moves as
the pass runs -- every transform's first statement is `input = input.clone()`. The
clusters are therefore fitted **once per pass, from the image as it stood when the first
region-selecting transform asked**, and not re-fitted afterwards. That is deliberate
twice over. Re-fitting is twenty-odd EM runs a batch, each with a 100 000-element
subsample draw and a host synchronisation per sample; and a single fit means two
region-selected transforms in the same pass cut the background along the same lines
instead of each along its own. What it costs is that a transform late in the pass
clusters an image a few transforms old. The units are a partition to switch on and off,
not a measurement, so that is the cheap half of the trade.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol, Union

import torch
from torch import Tensor

from smauglab.transforms.gpu.mixing import normalise_blend_range
from smauglab.transforms.kernels import gaussian_blur3d
from smauglab.transforms.rng import next_shared_seed

#: A fixed sigma in voxels, or a ``[lo, hi]`` range drawn from per call. Mirrors
#: `MixAlphaSpec` in gpu/mixing.py, which is the same shape of argument for a weight
#: rather than a width.
SegFeatherSpec = Union[float, Sequence[float]]

SEG_UNITS = ("whole", "label", "component")
BG_SPLITS = ("none", "em")

DEFAULT_SEG_UNIT = "whole"
DEFAULT_SEG_UNIT_PROB = 0.5
DEFAULT_BG_SPLIT = "none"
DEFAULT_BG_EM_CLUSTERS = (3, 10)
DEFAULT_SEG_FEATHER = 0.0

#: How many keep/drop uniforms a draw spends, per decomposition kind. Fixed per *kind*
#: so the count is a function of the configuration and never of the segmentation -- see
#: the module docstring. A unit id at or above the width folds into the last unit, which
#: only bites a patch with more than 32 label values or more than 256 components per
#: side, and then only by tying their fates together.
UNIT_DRAW_WIDTH = {"whole": 2, "label": 32, "component": 256}

#: Mode codes, as they travel in the `[N]` tensor `choose_region_modes` returns.
MODE_ALL = 0
MODE_IN = 1
MODE_OUT = 2
MODE_BOTH = 3

#: A label value no real label can take, for the gated shifts in
#: :func:`connected_components_3d`: the out-of-volume neighbour must never compare equal
#: to the voxel it is shifted past.
_NO_LABEL = -1

#: Where the attribute cache lives on a mask tensor. See :func:`region_units`.
_CACHE_ATTR = "_smauglab_region_units"

#: Appended to a `seg_feather` bound error, because "sigma in voxels" is not what the
#: name suggests to someone who has just read `mix_alpha`.
_FEATHER_HINT = "It is a Gaussian sigma in voxels, not a weight: 0 is a hard boundary, and there is no upper limit."


class RegionUnits(NamedTuple):
    """A segmentation decomposed into the units a mode can switch on and off.

    `ids` is dense in ``[0, n_units)`` and background units come **first**, so "is this
    unit foreground" is the scalar comparison ``id >= n_background`` rather than a
    per-unit tensor. Label units are numbered batch-wide, so a given label value is the
    same unit in every sample of a batch -- the choice `seg_region_masks` already
    documents for its channel ordering. Component and EM-cluster units cannot be:
    they are per-sample by nature, and sample `b`'s unit `k` is unrelated to sample
    `b+1`'s. The keep draw is per sample either way, so that costs nothing.
    """

    ids: Tensor
    n_units: int
    n_background: int


class UnitSpec(NamedTuple):
    """What a decomposition depends on, and therefore the cache key.

    `seg_unit_prob`, `in_seg`, `out_seg` and `seg_feather` are deliberately absent: they
    are redrawn per call and must not fragment the cache.
    """

    seg_unit: str
    bg_split: str
    bg_em_clusters: tuple[int, int]


class RegionSelecting(Protocol):
    """What the helpers below need off the transform they are handed.

    A Protocol rather than `ImageOnlyTransform`, for the same reason `mixing.Mixing` is
    one: these are plain attributes on an `nn.Module`, so reading them through the base
    class resolves via `Module.__getattr__`, which is typed as returning
    `Tensor | Module` -- and neither of those is a float. Naming them here is both what
    makes the arithmetic below type-check and a statement of the requirement.
    """

    in_seg: float
    out_seg: float
    seg_unit: str
    seg_unit_prob: float
    bg_split: str
    bg_em_clusters: tuple[int, int]
    seg_feather: tuple[float, float]


# ── validation ───────────────────────────────────────────────────────────────


def _one_of(value: Any, choices: tuple[str, ...], key: str, owner: str) -> str:
    where = f"{owner}: " if owner else ""
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{where}{key} must be one of {', '.join(choices)}, got {value!r}")
    return value


def _cluster_range(spec: Any, owner: str) -> tuple[int, int]:
    """A ``[lo, hi]`` pair of positive cluster counts.

    Not `normalise_blend_range`: these are counts, and silently accepting a float would
    put a non-integer into `torch.randint`'s bounds.
    """
    where = f"{owner}: " if owner else ""
    if isinstance(spec, (bool, int, float)):
        raise ValueError(f"{where}bg_em_clusters must be a [lo, hi] pair of counts, got {spec!r}")  # noqa: TRY004
    try:
        values = [int(v) for v in spec]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}bg_em_clusters must be a [lo, hi] pair of counts, got {spec!r}") from exc
    if len(values) != 2:
        raise ValueError(f"{where}bg_em_clusters must be a [lo, hi] pair of counts, got {len(values)} values: {spec!r}")
    lo, hi = values
    if lo < 1 or hi < lo:
        raise ValueError(f"{where}bg_em_clusters must be a non-decreasing pair of counts >= 1, got [{lo}, {hi}]")
    if hi > UNIT_DRAW_WIDTH["whole"] + max(UNIT_DRAW_WIDTH.values()):
        # Only a sanity bound: a background split into thousands of clusters is a
        # configuration mistake, and the keep draw would fold most of them together.
        raise ValueError(f"{where}bg_em_clusters upper bound {hi} is implausibly large")
    return (lo, hi)


def configure_regions(
    transform: RegionSelecting,
    in_seg: float,
    out_seg: float,
    seg_unit: str = DEFAULT_SEG_UNIT,
    seg_unit_prob: float = DEFAULT_SEG_UNIT_PROB,
    bg_split: str = DEFAULT_BG_SPLIT,
    bg_em_clusters: Sequence[int] = DEFAULT_BG_EM_CLUSTERS,
    seg_feather: SegFeatherSpec = DEFAULT_SEG_FEATHER,
) -> None:
    """Validate and assign the region parameters. One line per constructor.

    Every value check in this package happens in a constructor -- `smauglab/config.py`
    only ever asks whether a key is *accepted*, never what its value is. A pipeline is
    built before the first batch, so a bad value still fails long before training.

    `in_seg` and `out_seg` used to be clamped into range silently
    (``max(0.0, min(1.0, p))``), which quietly accepted `in_seg: 5.0` and
    `in_seg: true`. They are rejected now.
    """
    owner = type(transform).__name__
    transform.in_seg = normalise_blend_range(in_seg, key="in_seg", owner=owner, hint="")[0]
    transform.out_seg = normalise_blend_range(out_seg, key="out_seg", owner=owner, hint="")[0]
    transform.seg_unit = _one_of(seg_unit, SEG_UNITS, "seg_unit", owner)
    transform.seg_unit_prob = normalise_blend_range(seg_unit_prob, key="seg_unit_prob", owner=owner, hint="")[0]
    transform.bg_split = _one_of(bg_split, BG_SPLITS, "bg_split", owner)
    transform.bg_em_clusters = _cluster_range(bg_em_clusters, owner)
    transform.seg_feather = normalise_blend_range(seg_feather, key="seg_feather", owner=owner, hint=_FEATHER_HINT, bounds=(0.0, None))


def unit_spec(transform: RegionSelecting) -> UnitSpec:
    """The decomposition `transform` asks for, which is also its cache key."""
    return UnitSpec(transform.seg_unit, transform.bg_split, transform.bg_em_clusters)


def region_selection_active(transform: RegionSelecting) -> bool:
    """Whether either probability can fire. False means every helper here is a no-op."""
    return transform.in_seg > 0.0 or transform.out_seg > 0.0


# ── segmentation layout ──────────────────────────────────────────────────────


def label_map_from(mask: Tensor) -> Tensor:
    """A `[B, 1, *spatial]` integer label map, background 0, from any of three layouts.

    The layouts reach a transform interchangeably and the values alone cannot always
    tell them apart:

    * ``[B, 1, *spatial]`` -- an ordinary label map. The ids are the values and
      background is 0. Rounded first: a mask that has been through a geometric transform
      comes back as float, and unrounded floats would invent labels.
    * ``[B, C, *spatial]`` that leaves some voxel unset -- a background-*implicit*
      one-hot, this repository's convention (`collapse_onehot_to_index` in
      gpu/fromSeg.py): channel `c` encodes label `c + 1`.
    * ``[B, C, *spatial]`` that leaves no voxel unset -- a background-*inclusive*
      one-hot, which is what `seg_region_masks` emits for a single-channel label map and
      what `unit_tests/test_seg_layout.py` builds. Channel 0 **is** background.

    The third is the one `_foreground` already had to detect by coverage rather than by
    value, and the one `collapse_onehot_to_index` still gets wrong: it does `argmax + 1`
    unconditionally, so a background-inclusive one-hot comes back with every label
    shifted by one and background relabelled 1. That is why `em_subdivide_labels` carries
    a fallback for "the real background is no longer 0". Deriving the map here, once, is
    what lets everything else in this module be a read off it.

    One case no function can read off the values: a background-implicit one-hot patch
    labelled edge to edge is indistinguishable from a background-inclusive one. A real
    segmentation patch is not labelled edge to edge, which is the same admission
    `_foreground`'s docstring has always carried.
    """
    if mask.dim() < 3:
        raise ValueError(f"expected [B, C, *spatial], got {tuple(mask.shape)}")
    if mask.shape[1] == 1:
        return mask.round().long().clamp_min(0)

    covered = mask.amax(dim=1, keepdim=True) > 0
    index = mask.argmax(dim=1, keepdim=True).long()
    # `torch.where`, not `if bool(covered.all())`: reading that test back to Python
    # stalls the host on everything queued behind it, and both branches are a cheap
    # elementwise pass. Same reason `_foreground` is written this way.
    return torch.where(covered.all(), index, (index + 1) * covered)


def _foreground(mask: Tensor, dim: int) -> Tensor:
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

    Kept as its own function, rather than folded into :func:`label_map_from`, because
    `dim` is not always 1: `_apply_region_mode`'s unbatched path reduces a `[C, *spatial]`
    mask over axis 0. The two agree on every layout, which
    `unit_tests/test_foreground_layouts.py` is the check on.
    """
    covered = mask.amax(dim=dim) > 0
    if mask.shape[dim] == 1:
        return covered
    # `torch.where`, not `if bool(...)`: reading that test back to Python stalls the host
    # on everything queued behind it. Both branches are a reduction over the class axis,
    # so computing the unused one costs a fraction of what the stall does.
    without_background = mask.narrow(dim, 1, mask.shape[dim] - 1).amax(dim=dim) > 0
    return torch.where(covered.all(), without_background, covered)


def densify_region_ids(region_ids: Tensor) -> tuple[Tensor, int]:
    """Remap sparse non-negative integer region ids onto a contiguous [0, R) range.

    `em_subdivide_labels` encodes a fine id as `parent_idx * mult + assign`, which
    is sparse. Anything that indexes a per-region tensor of length R with these -- the
    keep/drop gather here, `signed_alpha_affine_remap` in the PALETTE blocks -- would
    index out of bounds on a sparse id.

    Lives here rather than in `gpu/palette/partitioners.py`, where it was written,
    because this module cannot import that one: `palette/base.py` reaches `gpu/base.py`,
    which imports this module for its "off" defaults. Re-exported from there.
    """
    # A lookup table off a bincount, not `torch.unique(return_inverse=True)`: the ids
    # are small non-negative integers, so counting the bins and numbering the
    # non-empty ones gives the same ascending remap without radix-sorting two
    # million voxels.
    counts = torch.bincount(region_ids.reshape(-1))
    present = counts.nonzero().flatten()
    lookup = torch.zeros(counts.numel(), dtype=torch.long, device=region_ids.device)
    lookup[present] = torch.arange(present.numel(), device=region_ids.device)
    return lookup[region_ids], int(present.numel())


# ── connected components ─────────────────────────────────────────────────────

#: Hard ceiling on the propagation below. Pointer jumping squares the reach each round,
#: so even a serpentine component spanning a 128^3 patch settles in about twenty; the cap
#: only exists so a bug cannot hang a training run.
_CC_MAX_ITERS = 64

#: How often the fixpoint test runs. `bool(...)` on a device tensor stalls the host until
#: the queue drains, and the loop body is six shifts over the volume, so testing every
#: round would spend a synchronisation to save work that is already cheap. Compact blobs
#: converge in a handful of rounds, which is why the test exists at all rather than
#: running a fixed count. Same trade `_kmeans_1d` made in the other direction.
_CC_CHECK_EVERY = 4


def _shift(volume: Tensor, dim: int, delta: int, fill: Any) -> Tensor:
    """`volume` shifted along `dim` so that index `i` holds what index `i - delta` held.

    `fill` enters at the face that is uncovered. Not `torch.roll`, which wraps the far
    face round to the near one and would join two components that only touch through
    opposite sides of the patch.
    """
    size = volume.shape[dim]
    step = abs(delta)
    kept = volume.narrow(dim, 0, size - step) if delta > 0 else volume.narrow(dim, step, size - step)
    pad_shape = list(volume.shape)
    pad_shape[dim] = step
    block = torch.full(pad_shape, fill, dtype=volume.dtype, device=volume.device)
    return torch.cat([block, kept] if delta > 0 else [kept, block], dim=dim)


def connected_components_3d(labels: Tensor, foreground: Tensor) -> tuple[Tensor, int]:
    """Connected components of each label value, 6-connected, per sample.

    `labels` is `[B, 1, D, H, W]` integer and `foreground` the bool mask of the voxels to
    partition. Returns `(ids, n_max)` where `ids` is `[B, 1, D, H, W]`, numbered from 0
    within each sample and `-1` outside `foreground`, and `n_max` is the largest
    per-sample count -- which is what sizes the keep/drop draw.

    **6-connected**, matching `scipy.ndimage.label`'s default structuring element
    (`generate_binary_structure(3, 1)`), which is also the conventional reading of
    "connected components of a label": two voxels touching only at a corner are two
    components. The obvious implementation -- one `max_pool3d(kernel=3)` -- is a 3^3
    neighbourhood and therefore 26-connected, which merges them. Six explicit shifts
    cost twice the kernels of three axis pools and buy the right answer.

    Components respect label boundaries without a pass per label value. The gate is the
    `same` mask below: a neighbour contributes only where it carries the *same* label, so
    one propagation partitions every label at once. Doing it with pooled values instead
    would need a pass per label (a spine patch has five to twenty-five), because a pool
    reports the largest value in the neighbourhood without saying which voxel it came
    from, and a higher-labelled neighbour would mask the same-labelled one.

    The algorithm is label-equivalence propagation: every voxel starts as its own
    component, named by its flat index; each round takes the largest name reachable from
    a same-label neighbour, then **jumps to that name's own current name**, which doubles
    the distance information travels per round and turns a linear walk along a long thin
    structure into a logarithmic one.

    Indices are `int64` throughout. The float alternative is tempting because
    `max_pool3d` requires it, but `float32` represents integers exactly only to 2^24, and
    a 256^3 patch has exactly that many voxels -- so a larger patch would silently merge
    components whose flat indices round together.
    """
    batch = labels.shape[0]
    spatial = labels.shape[2:]
    voxels = int(labels[0, 0].numel())
    device = labels.device

    # Gated once, outside the loop: the label map does not move, so the six
    # "neighbour carries the same label, and both of us are foreground" masks are
    # constants of the call. Recomputing them per round would double its cost.
    same = []
    for dim in (2, 3, 4):
        for delta in (1, -1):
            neighbour_label = _shift(labels, dim, delta, _NO_LABEL)
            neighbour_fg = _shift(foreground, dim, delta, False)
            same.append(((neighbour_label == labels) & neighbour_fg & foreground, dim, delta))

    # Globally unique seeds, not per-sample ones: the densify below counts over the whole
    # batch, so two samples whose components happened to root at the same within-sample
    # index would be counted as one.
    ids = torch.arange(batch * voxels, device=device, dtype=torch.long).view(batch, 1, *spatial)
    ids = torch.where(foreground, ids, torch.full_like(ids, -1))

    for iteration in range(1, _CC_MAX_ITERS + 1):
        best = ids
        for gate, dim, delta in same:
            best = torch.maximum(best, torch.where(gate, _shift(ids, dim, delta, -1), torch.full_like(ids, -1)))
        flat = best.flatten(1)
        # The name that our current name currently answers to. `clamp_min(0)` only keeps
        # the gather in range for the background entries, which the `where` then drops.
        jumped = torch.gather(flat, 1, (flat - batch_offset(batch, voxels, device)).clamp_min(0))
        advanced = torch.maximum(flat, torch.where(flat >= 0, jumped, flat)).view_as(ids)
        advanced = torch.where(foreground, advanced, torch.full_like(advanced, -1))
        settled = iteration % _CC_CHECK_EVERY == 0 and bool(torch.equal(advanced, ids))
        ids = advanced
        if settled:
            break

    # Densified over the batch, then rebased per sample. Global ids are ascending in the
    # root's flat index and every index of sample b is below every index of sample b+1, so
    # each sample's ids are a contiguous ascending block and subtracting its lowest gives
    # a 0-based numbering without a second densify per sample.
    dense, _total = densify_region_ids(ids.clamp_min(0))
    dense = torch.where(foreground, dense, torch.full_like(dense, -1))
    flat_dense = dense.flatten(1)
    sentinel = torch.full_like(flat_dense, torch.iinfo(torch.int64).max)
    lowest = torch.where(flat_dense >= 0, flat_dense, sentinel).min(dim=1).values
    highest = flat_dense.max(dim=1).values
    counts = torch.where(highest >= lowest, highest - lowest + 1, torch.zeros_like(highest))
    rebased = torch.where(foreground, dense - lowest.view(-1, *([1] * (dense.dim() - 1))), torch.full_like(dense, -1))
    return rebased, max(int(counts.max()), 1)


def batch_offset(batch: int, voxels: int, device: torch.device) -> Tensor:
    """`[B, 1]` of each sample's first global seed index.

    The names in :func:`connected_components_3d` are global so the densify can count them
    together, but `torch.gather` indexes within a row, so the row's offset comes off again
    before the lookup.
    """
    return (torch.arange(batch, device=device, dtype=torch.long) * voxels).view(batch, 1)


# ── background decomposition ─────────────────────────────────────────────────


def background_em_units(image: Tensor, labels: Tensor, clusters_range: tuple[int, int]) -> tuple[Tensor, int]:
    """The background split into intensity-coherent subregions by EM, SynthSeg-style.

    Returns `(ids, n)` with `ids` numbered from 0 on background voxels and 0 elsewhere
    (foreground ids are never read: the caller overwrites them with the foreground
    numbering).

    `em_subdivide_labels` does the work -- this is a two-label call that asks it to split
    label 0 into `clusters_range` clusters and leave the foreground as a single one, which
    is exactly the paper's background handling without its foreground handling.

    Two things are wrapped around it rather than left to it:

    * **The draw is forked.** It spends one `randint` per sample *and* a
      `max_fit_voxels`-element subsample index -- a hundred thousand uniforms, which the
      background of a 128^3 patch always triggers. Since the result is cached, a miss
      would spend them and a hit would not, which would make the global RNG stream a
      function of cache state: of which transform asked first, of which rows `p` selected,
      of whether a geometric transform replaced the mask. Forking keeps the stream exactly
      where it was found. `next_shared_seed` rather than an arbitrary seed so DDP ranks
      still agree.
    * **A patch with no background is left alone.** When the configured background label
      is absent, `em_subdivide_labels` falls back to the largest-area label, which here
      would silently split a *foreground* structure into three to ten pieces and call them
      background.

    `same_on_batch` is fixed at False: the decomposition is shared by every transform in
    the pass, so it cannot follow any one transform's flag.
    """
    # Imported here, not at module scope. `smauglab/transforms/synthseg/__init__.py`
    # imports synthseg/transforms.py, which imports gpu/base.py -- and gpu/base.py imports
    # this module for its "off" defaults. A module-scope import would close that loop and
    # break every `import smauglab.transforms.gpu.base`; `unit_tests/test_imports.py` is
    # the canary. By the time any transform calls this, the package is fully imported.
    from smauglab.transforms.synthseg.functional import em_subdivide_labels

    background = labels <= 0
    if not bool(background.any()):
        return torch.zeros_like(labels), 1

    binary = (~background).long()
    device = image.device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else [], enabled=True):
        torch.manual_seed(next_shared_seed())
        fine, _generation, _merge = em_subdivide_labels(
            image,
            binary,
            n_foreground_clusters=1,
            background_clusters_range=clusters_range,
            background_label=0,
            same_on_batch=False,
        )
    # Background is parent index 0, so its fine ids are `0 * mult + assign` -- already the
    # 0-based cluster numbering this returns. Foreground ids are zeroed because the caller
    # replaces them.
    ids = torch.where(background, fine, torch.zeros_like(fine))
    return ids, max(int(ids.max()) + 1, 1)


# ── the decomposition, and its cache ─────────────────────────────────────────


def _label_units(labels: Tensor) -> tuple[Tensor, int]:
    """One unit per distinct positive label value, numbered batch-wide from 0."""
    counts = torch.bincount(labels.reshape(-1).clamp_min(0))
    present = counts.nonzero().flatten()
    positive = present[present > 0]
    lookup = torch.zeros(counts.numel(), dtype=torch.long, device=labels.device)
    lookup[positive] = torch.arange(positive.numel(), device=labels.device)
    return lookup[labels.clamp_min(0)], max(int(positive.numel()), 1)


def _as_batched(volume: Tensor) -> Tensor:
    """`[C, *spatial]` or `[B, C, *spatial]` as `[B, C, *spatial]`.

    The region helpers are called from two shapes of `apply_transform`: the vectorised
    ones hand over the whole batch, and the per-sample ones (`RandomInverseGPU`,
    `RandomDomainTransferGPU`) hand over one sample with no batch axis.
    """
    return volume if volume.dim() == 5 else volume.unsqueeze(0)


def _build_units(mask: Tensor, image: Tensor | None, spec: UnitSpec) -> RegionUnits:
    labels = label_map_from(_as_batched(mask))
    foreground = labels > 0

    if spec.bg_split == "em" and image is not None:
        background_ids, n_background = background_em_units(_as_batched(image), labels, spec.bg_em_clusters)
    else:
        background_ids, n_background = torch.zeros_like(labels), 1

    if spec.seg_unit == "label":
        foreground_ids, n_foreground = _label_units(labels)
    elif spec.seg_unit == "component":
        foreground_ids, n_foreground = connected_components_3d(labels, foreground)
    else:
        foreground_ids, n_foreground = torch.zeros_like(labels), 1

    ids = torch.where(foreground, n_background + foreground_ids.clamp_min(0), background_ids)
    return RegionUnits(ids, n_background + n_foreground, n_background)


def region_units(mask: Tensor, image: Tensor | None, spec: UnitSpec) -> RegionUnits:
    """The decomposition of `mask`, computed once per mask tensor per spec.

    Connected components and EM each cost tens of milliseconds on a 128^3 patch, and a
    pipeline runs twenty-odd region-selecting transforms over one batch. Without a cache
    the feature is not trainable; with one, every transform of a pass shares the same
    units, which is also what makes two region-selected transforms cut the background
    along the same lines rather than each along its own.

    **The cache is an attribute on the mask tensor**, keyed by spec. That choice is load
    bearing in three ways:

    * **Lifetime.** It dies with the tensor, so there is nothing to bound, evict or clear.
    * **Invalidation.** `MaskSequentialOpsCustom.transform` hands back the *same* tensor
      object for an image-only module and a *new* one for a geometric module, so the
      identity changes exactly when the mask changes. A geometric transform mid-pipeline
      invalidates this for free.
    * **Not a dict keyed on tensors.** `Tensor.__hash__` is identity-based but
      `Tensor.__eq__` is elementwise, so a hash-bucket collision between two different
      masks makes `dict` evaluate `bool(a == b)` and raise *"Boolean value of Tensor with
      more than one value is ambiguous"*. Measured: a `WeakKeyDictionary` with five
      hundred same-shaped tensor keys raises reliably. An attribute has no buckets.

    Keyed on the mask **as the transform was handed it**, which is what keeps the image
    and the units row-aligned. kornia slices the image down to the rows that drew below
    `p` and `_seg_for_applied_rows` slices the mask to match, so a decomposition built
    from the full batch would pair B-row units with a B'-row image -- and `bg_split="em"`
    reads that image. When every row applies, which is the common case and always true at
    `p = 1`, the slice *is* the full mask object and every module of the pass hits the
    same entry. A module whose row selection is a strict subset pays for its own
    decomposition, which is the honest price of having the rows line up.
    """
    cached = getattr(mask, _CACHE_ATTR, None)
    if cached is not None and spec in cached:
        return cached[spec]
    units = _build_units(mask, image, spec)
    if cached is not None:
        cached[spec] = units
    else:
        try:
            setattr(mask, _CACHE_ATTR, {spec: units})
        except AttributeError:  # pragma: no cover -- a tensor subclass that forbids it
            pass
    return units


# ── the per-call draws ───────────────────────────────────────────────────────


def choose_region_modes(transform: RegionSelecting, batch: int, device: torch.device) -> Tensor | None:
    """One mode per sample, or None when neither probability can fire.

    Returns `MODE_ALL` / `MODE_IN` / `MODE_OUT` / `MODE_BOTH` codes. **Both fired means
    `MODE_BOTH`**, where it used to fall through to `MODE_ALL` -- so `in_seg=1, out_seg=1`
    stopped being indistinguishable from `in_seg=0, out_seg=0`.

    Returns None, **having drawn nothing**, when both probabilities are zero. The version
    this replaces drew its two uniforms whenever a mask was attached, which is the defect
    `984f037` removed for `mix_prob`: with the draws unconditional, a run with region
    selection off and a run with it on were never on the same stream, so no sweep could be
    compared against its own baseline at a fixed seed.

    Per sample, not per batch. Ten of the eleven call sites used to take one decision for
    the whole batch, so `in_seg=0.5` meant "half the batches" rather than "half the
    samples" -- while `p` and `mix_prob` were per-sample all along. On the image's device
    and with no `dtype=`, mirroring `mix_with_original`, so a half-precision batch stays on
    the same stream as a single-precision one.
    """
    if not region_selection_active(transform):
        return None
    draws = torch.rand(batch, 2, device=device)
    fired_in = draws[:, 0] < transform.in_seg
    fired_out = draws[:, 1] < transform.out_seg
    # The code arithmetic is the definition: 0 neither, 1 in, 2 out, 3 both.
    return fired_in.long() + 2 * fired_out.long()


def draw_feather(transform: RegionSelecting) -> float:
    """The feather sigma for this call, in voxels. Zero means a hard boundary.

    One draw per call rather than per sample: `gaussian_blur3d` reads its sigmas back to
    the host (`.tolist()`), so a per-sample sigma would be one blur call and one
    synchronisation per sample. The modes and the per-unit keeps stay per sample; a
    boundary *width* is not a decision about where. From the CPU default generator as a
    Python float, the way `blend_one` draws its gate, so nothing has to be read off the
    device.

    Spends nothing when `seg_feather` is zero, and otherwise spends exactly one uniform
    whatever the modes came out as. Gating the draw on "did any sample pick a mode that
    uses it" would make the stream a function of the data. Drawn even when `lo == hi`, for
    the reason `mix_alpha` is: otherwise the stream becomes a function of the range, and
    two runs differing only in `seg_feather` would not be comparable.
    """
    low, high = transform.seg_feather
    if high <= 0.0:
        return 0.0
    return float(torch.rand(1).item()) * (high - low) + low


def region_alpha(transform: RegionSelecting, units: RegionUnits, modes: Tensor) -> Tensor:
    """The `[B, 1, *spatial]` composite weight: 1 where the transform applies, 0 where not.

    Built as one gather off the unit id map. Nothing per-unit is materialised: a noisy
    mask under `seg_unit="component"` can hold hundreds of components, and
    `[2, 300, 128, 128, 128]` of bool is 1.2 GB.

    The keep/drop draw is a fixed-width block (:data:`UNIT_DRAW_WIDTH`) so the number of
    uniforms it spends depends on the configuration and not on how many labels this
    particular patch happens to contain -- otherwise every transform after this one would
    diverge between a four-label patch and a five-label one. Unit ids at or above the
    width fold into the last column, which ties those units' fates together rather than
    reading past the draw.

    The fall-back rule -- **if a side kept nothing, keep all of it** -- is applied per
    side, and costs no extra draw. It is what makes the degenerate cases come out right:
    an empty mask has no occupied foreground unit, so `MODE_IN` is a no-op and `MODE_OUT`
    covers everything, which is exactly what the binary version did.
    """
    device = units.ids.device
    width = UNIT_DRAW_WIDTH[transform.seg_unit] + transform.bg_em_clusters[1]
    columns = min(units.n_units, width)
    column = units.ids.clamp(0, columns - 1)

    keep = torch.rand(modes.numel(), width, device=device)[:, :columns] < transform.seg_unit_prob
    # Background units come first, and only the high end is folded, so a column is
    # foreground exactly when its index is at or past `n_background`.
    is_foreground = torch.arange(columns, device=device) >= units.n_background
    wants_foreground = (modes == MODE_IN) | (modes == MODE_BOTH)
    wants_background = (modes == MODE_OUT) | (modes == MODE_BOTH)
    wanted = torch.where(is_foreground.unsqueeze(0), wants_foreground.unsqueeze(1), wants_background.unsqueeze(1))

    selected = keep & wanted
    for side in (is_foreground, ~is_foreground):
        on_this_side = selected & side
        nothing_kept = ~on_this_side.any(dim=1, keepdim=True)
        selected = selected | (nothing_kept & wanted & side)

    alpha = torch.gather(selected, 1, column.flatten(1)).view(column.shape)
    everywhere = (modes == MODE_ALL).view(-1, *([1] * (alpha.dim() - 1)))
    return (alpha | everywhere).to(torch.float32)


def feather_alpha(alpha: Tensor, sigma: float) -> Tensor:
    """`alpha` with its boundaries turned from a step into a Gaussian ramp.

    Reflect padding, which is `gaussian_blur3d`'s default and the reason to use it rather
    than a local copy: zero padding pulls the weight toward 0 at the patch border, so a
    unit touching the border would get a soft edge that is an artefact of the patch rather
    than of the segmentation. `gpu/fromSeg.py` carries a note about exactly that bug in
    the local blur it replaced.

    Every boundary in `alpha` is feathered, not only foreground against background: under
    `seg_unit="label"` or `"component"` that includes the boundary between a kept unit and
    a dropped neighbour, and under `bg_split="em"` the boundaries between clusters -- which
    is what stops an EM split being visible as a hard intensity edge.

    The ramp is centred on the boundary and passes through 0.5 there, so feathering
    shrinks the full-strength region slightly. Dilating by `sigma` first would grow the
    ramp outward instead and keep each unit at full strength; that is one extra
    `max_pool3d` and a second parameter, and worth adding only if the shrink turns out to
    matter.

    Two-dimensional spatial inputs pass through unfeathered: the blur is 3-D. The sigma is
    still drawn, because the draw must not depend on the data.
    """
    if sigma <= 0.0 or alpha.dim() != 5:
        return alpha
    return gaussian_blur3d(alpha, sigma).clamp(0.0, 1.0)


# ── the public entry point ───────────────────────────────────────────────────


def _rescaled_into(source: Tensor, target: Tensor, reduce_dims: tuple[int, ...]) -> Tensor:
    """`source` linearly remapped onto `target`'s per-sample min-max range."""
    source_min = torch.amin(source, dim=reduce_dims, keepdim=True)
    source_max = torch.amax(source, dim=reduce_dims, keepdim=True)
    target_min = torch.amin(target, dim=reduce_dims, keepdim=True)
    target_max = torch.amax(target, dim=reduce_dims, keepdim=True)
    return (source - source_min) / (source_max - source_min + 1e-8) * (target_max - target_min) + target_min


def apply_region_selection(
    transform: RegionSelecting,
    orig: Tensor,
    transformed: Tensor,
    seg: Tensor | None,
    normalize: bool = False,
) -> Tensor:
    """Composite `transformed` over `orig` wherever the drawn region says it applies.

    `orig` and `transformed` are either `[N, *spatial]` -- one channel of a batch, which is
    what most `apply_transform` methods work in -- or `[*spatial]` for the transforms that
    assemble their result one sample at a time. `seg` is the segmentation in any of the
    three layouts :func:`label_map_from` reads.

    `normalize` rescales `transformed` into `orig`'s range first. That matters for the
    transforms whose output lives in a different space from their input: a SynthSeg or
    PALETTE synthesis is in `[0, 1]` while the patch it is composited onto is z-scored, so
    a raw composite would put an intensity step on the mask boundary that has nothing to do
    with the augmentation. The parameter existed on the function this replaces and had no
    caller; these are the callers it was written for.

    Returns `transformed` untouched, having drawn nothing, when there is no segmentation or
    when neither probability can fire.
    """
    if seg is None or not region_selection_active(transform):
        return transformed

    batched = orig.dim() >= 4
    reference = orig if batched else orig.unsqueeze(0)
    modes = choose_region_modes(transform, reference.shape[0], orig.device)
    assert modes is not None  # region_selection_active was just checked
    sigma = draw_feather(transform)

    units = region_units(seg, reference.unsqueeze(1), unit_spec(transform))
    if units.ids.shape[0] != reference.shape[0]:
        # kornia slices the image down to the rows that drew below `p`, and
        # `_seg_for_applied_rows` slices the mask to match, so these agree by
        # construction. Said out loud because the alternative is a broadcast that
        # silently pairs one sample's units with another sample's intensities.
        raise ValueError(
            f"{type(transform).__name__}: the segmentation has {units.ids.shape[0]} rows but the image has "
            f"{reference.shape[0]}; params['seg'] must be sliced to the rows apply_transform was handed."
        )
    alpha = feather_alpha(region_alpha(transform, units, modes), sigma)
    # `[B, 1, *spatial]` down to the caller's shape. The channel axis exists only so the
    # blur above sees a 5-D volume.
    weight = alpha.squeeze(1).to(transformed.dtype)
    if not batched:
        weight = weight.squeeze(0)

    if normalize:
        reduce_dims = tuple(range(1, orig.dim())) if batched else tuple(range(orig.dim()))
        transformed = _rescaled_into(transformed, orig, reduce_dims)
    return weight * transformed + (1.0 - weight) * orig
