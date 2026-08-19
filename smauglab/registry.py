"""The single source of truth for which augmentations exist.

Each augmentation class registers itself with `@register(...)`. The registry then
answers:

* what exists, per backend                      -- `names()`, `entries()`
* which class a config key maps to              -- `get()`
* what parameters that class accepts            -- `accepted_params()`
* which backends implement a given concept      -- `matrix()`, keyed on `AugId`

Deliberately stdlib-only, and deliberately importing nothing from
`smauglab.transforms` at module scope: that is what keeps the import graph acyclic
(transform modules import `register` from here) and lets a CLI ask "what exists?"
without paying for torch until it actually has to.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib
import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar

__all__ = [
    "AugEntry",
    "AugId",
    "AugType",
    "Backend",
    "InvalidConfigError",
    "RegistryError",
    "UnknownAugmentationError",
    "UnknownParameterError",
    "accepted_params",
    "entries",
    "get",
    "isolated",
    "matrix",
    "names",
    "register",
    "register_entry",
]


# `StrEnum` is 3.11+; pyproject promises `requires-python = ">=3.10"`.
class Backend(str, Enum):
    """Where an augmentation runs, which is also which config section it lives in."""

    GPU = "GPU"
    CPU = "CPU"
    MONAI = "MONAI"


class AugType(str, Enum):
    """Augmentation strength class.

    Member names are verbatim from segtransferaug's `AUG2GROUP`, which this
    replaces, so downstream `AugType.TA` keeps resolving. The GPU random-order
    pipelines bucket transforms by this to decide what goes inside each
    `RandomChooseXTransformsGPU`.
    """

    GEO = "GEO"  # geometric: applied in sequence, never bucketed
    GE = "GE"  # general enhancement, usually light
    TA = "TA"  # transfer augmentation, usually heavy


class AugId(str, Enum):
    """A backend-neutral augmentation *concept*.

    Config keys are class names and therefore backend-specific
    (`RandomGaussianNoiseGPU` vs batchgeneratorsv2's `GaussianNoiseTransform`), so
    a join key is needed to say "these are the same augmentation on two backends".
    That is what makes `matrix()` -- the CPU/GPU/MONAI coverage table -- possible.

    An enum rather than a free string on purpose: a typo'd `"gausian_noise"` would
    silently orphan a matrix row, whereas `AugId.GAUSIAN_NOISE` fails at import.
    The member list doubles as the authoritative inventory of concepts, which is
    what "track which augmentations have a MONAI version" actually means -- the row
    exists, and an empty cell is the record that no implementation does.
    """

    # -- geometric
    FLIP = "flip"
    AFFINE = "affine"
    CROP = "crop"
    SPATIAL = "spatial"  # nnU-Net's own SpatialTransform
    # -- general enhancement
    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAMMA = "gamma"
    INV_GAMMA = "inv_gamma"
    CLAMP = "clamp"
    LOW_RES = "low_res"
    ACQ = "acq"
    ZSCORE = "zscore"
    MIRROR = "mirror"
    # -- transfer augmentation
    SCHARR = "scharr"
    LAPLACE = "laplace"
    UNSHARP_MASK = "unsharp_mask"
    RAND_CONV = "rand_conv"
    BIAS_FIELD = "bias_field"
    INVERSE = "inverse"
    HISTOGRAM_EQUAL = "histogram_equal"
    REDISTRIBUTE_SEG = "redistribute_seg"
    PALETTE = "palette"
    DOMAIN_TRANSFER = "domain_transfer"
    SYNTHSEG = "synthseg"
    ARTIFACT = "artifact"
    SPATIAL_CUSTOM = "spatial_custom"
    SHAPE = "shape"
    # -- elementwise functions, one concept each so key <-> class stays 1:1
    FUNC_LOG1P = "func_log1p"
    FUNC_SQRT = "func_sqrt"
    FUNC_SIN = "func_sin"
    FUNC_EXP = "func_exp"
    FUNC_SIGMOID = "func_sigmoid"


class RegistryError(Exception):
    """Base class for every registry and config-resolution failure."""


class UnknownAugmentationError(RegistryError, KeyError):
    """A config named an augmentation that is not registered for that backend."""

    def __str__(self) -> str:  # KeyError.__str__ would repr() the message
        return self.args[0] if self.args else ""


class UnknownParameterError(RegistryError, TypeError):
    """A config passed a parameter the augmentation's constructor does not accept."""


class InvalidConfigError(RegistryError, ValueError):
    """A config has one or more problems. Every problem is reported at once, so a
    broken file is fixed in one pass rather than one pytest run at a time."""

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        joined = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"{source}: {len(problems)} problem(s)\n{joined}")


# Renamed *parameters*, for diagnostics only -- consulted when building an error
# message, NEVER when loading a config. difflib cannot bridge these on its own:
# "probability" vs "p" scores ~0.17, well under any usable cutoff, so without this
# table the single most common migration mistake would get no suggestion at all.
# A test asserts that no key here resolves through `get()`, which is what keeps it
# from becoming a back-compat path.
#
# Renamed augmentations are deliberately absent: stem matching in
# `_unknown_augmentation_message` already bridges "ScharrTransform" -> RandomScharrGPU
# and "SynthSeg" -> RandomSynthSegGPU without a table to maintain.
RENAMED_HINTS: Mapping[str, str] = MappingProxyType(
    {
        "probability": "p",
        "shear": "shears",
        "invert_image": "use RandomInvGammaGPU instead",
        "kernel_type": "the class name now carries the kernel (e.g. RandomScharrGPU)",
        "func": "the class name now carries the function (e.g. RandomLog1pGPU)",
        "one_dim": "use RandomAcqTransformGPU (one_dim) or RandomLowResTransformGPU",
    }
)


@dataclass(frozen=True, slots=True)
class AugEntry:
    """One registered augmentation: a class plus everything a config needs to know."""

    name: str
    cls: type
    backend: Backend
    aug_id: AugId
    group: AugType
    order: int
    # Set when the constructor legitimately forwards **kwargs to another class, so
    # `accepted_params` unions both signatures instead of giving up. RandomSynthSegGPU
    # forwards ~38 parameters to SynthSegGenerator.
    forwards_to: type | None = None
    # Supplied by the builder at runtime (nnU-Net hands over patch size, rotation and
    # mirror axes), so these are rejected if a config tries to set them.
    context_params: tuple[str, ...] = ()
    # CPU only: batchgeneratorsv2 puts the apply probability on a RandomTransform
    # wrapper rather than the transform, so `p` is a builder key, not a ctor kwarg.
    wrap_random: bool = True
    # Run in pipeline order even in the random-order pipelines, instead of going
    # into a shuffled RandomChooseX bucket. GEO transforms are always sequential;
    # this is for the ones whose group says otherwise. RandomLowResTransformGPU is
    # the only case: segtransferaug's AUG2GROUP calls it GE, but the random-order
    # builder has always run it in sequence, and the group is used downstream for
    # filtering, so the two meanings are kept apart rather than reconciled.
    force_sequential: bool = False
    # Name of an env var pointing at a large artefact the transform needs but the
    # wheel does not ship. Tests skip rather than fail when it is unset.
    external_asset: str | None = None
    # Parameters the builder must pass through a callable before handing them over.
    # batchgeneratorsv2 ranges are the reason: `BGContrast((0.7, 1.5))` samples 50/50
    # from [lo, 1] and [max(lo, 1), hi], where the bare tuple would be sampled
    # uniformly -- a different distribution, not a formatting detail. The config
    # stores the plain range; the adapter is applied on the way in.
    param_adapters: Mapping[str, Callable[[Any], Any]] = field(default_factory=lambda: MappingProxyType({}))
    # Constructor nudges that make the standalone smoke test exercise something.
    smoke_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    summary: str = ""

    def __post_init__(self) -> None:
        if self.cls.__name__ != self.name:
            raise RegistryError(
                f"registry name {self.name!r} does not match class name {self.cls.__name__!r}. "
                "The config key is the class name, so these cannot diverge."
            )
        if _has_var_keyword(self.cls) and self.forwards_to is None:
            raise RegistryError(
                f"{self.name}.__init__ takes **kwargs but declares no forwards_to. "
                "Parameter validation reads the signature, so **kwargs would silently "
                "accept anything. Name the parameters explicitly, or set forwards_to "
                "to the class the kwargs are passed on to."
            )


def _constructor_signature(cls: type) -> inspect.Signature | None:
    """The constructor signature, already without `self`.

    `inspect.signature(cls)` rather than `inspect.signature(cls.__init__)`: it drops
    `self` for us, and mypy rejects reading `__init__` off a `type` as unsound.
    """
    try:
        return inspect.signature(cls)
    except (TypeError, ValueError):  # builtins and C extensions have no signature
        return None


def _has_var_keyword(cls: type) -> bool:
    """True if the constructor ends in **kwargs."""
    signature = _constructor_signature(cls)
    if signature is None:
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def _named_params(cls: type) -> dict[str, inspect.Parameter]:
    """Named keyword-assignable constructor parameters, minus *args/**kwargs."""
    signature = _constructor_signature(cls)
    if signature is None:
        return {}
    skip = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return {name: p for name, p in signature.parameters.items() if p.kind not in skip}


# backend -> name -> entry. Insertion order is irrelevant; `entries()` sorts by `order`.
_REGISTRY: dict[Backend, dict[str, AugEntry]] = {backend: {} for backend in Backend}
# Held in a dict rather than a bare module global so `load_all` can flip it without
# a `global` statement.
_state: dict[str, bool] = {"loaded": False}

T = TypeVar("T", bound=type)


def register_entry(entry: AugEntry) -> AugEntry:
    """Add an entry, rejecting anything that would make lookups ambiguous.

    The call form exists for third-party classes that cannot be decorated -- the
    batchgeneratorsv2 transforms the CPU pipeline composes directly.
    """
    backend_entries = _REGISTRY[entry.backend]

    clash = backend_entries.get(entry.name)
    if clash is not None:
        raise RegistryError(f"{entry.backend.value} augmentation {entry.name!r} is already registered (as {clash.cls.__module__}).")

    same_order = next((e for e in backend_entries.values() if e.order == entry.order), None)
    if same_order is not None:
        raise RegistryError(
            f"{entry.backend.value} order {entry.order} is taken by {same_order.name!r}, so the "
            f"pipeline position of {entry.name!r} would be ambiguous. Orders are 10-spaced -- pick a free slot."
        )

    backend_entries[entry.name] = entry
    return entry


def register(
    *,
    aug_id: AugId,
    backend: Backend,
    group: AugType,
    order: int,
    forwards_to: type | None = None,
    context_params: tuple[str, ...] = (),
    wrap_random: bool = True,
    force_sequential: bool = False,
    external_asset: str | None = None,
    smoke_kwargs: Mapping[str, Any] | None = None,
    param_adapters: Mapping[str, Callable[[Any], Any]] | None = None,
    summary: str = "",
) -> Callable[[T], T]:
    """Register the decorated augmentation class. Returns the class unchanged."""

    def decorate(cls: T) -> T:
        register_entry(
            AugEntry(
                name=cls.__name__,
                cls=cls,
                backend=backend,
                aug_id=aug_id,
                group=group,
                order=order,
                forwards_to=forwards_to,
                context_params=context_params,
                wrap_random=wrap_random,
                force_sequential=force_sequential,
                external_asset=external_asset,
                smoke_kwargs=MappingProxyType(dict(smoke_kwargs or {})),
                param_adapters=MappingProxyType(dict(param_adapters or {})),
                summary=summary or _first_docstring_line(cls),
            )
        )
        return cls

    return decorate


def _first_docstring_line(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.strip().split("\n", 1)[0]


def load_all() -> None:
    """Import every transform module so the decorators have run. Idempotent."""
    if _state["loaded"]:
        return
    # Set before importing, not after: a transform module that queries the registry at
    # import time would otherwise re-enter this and recurse.
    _state["loaded"] = True
    importlib.import_module("smauglab.transforms")


def _ensure_loaded() -> None:
    if not _state["loaded"]:
        load_all()


def entries(
    backend: Backend | None = None,
    group: AugType | None = None,
    aug_id: AugId | None = None,
) -> list[AugEntry]:
    """Matching entries, in pipeline order.

    Order is a property of the augmentation, not of whoever last edited a config,
    so it comes from the registry rather than from config key order.
    """
    _ensure_loaded()
    backends = [backend] if backend is not None else list(Backend)
    found = [e for b in backends for e in _REGISTRY[b].values()]
    if group is not None:
        found = [e for e in found if e.group is group]
    if aug_id is not None:
        found = [e for e in found if e.aug_id is aug_id]
    return sorted(found, key=lambda e: (e.backend.value, e.order))


def names(backend: Backend | None = None, group: AugType | None = None) -> list[str]:
    """Registered config keys, in pipeline order."""
    return [e.name for e in entries(backend=backend, group=group)]


def get(name: str, backend: Backend | None = None) -> AugEntry:
    """Resolve a config key to its entry, or raise with a suggestion."""
    _ensure_loaded()
    backends = [backend] if backend is not None else list(Backend)
    for b in backends:
        entry = _REGISTRY[b].get(name)
        if entry is not None:
            return entry
    raise UnknownAugmentationError(_unknown_augmentation_message(name, backend))


def _unknown_augmentation_message(name: str, backend: Backend | None) -> str:
    candidates = names(backend)
    where = f"{backend.value} " if backend is not None else ""
    lines = [f"unknown {where}augmentation {name!r}."]

    # Two kinds of near-miss, and they find different things. difflib catches
    # typos; matching on the distinctive middle of the name catches a renamed key
    # such as "ScharrTransform" -> "RandomScharrGPU", which shares too little with
    # its replacement for any usable cutoff. Merged rather than used as a fallback:
    # searching across backends, difflib alone would fill the list with CPU
    # candidates and crowd out the GPU rename the caller is probably after.
    stem = name.removeprefix("Random").removesuffix("Transform").removesuffix("GPU")
    close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
    close += [c for c in candidates if stem and stem.lower() in c.lower() and c not in close]
    if close:
        lines.append(f"  Did you mean: {', '.join(close[:4])}?")

    lines.append(f"  {len(candidates)} registered: run `smauglab list` to see them.")
    return "\n".join(lines)


def accepted_params(entry: AugEntry) -> dict[str, inspect.Parameter]:
    """Every parameter a config may set for this augmentation.

    The constructor signature is the whole truth -- there is no hand-maintained
    schema to drift out of sync. `forwards_to` widens it where a class genuinely
    passes kwargs on; `context_params` narrows it where the builder supplies the
    value; and `p` is added for CPU entries whose probability lives on the
    RandomTransform wrapper rather than on the transform itself.
    """
    params = _named_params(entry.cls)
    if entry.forwards_to is not None:
        params = {**_named_params(entry.forwards_to), **params}
    for name in entry.context_params:
        params.pop(name, None)
    if entry.backend is Backend.CPU and entry.wrap_random and "p" not in params:
        params["p"] = inspect.Parameter("p", inspect.Parameter.KEYWORD_ONLY, default=1.0, annotation=float)
    return params


def required_params(entry: AugEntry) -> set[str]:
    """Parameters with no default, which a config must therefore supply."""
    return {name for name, p in accepted_params(entry).items() if p.default is inspect.Parameter.empty}


def unknown_parameter_message(entry: AugEntry, name: str) -> str:
    """Explain an unaccepted parameter, with a suggestion where one exists."""
    allowed = sorted(accepted_params(entry))
    lines = [f"{entry.name}: unknown parameter {name!r}."]
    close = difflib.get_close_matches(name, allowed, n=3, cutoff=0.6)
    if close:
        lines.append(f"  Did you mean: {', '.join(close)}?")
    elif name in RENAMED_HINTS:
        lines.append(f"  {name!r} -> {RENAMED_HINTS[name]}")
    if name in entry.context_params:
        lines[-1:] = [f"  {name!r} is supplied by the trainer at runtime and must not appear in the config."]
    lines.append(f"  Accepted: {', '.join(allowed)}")
    return "\n".join(lines)


def matrix() -> dict[AugId, dict[Backend, AugEntry | None]]:
    """Concept -> backend -> implementing entry, or None where there is none.

    Every `AugId` gets a row even with no implementations anywhere, because an
    empty cell is exactly the thing worth seeing.
    """
    _ensure_loaded()
    table: dict[AugId, dict[Backend, AugEntry | None]] = {aug_id: dict.fromkeys(Backend) for aug_id in AugId}
    for entry in entries():
        table[entry.aug_id][entry.backend] = entry
    return table


def render_matrix(fmt: str = "md") -> str:
    """The CPU/GPU/MONAI coverage table."""
    table = matrix()
    if fmt not in {"md", "table"}:
        raise ValueError(f"unknown format {fmt!r}; expected 'md' or 'table'")

    def cell(entry: AugEntry | None) -> str:
        if entry is None:
            return "—" if fmt == "md" else "-"
        return f"`{entry.name}`" if fmt == "md" else entry.name

    def group_of(row: dict[Backend, AugEntry | None]) -> str:
        found = next((e for e in row.values() if e is not None), None)
        return found.group.value if found else ""

    header = ["Augmentation", "Group", *(b.value for b in Backend)]
    rows = [[aug_id.value, group_of(row), *(cell(row[b]) for b in Backend)] for aug_id, row in table.items()]

    if fmt == "md":
        out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)

    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in [header, *rows])


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a default into something JSON can hold."""
    if value is inspect.Parameter.empty:
        # Required: no default to show. Null makes the hole visible, and strict
        # loading will reject it until someone fills it in.
        return None
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def render_template(backend: Backend) -> dict[str, dict[str, Any]]:
    """A config section naming every registered augmentation at its defaults.

    Round-trips through the same `accepted_params`/`inspect.signature` path that
    validation uses, so a template that fails to load is a real bug rather than a
    documentation slip. Checked into the repo, which is what makes "an augmentation
    exists but no config can reach it" a test failure.
    """
    section: dict[str, dict[str, Any]] = {}
    for entry in entries(backend):
        section[entry.name] = {name: _json_safe(param.default) for name, param in sorted(accepted_params(entry).items())}
    return section


def clear(mark_loaded: bool = True) -> None:
    """Drop every entry. For tests only -- never call this from library code.

    `mark_loaded` defaults to True so a test that clears the registry and registers
    its own synthetic entries does not have the real augmentations imported back in
    underneath it on the next lookup. Pass False to restore normal lazy loading.

    Note this is NOT undoable on its own: `load_all()` re-imports
    `smauglab.transforms`, but the module is already in `sys.modules` by then, so the
    `@register` decorators do not run a second time and the registry would stay empty
    for the rest of the process. Use `isolated()` unless you mean that.
    """
    for backend_entries in _REGISTRY.values():
        backend_entries.clear()
    _state["loaded"] = mark_loaded


@contextlib.contextmanager
def isolated() -> Iterator[None]:
    """Empty the registry for the duration of the block, then put it back.

    For tests that register synthetic augmentations and must not see the real ones.
    Restores by hand rather than by reloading, because re-importing the transform
    modules would not re-run their decorators -- see `clear()`.
    """
    saved = {backend: dict(entries_) for backend, entries_ in _REGISTRY.items()}
    was_loaded = _state["loaded"]
    clear(mark_loaded=True)
    try:
        yield
    finally:
        for backend, entries_ in saved.items():
            _REGISTRY[backend].clear()
            _REGISTRY[backend].update(entries_)
        _state["loaded"] = was_loaded
