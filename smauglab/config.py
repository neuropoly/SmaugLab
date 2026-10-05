"""Load and validate a SmaugLab augmentation config.

A config is a JSON document with reserved top-level sections:

    {
      "_comment": "...",              // any key starting with '_' is ignored
      "GPU":      { "<ClassName>": { <constructor kwargs> }, ... },
      "CPU":      { ... },
      "pipeline": { "random_choose": { ... } }
    }

A key is a class name, exactly, and a parameter is a constructor argument, exactly.
Both are checked against the registry, and every problem in a file is reported at
once rather than one per run.

A flat document with no section is rejected: it used to be interpreted as "GPU or
CPU, whichever the keys look like", and the two namespaces overlapped enough that
`GaussianBlurTransform` meant different transforms depending on which builder read
it. `SECTION_HINT` below is what the errors tell the author to do about it.
"""

from __future__ import annotations

import copy
import difflib
import functools
import json
from enum import Enum
from pathlib import Path
from typing import Any

from smauglab import registry
from smauglab.registry import Backend, InvalidConfigError


class PipelineMode(str, Enum):
    """How the GPU pipeline arranges what the registry gives it.

    This used to be encoded in the *trainer class* -- a separate subclass per
    arrangement -- which baked the choice into the name of every run directory.
    """

    #: Everything in pipeline order. What `AugTransformsGPU` has always done.
    SEQUENTIAL = "sequential"
    #: Geometry in order, then TA and GE each shuffled inside a RandomChooseX.
    RANDOM_ORDER = "random_order"
    #: As above but the TA bucket keeps its order; GE is not bucketed separately.
    RANDOM_ORDER_TA = "random_order_ta"


class OrderSource(str, Enum):
    """Where a pipeline's transform order comes from.

    `registry` -- the default -- is `registry.PIPELINE_ORDER`, which is fixed and the
    same for every config. `config` takes the order the keys appear in the file
    instead, for callers who want to control the sequence per experiment.

    Not the default, because it makes the pipeline sensitive to something people
    reasonably treat as cosmetic: reordering or reformatting a config would silently
    change what it does. Opting in makes that intent explicit and greppable.
    """

    REGISTRY = "registry"
    CONFIG = "config"


#: Keys a config section may hold that are not augmentations.
NON_AUGMENTATION_KEYS = ("_",)


def validate_section(section: dict, backend: Backend) -> list[str]:
    """Return every problem in one backend's section. Empty means it will build."""
    problems: list[str] = []
    for name, params in section.items():
        if name.startswith(NON_AUGMENTATION_KEYS):
            continue
        try:
            entry = registry.get(name, backend)
        except registry.UnknownAugmentationError as exc:
            problems.append(str(exc).replace("\n", "\n    "))
            continue
        if not isinstance(params, dict):
            problems.append(f"{name}: expected a block of parameters, got {type(params).__name__}")
            continue

        accepted = registry.accepted_params(entry)
        problems.extend(registry.unknown_parameter_message(entry, key).replace("\n", "\n    ") for key in params if key not in accepted)
        missing = registry.required_params(entry) - set(params) - set(entry.context_params)
        if missing:
            problems.append(f"{name}: missing required parameter(s) {', '.join(sorted(missing))}")
    return problems


#: Top-level keys that are not augmentation sections.
RESERVED_SECTIONS = ("pipeline",)

#: Keys the `pipeline` section may hold.
PIPELINE_KEYS = ("mode", "order", "random_choose", "same_on_batch")

#: What to do about a config that has no backend section. Quoted by both errors that
#: can report it, so the two cannot drift apart.
SECTION_HINT = "move the augmentation blocks into a 'GPU' or 'CPU' section, and spell each key as its class name."


class SmaugConfig:
    """A parsed, validated config document."""

    def __init__(self, payload: dict, source: str = "<config>") -> None:
        self.payload = payload
        self.source = source
        self.validate()

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path) -> SmaugConfig:
        path = Path(path)
        return cls(json.loads(path.read_text()), source=path.name)

    # -- access ------------------------------------------------------------------

    def section(self, backend: Backend) -> dict[str, Any]:
        """The augmentation blocks for a backend.

        A deepcopy, because `load_config` caches documents and a caller that mutated
        what it got back would poison every later read of the same file.
        """
        return copy.deepcopy(self.payload.get(backend.value, {}))

    def pipeline_options(self, name: str) -> dict[str, Any]:
        """Options for a named pipeline feature, e.g. `random_choose`.

        Always returns a dict. The old builder did `config.get("RandomChooseXTransforms")`
        and then `.get()` on the result, which raised AttributeError on every config
        that omitted the block.
        """
        return copy.deepcopy(self.payload.get("pipeline", {}).get(name, {}))

    def pipeline_mode(self) -> PipelineMode:
        """How the GPU pipeline arranges its transforms.

        This used to be encoded in the *trainer class* -- a separate subclass per
        arrangement -- which meant the choice was baked into the name of every run
        directory and could not be varied without a new class. It is a property of
        the augmentation setup, so it belongs in the config next to it.
        """
        raw = self.payload.get("pipeline", {}).get("mode")
        return PipelineMode(raw) if raw else PipelineMode.SEQUENTIAL

    def order_source(self) -> OrderSource:
        """Where this config's transform order comes from.

        Defaults to the registry's fixed PIPELINE_ORDER. `"pipeline": {"order":
        "config"}` switches to the order the keys appear in the file.
        """
        raw = self.payload.get("pipeline", {}).get("order")
        return OrderSource(raw) if raw else OrderSource.REGISTRY

    def same_on_batch(self) -> bool:
        """Whether every sample of a batch shares one set of augmentation draws.

        Defaults to True, which is what the GPU pipeline has always forced. That is
        not only about sharing parameters: kornia samples the per-sample
        application mask with the same flag
        (`_adapted_sampling((B,), p, same_on_batch)`), so with True a transform's
        `p` is spent once for the whole batch -- either every sample is augmented
        or none is. Setting it to False restores the per-transform
        `"same_on_batch"` values the configs already carry, and makes `p`
        per-sample.

        It defaults to True rather than to the configs' own values because every
        published run was trained under the batch-wise behaviour; flipping it by
        default would silently stop the shipped configs reproducing their results.
        """
        raw = self.payload.get("pipeline", {}).get("same_on_batch")
        return True if raw is None else bool(raw)

    def names(self, backend: Backend) -> list[str]:
        return [k for k in self.section(backend) if not k.startswith("_")]

    # -- validation --------------------------------------------------------------

    def validate(self) -> None:
        problems: list[str] = []

        known_sections = {b.value for b in Backend} | set(RESERVED_SECTIONS)
        for key in self.payload:
            if key.startswith("_") or key in known_sections:
                continue
            problems.append(
                f"unknown top-level key {key!r}. Expected one of "
                f"{', '.join(sorted(known_sections))}, or a '_'-prefixed comment. "
                f"A flat config without a backend section is no longer accepted -- {SECTION_HINT}"
            )

        if not any(b.value in self.payload for b in Backend):
            problems.append(f"no GPU or CPU section; this looks like a pre-registry config. {SECTION_HINT}")

        problems.extend(self._pipeline_problems())

        for backend in Backend:
            section = self.payload.get(backend.value)
            if section is None:
                continue
            if not isinstance(section, dict):
                problems.append(f"{backend.value}: expected an object of augmentation blocks")
                continue
            problems.extend(f"{backend.value}.{p}" for p in validate_section(section, backend))

        if problems:
            raise InvalidConfigError(self.source, problems)

    def _pipeline_problems(self) -> list[str]:
        """Check the `pipeline` section, which was previously accepted unchecked."""
        pipeline = self.payload.get("pipeline")
        if pipeline is None:
            return []
        if not isinstance(pipeline, dict):
            return ["pipeline: expected an object"]

        problems = []
        for key in pipeline:
            if key.startswith("_") or key in PIPELINE_KEYS:
                continue
            close = difflib.get_close_matches(key, PIPELINE_KEYS, n=2, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            problems.append(f"pipeline: unknown key {key!r}. Accepted: {', '.join(PIPELINE_KEYS)}.{hint}")

        order = pipeline.get("order")
        if order is not None:
            valid_orders = [o.value for o in OrderSource]
            if order not in valid_orders:
                close = difflib.get_close_matches(str(order), valid_orders, n=2, cutoff=0.5)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                problems.append(f"pipeline.order: unknown source {order!r}. Accepted: {', '.join(valid_orders)}.{hint}")

        same_on_batch = pipeline.get("same_on_batch")
        if same_on_batch is not None and not isinstance(same_on_batch, bool):
            problems.append(f"pipeline.same_on_batch: expected true or false, got {same_on_batch!r}")

        mode = pipeline.get("mode")
        if mode is not None:
            valid = [m.value for m in PipelineMode]
            if mode not in valid:
                close = difflib.get_close_matches(str(mode), valid, n=2, cutoff=0.5)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                problems.append(f"pipeline.mode: unknown mode {mode!r}. Accepted: {', '.join(valid)}.{hint}")
        return problems


@functools.lru_cache(maxsize=8)
def load_config(path: str) -> SmaugConfig:
    """Parse and validate a config, once per path.

    Cached because the nnU-Net trainer reads the same file twice: its
    `get_training_transforms` is a staticmethod (nnU-Net's contract), so it cannot
    reach the instance's already-parsed config and has to open the file itself.
    `SmaugConfig.section` hands out copies, so sharing the parsed document is safe.
    """
    return SmaugConfig.from_path(path)


def validate_file(path: str | Path) -> list[str]:
    """Every problem in a config file, without raising. Empty means it is valid."""
    try:
        SmaugConfig.from_path(path)
    except InvalidConfigError as exc:
        return exc.problems
    except json.JSONDecodeError as exc:
        return [f"not valid JSON: {exc}"]
    return []


def config_hash(payload: dict, algo: str = "sha256") -> str:
    """Content-addressed identity for a config.

    Byte-for-byte the same canonicalisation segtransferaug has always used, because
    experiment directories are named `...-aug-<transform_hash>-c-<config_hash>` and
    changing it would orphan every existing run folder.
    """
    import hashlib

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.new(algo)
    digest.update(canonical)
    return digest.hexdigest()


def file_hash(path: str | Path, algo: str = "sha256") -> str:
    """Hash a source file's text.

    Used downstream to name experiment directories after the implementation that
    produced them, so a change to the transforms is visible in the run name.
    """
    import hashlib

    digest = hashlib.new(algo)
    digest.update(Path(path).read_text().encode("utf-8"))
    return digest.hexdigest()


def write_temp_config(payload: dict, directory: str | Path | None = None) -> str:
    """Materialise a config so it can be handed to a subprocess by path.

    Named after its content hash, so the same config reuses the same file and a
    sweep does not fill the directory with near-duplicates.
    """
    import tempfile

    target_dir = Path(directory) if directory else Path(tempfile.gettempdir()) / "smauglab_configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"transform_params_{config_hash(payload)[:8]}.json"
    path.write_text(json.dumps(payload, indent=4) + "\n")
    return str(path)
