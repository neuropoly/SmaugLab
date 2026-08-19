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
it. See `migration/` in the repository to bring an old file forward -- `MIGRATE_HINT`
below is the single source of truth for that pointer, and the error messages quote it.
"""

from __future__ import annotations

import copy
import difflib
import functools
import json
from pathlib import Path
from typing import Any

from smauglab import registry
from smauglab.registry import Backend, InvalidConfigError
from smauglab.transforms.build import PipelineMode, validate_section

#: Top-level keys that are not augmentation sections.
RESERVED_SECTIONS = ("pipeline",)

#: Keys the `pipeline` section may hold.
PIPELINE_KEYS = ("mode", "random_choose")

#: How to bring a pre-registry config forward. The migrator is a one-time tool kept
#: in the repository rather than shipped in the wheel, so this points at the repo.
MIGRATE_HINT = "see migration/ in the SmaugLab repository to bring an old config forward."


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
                f"A flat config without a backend section is no longer accepted -- {MIGRATE_HINT}"
            )

        if not any(b.value in self.payload for b in Backend):
            problems.append(f"no GPU or CPU section; this looks like a pre-registry config. {MIGRATE_HINT}")

        problems.extend(self._pipeline_problems())

        for backend in Backend:
            section = self.payload.get(backend.value)
            if section is None:
                continue
            if not isinstance(section, dict):
                problems.append(f"{backend.value}: expected an object of augmentation blocks")
                continue
            problems.extend(f"{backend.value}.{p}" for p in validate_section(section, backend, source=self.source))

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


def registered_names(backend: Backend) -> list[str]:
    """Convenience re-export so callers need not import the registry directly."""
    return registry.names(backend)


# --- config manipulation ----------------------------------------------------------
#
# Upstreamed from segtransferaug/utils/smauglab_config.py, which drove the sweep
# scripts. Three module-level absolute paths went away (the packaged config is
# resolved through importlib.resources now), and the hardcoded AUG2GROUP table
# became the registry's `group` field, so a new augmentation no longer has to be
# added to a dict in a different repository before the sweeps can see it.


def default_config_path() -> Path:
    """The packaged default GPU config."""
    import importlib.resources

    from smauglab import configs

    return Path(str(importlib.resources.files(configs))) / "transform_params_gpu.json"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def transform_names(payload: dict, backend: Backend = Backend.GPU) -> list[str]:
    """The augmentations a config actually names."""
    return [k for k in payload.get(backend.value, {}) if not k.startswith("_")]


def single_transform_config(name: str, payload: dict, backend: Backend = Backend.GPU) -> dict:
    """A copy of the config with only one augmentation left enabled."""
    registry.get(name, backend)  # raises with a suggestion if the name is wrong
    out = copy.deepcopy(payload)
    section = out.get(backend.value, {})
    out[backend.value] = {k: v for k, v in section.items() if k == name or k.startswith("_")}
    return out


def drop_zero_probability(payload: dict, backend: Backend = Backend.GPU) -> dict:
    """Remove augmentations that would never fire, so the config says what it does."""
    out = copy.deepcopy(payload)
    section = out.get(backend.value, {})
    out[backend.value] = {k: v for k, v in section.items() if k.startswith("_") or v.get("p", 1.0) != 0}
    return out


def filter_by_group(payload: dict, group, backend: Backend = Backend.GPU) -> dict:
    """Keep only the augmentations in one GEO/GE/TA group."""
    keep = set(registry.names(backend, group=group))
    out = copy.deepcopy(payload)
    section = out.get(backend.value, {})
    out[backend.value] = {k: v for k, v in section.items() if k in keep or k.startswith("_")}
    return out


def set_probabilities(payload: dict, p: float, group=None, backend: Backend = Backend.GPU) -> dict:
    """Set `p` on every augmentation, or only on one group."""
    targets = set(registry.names(backend, group=group)) if group is not None else None
    out = copy.deepcopy(payload)
    for name, block in out.get(backend.value, {}).items():
        if name.startswith("_") or not isinstance(block, dict):
            continue
        if targets is None or name in targets:
            block["p"] = p
    return out


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


def remove_temp_configs(directory: str | Path | None = None) -> int:
    """Delete configs written by `write_temp_config`. Returns how many went."""
    import tempfile

    target_dir = Path(directory) if directory else Path(tempfile.gettempdir()) / "smauglab_configs"
    if not target_dir.is_dir():
        return 0
    removed = 0
    for path in target_dir.glob("transform_params_*.json"):
        path.unlink()
        removed += 1
    return removed
