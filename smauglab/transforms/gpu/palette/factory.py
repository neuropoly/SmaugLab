"""Turn the config's nested blocks into partitioner objects.

`smauglab.registry` validates a config against the constructor signature, which
gets it as far as "`initial_partitioner` is a parameter `PaletteSynthesisGPU`
accepts". It cannot look inside the block, because the block's shape depends on
its own `type` field. That check lives here, and it runs at construction time --
before the first batch, not on the thousandth step of a training run.

The two registries are kept apart on purpose. `voronoi` genuinely cannot be an
initial partitioner (it subdivides a partition it is handed) and `kmeans1d`
cannot be a refinement, so naming one in the other's slot is a config error with
a specific fix, and says so rather than raising `KeyError` on a lookup.
"""

from typing import Any

from smauglab.transforms.gpu.palette.base import InitialPartitioner, RefinementPartitioner
from smauglab.transforms.gpu.palette.overlay import AnatomicalLabelOverlay
from smauglab.transforms.gpu.palette.partitioners import (
    EMGMMInitial,
    EMGMMRefinement,
    IdentityRefinement,
    KMeans1DInitial,
    VoronoiRefinement,
)

#: Config `type` -> block, for the partitioner that reads the raw image.
INITIAL_REGISTRY: dict[str, type[InitialPartitioner]] = {
    "kmeans1d": KMeans1DInitial,
    "em_gmm": EMGMMInitial,
}

#: Config `type` -> block, for the partitioners that subdivide an existing map.
REFINEMENT_REGISTRY: dict[str, type[RefinementPartitioner]] = {
    "voronoi": VoronoiRefinement,
    "em_gmm": EMGMMRefinement,
    "identity": IdentityRefinement,
}


def make_initial(spec: InitialPartitioner | dict[str, Any]) -> InitialPartitioner:
    """Coerce an `initial_partitioner` config block into a block instance."""
    if isinstance(spec, InitialPartitioner):
        return spec
    cfg = dict(spec)
    type_name = cfg.pop("type", None)
    if type_name is None:
        raise ValueError(f"initial_partitioner must name a 'type' (one of: {sorted(INITIAL_REGISTRY)})")
    if type_name not in INITIAL_REGISTRY:
        raise ValueError(_bad_type_message(type_name, "initial_partitioner"))
    return INITIAL_REGISTRY[type_name](**cfg)


def make_refinement(spec: RefinementPartitioner | dict[str, Any], index: int) -> RefinementPartitioner:
    """Coerce one `refinement_partitioners` entry into a block instance."""
    if isinstance(spec, RefinementPartitioner):
        return spec
    cfg = dict(spec)
    type_name = cfg.pop("type", None)
    if type_name is None:
        raise ValueError(f"refinement_partitioners[{index}] must name a 'type' (one of: {sorted(REFINEMENT_REGISTRY)})")
    if type_name not in REFINEMENT_REGISTRY:
        raise ValueError(_bad_type_message(type_name, f"refinement_partitioners[{index}]"))
    return REFINEMENT_REGISTRY[type_name](**cfg)


def make_overlay(spec: AnatomicalLabelOverlay | dict[str, Any] | None) -> AnatomicalLabelOverlay | None:
    """Coerce an `overlay` config block, honouring its `enabled` flag.

    `enabled` is a block key rather than an absent block so that a config can turn
    the overlay off without losing the settings it would use when turned back on.
    """
    if spec is None or isinstance(spec, AnatomicalLabelOverlay):
        return spec
    cfg = dict(spec)
    if not cfg.pop("enabled", True):
        return None
    return AnatomicalLabelOverlay(**cfg)


def _bad_type_message(type_name: str, where: str) -> str:
    """Why the type is wrong, and -- for a block in the other slot -- where it goes."""
    slot_is_initial = where == "initial_partitioner"
    valid = INITIAL_REGISTRY if slot_is_initial else REFINEMENT_REGISTRY
    other = REFINEMENT_REGISTRY if slot_is_initial else INITIAL_REGISTRY

    if type_name in other:
        other_slot = "refinement_partitioners" if slot_is_initial else "initial_partitioner"
        return f"{where}: {type_name!r} belongs in {other_slot!r}, not here. Valid types here: {sorted(valid)}"
    return f"{where}: unknown partitioner type {type_name!r}. Valid types: {sorted(valid)}"
