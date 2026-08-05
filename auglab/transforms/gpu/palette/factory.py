"""Config → PaletteSynthesisGPU factory with typed partitioner registries."""
from __future__ import annotations

from typing import Any, Dict

from auglab.transforms.gpu.palette.overlay import AnatomicalLabelOverlay
from auglab.transforms.gpu.palette.partitioners import (
    EMGMMInitial,
    EMGMMRefinement,
    IdentityRefinement,
    KMeans1DInitial,
    VoronoiRefinement,
)
from auglab.transforms.gpu.palette.transform import PaletteSynthesisGPU


INITIAL_REGISTRY: Dict[str, type] = {
    "kmeans1d": KMeans1DInitial,
    "em_gmm":   EMGMMInitial,
}

REFINEMENT_REGISTRY: Dict[str, type] = {
    "voronoi":  VoronoiRefinement,
    "em_gmm":   EMGMMRefinement,
    "identity": IdentityRefinement,
}

# Top-level config keys consumed directly by PaletteSynthesisGPU (everything
# else must be routed into a nested block or is rejected as unknown).
_TOP_KEYS = {
    "alpha_magnitude_range",
    "dark_threshold",
    "blur_sigmas_pre",
    "blur_sigmas_post",
    "p",
}

# Keys accepted at the top of the config but not passed to the transform ctor.
_TOP_META_KEYS = {"probability", "initial_partitioner", "refinement_partitioners", "overlay"}


def build_palette_from_cfg(cfg: Dict[str, Any]) -> PaletteSynthesisGPU:
    """Build a ``PaletteSynthesisGPU`` from a nested config dict.

    Raises ``ValueError`` if the initial partitioner is missing, if a type
    string is placed in the wrong registry slot, or if unknown top-level keys
    are present.
    """
    if "initial_partitioner" not in cfg:
        raise ValueError(
            "PaletteSynthesisTransform config requires an 'initial_partitioner' block "
            f"(one of: {list(INITIAL_REGISTRY)})"
        )

    unknown = set(cfg) - _TOP_KEYS - _TOP_META_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in PaletteSynthesisTransform: {sorted(unknown)}. "
            f"Expected any of: {sorted(_TOP_KEYS | _TOP_META_KEYS)}"
        )

    init_cfg = dict(cfg["initial_partitioner"])
    init_type = init_cfg.pop("type", None)
    if init_type is None:
        raise ValueError("initial_partitioner must specify a 'type' field")
    if init_type in REFINEMENT_REGISTRY and init_type not in INITIAL_REGISTRY:
        raise ValueError(
            f"'{init_type}' is a refinement partitioner, not an initial one. "
            f"Move it into 'refinement_partitioners'. "
            f"Valid initial types: {list(INITIAL_REGISTRY)}"
        )
    if init_type not in INITIAL_REGISTRY:
        raise ValueError(
            f"Unknown initial partitioner type '{init_type}'. "
            f"Valid types: {list(INITIAL_REGISTRY)}"
        )
    initial = INITIAL_REGISTRY[init_type](**init_cfg)

    refinements = []
    for i, r_cfg in enumerate(cfg.get("refinement_partitioners", []) or []):
        r_cfg = dict(r_cfg)
        r_type = r_cfg.pop("type", None)
        if r_type is None:
            raise ValueError(f"refinement_partitioners[{i}] must specify a 'type' field")
        if r_type in INITIAL_REGISTRY and r_type not in REFINEMENT_REGISTRY:
            raise ValueError(
                f"'{r_type}' is an initial partitioner and cannot be used as a refinement. "
                f"Valid refinement types: {list(REFINEMENT_REGISTRY)}"
            )
        if r_type not in REFINEMENT_REGISTRY:
            raise ValueError(
                f"Unknown refinement partitioner type '{r_type}'. "
                f"Valid types: {list(REFINEMENT_REGISTRY)}"
            )
        refinements.append(REFINEMENT_REGISTRY[r_type](**r_cfg))

    ov_cfg = cfg.get("overlay")
    overlay = None
    if ov_cfg is not None:
        ov_kwargs = {k: v for k, v in ov_cfg.items() if k != "enabled"}
        if ov_cfg.get("enabled", True):
            overlay = AnatomicalLabelOverlay(**ov_kwargs)

    top_kwargs = {k: cfg[k] for k in _TOP_KEYS if k in cfg}
    if "probability" in cfg and "p" not in top_kwargs:
        top_kwargs["p"] = cfg["probability"]

    return PaletteSynthesisGPU(
        initial_partitioner=initial,
        refinement_partitioners=refinements,
        overlay=overlay,
        **top_kwargs,
    )
