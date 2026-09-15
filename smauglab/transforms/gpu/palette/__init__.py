"""PALETTE synthesis, taken apart into swappable partitioning blocks.

`RandomPaletteGPU` in `gpu/fromSeg.py` stays as it is -- it is the composition
every shipped config names, and rewriting it in terms of these blocks would put a
refactor in the way of reading what the published experiments ran. This package is
the composable form: same remap, same overlay, but the partitioning steps are
config.
"""

from smauglab.transforms.gpu.palette.base import (
    BlockContext,
    InitialPartitioner,
    RefinementPartitioner,
    signed_alpha_affine_remap,
)
from smauglab.transforms.gpu.palette.factory import INITIAL_REGISTRY, REFINEMENT_REGISTRY
from smauglab.transforms.gpu.palette.overlay import AnatomicalLabelOverlay
from smauglab.transforms.gpu.palette.partitioners import (
    EMGMMInitial,
    EMGMMRefinement,
    IdentityRefinement,
    KMeans1DInitial,
    VoronoiRefinement,
)
from smauglab.transforms.gpu.palette.transform import PaletteSynthesisGPU

__all__ = [
    "INITIAL_REGISTRY",
    "REFINEMENT_REGISTRY",
    "AnatomicalLabelOverlay",
    "BlockContext",
    "EMGMMInitial",
    "EMGMMRefinement",
    "IdentityRefinement",
    "InitialPartitioner",
    "KMeans1DInitial",
    "PaletteSynthesisGPU",
    "RefinementPartitioner",
    "VoronoiRefinement",
    "signed_alpha_affine_remap",
]
