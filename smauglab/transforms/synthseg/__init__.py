"""SynthSeg generative augmentation for SmaugLab.

A faithful torch re-implementation of the SynthSeg "brain generator"
(Billot et al., Medical Image Analysis 2023; BBillot/SynthSeg, BBillot/lab2im):
synthesise a randomly-contrasted, randomly-resolved image from an anatomical
label map, together with the matching ground-truth segmentation.

Public API:
    SynthSegGenerator      -- the full generative model as an nn.Module
                              (``forward(label_map) -> (image, label)``).
    SynthSegTransformsGPU  -- config-driven driver, ``forward(data, target) ->
                              (image, target)`` (SmaugLab calling convention).
    RandomSynthSegGPU      -- ImageOnlyTransform that replaces the image with a
                              GMM synthesis of ``params['seg']`` (composes inside
                              AugmentationSequentialCustom pipelines).
"""

from smauglab.transforms.synthseg import functional
from smauglab.transforms.synthseg.generator import SynthSegGenerator
from smauglab.transforms.synthseg.transforms import RandomSynthSegGPU, SynthSegTransformsGPU

__all__ = [
    "RandomSynthSegGPU",
    "SynthSegGenerator",
    "SynthSegTransformsGPU",
    "functional",
]
