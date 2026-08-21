"""Registry entries for the batchgeneratorsv2 transforms the CPU pipeline composes.

These are third-party classes, so they cannot carry an `@register` decorator; the
entries are built here instead. Everything else about them is identical to a
decorated augmentation -- same config key rules, same signature-derived parameter
validation.

Two things differ from the GPU side and are declared per entry:

* `wrap_random=False` for transforms that are appended directly rather than inside
  a `RandomTransform(...)`. Those own no application probability, so `p` is rejected
  in a config for them.
* `context_params` for values nnU-Net supplies at runtime (patch size, rotation
  range). A config must not set those, and the builder injects them.
"""

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from smauglab.registry import AugEntry, AugId, AugType, Backend, register_entry


def _entry(cls: type, aug_id: AugId, group: AugType, **kwargs) -> AugEntry:
    return register_entry(
        AugEntry(
            name=cls.__name__,
            cls=cls,
            backend=Backend.CPU,
            aug_id=aug_id,
            group=group,
            summary=(cls.__doc__ or "").strip().split("\n", 1)[0],
            **kwargs,
        )
    )


# Pipeline positions for these live in registry.PIPELINE_ORDER[Backend.CPU], taken
# from the sequence in AugTransforms._build_transforms.
_entry(
    SpatialTransform,
    AugId.SPATIAL,
    AugType.GEO,
    wrap_random=False,
    # patch_size is positional and rotation comes from nnU-Net's
    # configure_rotation_dummyDA_mirroring_and_inital_patch_size.
    context_params=("patch_size", "rotation"),
)
_entry(GaussianNoiseTransform, AugId.GAUSSIAN_NOISE, AugType.GE)
_entry(GaussianBlurTransform, AugId.GAUSSIAN_BLUR, AugType.GE)
_entry(
    MultiplicativeBrightnessTransform,
    AugId.BRIGHTNESS,
    AugType.GE,
    param_adapters={"multiplier_range": BGContrast},
)
_entry(
    ContrastTransform,
    AugId.CONTRAST,
    AugType.GE,
    param_adapters={"contrast_range": BGContrast},
)
_entry(SimulateLowResolutionTransform, AugId.LOW_RES, AugType.GE)
_entry(GammaTransform, AugId.GAMMA, AugType.GE, param_adapters={"gamma": BGContrast})
# Appended bare, and its allowed_axes comes from the config rather than the trainer:
# AugTransforms reads transform_params["mirror_axes"], not its own mirror_axes argument.
_entry(MirrorTransform, AugId.MIRROR, AugType.GEO, wrap_random=False)
