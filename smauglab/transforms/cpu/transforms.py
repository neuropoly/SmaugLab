"""The CPU augmentation pipeline, built from a config section via the registry.

The ~250-line `if` ladder this replaces also read three values from outside the
config: `patch_size` and `rotation` come from nnU-Net at runtime (declared as
`context_params` on the registry entries), a top-level `retain_stats` was pushed
into several blocks, and `mode_seg="nearest"` was hardcoded. The first is injected
by the builder; the other two were folded into the configs by the `migration/` script.
"""

from typing import Union

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from smauglab.config import load_config
from smauglab.registry import Backend
from smauglab.transforms.build import build_cpu_pipeline
from smauglab.transforms.cpu.contrast import ScharrConvTransform
from smauglab.transforms.cpu.spatial import SpatialCustomTransform


class AugTransforms(ComposeTransforms):
    """Dataloader-side augmentations, in registry order."""

    def __init__(
        self,
        json_path: str,
        do_dummy_2d_data_aug: bool,
        # tuple[int, ...], not tuple[int]: these are 3D patch shapes and axis
        # tuples, so the one-element form was never what callers pass.
        patch_size: Union[np.ndarray, tuple[int, ...]],
        rotation_for_DA: RandomScalar,
        # Accepted for signature compatibility with the nnU-Net trainers but unused:
        # the mirror axes come from the config's MirrorTransform block, which is
        # where the old builder read them from too (transform_params["mirror_axes"],
        # never its own argument).
        mirror_axes: tuple[int, ...] | None = None,
    ):
        config = load_config(str(json_path))
        self.transform_params = config.section(Backend.CPU)
        self.transforms = build_cpu_pipeline(
            self.transform_params,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            patch_size=patch_size,
            rotation=rotation_for_DA,
            source=config.source,
            order_source=config.order_source(),
        )
        super().__init__(transforms=self.transforms)


class AugTransformsTest(ComposeTransforms):
    def __init__(self):
        self.transforms = self._build_transforms()
        super().__init__(transforms=self.transforms)

    def _build_transforms(self):
        transforms = []

        # Scharr filter
        transforms.append(
            RandomTransform(
                ScharrConvTransform(absolute=True),
                apply_probability=0.9,
            )
        )

        # Affine transforms
        transforms.append(
            RandomTransform(
                SpatialCustomTransform(
                    affine=True,
                ),
                apply_probability=0.9,
            )
        )

        return transforms
