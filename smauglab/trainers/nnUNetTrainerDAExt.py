"""The SmaugLab nnU-Net trainer.

One class, because the config already says everything the three previous trainers
encoded between them. `nnUNetTrainerDAExt` built the CPU pipeline, `...GPU` built the
GPU one plus nnU-Net's SpatialTransform, and `...Hybrid` built both -- but a config is
sectioned into "CPU" and "GPU", so which sections are populated decides that on its
own:

    transform_params.json         CPU: 19  GPU: 0   -> CPU-only, as ...DAExt did
    transform_params_gpu.json     CPU: 1   GPU: 26  -> GPU-only, as ...DAExtGPU did
    transform_params_hybrid.json  CPU: 19  GPU: 24  -> both, as ...DAExtHybrid did

The class keeps the name `nnUNetTrainerDAExtGPU` whatever the config contains. That is
not cosmetic: nnU-Net writes the trainer class name into every checkpoint
(`checkpoint['trainer_name']`) and resolves the class from it at inference, so
renaming it would make several hundred trained models unloadable.
"""

import importlib
import os
import shutil
import warnings
from typing import Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

from smauglab import configs
from smauglab.config import load_config
from smauglab.registry import Backend
from smauglab.trainers.utils import DownsampleSegForDSTransformCustom, nnunet_tail_transforms
from smauglab.transforms.build import build_cpu_pipeline, build_gpu_pipeline
from smauglab.transforms.gpu.base import AugmentationSequentialCustom

#: Env var naming the config to train with.
CONFIG_ENV = "SMAUGLAB_PARAMS_JSON"

#: Previous name, still honoured so existing sweep scripts keep working. The three
#: old variables (_CPU_JSON, _GPU_JSON, _HYBRID_JSON) picked a trainer as much as a
#: file; only this one ever had an external caller.
LEGACY_CONFIG_ENV = "SMAUGLAB_PARAMS_GPU_JSON"

DEFAULT_CONFIG = "transform_params_gpu.json"


def resolve_config_path() -> str:
    """Locate the config: new env var, then the deprecated one, then the default."""
    path = os.environ.get(CONFIG_ENV)
    if path:
        return path
    legacy = os.environ.get(LEGACY_CONFIG_ENV)
    if legacy:
        warnings.warn(
            f"{LEGACY_CONFIG_ENV} is deprecated; use {CONFIG_ENV}. The config's CPU and GPU "
            "sections now decide which augmentations run, so the name no longer implies a backend.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return str(importlib.resources.files(configs) / DEFAULT_CONFIG)


def _has_gpu_augmentations(config) -> bool:
    """Whether this config asks for anything on the GPU side."""
    return bool(config.names(Backend.GPU))


class nnUNetTrainerDAExtGPU(nnUNetTrainer):
    """nnU-Net trainer driven entirely by a SmaugLab config.

    CPU-section augmentations run in the dataloader worker; GPU-section ones run on
    the batch in `train_step`.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        json_path = resolve_config_path()
        config = load_config(json_path)

        # Built only when the config actually asks for GPU augmentations, so a
        # CPU-only config costs nothing per step and train_step stays a no-op.
        self.transforms: AugmentationSequentialCustom | None = None
        if _has_gpu_augmentations(config):
            self.transforms = AugmentationSequentialCustom(
                *build_gpu_pipeline(
                    config.section(Backend.GPU),
                    mode=config.pipeline_mode(),
                    options=config.pipeline_options("random_choose"),
                    source=config.source,
                ),
                data_keys=["input", "mask"],
                same_on_batch=True,
            ).to(self.device)

        print(f"Using SmaugLab transforms from: {json_path}")
        print(
            f"  CPU: {len(config.names(Backend.CPU))} augmentations, GPU: {len(config.names(Backend.GPU))}, mode: {config.pipeline_mode().value}"
        )

        shutil.copy(json_path, os.path.join(self.output_folder, "transform_params_used_for_training.json"))

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, tuple[int, ...]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[list, tuple, None],
        mirror_axes: tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: list[bool] | None = None,
        is_cascaded: bool = False,
        foreground_labels: Union[tuple[int, ...], list[int]] | None = None,
        regions: list[Union[list[int], tuple[int, ...], int]] | None = None,
        ignore_label: int | None = None,
    ) -> BasicTransform:
        """Dataloader-side augmentations: whatever the config's CPU section names.

        A staticmethod because that is nnU-Net's contract, so it cannot reach the
        instance's parsed config and resolves the path itself. `load_config` is
        cached, so the file is still read and validated once.
        """
        transforms = []

        config = load_config(resolve_config_path())
        transforms.extend(
            build_cpu_pipeline(
                config.section(Backend.CPU),
                do_dummy_2d_data_aug=do_dummy_2d_data_aug,
                patch_size=patch_size,
                rotation=rotation_for_DA,
                source=config.source,
            )
        )

        # Deep supervision has to come after whatever last deformed the mask. With GPU
        # augmentations that is train_step, so the downsampling happens there; without
        # them nothing touches the mask after this point and it belongs here, which is
        # where nnU-Net puts it. Passing None is how the tail is told to skip it.
        transforms.extend(
            nnunet_tail_transforms(
                use_mask_for_norm=use_mask_for_norm,
                deep_supervision_scales=None if _has_gpu_augmentations(config) else deep_supervision_scales,
                is_cascaded=is_cascaded,
                foreground_labels=foreground_labels,
                regions=regions,
                ignore_label=ignore_label,
            )
        )

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[list, tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[tuple[int, ...], list[int]] | None = None,
        regions: list[Union[list[int], tuple[int, ...], int]] | None = None,
        ignore_label: int | None = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True)
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=[*list(regions), ignore_label] if ignore_label is not None else regions, channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        # Now target should be a single tensor, not a list
        target = target.to(self.device, non_blocking=True)
        # if isinstance(target, list):
        #     target = [i.to(self.device, non_blocking=True) for i in target]
        # else:
        #     target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Apply GPU augmentations to full-resolution data/target, then build the
            # deep-supervision targets from the *augmented* mask. A CPU-only config
            # builds no GPU pipeline; nothing has touched the mask since the
            # dataloader, which already produced those targets.
            if self.transforms is not None:
                data, target = self.transforms(data, target)

                deep_supervision_scales = self._get_deep_supervision_scales()
                if deep_supervision_scales is not None:
                    ds_transform = DownsampleSegForDSTransformCustom(ds_scales=deep_supervision_scales)
                    target = ds_transform(target)

            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
