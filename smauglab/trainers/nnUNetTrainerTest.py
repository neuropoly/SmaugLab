import importlib
from typing import Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

from smauglab import configs
from smauglab.trainers.utils import DownsampleSegForDSTransformCustom, nnunet_tail_transforms
from smauglab.transforms.cpu.transforms import AugTransformsTest
from smauglab.transforms.gpu.transforms import AugTransformsGPU


class nnUNetTrainerTest(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

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
        retain_stats: bool = False,
    ) -> BasicTransform:
        transforms = []

        ### Adds transforms
        transforms.append(AugTransformsTest())

        ### Keep some nnunet transforms
        if do_dummy_2d_data_aug:
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0,
                rotation=rotation_for_DA,
                p_scaling=0,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.extend(
            nnunet_tail_transforms(
                use_mask_for_norm=use_mask_for_norm,
                deep_supervision_scales=deep_supervision_scales,
                is_cascaded=is_cascaded,
                foreground_labels=foreground_labels,
                regions=regions,
                ignore_label=ignore_label,
            )
        )

        return ComposeTransforms(transforms)


class nnUNetTrainerTestGPU(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = configs_path / "transform_params_gpu.json"
        self.transforms = AugTransformsGPU(json_path=str(json_path)).to(self.device)

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
        retain_stats: bool = False,
    ) -> BasicTransform:
        transforms = []

        ### Keep some nnunet transforms
        if do_dummy_2d_data_aug:
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0,
                rotation=rotation_for_DA,
                p_scaling=0,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # deep_supervision_scales=None: this trainer always runs GPU augmentations, so
        # the mask is still being deformed after this point and train_step builds the
        # multi-scale targets from the augmented mask instead.
        transforms.extend(
            nnunet_tail_transforms(
                use_mask_for_norm=use_mask_for_norm,
                deep_supervision_scales=None,
                is_cascaded=is_cascaded,
                foreground_labels=foreground_labels,
                regions=regions,
                ignore_label=ignore_label,
            )
        )

        return ComposeTransforms(transforms)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        # Now target should be a single tensor, not a list
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Apply GPU augmentations to full-resolution data/target
            data, target = self.transforms(data, target)

            # Create multi-scale targets for deep supervision after augmentation
            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
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
