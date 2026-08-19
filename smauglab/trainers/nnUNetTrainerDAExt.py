import importlib
import json
import os
import shutil
from typing import Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from torch import distributed as dist

from smauglab import configs
from smauglab.trainers.utils import DownsampleSegForDSTransformCustom
from smauglab.transforms.cpu.transforms import AugTransforms
from smauglab.transforms.gpu.transforms import AugTransformsGPU


class nnUNetTrainerDAExt(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

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
        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("SMAUGLAB_PARAMS_CPU_JSON", str(configs_path / "transform_params.json"))
        transforms.append(
            AugTransforms(
                json_path=json_path,
                do_dummy_2d_data_aug=do_dummy_2d_data_aug,
                patch_size=patch_size,
                rotation_for_DA=rotation_for_DA,
                mirror_axes=mirror_axes,
            )
        )

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

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, "We need foreground_labels for cascade augmentations"
            transforms.append(
                MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True)
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=[*list(regions), ignore_label] if ignore_label is not None else regions, channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainerDAExtGPU(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1000

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("SMAUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        self.transforms = AugTransformsGPU(json_path=json_path).to(self.device)
        print(f"Using SmaugLab GPU transforms with parameters from: {json_path}")

        # Load JSON parameters for validation augmentation checkpoints
        with open(json_path) as f:
            config = json.load(f)
        self.validation_augmentation_checkpoints = config.get("ValidationAugmentationCheckpoints", {}).get("checkpoints", [0])
        self.ema_dice_validation = [None] * len(self.validation_augmentation_checkpoints)
        self.best_ema_dice_validation = [None] * len(self.validation_augmentation_checkpoints)

        # Copy json transfrom parameters to output folder
        shutil.copy(json_path, os.path.join(self.output_folder, "transform_params_gpu_used_for_training.json"))

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

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

        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("SMAUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        with open(json_path) as f:
            config = json.load(f)

        ### Keep some nnunet transforms
        if do_dummy_2d_data_aug:
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size

        spatial_params = config.get("nnUNetSpatialTransform", {})

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=spatial_params.get("patch_center_dist_from_border", 0),
                random_crop=spatial_params.get("random_crop", False),
                p_elastic_deform=spatial_params.get("p_elastic_deform", 0),
                p_rotation=spatial_params.get("p_rotation", 0),
                rotation=rotation_for_DA,
                p_scaling=spatial_params.get("p_scaling", 0),
                scaling=spatial_params.get("scaling", (0.7, 1.4)),
                p_synchronize_scaling_across_axes=spatial_params.get("p_synchronize_scaling_across_axes", 1),
                bg_style_seg_sampling=False,
                mode_seg="nearest",
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, "We need foreground_labels for cascade augmentations"
            transforms.append(
                MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True)
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=[*list(regions), ignore_label] if ignore_label is not None else regions, channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

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

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

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
            # Apply GPU augmentations to full-resolution data/target
            data, target = self.transforms(data, target)

            # Create multi-scale targets for deep supervision after augmentation
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

    def validation_step(self, batch: dict, augmentation_prob: float) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        # Now target should be a single tensor, not a list
        target = target.to(self.device, non_blocking=True)
        # if isinstance(target, list):
        #     target = [i.to(self.device, non_blocking=True) for i in target]
        # else:
        #     target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # Apply GPU augmentations to full-resolution data/target
            if torch.rand(1).item() < augmentation_prob:
                data, target = self.transforms(data, target)

            # Create multi-scale targets for deep supervision after augmentation
            deep_supervision_scales = self._get_deep_supervision_scales()
            if deep_supervision_scales is not None:
                ds_transform = DownsampleSegForDSTransformCustom(ds_scales=deep_supervision_scales)
                target = ds_transform(target)

            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0, *list(range(2, output.ndim))]

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = ~target[:, -1:] if target.dtype == torch.bool else 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}

    @staticmethod
    def _valaug_sidecar_path(checkpoint_path: str) -> str:
        return os.path.join(os.path.dirname(checkpoint_path), "checkpoint_best_validation_aug_parameters.json")

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing:
            payload = {
                "ema_dice_validation": [float(x) if x is not None else None for x in self.ema_dice_validation],
                "best_ema_dice_validation": [float(x) if x is not None else None for x in self.best_ema_dice_validation],
                "validation_augmentation_checkpoints": list(self.validation_augmentation_checkpoints),
            }
            with open(self._valaug_sidecar_path(filename), "w") as f:
                json.dump(payload, f, indent=2)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if isinstance(filename_or_checkpoint, str):
            sidecar = self._valaug_sidecar_path(filename_or_checkpoint)
            if os.path.isfile(sidecar):
                with open(sidecar) as f:
                    state = json.load(f)
                # Only adopt persisted EMA state if the configured checkpoints list is unchanged;
                # otherwise the per-index slots no longer correspond and we restart tracking.
                if state.get("validation_augmentation_checkpoints") == self.validation_augmentation_checkpoints:
                    self.ema_dice_validation = state["ema_dice_validation"]
                    self.best_ema_dice_validation = state["best_ema_dice_validation"]

    def compute_validation_metrics(self, val_outputs: list[dict]):
        """
        Based on on_validation_epoch_end nnUNetTrainer
        """
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        val_losses = loss_here
        return mean_fg_dice, global_dc_per_class, val_losses

    def run_training(self):
        self.on_train_start()

        for _epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train)) for _batch_id in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [[] for _ in range(len(self.validation_augmentation_checkpoints))]
                for _batch_id in range(self.num_val_iterations_per_epoch):
                    batch = next(self.dataloader_val)
                    for prob_id, augmentation_prob in enumerate(self.validation_augmentation_checkpoints):
                        val_outputs[prob_id].append(self.validation_step(batch, augmentation_prob))
                for val_idx, val_output in enumerate(val_outputs):
                    mean_fg_dice, global_dc_per_class, val_losses = self.compute_validation_metrics(val_output)
                    self.ema_dice_validation[val_idx] = (
                        self.ema_dice_validation[val_idx] * 0.9 + 0.1 * mean_fg_dice
                        if self.ema_dice_validation[val_idx] is not None
                        else mean_fg_dice
                    )
                    if val_idx == 0:
                        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
                        self.logger.log("dice_per_class_or_region", global_dc_per_class, self.current_epoch)
                        self.logger.log("val_losses", val_losses, self.current_epoch)

            self.on_epoch_end()
            for val_idx, augmentation_prob in enumerate(self.validation_augmentation_checkpoints):
                ema_dice = self.ema_dice_validation[val_idx]
                best = self.best_ema_dice_validation[val_idx]
                if ema_dice is not None and (best is None or ema_dice > best):
                    self.best_ema_dice_validation[val_idx] = ema_dice
                    self.save_checkpoint(join(self.output_folder, f"checkpoint_best_validation_aug_{augmentation_prob}.pth"))

        self.on_train_end()


class nnUNetTrainerDAExtHybrid(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("SMAUGLAB_PARAMS_HYBRID_JSON", str(configs_path / "transform_params_hybrid.json"))
        self.transforms = AugTransformsGPU(json_path=json_path).to(self.device)
        print(f"Using SmaugLab hybrid transforms with parameters from: {json_path}")

        # Copy json transfrom parameters to output folder
        shutil.copy(json_path, os.path.join(self.output_folder, "transform_params_hybrid_used_for_training.json"))

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

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
        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("SMAUGLAB_PARAMS_HYBRID_JSON", str(configs_path / "transform_params_hybrid.json"))
        transforms.append(
            AugTransforms(
                json_path=json_path,
                do_dummy_2d_data_aug=do_dummy_2d_data_aug,
                patch_size=patch_size,
                rotation_for_DA=rotation_for_DA,
                mirror_axes=mirror_axes,
            )
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, "We need foreground_labels for cascade augmentations"
            transforms.append(
                MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True)
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=[*list(regions), ignore_label] if ignore_label is not None else regions, channel_in_seg=0
                )
            )

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

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
            # Apply GPU augmentations to full-resolution data/target
            data, target = self.transforms(data, target)

            # Create multi-scale targets for deep supervision after augmentation
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
