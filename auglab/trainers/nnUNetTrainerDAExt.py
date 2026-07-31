from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from typing import Tuple, Union, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

import os
import torch
import importlib
from torch import autocast
import json
import shutil

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from auglab.transforms.cpu.transforms import AugTransforms
from auglab.transforms.cpu.contrast import ZscoreNormalization
import auglab.configs as configs
from auglab.transforms.gpu.transforms import AugTransformsGPU
from auglab.trainers.utils import DownsampleSegForDSTransformCustom

class nnUNetTrainerDAExt(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
    
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        ### Adds transforms
        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_CPU_JSON", str(configs_path / "transform_params.json"))
        transforms.append(AugTransforms(json_path=json_path, do_dummy_2d_data_aug=do_dummy_2d_data_aug, patch_size=patch_size, rotation_for_DA=rotation_for_DA, mirror_axes=mirror_axes))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
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
                mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainerDAExtGPU(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1000

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        self.transforms = AugTransformsGPU(json_path=json_path).to(self.device)
        print(f'Using AugLab GPU transforms with parameters from: {json_path}')

        # Load JSON parameters for validation augmentation checkpoints
        with open(json_path, 'r') as f:
            config = json.load(f)
        self.validation_augmentation_checkpoints = config.get("ValidationAugmentationCheckpoints", {}).get("checkpoints", [0])

        # Copy json transfrom parameters to output folder
        shutil.copy(
            json_path,
            os.path.join(self.output_folder, 'transform_params_gpu_used_for_training.json')
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_GPU_JSON", str(configs_path / "transform_params_gpu.json"))
        with open(json_path, 'r') as f:
            config = json.load(f)

        ### Keep some nnunet transforms
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        if 'nnUNetSpatialTransform' in config:
            spatial_params = config['nnUNetSpatialTransform']
        else:
            spatial_params = {}

        transforms.append(
            SpatialTransform(
                patch_size_spatial, 
                patch_center_dist_from_border=spatial_params.get('patch_center_dist_from_border', 0),
                random_crop=spatial_params.get('random_crop', False),
                p_elastic_deform=spatial_params.get('p_elastic_deform', 0),
                p_rotation=spatial_params.get('p_rotation', 0),
                rotation=rotation_for_DA, 
                p_scaling=spatial_params.get('p_scaling', 0), 
                scaling=spatial_params.get('scaling', (0.7, 1.4)), 
                p_synchronize_scaling_across_axes=spatial_params.get('p_synchronize_scaling_across_axes', 1),
                bg_style_seg_sampling=False, 
                mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

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
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():            
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
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict, augmentation_prob: float) -> dict:
            data = batch['data']
            target = batch['target']
    
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
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
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
            axes = [0] + list(range(2, output.ndim))
    
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
                    if target.dtype == torch.bool:
                        mask = ~target[:, -1:]
                    else:
                        mask = 1 - target[:, -1:]
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
    
            return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
class nnUNetTrainerDAExtHybrid(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_HYBRID_JSON", str(configs_path / "transform_params_hybrid.json"))
        self.transforms = AugTransformsGPU(json_path=json_path).to(self.device)
        print(f'Using AugLab hybrid transforms with parameters from: {json_path}')

        # Copy json transfrom parameters to output folder
        shutil.copy(
            json_path,
            os.path.join(self.output_folder, 'transform_params_hybrid_used_for_training.json')
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        ### Adds transforms
        # Load transform parameters from json file
        configs_path = importlib.resources.files(configs)
        json_path = os.environ.get("AUGLAB_PARAMS_HYBRID_JSON", str(configs_path / "transform_params_hybrid.json"))
        transforms.append(AugTransforms(json_path=json_path, do_dummy_2d_data_aug=do_dummy_2d_data_aug, patch_size=patch_size, rotation_for_DA=rotation_for_DA, mirror_axes=mirror_axes))
        
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        # The following augmentations are related to special nnunet executions
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )
        
        # NOTE: DownsampleSegForDSTransform is now handled in train_step for GPU augmentations
        # if deep_supervision_scales is not None:
        #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

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
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():            
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
        return {'loss': l.detach().cpu().numpy()}
