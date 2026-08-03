import os
import json
import torch
import numpy as np
from typing import Union, Tuple

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform

from auglab.transforms.cpu.artifact import ArtifactTransform
from auglab.transforms.cpu.contrast import ConvTransform, HistogramEqualTransform, FunctionTransform
from auglab.transforms.cpu.fromSeg import RedistributeTransform
from auglab.transforms.cpu.spatial import SpatialCustomTransform, ShapeTransform

class AugTransforms(ComposeTransforms):
    def __init__(self, json_path: str, do_dummy_2d_data_aug: bool, patch_size: Union[np.ndarray, Tuple[int]],
                 rotation_for_DA: RandomScalar, mirror_axes: Tuple[int]):
        # Load transform parameters from JSON
        config_path = os.path.join(json_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'CPU' in config.keys():
            self.transform_params = config['CPU']
        else:
            self.transform_params = config
            
        self.transforms = self._build_transforms(
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            mirror_axes=mirror_axes
        )
        super().__init__(transforms=self.transforms)

    def _build_transforms(self, do_dummy_2d_data_aug: bool, patch_size: Union[np.ndarray, Tuple[int]],
                          rotation_for_DA: RandomScalar, mirror_axes: Tuple[int]):
        transform_params = self.transform_params
        transforms = []

        # Scharr filter
        conv_params = transform_params.get('ConvTransform')
        if conv_params is not None:
            transforms.append(RandomTransform(
                ConvTransform(
                    kernel_type=conv_params.get('kernel_type', 'Scharr'),
                    absolute=conv_params.get('absolute', True),
                    retain_stats=transform_params.get('retain_stats', False)
                ), apply_probability=conv_params.get('probability', 0)
            ))

        # Apply functions
        func_list = [
            lambda x: torch.log(1 + x),
            torch.sqrt,
            torch.sin,
            torch.exp,
            lambda x: 1/(1 + torch.exp(-x)),
        ]
        func_params = transform_params.get('FunctionTransform')
        if func_params is not None:
            for func in func_list:
                transforms.append(RandomTransform(
                    FunctionTransform(
                        function=func,
                        retain_stats=transform_params.get('retain_stats', False)
                    ), apply_probability=func_params.get('probability', 0)
                ))
        
        # Histogram manipulations
        hist_params = transform_params.get('HistogramEqualTransform')
        if hist_params is not None:
            transforms.append(RandomTransform(
                HistogramEqualTransform(
                    retain_stats=transform_params.get('retain_stats', False)
                ), apply_probability=hist_params.get('probability', 0)
            ))

        # Redistribute segmentation values
        redist_params = transform_params.get('RedistributeTransform')
        if redist_params is not None:
            transforms.append(RandomTransform(
                RedistributeTransform(
                    in_seg=redist_params.get('in_seg', 0),
                    retain_stats=transform_params.get('retain_stats', False)
                ), apply_probability=redist_params.get('probability', 0)
            ))

        # Resolution transforms
        shape_params = transform_params.get('ShapeTransform')
        if shape_params is not None:
            transforms.append(RandomTransform(
                ShapeTransform(
                    shape_min=shape_params.get('shape_min'),
                    ignore_axes=tuple(shape_params.get('ignore_axes', None)) if shape_params.get('ignore_axes', None) is not None else None,
                ), apply_probability=shape_params.get('probability', 0)
        ))

        # Artifacts generation
        artifact_params = transform_params.get('ArtifactTransform')
        if artifact_params is not None:
            transforms.append(RandomTransform(
                ArtifactTransform(
                    motion=artifact_params.get('motion', False),
                    ghosting=artifact_params.get('ghosting', False),
                    spike=artifact_params.get('spike', False),
                    bias_field=artifact_params.get('bias_field', False),
                    blur=artifact_params.get('blur', False),
                    noise=artifact_params.get('noise', False),
                    swap=artifact_params.get('swap', False),
                    random_pick=artifact_params.get('random_pick', False)
                ), apply_probability=artifact_params.get('probability', 0)
            ))

        # Spatial transforms
        spatial_custom_params = transform_params.get('SpatialCustomTransform')
        if spatial_custom_params is not None:
            transforms.append(RandomTransform(
                SpatialCustomTransform(
                    flip=spatial_custom_params.get('flip', False),
                    affine=spatial_custom_params.get('affine', False),
                    elastic=spatial_custom_params.get('elastic', False),
                    anisotropy=spatial_custom_params.get('anisotropy', False),
                    random_pick=spatial_custom_params.get('random_pick', False)
                ), apply_probability=spatial_custom_params.get('probability', 0)
            ))
        
        # Spatial nnunet transform
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        
        spatial_params = transform_params.get('SpatialTransform')
        if spatial_params is not None:
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
                    bg_style_seg_sampling=spatial_params.get('bg_style_seg_sampling', False),
                    mode_seg='nearest'
                )
            )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        
        # Noise transforms
        noise_params = transform_params.get('GaussianNoiseTransform')
        if noise_params is not None:
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=tuple(noise_params.get('noise_variance', (0, 0.1))),
                    p_per_channel=noise_params.get('p_per_channel', 1),
                    synchronize_channels=noise_params.get('synchronize_channels', True)
                ), apply_probability=noise_params.get('probability', 0)
            ))

        # Gaussian blur
        blur_params = transform_params.get('GaussianBlurTransform')
        if blur_params is not None:
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=tuple(blur_params.get('blur_sigma', (0.5, 1.))),
                    synchronize_channels=blur_params.get('synchronize_channels', False),
                    synchronize_axes=blur_params.get('synchronize_axes', False),
                    p_per_channel=blur_params.get('p_per_channel', 0.5),
                    benchmark=blur_params.get('benchmark', True)
                ), apply_probability=blur_params.get('probability', 0)
            ))

        # Brightness transforms
        bright_params = transform_params.get('MultiplicativeBrightnessTransform')
        if bright_params is not None:
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast(tuple(bright_params.get('multiplier_range', (0.75, 1.25)))),
                    synchronize_channels=bright_params.get('synchronize_channels', False),
                    p_per_channel=bright_params.get('p_per_channel', 1)
                ), apply_probability=bright_params.get('probability', 0)
            ))

        # Contrast transforms
        contrast_params = transform_params.get('ContrastTransform')
        if contrast_params is not None:
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast(tuple(contrast_params.get('contrast_range', (0.75, 1.25)))),
                    preserve_range=contrast_params.get('preserve_range', True),
                    synchronize_channels=contrast_params.get('synchronize_channels', False),
                    p_per_channel=contrast_params.get('p_per_channel', 1)
                ), apply_probability=contrast_params.get('probability', 0)
            ))

        # Simulate low resolution
        lowres_params = transform_params.get('SimulateLowResolutionTransform')
        if lowres_params is not None:
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(
                    scale=tuple(lowres_params.get('scale', (0.3, 1))),
                    synchronize_channels=lowres_params.get('synchronize_channels', True),
                    synchronize_axes=lowres_params.get('synchronize_axes', False),
                    ignore_axes=tuple(lowres_params.get('ignore_axes', ())),
                    allowed_channels=lowres_params.get('allowed_channels', None),
                    p_per_channel=lowres_params.get('p_per_channel', 0.5)
                ), apply_probability=lowres_params.get('probability', 0)
            ))

        # Gamma transforms
        gamma_inv_params = transform_params.get('GammaTransform_invert')
        if gamma_inv_params is not None:
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast(tuple(gamma_inv_params.get('gamma', (0.7, 1.5)))),
                    p_invert_image=gamma_inv_params.get('p_invert_image', 1),
                    synchronize_channels=gamma_inv_params.get('synchronize_channels', False),
                    p_per_channel=gamma_inv_params.get('p_per_channel', 1),
                    p_retain_stats=gamma_inv_params.get('p_retain_stats', 1)
                ), apply_probability=gamma_inv_params.get('probability', 0)
        ))

        gamma_params = transform_params.get('GammaTransform')
        if gamma_params is not None:
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast(tuple(gamma_params.get('gamma', (0.7, 1.5)))),
                    p_invert_image=gamma_params.get('p_invert_image', 0),
                    synchronize_channels=gamma_params.get('synchronize_channels', False),
                    p_per_channel=gamma_params.get('p_per_channel', 1),
                    p_retain_stats=gamma_params.get('p_retain_stats', 1)
                ), apply_probability=gamma_params.get('probability', 0)
        ))

        # Mirroring transforms
        if transform_params.get('mirror_axes') is not None and len(transform_params['mirror_axes']) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=transform_params.get('mirror_axes') 
                )
            )

        return transforms

class AugTransformsTest(ComposeTransforms):
    def __init__(self):
        self.transforms = self._build_transforms()
        super().__init__(transforms=self.transforms)

    def _build_transforms(self):
        transforms = []

        # Scharr filter
        transforms.append(RandomTransform(
            ConvTransform(
                kernel_type="Scharr",
                absolute=True,
            ), apply_probability=0.9
        ))

        # Affine transforms
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                affine=True,
            ), apply_probability=0.9
        ))

        return transforms

if __name__ == "__main__":
    # Example usage
    import importlib
    import auglab.configs as configs
    from auglab.utils.image import Image, resample_nib
    import cv2
    from auglab.utils.utils import normalize
    from auglab.transforms.gpu.transforms import AugTransformsGPU
    
    configs_path = importlib.resources.files(configs)
    json_path = configs_path / "transform_params_hybrid_TAGE.json"

    # Load images and masks tensors
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/sub-amu02/anat/sub-amu02_T1w.nii.gz'
    img = Image(img_path).change_orientation('RSP')
    img = resample_nib(img, new_size=[1,1,1], new_size_type='mm', interpolation='linear')
    img_tensor = torch.from_numpy(img.data.copy()).to(torch.float32).unsqueeze(0)

    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/data-multi-subject/derivatives/labels/sub-amu02/anat/sub-amu02_T1w_label-spine_dseg.nii.gz'
    seg = Image(seg_path).change_orientation('RSP')
    seg = resample_nib(seg, new_size=[1,1,1], new_size_type='mm', interpolation='nn')
    seg_tensor_all = torch.from_numpy(seg.data.copy())

    # Add segmentation values to different channels
    seg_tensor = torch.zeros((5, *seg_tensor_all.shape))
    for i, value in enumerate([12, 13, 14, 15, 16]):
        seg_tensor[i] = (seg_tensor_all == value)

    # Example usage
    aug_transforms = AugTransforms(
        json_path=json_path,
        do_dummy_2d_data_aug=False,
        patch_size=(128, 128, 128),
        rotation_for_DA=(-10, 10),
        mirror_axes=None
    )

    augmentor_gpu = AugTransformsGPU(json_path)

    # Apply transforms
    tensor_dict = {}
    gpu = False
    for i in range(24):
        tensor_dict[f'transfo_{str(i+1)}'] = aug_transforms(**{'image': img_tensor.detach().clone(), 'segmentation': seg_tensor.detach().clone()})

        if gpu:
            augmented_img, augmented_seg = augmentor_gpu(tensor_dict[f'transfo_{str(i+1)}']['image'].cuda().unsqueeze(0).clone(), tensor_dict[f'transfo_{str(i+1)}']['segmentation'].cuda().unsqueeze(0).clone())
            tensor_dict[f'transfo_{str(i+1)}']['image'] = augmented_img.cpu().squeeze(0)
            tensor_dict[f'transfo_{str(i+1)}']['segmentation'] = augmented_seg.cpu().squeeze(0)
    
    nb_img = len(tensor_dict.keys())
    nb_col = 6
    for key in ['image', 'segmentation']:
        output = []
        line = []
        aug = [[]]
        for idx, (augment, dic) in enumerate(tensor_dict.items()):
            if len(line) < nb_col:
                img = 255*normalize(np.sum(tensor_dict[augment][key].detach().numpy(), axis=0, keepdims=True)[0,64])
                line.append(img)
                aug[-1].append(augment)
            else:
                output.append(np.concatenate(line, axis=1))
                img = 255*normalize(np.sum(tensor_dict[augment][key].detach().numpy(), axis=0, keepdims=True)[0,64])
                line = [img]
                aug.append([augment])
        output.append(np.concatenate(line, axis=1))

        out_img = np.concatenate(output, axis=0)
        cv2.imwrite(f'img/transforms_default+plus_{key}.png', out_img)    
    print(aug_transforms)