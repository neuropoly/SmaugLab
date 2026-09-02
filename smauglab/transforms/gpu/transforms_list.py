import json
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from smauglab.transforms.gpu.base import AugmentationSequentialCustom, ImageOnlyTransform, TransformType
from smauglab.transforms.gpu.contrast import (
    RandomBiasFieldGPU,
    RandomBrightnessGPU,
    RandomClampGPU,
    RandomContrastGPU,
    RandomConvTransformGPU,
    RandomFunctionGPU,
    RandomGaussianNoiseGPU,
    RandomHistogramEqualizationGPU,
    RandomInverseGPU,
    ZscoreNormalizationGPU,
    _RandomGammaWithInvertGPU,
)
from smauglab.transforms.gpu.fromSeg import RandomRedistributeSegGPU
from smauglab.transforms.gpu.spatial import RandomAcqTransformGPU, RandomAffineGPU, RandomFlipTransformGPU, RandomLowResTransformGPU


class AugTransformsGPURandomOrder(AugmentationSequentialCustom):
    """
    Module to perform data augmentation on GPU.
    """

    def __init__(self, json_path: str):
        # Load transform parameters from JSON
        config_path = os.path.join(json_path)
        with open(config_path) as f:
            config = json.load(f)

        if "GPU" in config.keys():
            self.transform_params = config["GPU"]
        else:
            self.transform_params = config

        transforms = self._build_transforms()
        super().__init__(
            *transforms, data_keys=["input", "mask"], same_on_batch=True
        )  # Same_on_batch to ensure mask are aligned with images correctly (custom) see AugmentationSequentialOpsCustom in base.py

    def _build_transforms(self) -> list[TransformType]:
        # Annotated rather than inferred: an empty list takes its element type from
        # the first append, which would pin it to whichever transform the config
        # happens to enable first and reject every sibling appended after it.
        transforms: list[TransformType] = []

        # Flipping transforms
        flip_params = self.transform_params.get("FlipTransform")
        if flip_params is not None:
            transforms.append(
                RandomFlipTransformGPU(
                    flip_axis=flip_params.get("flip_axis", [0]),
                    p=flip_params.get("probability", 0),
                    same_on_batch=flip_params.get("same_on_batch", False),
                    keepdim=flip_params.get("keepdim", True),
                )
            )

        # Spatial transforms
        affine_params = self.transform_params.get("AffineTransform")
        if affine_params is not None:
            transforms.append(
                RandomAffineGPU(
                    degrees=affine_params.get("degrees", 10),
                    translate=affine_params.get("translate", [0.1, 0.1, 0.1]),
                    scale=affine_params.get("scale", [0.9, 1.1]),
                    shears=affine_params.get("shear", [-10, 10, -10, 10, -10, 10]),
                    resample=affine_params.get("resample", "bilinear"),
                    p=affine_params.get("probability", 0),
                )
            )

        ## Transfer augmentations (TA)
        ta_transforms: list[ImageOnlyTransform] = []
        # Inverse transform (max - pixel_value)
        inverse_params = self.transform_params.get("InverseTransform")
        if inverse_params is not None:
            ta_transforms.append(
                RandomInverseGPU(
                    p=inverse_params.get("probability", 0),
                    in_seg=inverse_params.get("in_seg", 0.0),
                    out_seg=inverse_params.get("out_seg", 0.0),
                    mix_in_out=inverse_params.get("mix_in_out", False),
                    mix_prob=inverse_params.get("mix_prob", 0.0),
                    retain_stats=inverse_params.get("retain_stats", False),
                )
            )

        # Histogram manipulations
        histo_params = self.transform_params.get("HistogramEqualizationTransform")
        if histo_params is not None:
            ta_transforms.append(
                RandomHistogramEqualizationGPU(
                    p=histo_params.get("probability", 0),
                    in_seg=histo_params.get("in_seg", 0.0),
                    out_seg=histo_params.get("out_seg", 0.0),
                    mix_in_out=histo_params.get("mix_in_out", False),
                    mix_prob=histo_params.get("mix_prob", 0.0),
                    retain_stats=histo_params.get("retain_stats", False),
                )
            )

        # Redistribute segmentation values transform
        redistribute_params = self.transform_params.get("RedistributeSegTransform")
        if redistribute_params is not None:
            ta_transforms.append(
                RandomRedistributeSegGPU(
                    in_seg=redistribute_params.get("in_seg", 0.2),
                    retain_stats=redistribute_params.get("retain_stats", False),
                    p=redistribute_params.get("probability", 0),
                )
            )

        # Scharr filter
        scharr_params = self.transform_params.get("ScharrTransform")
        if scharr_params is not None:
            ta_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=scharr_params.get("kernel_type", "Scharr"),
                    p=scharr_params.get("probability", 0),
                    in_seg=scharr_params.get("in_seg", 0.0),
                    out_seg=scharr_params.get("out_seg", 0.0),
                    mix_in_out=scharr_params.get("mix_in_out", False),
                    retain_stats=scharr_params.get("retain_stats", True),
                    absolute=scharr_params.get("absolute", True),
                    mix_prob=scharr_params.get("mix_prob", 0.0),
                )
            )

        # Unsharp masking
        unsharp_params = self.transform_params.get("UnsharpMaskTransform")
        if unsharp_params is not None:
            ta_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=unsharp_params.get("kernel_type", "UnsharpMask"),
                    p=unsharp_params.get("probability", 0),
                    in_seg=unsharp_params.get("in_seg", 0.0),
                    out_seg=unsharp_params.get("out_seg", 0.0),
                    mix_in_out=unsharp_params.get("mix_in_out", False),
                    sigma=unsharp_params.get("sigma", 1.0),
                    unsharp_amount=unsharp_params.get("unsharp_amount", 1.5),
                    mix_prob=unsharp_params.get("mix_prob", 0.0),
                )
            )

        # RandomConv transform
        randconv_params = self.transform_params.get("RandomConvTransform")
        if randconv_params is not None:
            ta_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=randconv_params.get("kernel_type", "RandConv"),
                    p=randconv_params.get("probability", 0),
                    in_seg=randconv_params.get("in_seg", 0.0),
                    out_seg=randconv_params.get("out_seg", 0.0),
                    mix_in_out=randconv_params.get("mix_in_out", False),
                    retain_stats=randconv_params.get("retain_stats", False),
                    kernel_sizes=randconv_params.get("kernel_sizes", [1, 3, 5, 7]),
                    mix_prob=randconv_params.get("mix_prob", 0.0),
                )
            )

        # Apply functions
        func_list = [
            lambda x: torch.log(1 + x),
            torch.sqrt,
            torch.sin,
            torch.exp,
            lambda x: 1 / (1 + torch.exp(-x)),
        ]
        function_params = self.transform_params.get("FunctionTransform")
        if function_params is not None:
            ta_transforms.extend(
                RandomFunctionGPU(
                    func=func,
                    p=function_params.get("probability", 0),
                    in_seg=function_params.get("in_seg", 0.0),
                    out_seg=function_params.get("out_seg", 0.0),
                    mix_in_out=function_params.get("mix_in_out", False),
                    retain_stats=function_params.get("retain_stats", False),
                )
                for func in func_list
            )

        # Bias field artifact
        bias_field_params = self.transform_params.get("BiasFieldTransform")
        if bias_field_params is not None:
            ta_transforms.append(
                RandomBiasFieldGPU(
                    p=bias_field_params.get("probability", 0),
                    in_seg=bias_field_params.get("in_seg", 0.0),
                    out_seg=bias_field_params.get("out_seg", 0.0),
                    mix_in_out=bias_field_params.get("mix_in_out", False),
                    retain_stats=bias_field_params.get("retain_stats", False),
                    coefficients=bias_field_params.get("coefficients", 0.5),
                )
            )

        ## General enhancement (GE)
        ge_transforms: list[ImageOnlyTransform] = []
        # Clamping transform
        clamp_params = self.transform_params.get("ClampTransform")
        if clamp_params is not None:
            ge_transforms.append(
                RandomClampGPU(
                    max_clamp_amount=clamp_params.get("max_clamp_amount", 0.0),
                    in_seg=clamp_params.get("in_seg", 0.0),
                    out_seg=clamp_params.get("out_seg", 0.0),
                    mix_in_out=clamp_params.get("mix_in_out", False),
                    retain_stats=clamp_params.get("retain_stats", False),
                    p=clamp_params.get("probability", 0),
                )
            )

        # Noise transforms
        noise_params = self.transform_params.get("GaussianNoiseTransform")
        if noise_params is not None:
            ge_transforms.append(
                RandomGaussianNoiseGPU(
                    mean=noise_params.get("mean", 0.0),
                    std=noise_params.get("std", 1.0),
                    in_seg=noise_params.get("in_seg", 0.0),
                    out_seg=noise_params.get("out_seg", 0.0),
                    mix_in_out=noise_params.get("mix_in_out", False),
                    p=noise_params.get("probability", 0),
                )
            )

        # Gaussian blur
        gaussianblur_params = self.transform_params.get("GaussianBlurTransform")
        if gaussianblur_params is not None:
            ge_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=gaussianblur_params.get("kernel_type", "GaussianBlur"),
                    in_seg=gaussianblur_params.get("in_seg", 0.0),
                    out_seg=gaussianblur_params.get("out_seg", 0.0),
                    mix_in_out=gaussianblur_params.get("mix_in_out", False),
                    p=gaussianblur_params.get("probability", 0),
                    sigma=gaussianblur_params.get("sigma", 1.0),
                )
            )

        # Brightness transforms
        brightness_params = self.transform_params.get("BrightnessTransform")
        if brightness_params is not None:
            ge_transforms.append(
                RandomBrightnessGPU(
                    brightness_range=brightness_params.get("brightness_range", [0.5, 1.5]),
                    in_seg=brightness_params.get("in_seg", 0.0),
                    out_seg=brightness_params.get("out_seg", 0.0),
                    mix_in_out=brightness_params.get("mix_in_out", False),
                    p=brightness_params.get("probability", 0),
                )
            )

        # Gamma transforms
        gamma_params = self.transform_params.get("GammaTransform")
        if gamma_params is not None:
            ge_transforms.append(
                _RandomGammaWithInvertGPU(
                    gamma_range=gamma_params.get("gamma_range", [0.7, 1.5]),
                    p=gamma_params.get("probability", 0),
                    invert_image=False,
                    in_seg=gamma_params.get("in_seg", 0.0),
                    out_seg=gamma_params.get("out_seg", 0.0),
                    mix_in_out=gamma_params.get("mix_in_out", False),
                    retain_stats=gamma_params.get("retain_stats", False),
                )
            )

        inv_gamma_params = self.transform_params.get("InvGammaTransform")
        if inv_gamma_params is not None:
            ge_transforms.append(
                _RandomGammaWithInvertGPU(
                    gamma_range=inv_gamma_params.get("gamma_range", [0.7, 1.5]),
                    p=inv_gamma_params.get("probability", 0),
                    in_seg=inv_gamma_params.get("in_seg", 0.0),
                    out_seg=inv_gamma_params.get("out_seg", 0.0),
                    mix_in_out=inv_gamma_params.get("mix_in_out", False),
                    invert_image=True,
                    retain_stats=inv_gamma_params.get("retain_stats", False),
                )
            )

        # nnUNetV2 Contrast transforms
        contrast_params = self.transform_params.get("ContrastTransform")
        if contrast_params is not None:
            ge_transforms.append(
                RandomContrastGPU(
                    contrast_range=contrast_params.get("contrast_range", [0.75, 1.25]),
                    p=contrast_params.get("probability", 0),
                    in_seg=contrast_params.get("in_seg", 0.0),
                    out_seg=contrast_params.get("out_seg", 0.0),
                    mix_in_out=contrast_params.get("mix_in_out", False),
                    retain_stats=contrast_params.get("retain_stats", False),
                )
            )

        # Shape transforms (Cropping and Simulating low resolution)
        lowres_params = self.transform_params.get("SimulateLowResTransform")
        if lowres_params is not None:
            transforms.append(
                RandomLowResTransformGPU(
                    p=lowres_params.get("probability", 0),
                    scale=lowres_params.get("scale", [0.3, 1.0]),
                    same_on_batch=lowres_params.get("same_on_batch", False),
                )
            )

        acq_params = self.transform_params.get("AcqTransform")
        if acq_params is not None:
            ge_transforms.append(
                RandomAcqTransformGPU(
                    p=acq_params.get("probability", 0),
                    scale=acq_params.get("scale", [0.3, 1.0]),
                    same_on_batch=acq_params.get("same_on_batch", False),
                )
            )

        ## Random Z-score normalization
        zscore_params = self.transform_params.get("ZscoreNormalizationTransform")
        if zscore_params is not None:
            ge_transforms.append(ZscoreNormalizationGPU(p=zscore_params.get("probability", 0)))

        choose_x_params = self.transform_params.get("RandomChooseXTransforms")
        transforms.append(
            RandomChooseXTransformsGPU(
                transforms_list=ta_transforms,
                num_transforms=len(ta_transforms),
                p=choose_x_params.get("ta_probability", 1.0),
                random_order=choose_x_params.get("ta_random_order", True),
            )
        )
        transforms.append(
            RandomChooseXTransformsGPU(
                transforms_list=ge_transforms,
                num_transforms=len(ge_transforms),
                p=choose_x_params.get("ge_probability", 1.0),
                random_order=choose_x_params.get("ge_random_order", True),
            )
        )

        return transforms


class AugTransformsGPURandomOrderTA(AugmentationSequentialCustom):
    """
    Module to perform data augmentation on GPU.
    """

    def __init__(self, json_path: str):
        # Load transform parameters from JSON
        config_path = os.path.join(json_path)
        with open(config_path) as f:
            config = json.load(f)

        if "GPU" in config.keys():
            self.transform_params = config["GPU"]
        else:
            self.transform_params = config

        transforms = self._build_transforms()
        super().__init__(
            *transforms, data_keys=["input", "mask"], same_on_batch=True
        )  # Same_on_batch to ensure mask are aligned with images correctly (custom) see AugmentationSequentialOpsCustom in base.py

    def _build_transforms(self) -> list[TransformType]:
        # Annotated rather than inferred: an empty list takes its element type from
        # the first append, which would pin it to whichever transform the config
        # happens to enable first and reject every sibling appended after it.
        transforms: list[TransformType] = []

        # Flipping transforms
        flip_params = self.transform_params.get("FlipTransform")
        if flip_params is not None:
            transforms.append(
                RandomFlipTransformGPU(
                    flip_axis=flip_params.get("flip_axis", [0]),
                    p=flip_params.get("probability", 0),
                    same_on_batch=flip_params.get("same_on_batch", False),
                    keepdim=flip_params.get("keepdim", True),
                )
            )

        # Spatial transforms
        affine_params = self.transform_params.get("AffineTransform")
        if affine_params is not None:
            transforms.append(
                RandomAffineGPU(
                    degrees=affine_params.get("degrees", 10),
                    translate=affine_params.get("translate", [0.1, 0.1, 0.1]),
                    scale=affine_params.get("scale", [0.9, 1.1]),
                    shears=affine_params.get("shear", [-10, 10, -10, 10, -10, 10]),
                    resample=affine_params.get("resample", "bilinear"),
                    p=affine_params.get("probability", 0),
                )
            )

        ## Transfer augmentations (TA)
        ta_transforms: list[ImageOnlyTransform] = []
        # Inverse transform (max - pixel_value)
        inverse_params = self.transform_params.get("InverseTransform")
        if inverse_params is not None:
            ta_transforms.append(
                RandomInverseGPU(
                    p=inverse_params.get("probability", 0),
                    in_seg=inverse_params.get("in_seg", 0.0),
                    out_seg=inverse_params.get("out_seg", 0.0),
                    mix_in_out=inverse_params.get("mix_in_out", False),
                    mix_prob=inverse_params.get("mix_prob", 0.0),
                    retain_stats=inverse_params.get("retain_stats", False),
                )
            )

        # Histogram manipulations
        histo_params = self.transform_params.get("HistogramEqualizationTransform")
        if histo_params is not None:
            ta_transforms.append(
                RandomHistogramEqualizationGPU(
                    p=histo_params.get("probability", 0),
                    in_seg=histo_params.get("in_seg", 0.0),
                    out_seg=histo_params.get("out_seg", 0.0),
                    mix_in_out=histo_params.get("mix_in_out", False),
                    mix_prob=histo_params.get("mix_prob", 0.0),
                    retain_stats=histo_params.get("retain_stats", False),
                )
            )

        # Redistribute segmentation values transform
        redistribute_params = self.transform_params.get("RedistributeSegTransform")
        if redistribute_params is not None:
            ta_transforms.append(
                RandomRedistributeSegGPU(
                    in_seg=redistribute_params.get("in_seg", 0.2),
                    retain_stats=redistribute_params.get("retain_stats", False),
                    p=redistribute_params.get("probability", 0),
                )
            )

        # Scharr filter
        scharr_params = self.transform_params.get("ScharrTransform")
        if scharr_params is not None:
            ta_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=scharr_params.get("kernel_type", "Scharr"),
                    p=scharr_params.get("probability", 0),
                    in_seg=scharr_params.get("in_seg", 0.0),
                    out_seg=scharr_params.get("out_seg", 0.0),
                    mix_in_out=scharr_params.get("mix_in_out", False),
                    retain_stats=scharr_params.get("retain_stats", True),
                    absolute=scharr_params.get("absolute", True),
                    mix_prob=scharr_params.get("mix_prob", 0.0),
                )
            )

        # Unsharp masking
        unsharp_params = self.transform_params.get("UnsharpMaskTransform")
        if unsharp_params is not None:
            ta_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=unsharp_params.get("kernel_type", "UnsharpMask"),
                    p=unsharp_params.get("probability", 0),
                    in_seg=unsharp_params.get("in_seg", 0.0),
                    out_seg=unsharp_params.get("out_seg", 0.0),
                    mix_in_out=unsharp_params.get("mix_in_out", False),
                    sigma=unsharp_params.get("sigma", 1.0),
                    unsharp_amount=unsharp_params.get("unsharp_amount", 1.5),
                    mix_prob=unsharp_params.get("mix_prob", 0.0),
                )
            )

        # RandomConv transform
        randconv_params = self.transform_params.get("RandomConvTransform")
        if randconv_params is not None:
            ta_transforms.append(
                RandomConvTransformGPU(
                    kernel_type=randconv_params.get("kernel_type", "RandConv"),
                    p=randconv_params.get("probability", 0),
                    in_seg=randconv_params.get("in_seg", 0.0),
                    out_seg=randconv_params.get("out_seg", 0.0),
                    mix_in_out=randconv_params.get("mix_in_out", False),
                    retain_stats=randconv_params.get("retain_stats", False),
                    kernel_sizes=randconv_params.get("kernel_sizes", [1, 3, 5, 7]),
                    mix_prob=randconv_params.get("mix_prob", 0.0),
                )
            )

        # Apply functions
        func_list = [
            lambda x: torch.log(1 + x),
            torch.sqrt,
            torch.sin,
            torch.exp,
            lambda x: 1 / (1 + torch.exp(-x)),
        ]
        function_params = self.transform_params.get("FunctionTransform")
        if function_params is not None:
            ta_transforms.extend(
                RandomFunctionGPU(
                    func=func,
                    p=function_params.get("probability", 0),
                    in_seg=function_params.get("in_seg", 0.0),
                    out_seg=function_params.get("out_seg", 0.0),
                    mix_in_out=function_params.get("mix_in_out", False),
                    retain_stats=function_params.get("retain_stats", False),
                )
                for func in func_list
            )

        # Bias field artifact
        bias_field_params = self.transform_params.get("BiasFieldTransform")
        if bias_field_params is not None:
            ta_transforms.append(
                RandomBiasFieldGPU(
                    p=bias_field_params.get("probability", 0),
                    in_seg=bias_field_params.get("in_seg", 0.0),
                    out_seg=bias_field_params.get("out_seg", 0.0),
                    mix_in_out=bias_field_params.get("mix_in_out", False),
                    retain_stats=bias_field_params.get("retain_stats", False),
                    coefficients=bias_field_params.get("coefficients", 0.5),
                )
            )

        choose_x_params = self.transform_params.get("RandomChooseXTransforms")
        transforms.append(
            RandomChooseXTransformsGPU(
                transforms_list=ta_transforms,
                num_transforms=len(ta_transforms),
                p=choose_x_params.get("ta_probability", 1.0),
                random_order=False,
            )
        )

        ## General enhancement (GE)
        # Clamping transform
        clamp_params = self.transform_params.get("ClampTransform")
        if clamp_params is not None:
            transforms.append(
                RandomClampGPU(
                    max_clamp_amount=clamp_params.get("max_clamp_amount", 0.0),
                    in_seg=clamp_params.get("in_seg", 0.0),
                    out_seg=clamp_params.get("out_seg", 0.0),
                    mix_in_out=clamp_params.get("mix_in_out", False),
                    retain_stats=clamp_params.get("retain_stats", False),
                    p=clamp_params.get("probability", 0),
                )
            )

        # Noise transforms
        noise_params = self.transform_params.get("GaussianNoiseTransform")
        if noise_params is not None:
            transforms.append(
                RandomGaussianNoiseGPU(
                    mean=noise_params.get("mean", 0.0),
                    std=noise_params.get("std", 1.0),
                    in_seg=noise_params.get("in_seg", 0.0),
                    out_seg=noise_params.get("out_seg", 0.0),
                    mix_in_out=noise_params.get("mix_in_out", False),
                    p=noise_params.get("probability", 0),
                )
            )

        # Gaussian blur
        gaussianblur_params = self.transform_params.get("GaussianBlurTransform")
        if gaussianblur_params is not None:
            transforms.append(
                RandomConvTransformGPU(
                    kernel_type=gaussianblur_params.get("kernel_type", "GaussianBlur"),
                    in_seg=gaussianblur_params.get("in_seg", 0.0),
                    out_seg=gaussianblur_params.get("out_seg", 0.0),
                    mix_in_out=gaussianblur_params.get("mix_in_out", False),
                    p=gaussianblur_params.get("probability", 0),
                    sigma=gaussianblur_params.get("sigma", 1.0),
                )
            )

        # Brightness transforms
        brightness_params = self.transform_params.get("BrightnessTransform")
        if brightness_params is not None:
            transforms.append(
                RandomBrightnessGPU(
                    brightness_range=brightness_params.get("brightness_range", [0.5, 1.5]),
                    in_seg=brightness_params.get("in_seg", 0.0),
                    out_seg=brightness_params.get("out_seg", 0.0),
                    mix_in_out=brightness_params.get("mix_in_out", False),
                    p=brightness_params.get("probability", 0),
                )
            )

        # Gamma transforms
        gamma_params = self.transform_params.get("GammaTransform")
        if gamma_params is not None:
            transforms.append(
                _RandomGammaWithInvertGPU(
                    gamma_range=gamma_params.get("gamma_range", [0.7, 1.5]),
                    p=gamma_params.get("probability", 0),
                    invert_image=False,
                    in_seg=gamma_params.get("in_seg", 0.0),
                    out_seg=gamma_params.get("out_seg", 0.0),
                    mix_in_out=gamma_params.get("mix_in_out", False),
                    retain_stats=gamma_params.get("retain_stats", False),
                )
            )

        inv_gamma_params = self.transform_params.get("InvGammaTransform")
        if inv_gamma_params is not None:
            transforms.append(
                _RandomGammaWithInvertGPU(
                    gamma_range=inv_gamma_params.get("gamma_range", [0.7, 1.5]),
                    p=inv_gamma_params.get("probability", 0),
                    in_seg=inv_gamma_params.get("in_seg", 0.0),
                    out_seg=inv_gamma_params.get("out_seg", 0.0),
                    mix_in_out=inv_gamma_params.get("mix_in_out", False),
                    invert_image=True,
                    retain_stats=inv_gamma_params.get("retain_stats", False),
                )
            )

        # nnUNetV2 Contrast transforms
        contrast_params = self.transform_params.get("ContrastTransform")
        if contrast_params is not None:
            transforms.append(
                RandomContrastGPU(
                    contrast_range=contrast_params.get("contrast_range", [0.75, 1.25]),
                    p=contrast_params.get("probability", 0),
                    in_seg=contrast_params.get("in_seg", 0.0),
                    out_seg=contrast_params.get("out_seg", 0.0),
                    mix_in_out=contrast_params.get("mix_in_out", False),
                    retain_stats=contrast_params.get("retain_stats", False),
                )
            )

        # Shape transforms (Cropping and Simulating low resolution)
        lowres_params = self.transform_params.get("SimulateLowResTransform")
        if lowres_params is not None:
            transforms.append(
                RandomLowResTransformGPU(
                    p=lowres_params.get("probability", 0),
                    scale=lowres_params.get("scale", [0.3, 1.0]),
                    same_on_batch=lowres_params.get("same_on_batch", False),
                )
            )

        acq_params = self.transform_params.get("AcqTransform")
        if acq_params is not None:
            transforms.append(
                RandomAcqTransformGPU(
                    p=acq_params.get("probability", 0),
                    scale=acq_params.get("scale", [0.3, 1.0]),
                    same_on_batch=acq_params.get("same_on_batch", False),
                )
            )

        ## Random Z-score normalization
        zscore_params = self.transform_params.get("ZscoreNormalizationTransform")
        if zscore_params is not None:
            transforms.append(ZscoreNormalizationGPU(p=zscore_params.get("probability", 0)))

        return transforms


class RandomChooseXTransformsGPU(ImageOnlyTransform):
    """Randomly choose X transforms to apply from a given list of ImageOnlyTransform transforms (GPU version).

    Args:
        transforms_list: List of initialized ImageOnlyTransform to choose from.
        num_transforms: Number of transforms to randomly select and apply.
        same_on_batch: apply the same transformation across the batch.
        p: probability for applying the X transforms to a batch. This param controls the augmentation
          probabilities batch-wise.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    """

    def __init__(
        self,
        transforms_list: list[ImageOnlyTransform],
        num_transforms: int = 1,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = True,
        random_order: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        if not isinstance(num_transforms, int) or num_transforms < 0:
            raise ValueError(f"num_transforms must be a non-negative int. Got {num_transforms!r}.")
        self.transforms_list = nn.ModuleList(transforms_list)
        self.num_transforms = num_transforms
        self.random_order = random_order

    def _apply_mix(self, x: Tensor, seg: Tensor | None) -> Tensor:
        if self.num_transforms == 0 or len(self.transforms_list) == 0:
            return x

        k = min(self.num_transforms, len(self.transforms_list))
        # sample without replacement
        if self.random_order:
            idx = torch.randperm(len(self.transforms_list), device=x.device)[:k]
        else:
            idx = torch.arange(len(self.transforms_list), device=x.device)[:k]

        child_params: dict[str, Tensor] = {}
        if seg is not None:
            child_params["seg"] = seg

        for j in idx.tolist():
            t = self.transforms_list[j]
            if torch.rand(1, device=x.device, dtype=x.dtype) > t.p:
                continue
            if not hasattr(t, "apply_transform"):
                raise TypeError(f"All transforms must implement apply_transform like ImageOnlyTransform. Got {type(t)}")
            # Most contrast transforms perform their random sampling inside
            # apply_transform, so an empty params dict is all they need. The ones with a
            # kornia `_param_generator` (the spatial transforms) read their draw out of
            # `params` instead, and calling apply_transform directly skips the
            # forward_parameters step that fills it -- they used to raise
            # "params must contain 'scale'" from inside a bucket. Sampling here keeps
            # the bucket usable for both kinds.
            t_params = child_params
            if getattr(t, "_param_generator", None) is not None:
                t_params = {**child_params, **t.forward_parameters(x.shape)}
            t_flags = getattr(t, "flags", {})
            x = t.apply_transform(x, t_params, t_flags, transform=None)
        return x

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        seg = params.get("seg")

        if self.same_on_batch:
            return self._apply_mix(input, seg)

        batch_size = input.shape[0]
        # A clone, not `out = input`: the loop writes back through `out[i:i+1]`, so
        # without it the caller's batch is modified in place. Every sibling transform
        # in gpu/spatial.py clones.
        out = input.clone()
        for i in range(batch_size):
            xi = out[i : i + 1]
            seg_i = None
            seg_i = seg[i : i + 1] if seg is not None and isinstance(seg, torch.Tensor) and seg.shape[0] == batch_size else seg
            xi = self._apply_mix(xi, seg_i)
            out[i : i + 1] = xi
        return out


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a tensor to the range [0, 1].
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val + 1e-8)
    return normalized_arr


def pad_numpy_array(arr, shape):
    """
    Pad a numpy array to the desired shape with zeros.
    """
    # Calculate padding needed for each dimension
    pad_width = [
        (max(0, shape[i] - arr.shape[i]) // 2, max(0, shape[i] - arr.shape[i]) - max(0, shape[i] - arr.shape[i]) // 2)
        for i in range(len(shape))
    ]
    padded_arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return padded_arr
