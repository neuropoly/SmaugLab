"""nnUNetTrainerPA: nnU-Net trained with Palette Augmentation (PA), with or without the Auglab pipeline.

Thin preset over nnUNetTrainerDAExtGPU: it only selects the GPU config and reads the epoch budget
from the environment. The PA transform (auglab/transforms/gpu/fromSeg.py, RandomV26_6_2ContrastGPU)
and the GPU augmentation/train step (nnUNetTrainerDAExtGPU) are used unchanged.

Register once per nnU-Net install:  auglab_add_nnunettrainer -t nnUNetTrainerPA
Train:                              nnUNetv2_train <DATASET> 3d_fullres <FOLD> -tr nnUNetTrainerPA
See PALETTE.md.

Environment variables
- AUGLAB_PARAMS_GPU_JSON   another GPU config (default: PA+Auglab). PA alone:
                           auglab/configs/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json
- NNUNET_NUM_EPOCHS        epochs (default 1000, nnU-Net's default)
- NNUNET_ITERS_PER_EPOCH   iterations per epoch (default 250, nnU-Net's default)
"""
import importlib.resources
import os

import torch

import auglab.configs as configs
from auglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

PA_AUGLAB_CONFIG = str(importlib.resources.files(configs)
                       / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json")


class nnUNetTrainerPA(nnUNetTrainerDAExtGPU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # nnUNetTrainerDAExtGPU (and its static get_training_transforms) read the config path from here.
        # An empty value counts as unset: job scripts often export AUGLAB_PARAMS_GPU_JSON="".
        if not os.environ.get("AUGLAB_PARAMS_GPU_JSON"):
            os.environ["AUGLAB_PARAMS_GPU_JSON"] = PA_AUGLAB_CONFIG
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", 1000))
        self.num_iterations_per_epoch = int(os.environ.get("NNUNET_ITERS_PER_EPOCH", 250))
