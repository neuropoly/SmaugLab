"""nnUNetTrainerSRCSM: nnU-Net trained with the SRCSM augmentation (Thaler et al., IEEE Access 2025).

Thin preset over nnUNetTrainerDAExtGPU: it only selects the SRCSM config
(auglab/configs/transform_params_gpu_srcsm_semrandconv.json) and reads the epoch budget from the
environment. The SRCSM operator itself (srcsm_auglab_port/, auglab/transforms/gpu/srcsm_semrandconv.py)
and the GPU augmentation/train step (nnUNetTrainerDAExtGPU) are used unchanged.

Register once per nnU-Net install:  auglab_add_nnunettrainer -t nnUNetTrainerSRCSM
Train:                              nnUNetv2_train <DATASET> 3d_fullres <FOLD> -tr nnUNetTrainerSRCSM
Predict (with SRCSM's test-time source matching): auglab_srcsm_predict, see SRCSM.md.

Environment variables
- AUGLAB_PARAMS_GPU_JSON   another GPU config instead of the SRCSM one (default: the SRCSM config)
- NNUNET_NUM_EPOCHS        epochs (default 1000, nnU-Net's default)
- NNUNET_ITERS_PER_EPOCH   iterations per epoch (default 250, nnU-Net's default)
"""
import importlib.resources
import os

import torch

import auglab.configs as configs
from auglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

SRCSM_CONFIG = str(importlib.resources.files(configs) / "transform_params_gpu_srcsm_semrandconv.json")


class nnUNetTrainerSRCSM(nnUNetTrainerDAExtGPU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # nnUNetTrainerDAExtGPU (and its static get_training_transforms) read the config path from here
        os.environ.setdefault("AUGLAB_PARAMS_GPU_JSON", SRCSM_CONFIG)
        if os.path.abspath(os.environ["AUGLAB_PARAMS_GPU_JSON"]) != os.path.abspath(SRCSM_CONFIG):
            print(f"WARNING: AUGLAB_PARAMS_GPU_JSON={os.environ['AUGLAB_PARAMS_GPU_JSON']} overrides the SRCSM config")
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", 1000))
        self.num_iterations_per_epoch = int(os.environ.get("NNUNET_ITERS_PER_EPOCH", 250))
