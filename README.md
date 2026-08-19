[![arXiv](https://img.shields.io/badge/Preprint-arXiv:2605.03098-orange)](https://arxiv.org/abs/2605.03098)
[![PyPI](https://img.shields.io/pypi/v/smauglab)](https://pypi.org/project/smauglab/)
[![Python Versions](https://img.shields.io/pypi/pyversions/smauglab)](https://pypi.org/project/smauglab/)
[![tests](https://github.com/neuropoly/SmaugLab/actions/workflows/tests.yml/badge.svg)](https://github.com/neuropoly/SmaugLab/actions/workflows/tests.yml)
[![lint](https://github.com/neuropoly/SmaugLab/actions/workflows/lint.yml/badge.svg)](https://github.com/neuropoly/SmaugLab/actions/workflows/lint.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# SmaugLab
This repository investigates the influence of different data augmentation strategies on MRI training performance.

## Citation

If you use SmaugLab, please make sure to cite the following paper:

```
@article{molinier2026one,
  title={One Sequence to Segment Them All: Efficient Data Augmentation for CT and MRI Cross-Domain 3D Spine Segmentation},
  author={Molinier, Nathan and M{\"o}ller, Hendrik and Dagonneau, Thomas and Curto-Vilalta, Anna and Graf, Robert and Atad, Matan and Rueckert, Daniel and Kirschke, Jan S and Cohen-Adad, Julien},
  journal={arXiv preprint arXiv:2605.03098},
  year={2026}
}
```

## What is available ?

This repository contains:
- A nnUNet [trainer](https://github.com/neuropoly/SmaugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/smauglab/trainers/nnUNetTrainerDAExt.py) with extensive data augmentations
- A basic Monai segmentation [script](https://github.com/neuropoly/SmaugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/scripts/train_monai.py) incorporating data augmentations
- A [script](https://github.com/neuropoly/SmaugLab/blob/bed6c1b5cf8ec3dbe6165daca507bf431cad65e5/scripts/generate_augmentations.py) generating augmentations from input images and segmentations

## How to install ?

1. Open a `bash` terminal in the directory where you want to work.

2. Create and activate a virtual environment using python >=3.10 (highly recommended):
   - venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda env
   ```
   conda create -n myenv python=3.10
   conda activate myenv
   ```

3. Clone this repository:
   - Git clone
   ```bash
   git clone git@github.com:neuropoly/SmaugLab.git
   cd SmaugLab
   ```

4. Install SmaugLab using one of the following commands:
   > **Note:** If you pull a new version from GitHub, make sure to rerun this command with the flag `--upgrade`
   - nnunetv2 only usage (tested with nnunetv2==2.6.2)
   ```bash
   python3 -m pip install -e . nnunetv2==2.6.2
   ```
   - full usage (with Monai and other dependencies)
   ```bash
   python3 -m pip install -e .[all]
   ```

5. Install PyTorch following the instructions on their [website](https://pytorch.org/). Be sure to add the `--upgrade` flag to your installation command to replace any existing PyTorch installation.
   Example:
```bash
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

## Run nnunet training with SmaugLab trainer

To use the SmaugLab trainer with nnUNet, first add the trainer to your nnUNet installation by running:
```bash
smauglab_add_nnunettrainer --trainer nnUNetTrainerDAExt
```

Then, when you run nnUNet training as usual, specifying the SmaugLab trainer, for example:
```bash
nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDAExtGPU -p nnUNetPlans
```

There is one trainer. Which augmentations run is decided by the config, not by the
trainer: a config's `CPU` section runs in the dataloader worker and its `GPU` section
runs on the batch, so the same trainer covers CPU-only, GPU-only and mixed setups.

Point it at a config with `SMAUGLAB_PARAMS_JSON`:
> **Note:** By default [smauglab/configs/transform_params_gpu.json](https://github.com/neuropoly/SmaugLab/blob/main/smauglab/configs/transform_params_gpu.json) is used if no file is specified.
```bash
SMAUGLAB_PARAMS_JSON=/path/to/your/params.json nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDAExtGPU -p nnUNetPlans
```

> ⚠️ **Warning** : To avoid any paths issues, please specify an absolute path to your JSON file.

## Run Monai training with SmaugLab augmentations

> To use SmaugLab augmentations in a MONAI training pipeline, refer to the example [training script](https://github.com/neuropoly/SmaugLab/blob/main/scripts/train_monai.py). Key implementation lines required for proper integration are marked with a 🐞 emoji in the comments.

To run the Monai training script directly, you need to provide a config JSON (`config.json`) file with paths to the images and labels (ground truth) for TRAINING, VALIDATION and TESTING sets like this:
```json
{
   "TYPE": "LABEL",
   "TRAINING": [
      {
         "IMAGE": "/path/to/image1.nii.gz",
         "LABEL": "/path/to/label1.nii.gz"
      },
      {
         "IMAGE": "/path/to/image2.nii.gz",
         "LABEL": "/path/to/label2.nii.gz"
      }
   ],
   "VALIDATION": [
      {
         "IMAGE": "/path/to/image3.nii.gz",
         "LABEL": "/path/to/label3.nii.gz"
      },
      {
         "IMAGE": "/path/to/image4.nii.gz",
         "LABEL": "/path/to/label4.nii.gz"
      }
   ],
   "TESTING": [
      {
         "IMAGE": "/path/to/image5.nii.gz",
         "LABEL": "/path/to/label5.nii.gz"
      },
   ]
}
```

Then run the training script with the following command, specifying the path to your config JSON file and the path to your data augmentation parameters JSON file (if you want to use custom parameters, otherwise the default [transform_params_gpu.json](https://github.com/neuropoly/SmaugLab/blob/main/smauglab/configs/transform_params_gpu.json) is used):
```bash
python scripts/train_monai.py --config <your_path>/config.json --transforms <your_path>/transform_params_gpu.json
```

Additional parameters can be specified—see `python scripts/train_monai.py -h` for details. If anything is unclear, feel free to open an issue.

## Contributing

Development setup, the test suite, and the release process are documented in
[CONTRIBUTING.md](CONTRIBUTING.md). The short version:

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

Pull requests are gated on Ruff (lint + format) and the test suite across
Python 3.10–3.12.

## How to use my data ?

Scripts developped in this repository use JSON files to specify image and segmentation paths: see this [example](https://github.com/neuropoly/SmaugLab/blob/16653a84e031c40e25a72e946c2724494606b21c/smauglab/configs/data/data.json).

## How do I specify my parameters ?

To track parameters used during data augmentation, JSON files are also used: see this [example](https://github.com/neuropoly/SmaugLab/blob/16653a84e031c40e25a72e946c2724494606b21c/smauglab/configs/transform_params.json)


## Citation

If you use SmaugLab, please make sure to cite the following paper:

```
@article{molinier2026one,
  title={One Sequence to Segment Them All: Efficient Data Augmentation for CT and MRI Cross-Domain 3D Spine Segmentation},
  author={Molinier, Nathan and M{\"o}ller, Hendrik and Dagonneau, Thomas and Curto-Vilalta, Anna and Graf, Robert and Atad, Matan and Rueckert, Daniel and Kirschke, Jan S and Cohen-Adad, Julien},
  journal={arXiv preprint arXiv:2605.03098},
  year={2026}
}
```

## Working with augmentations

The `smauglab` command answers what exists and whether a config is valid, reading
the registry directly so it cannot go out of date:

```bash
smauglab list --backend gpu            # what a GPU config can name, in pipeline order
smauglab list --group TA               # just the transfer augmentations
smauglab show RandomScharrGPU          # one augmentation's parameters and defaults
smauglab validate my_config.json       # strict check, every problem reported at once
smauglab template --backend gpu        # a config naming everything, at defaults
smauglab hash my_config.json           # content-addressed config identity
```

Bringing a pre-registry config forward is a one-time job, so it is not a subcommand:
the migrator lives in `migration/` in this repository rather than in the wheel.

Config keys are class names, exactly, and parameters are constructor arguments,
exactly. Anything else is an error rather than a silent no-op:

```
my_config.json: 2 problem(s)
  - GPU.unknown GPU augmentation 'ScharrTransform'.
      Did you mean: RandomScharrGPU?
  - GPU.RandomGaussianNoiseGPU: unknown parameter 'probability'.
      'probability' -> p
```

## Augmentation behaviour changed in the registry release

Several augmentations were silently producing something other than what they claimed.
Fixing them changes what the pipeline emits, so **models trained before this release
saw different augmentations and are not bit-reproducible against it**. Configs are
unaffected -- no key, parameter or default changed. The hash discontinuity is already
recorded in `migration/hash_migration.json`.

What changed, and why each was wrong:

| Augmentation | Was | Now |
|---|---|---|
| `RandomFlipTransformGPU` | Ignored the flip flags its generator sampled, so it flipped every configured axis, identically, on every call and for every batch element. Seeded runs gave byte-identical output. | Reads `params["flip"]`: each batch element flips an independently sampled subset. |
| `RandomGaussianBlurGPU`, `RandomUnsharpMaskGPU` | The Gaussian kernel was sampled at `arange(k)` rather than a centred range, so its peak sat at index 0 and the "blur" also translated the image about a voxel — relative to a mask that was not translated. | Centred kernel; blurring an impulse leaves its centre of mass in place. |
| Every GPU contrast transform with `in_seg` / `out_seg` | Reduced the mask's class axis with `argmax(...) > 0`. For an ordinary single-channel mask that is always false, so `in_seg` applied the transform *nowhere* and `out_seg` applied it *everywhere*; for a one-hot mask it dropped the first foreground class. | `amax(...) > 0`, matching `collapse_onehot_to_index`. Both knobs now do what they say. |
| `RandomAcqTransformGPU` (and any `one_dim` generator) | Drew its "random" axis in `make_samplers`, which kornia calls once and caches — the same axis was degraded for the entire training run. `CropGenerator3D` also drew *separate* axes for the crop and for its position. | Drawn per call and per batch element, with one axis shared by crop and position. |
| `ScharrConvTransform` (CPU, 2-D only) | The x-kernel's middle row was `[-10, 0, -10]`, so it summed to −20 and was not a gradient operator. The 3-D CPU and both GPU kernels were correct. | Middle row is `[-10, 0, 10]`. |
| `RandomLog1pGPU`, `RandomSqrtGPU`, `RandomSinGPU`, `RandomExpGPU`, `RandomSigmoidGPU` | Normalised with a batch-wide `x.min()` / `x.max()`, so a volume's augmentation depended on which other volumes shared its batch. | Per-sample min/max, as every other transform in the file. |
| `RandomHistogramEqualizationGPU` | Wrote through a view of the input, so its non-finite guard skipped over values already in the batch. | Works on a clone; the guard is effective. |
| `RandomChooseXTransformsGPU` | Wrote into the caller's batch in place. Transforms with a kornia parameter generator raised `params must contain 'scale'` inside a bucket. | Clones; samples parameters for generator-based transforms. |
| Transforms drawing blur sigmas / kernel sizes | Used Python's `random`, which `torch.manual_seed` does not reach and which diverges across DDP ranks. | `smauglab.transforms.rng.shared_choice`, driven by torch's RNG. |

The regression tests are in `unit_tests/test_region_mode.py` and
`unit_tests/test_transform_randomness.py`; each one fails against the previous
implementation.

## Available augmentations

Which augmentations exist, and which backends implement each one. An empty cell
means no implementation on that backend yet. Regenerate with `smauglab matrix --write`.

<!-- BEGIN AUG MATRIX (generated by `smauglab matrix --write`; do not edit) -->
| Augmentation | Group | GPU | CPU | MONAI |
| --- | --- | --- | --- | --- |
| flip | GEO | `RandomFlipTransformGPU` | — | — |
| affine | GEO | `RandomAffineGPU` | — | — |
| crop | GEO | `RandomCropTransformGPU` | — | — |
| spatial | GEO | — | `SpatialTransform` | — |
| gaussian_noise | GE | `RandomGaussianNoiseGPU` | `GaussianNoiseTransform` | — |
| gaussian_blur | GE | `RandomGaussianBlurGPU` | `GaussianBlurTransform` | — |
| brightness | GE | `RandomBrightnessGPU` | `MultiplicativeBrightnessTransform` | — |
| contrast | GE | `RandomContrastGPU` | `ContrastTransform` | — |
| gamma | GE | `RandomGammaGPU` | `GammaTransform` | — |
| inv_gamma | GE | `RandomInvGammaGPU` | `InvertedGammaTransform` | — |
| clamp | GE | `RandomClampGPU` | — | — |
| low_res | GE | `RandomLowResTransformGPU` | `SimulateLowResolutionTransform` | — |
| acq | GE | `RandomAcqTransformGPU` | — | — |
| zscore | GE | `ZscoreNormalizationGPU` | `ZscoreNormalization` | — |
| mirror | GEO | — | `MirrorTransform` | — |
| scharr | TA | `RandomScharrGPU` | `ScharrConvTransform` | — |
| laplace | TA | `RandomLaplaceGPU` | `LaplaceConvTransform` | — |
| unsharp_mask | TA | `RandomUnsharpMaskGPU` | — | — |
| rand_conv | TA | `RandomRandConvGPU` | — | — |
| bias_field | TA | `RandomBiasFieldGPU` | — | — |
| inverse | TA | `RandomInverseGPU` | — | — |
| histogram_equal | TA | `RandomHistogramEqualizationGPU` | `HistogramEqualTransform` | — |
| redistribute_seg | TA | `RandomRedistributeSegGPU` | `RedistributeTransform` | — |
| palette | TA | `RandomPaletteGPU` | — | — |
| domain_transfer | TA | `RandomDomainTransferGPU` | — | — |
| synthseg | TA | `RandomSynthSegGPU` | — | — |
| artifact | TA | — | `ArtifactTransform` | — |
| spatial_custom | GEO | — | `SpatialCustomTransform` | — |
| shape | GE | — | `ShapeTransform` | — |
| func_log1p | TA | `RandomLog1pGPU` | `Log1pTransform` | — |
| func_sqrt | TA | `RandomSqrtGPU` | `SqrtTransform` | — |
| func_sin | TA | `RandomSinGPU` | `SinTransform` | — |
| func_exp | TA | `RandomExpGPU` | `ExpTransform` | — |
| func_sigmoid | TA | `RandomSigmoidGPU` | `SigmoidTransform` | — |
<!-- END AUG MATRIX -->
