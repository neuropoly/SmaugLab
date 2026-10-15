[![paper](https://img.shields.io/badge/Paper-miccai2026-orange)](https://papers.miccai.org/miccai-2026/paper/1360_paper.pdf)
[![PyPI](https://img.shields.io/pypi/v/smauglab)](https://pypi.org/project/smauglab/)
[![Python Versions](https://img.shields.io/pypi/pyversions/smauglab)](https://pypi.org/project/smauglab/)
[![tests](https://github.com/neuropoly/SmaugLab/actions/workflows/tests.yml/badge.svg)](https://github.com/neuropoly/SmaugLab/actions/workflows/tests.yml)
[![lint](https://github.com/neuropoly/SmaugLab/actions/workflows/lint.yml/badge.svg)](https://github.com/neuropoly/SmaugLab/actions/workflows/lint.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<h1 align="center">
<img src="smauglab_logo.png" width="500">
</h1>

This repository investigates the influence of different data augmentation strategies on MRI training performance.

## Citation

If you use SmaugLab, please make sure to cite the following paper:

```
@InProceedings{moliniermoller_MICCAI2026_SmaugLab,
        author = { Molinier, Nathan AND Möller, Hendrik AND Dagonneau, Thomas AND Curto-Vilalta, Anna AND Graf, Robert AND Atad, Matan AND Rueckert, Daniel AND Kirschke, Jan S. AND Cohen-Adad, Julien},
        title = { { One Sequence to Segment Them All: Efficient Data Augmentation for CT and MRI Cross-Domain 3D Spine Segmentation } },
        booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
        year = {2026},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 16881},
        month = {September},
        page = {pending}
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

You can also specify your data augmentation parameters by providing a JSON file using the environment variable `SMAUGLAB_PARAMS_GPU_JSON`:
> **Note:** By default [smauglab/configs/transform_params_gpu.json](https://github.com/neuropoly/SmaugLab/blob/main/smauglab/configs/transform_params_gpu.json) is used if no file is specified.
```bash
SMAUGLAB_PARAMS_GPU_JSON=/path/to/your/params.json nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDAExtGPU -p nnUNetPlans
```

> ⚠️ **Warning** : To avoid any paths issues, please specify an absolute path to your JSON file.

### Mixed-precision gradient scale

The trainer floors the AMP gradient scale at `1.0`. `torch`'s `GradScaler` halves the scale on every
step with a non-finite gradient and only doubles it back after 2000 *consecutive* clean steps, with no
floor underneath, so a bad-batch rate above 1-in-2000 drives it to exactly `0.0` -- at which point every
gradient becomes `NaN`, clean batches included, and the run cannot recover while still logging epochs.
Below `1.0` the scaler can only shrink gradients, so the floor costs nothing.

`SMAUGLAB_MIN_GRAD_SCALE` overrides it, for the one case the default forecloses: gradients that genuinely
overflow fp16 at scale 1, where halving is the correct response.

```bash
SMAUGLAB_MIN_GRAD_SCALE=0.25 nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainerDAExtGPU -p nnUNetPlans
```

A value at or below zero is rejected, since that is the collapse the floor exists to prevent, and a value
below `1.0` warns: a floor deep in the denormals trains on all-zero gradients without ever reporting a
non-finite step. Resets are written to the training log and counted in the epoch summary either way.

## Run Monai training with SmaugLab augmentations

> To use SmaugLab augmentations in a MONAI training pipeline, refer to the example [training script](https://github.com/neuropoly/SmaugLab/blob/main/scripts/train_monai.py). Key implementation lines required for proper integration are marked with a 🐞 emoji in the comments.

SmaugLab's augmentations reach a MONAI pipeline *as* the GPU pipeline: MONAI loads the data and does the
deterministic preprocessing, then `AugTransformsGPU(...).to(device)` augments each collated batch on device
inside the training loop. Every augmentation in the GPU column of [Available augmentations](#available-augmentations)
is therefore available here. There is no MONAI-native backend, and none is planned: the CPU column is
batchgeneratorsv2 transforms for the nnU-Net dataloader, not MONAI `MapTransform`s.

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

### Random normalization

The `NORM` group in the table below -- z-score, min-max, the CT-style percentile clip, a
robust median/IQR scaling, divide-by-max, divide-by-mean -- is the intensity
standardization step. Its members are alternatives, not additions: a min-max rescale on
top of a z-score does not apply two augmentations, it discards the first. So a config
naming more than one of them does not get a sequence. The builder puts them in a single
stage at the end of the pipeline that picks **one scheme per draw**, which makes the
choice of normalization itself an augmentation.

```json
{
  "GPU": {
    "ZscoreNormalizationGPU": { "p": 1.0 },
    "MinMaxNormalizationGPU": { "out_range": [0.0, 1.0], "p": 1.0 },
    "PercentileNormalizationGPU": { "percentiles": [0.005, 0.995], "p": 1.0 },
    "RobustZscoreNormalizationGPU": { "spread": "iqr", "p": 1.0 },
    "MaxScaleNormalizationGPU": { "p": 0.5 }
  },
  "pipeline": { "random_choose": { "norm_probability": 1.0 } }
}
```

Inside that stage a scheme's `p` is its **weight**, not its frequency: one scheme is drawn
uniformly from the `n` named, and its own `p` is then spent, so `P(scheme) = p / n`. The
config above gives each of the first four a 1-in-5 share and `MaxScaleNormalizationGPU` a
1-in-10 one, and in the remaining 1-in-10 of samples nothing is applied -- which is how
"leave it at the preprocessing normalization" is expressed. There is deliberately no
no-op transform to name; a scheme at `p: 0.0` is a share of the choice that does nothing.

| key | default | what it does |
| --- | --- | --- |
| `stats_from` | `"all"` | Which voxels the statistics are read from: `"all"`, `"seg"` (the labelled anatomy) or `"nonzero"` (everything preprocessing did not zero out, the closest thing here to nnU-Net's `use_mask_for_norm`). It changes what is *measured*, never what is written -- `in_seg` / `out_seg` still decide that. |
| `min_scale` | `0.001` | Leave a sample alone when the divisor is below this fraction of its standard deviation. Without it, `MeanScaleNormalizationGPU` on an already-z-scored patch divides rounding noise by a mean of ~0. |
| `norm_probability` | `1.0` | Probability that the stage runs at all, under `pipeline.random_choose`. |

A draw is one per **batch** at the default `pipeline.same_on_batch: true`, which the
pipeline forces onto every stage it holds -- the `ta` and `ge` buckets behave the same
way. Set `pipeline.same_on_batch: false` and each sample draws its own scheme, which is
also what makes every transform's `p` per-sample rather than batch-wide.

Three things to know before turning this on:

* The patch has already been normalized once, by nnU-Net's preprocessing, and inference
  re-applies only that fixed preprocessing. Randomizing the final standardization is
  therefore a deliberate train/test difference in input scale, on the same logic as every
  other appearance augmentation here: the network is being asked to stop depending on one
  scale. There is no compensating re-normalization at the end -- if that is what you want,
  it is the statistics bracket below, which runs *before* this stage.
* The schemes act on preprocessed intensities, not on HU. A CT window in Hounsfield units
  means nothing after preprocessing unless its bounds are stated, which is why
  `PercentileNormalizationGPU` takes an explicit `clip_range`, `mean` and `std` -- given
  all three it reproduces nnU-Net's `CTNormalization` exactly, and
  `MinMaxNormalizationGPU` at its defaults reproduces `RescaleTo01Normalization`.
  `unit_tests/test_normalization.py` pins both against the installed nnU-Net.

### Statistics restore (`pipeline.retain_stats`)

The per-transform `retain_stats` flag puts a channel's mean and standard deviation back
after one transform. This is the same idea for the whole pipeline, with two more
statistics and a jitter: the profile is snapshotted once after the geometry and restored
once at the end, so the augmentations are free to do what they like in between and the
network still sees an intensity profile that is plausible -- but never exactly the one the
patch arrived with.

```json
{
  "pipeline": {
    "retain_stats": {
      "stats": ["mean", "std", "p1", "p99"],
      "percentiles": [1.0, 99.0],
      "std_scale": [0.9, 1.1],
      "percentile_scale": [0.9, 1.1],
      "stats_from": "all",
      "p": 1.0
    }
  }
}
```

An absent block is off and spends no random numbers. A block that is present but empty is
on with every default, which restores the snapshot exactly.

**Two of the four, drawn per batch.** The snapshot is the per-sample, per-channel mean,
standard deviation, 1st and 99th percentile. An affine map `y = ax + b` has two degrees of
freedom, so all four cannot be restored at once -- each pass therefore draws **two** of
them and solves for the map that puts exactly those two on target, leaving the other two
to fall where the augmentations left them. The six pairs span the two useful extremes:

| drawn pair | what it pins | what it is |
| --- | --- | --- |
| mean, std | the moments | what per-transform `retain_stats` does |
| p1, p99 | the robust dynamic range | immune to the outliers an edge filter or a random convolution manufactures, which are exactly what make the standard deviation a bad target |
| mean, p1 / mean, p99 / std, p1 / std, p99 | one of each | the mixed cases in between |

`stats` is the pool the pair is drawn from, so `["mean", "std"]` reproduces the old
behaviour at pipeline level and the default draws from all six pairs.

**The jitter.** `std_scale` multiplies the standard-deviation target; `percentile_scale`
multiplies each percentile's distance from the mean, independently, so the span can widen,
narrow or go asymmetric. Both are a scalar or a `[lo, hi]` range, and `1.0` restores the
snapshot exactly. The mean target is never jittered -- it is the anchor every pair shares,
and moving it as well would put two unrelated augmentations behind one knob. `p` is the
per-sample probability that the restore applies at all.

**Where it sits.** With the block on, the pipeline is

```
[geometry] [snapshot] [augmentations] [restore] [crop] [normalization]
```

in all three pipeline modes. Two groups move to the tail: the crop, because it occludes a
random box with the volume's own minimum rather than changing intensities, so a restore
measured before it would be undoing the occlusion; and the normalization stage, because
deciding the scale the network sees is its entire job. In `random_order` mode the snapshot
lands after the geometry but *before* the hoisted `RandomLowResTransformGPU`, which is a
general enhancement that merely runs early.

A config should pick one mechanism or the other: the two do not compose, and a transform
that restores its own moments inside the bracket is doing work the restore will redo.
`smauglab/configs/transform_params_gpu_stats-restore.json` is the paper's `Ours` setup with
the bracket on and every per-transform `retain_stats` turned off.

### Paper configs

Two shipped configs are the setups behind the paper's tables. `smauglab hash <config>`
prints the identity below, which is how a run folder named `...-c-<config_hash>` is
traced back to the file that produced it.

| shipped config | config hash | paper setup |
| --- | --- | --- |
| `smauglab/configs/transform_params_one-sequence-to-segment-them-all.json` | `c69b9872` | Ours |
| `smauglab/configs/transform_params_paper-synthseg.json` | `580beb13` | SynthSeg |

The SynthSeg config hashed to `0d5d802f` until `em_merge_prob: 0.0` was added to it;
the two files generate identically, so a run folder named `...-c-0d5d802f` is the same
setup.

Despite the name, "SynthSeg" is not "Ours plus SynthSeg": it is Ours with all six
transfer augmentations (`RandomScharrGPU`, `RandomUnsharpMaskGPU`, `RandomRandConvGPU`,
`RandomRedistributeSegGPU`, `RandomInverseGPU`, `RandomHistogramEqualizationGPU`) at
`p=0` and `RandomSynthSegGPU` at `p=1.0` in their place. The general-enhancement half is
identical between the two.

## Available augmentations

Which augmentations exist, and which of the two backends implements each one. An empty
cell means no implementation on that backend yet. MONAI training draws from the GPU
column -- see "Run Monai training with SmaugLab augmentations" above. Regenerate with
`smauglab matrix --write`.

<!-- BEGIN AUG MATRIX (generated by `smauglab matrix --write`; do not edit) -->
| Augmentation | Group | GPU | CPU |
| --- | --- | --- | --- |
| flip | GEO | `RandomFlipTransformGPU` | — |
| affine | GEO | `RandomAffineGPU` | — |
| crop | GEO | `RandomCropTransformGPU` | — |
| spatial | GEO | — | `SpatialTransform` |
| gaussian_noise | GE | `RandomGaussianNoiseGPU` | `GaussianNoiseTransform` |
| gaussian_blur | GE | `RandomGaussianBlurGPU` | `GaussianBlurTransform` |
| brightness | GE | `RandomBrightnessGPU` | `MultiplicativeBrightnessTransform` |
| contrast | GE | `RandomContrastGPU` | `ContrastTransform` |
| gamma | GE | `RandomGammaGPU` | `GammaTransform` |
| inv_gamma | GE | `RandomInvGammaGPU` | `InvertedGammaTransform` |
| clamp | GE | `RandomClampGPU` | — |
| low_res | GE | `RandomLowResTransformGPU` | `SimulateLowResolutionTransform` |
| acq | GE | `RandomAcqTransformGPU` | — |
| mirror | GEO | — | `MirrorTransform` |
| zscore | NORM | `ZscoreNormalizationGPU` | `ZscoreNormalization` |
| norm_minmax | NORM | `MinMaxNormalizationGPU` | — |
| norm_percentile | NORM | `PercentileNormalizationGPU` | — |
| norm_robust_zscore | NORM | `RobustZscoreNormalizationGPU` | — |
| norm_max_scale | NORM | `MaxScaleNormalizationGPU` | — |
| norm_mean_scale | NORM | `MeanScaleNormalizationGPU` | — |
| scharr | TA | `RandomScharrGPU` | `ScharrConvTransform` |
| laplace | TA | `RandomLaplaceGPU` | `LaplaceConvTransform` |
| unsharp_mask | TA | `RandomUnsharpMaskGPU` | — |
| rand_conv | TA | `RandomRandConvGPU` | — |
| bias_field | TA | `RandomBiasFieldGPU` | — |
| inverse | TA | `RandomInverseGPU` | — |
| histogram_equal | TA | `RandomHistogramEqualizationGPU` | `HistogramEqualTransform` |
| redistribute_seg | TA | `RandomRedistributeSegGPU` | `RedistributeTransform` |
| palette | TA | `RandomPaletteGPU` | — |
| palette_composed | TA | `PaletteSynthesisGPU` | — |
| domain_transfer | TA | `RandomDomainTransferGPU` | — |
| synthseg | TA | `RandomSynthSegGPU` | — |
| artifact | TA | — | `ArtifactTransform` |
| spatial_custom | GEO | — | `SpatialCustomTransform` |
| shape | GE | — | `ShapeTransform` |
| func_log1p | TA | `RandomLog1pGPU` | `Log1pTransform` |
| func_sqrt | TA | `RandomSqrtGPU` | `SqrtTransform` |
| func_sin | TA | `RandomSinGPU` | `SinTransform` |
| func_exp | TA | `RandomExpGPU` | `ExpTransform` |
| func_sigmoid | TA | `RandomSigmoidGPU` | `SigmoidTransform` |
<!-- END AUG MATRIX -->
