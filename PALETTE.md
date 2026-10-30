# Palette Augmentation (PA) with nnU-Net

**Palette Augmentation** re-colours each training image: it splits the image into regions (1-D K-means
intensity clusters, sub-divided into Voronoi cells, plus the anatomical labels) and gives every region a
new random intensity while keeping the real image's texture inside it. A network trained this way sees
many synthetic contrasts of the same anatomy, which helps it segment contrasts it was never trained on.

Two configs, both in `auglab/configs/`:

| Name | Config | What it does |
|---|---|---|
| **PA+Auglab** (default) | `transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json` | PA on half of the training batches, plus the Auglab intensity and resolution augmentations, flips and nnU-Net's spatial augmentation |
| **PA** | `transform_params_gpu_v26_6_2_synth_spatialDA_train050.json` | PA on half of the training batches, plus flips and nnU-Net's spatial augmentation only |

The transform is `RandomV26_6_2ContrastGPU` (`auglab/transforms/gpu/fromSeg.py`); it runs on the GPU inside
nnU-Net's train step (`nnUNetTrainerDAExtGPU`). nnU-Net's own mirroring is off, as in every AugLab GPU
trainer (flips come from the config).

## Install

```bash
python3 -m venv venv && source venv/bin/activate
# install PyTorch for your CUDA first (https://pytorch.org)
git clone -b palette https://github.com/neuropoly/AugLab.git
cd AugLab
python3 -m pip install -e .[nnunetv2]
auglab_add_nnunettrainer -t nnUNetTrainerPA    # copies the trainer into nnU-Net (once per environment)
```
AugLab needs kornia < 0.8 (0.8 removed `kornia.core.Tensor`; the pin is in `pyproject.toml`).
nnU-Net compiles the network by default, which needs a working `triton` (PyPI torch wheels include it);
without one, set `nnUNet_compile=f`.

## Train

Your dataset must be an ordinary nnU-Net dataset (`nnUNetv2_plan_and_preprocess` done).

```bash
export NNUNET_NUM_EPOCHS=1000        # default 1000 (nnU-Net's default)
export NNUNET_ITERS_PER_EPOCH=250    # default 250 (nnU-Net's default)
nnUNetv2_train <DATASET> 3d_fullres <FOLD> -tr nnUNetTrainerPA                       # PA+Auglab

AUGLAB_PARAMS_GPU_JSON=$(python -c "import auglab.configs as c, importlib.resources as r; \
print(r.files(c) / 'transform_params_gpu_v26_6_2_synth_spatialDA_train050.json')") \
nnUNetv2_train <DATASET> 3d_fullres <FOLD> -tr nnUNetTrainerPA                       # PA
```
`AUGLAB_PARAMS_GPU_JSON` takes any GPU config (absolute path). The config actually used is copied to
`transform_params_gpu_used_for_training.json` in each fold folder.

## Predict

```bash
nnUNetv2_predict -i <imagesTs> -o <predictions> -d <DATASET> -c 3d_fullres -f 0 1 2 \
    -tr nnUNetTrainerPA -chk checkpoint_best.pth --disable_tta
```
No test-time processing is needed. The trainer disables nnU-Net's mirroring, so `--disable_tta` only makes
that explicit.

## What the paper's runs used

- 150 iterations per epoch (not nnU-Net's 250) and a per-dataset epoch count: 2000 on most tasks
  (e.g. BraTS 2500, I-SPY2 1000, CHAOS 200). Folds 0-2.
- `checkpoint_best.pth`, chosen on real validation images: nnU-Net's validation does not augment,
  so `nnUNetTrainerPA` already selects its checkpoint this way.
