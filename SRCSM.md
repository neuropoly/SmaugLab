# SRCSM with nnU-Net (branch `srcsm-standalone`)

Train and run nnU-Net with **SRCSM** (Semantic-aware Random Convolution + Source Matching;
Thaler et al., IEEE Access 2025, [arXiv 2512.01510](https://arxiv.org/abs/2512.01510),
code: [imigraz/SRCSM_Domain_Generalization](https://github.com/imigraz/SRCSM_Domain_Generalization)).

- **Training**: the SRCSM augmentation (per-label random convolutions, `SemRandConvGPU`) applied on the
  GPU inside nnU-Net's train step, plus nnU-Net's own spatial augmentation and an affine transform
  (config `auglab/configs/transform_params_gpu_srcsm_semrandconv.json`). Mirroring is off.
- **Inference**: SRCSM's test-time source matching. Every test image is histogram-matched to the
  average intensity CDF of the training images, then predicted.

The SRCSM code is the one used in the PALETTE benchmark, copied unchanged: `srcsm_auglab_port/`
(the operator and the source matching) and `auglab/transforms/gpu/srcsm_semrandconv.py`. This branch
only adds a trainer preset (`auglab/trainers/nnUNetTrainerSRCSM.py`), a predict command
(`auglab/srcsm_predict.py`) and packaging. SRCSM is GPL-3.0, as is AugLab.

## Install

```bash
python3 -m venv venv && source venv/bin/activate
# install PyTorch for your CUDA first (https://pytorch.org)
git clone -b srcsm-standalone https://github.com/neuropoly/AugLab.git
cd AugLab
python3 -m pip install -e .[nnunetv2]
auglab_add_nnunettrainer -t nnUNetTrainerSRCSM    # copies the trainer into nnU-Net (once per environment)
```
Tested with nnunetv2 2.7.0 and torch 2.x on NVIDIA L40S / H100.

## Train

Your dataset must be an ordinary nnU-Net dataset (`nnUNetv2_plan_and_preprocess` done).

```bash
export NNUNET_NUM_EPOCHS=1000        # default 1000
export NNUNET_ITERS_PER_EPOCH=250    # default 250 (nnU-Net's default)
nnUNetv2_train <DATASET> 3d_fullres <FOLD> -tr nnUNetTrainerSRCSM
```
The config actually used is copied to `transform_params_gpu_used_for_training.json` in each fold folder.
`num_labels` of the SRCSM operator is set automatically (background + number of foreground classes).

For reference, the PALETTE benchmark trained 2000 epochs x 150 iterations on most tasks (BraTS 2500,
CHAOS 200), folds 0-2, and took `checkpoint_best.pth`.

## Predict (with source matching)

```bash
auglab_srcsm_predict -i <imagesTs> -o <predictions> -d <DATASET> -c 3d_fullres -f 0 1 2
```
- Source CDF: built from `$nnUNet_raw/<Dataset>/imagesTr/*_0000.nii.gz` on first use and cached as
  `srcsm_source_cdf.json` in the model folder. Use `--source-images <dir>` for another folder and
  `--exclude-cases cases.json` (a JSON list of case ids) if test cases sit inside `imagesTr`.
- Modality (`mr` or `ct`) is read from the dataset's channel 0 name; override with `--modality`.
- Defaults: `-chk checkpoint_best.pth`, `--disable_tta`. Any other flag is passed on to `nnUNetv2_predict`.
- `--matched-dir <dir>` keeps the matched images; `--no-source-matching` predicts on the raw inputs
  (SRC only, the authors' ablation).
- Single-channel models only (inputs `<case>_0000.nii.gz`).

The matching tool can also run by itself:
```bash
python -m srcsm_auglab_port.source_matching cdf   --images imagesTr --modality mr --out cdf.json
python -m srcsm_auglab_port.source_matching match --cdf cdf.json --in-dir imagesTs --out-dir matched --modality mr
python -m srcsm_auglab_port.source_matching qc    --cdf cdf.json --in-dir imagesTs --out-dir matched --modality mr
```
It follows the authors' code with two documented deviations (header of `srcsm_auglab_port/source_matching.py`):
MR intensities are clipped at 0 before scaling, and the CT window is [-1024, 3071] HU.
