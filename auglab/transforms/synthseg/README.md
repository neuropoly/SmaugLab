# SynthSeg generative augmentation

A faithful [torch] re-implementation of the **SynthSeg** "brain generator" as an
AugLab augmentation. Unlike every other transform in AugLab — which perturbs a
*real* image — SynthSeg **ignores the input image entirely and synthesises a new
image from a label map**, using domain randomisation (a per-label Gaussian
mixture model plus random spatial, bias, intensity and resolution corruptions).
A network trained on these synthetic images becomes agnostic to MRI contrast and
resolution.

Because the method is fundamentally different from the intensity/geometry
transforms in `gpu/` and `cpu/`, it lives in its own package.

References:
- B. Billot et al., *SynthSeg: Segmentation of brain MRI scans of any contrast
  and resolution without retraining*, Medical Image Analysis, 2023.
- B. Billot et al., *A Learning Strategy for Contrast-agnostic MRI Segmentation*
  (MICCAI 2020) and *Partial Volume Segmentation of Brain MRI Scans of any
  Resolution and Contrast* (MedIA 2021).
- Reference code: [`BBillot/SynthSeg`](https://github.com/BBillot/SynthSeg),
  [`BBillot/lab2im`](https://github.com/BBillot/lab2im).

## Pipeline

`SynthSegGenerator` reproduces the exact order of `labels_to_image_model`:

```
label map
  └─ spatial deformation (affine + diffeomorphic SVF, on labels, nearest)
  └─ [optional random crop to output_shape]
  └─ left/right flip with anatomical label swap
  └─ GMM intensity sampling           image[v] = mean[label_v] + std[label_v]·N(0,1)
  └─ bias field                       × exp(smooth Gaussian field)
  └─ intensity augmentation           clip[0,300] → min-max to [0,1] → image^exp(N(0,γ))
  └─ resolution randomisation         blur → subsample (nearest) → resample (linear), per channel
  └─ map generation labels → output labels
=> (synthetic image, deformed label map)
```

Each step is a small function in [`functional.py`](functional.py), each citing
the corresponding `lab2im`/`SynthSeg` layer.

## Default hyper-parameters

Defaults match `BrainGenerator.__init__` (which overrides several
`labels_to_image_model` signature defaults). Notable, easy-to-miss values:

| Parameter | Default | Notes |
|---|---|---|
| `prior_means` / `prior_stds` | `None` → `U(0, 250)` / `U(0, 30)` | full domain randomisation. The often-quoted `[25,225]`/`[5,25]` are a stale docstring; the code uses `centre=125,range=125` and `centre=15,range=15`. |
| `scaling_bounds` | `0.2` | per-axis `U(0.8, 1.2)` |
| `rotation_bounds` | `15` | per-axis `U(-15°, 15°)` |
| `shearing_bounds` | `0.012` | per off-diagonal `U(-0.012, 0.012)` |
| `translation_bounds` | `false` | off |
| `nonlin_std` / `nonlin_scale` | `4.0` / `0.04` | SVF std `~U(0,4)`; coarse grid `ceil(shape·0.04)` |
| `svf_integration_steps` | `7` | scaling-and-squaring (`VecInt`, `ss`) |
| `bias_field_std` / `bias_scale` | `0.7` / `0.025` | std `~U(0,0.7)`; coarse grid `ceil(shape·0.025)` |
| `gamma_std` / `clip` | `0.5` / `300` | hard-coded in SynthSeg's `IntensityAugmentation` |
| `randomise_res` | `true` | per-channel random acquisition resolution |
| `max_res_iso` / `max_res_aniso` | `4.0` / `8.0` | mm ceilings |
| `blur_range` | `1.03` | sigma jitter `U(1/1.03, 1.03)` (`1.15` in the 2020 model) |
| `atlas_res` | `1.0` | native resolution of the input label map (mm) |

## Implementation notes / deviations

- **3D only** (5D `(B, C, D, H, W)` tensors), matching AugLab's GPU transforms.
- Affine transforms are applied **about the volume centre** (like AugLab's
  `RandomAffine3DCustom`), rather than the corner-origin used by neuron's
  `affine_to_shift`. This keeps the anatomy in frame and is the standard choice;
  the visual augmentation is equivalent.
- The SVF is integrated at full resolution after upsampling the coarse velocity
  field (lab2im integrates at half resolution then upsamples — equivalent in
  effect for a smooth field, simpler and less error-prone here).
- The background special-casing in GMM parameter sampling (5% black / 25%
  dark-low-variance / 70% normal) is reproduced. The internal 0.95 "apply" prob
  of the bias-field layer is folded into the transform-level probability.

## ⚠️ Label maps must be dense for realistic synthesis

SynthSeg's realism comes from a **dense anatomical label map** (e.g. a
FreeSurfer/SAMSEG segmentation covering every tissue: WM, GM, CSF, ventricles,
sub-cortical structures, extra-cerebral tissue, ...). If you feed it a *sparse*
target segmentation (a few foreground structures over a `0` background — as is
common for spinal-cord/lesion tasks), it still runs, but without the EM
completion below the synthetic image will only contain those structures over a
single-Gaussian background.

### EM label completion for sparse maps (`em_label_completion`)

This is exactly the situation the SynthSeg paper addresses in §5.4:

> *"we enhance the training segmentations by subdividing all their labels
> (background and foreground) into finer subregions. This is achieved by
> clustering the intensities of the associated image with the Expectation
> Maximisation algorithm."*

Enable `em_label_completion=True` and the generator will, **using the paired real
image** that is already available on-the-fly (the `data` / `input` tensor):

- split every **foreground** label into `em_n_foreground_clusters` subregions (2 in the paper);
- split the **background** label into a random `N ∈ em_background_clusters_range` subregions ([3, 10] in the paper);
- give each subregion its own generation Gaussian (so the formerly single-Gaussian
  background becomes an intensity-coherent patchwork — realistic extra-cerebral / unlabelled tissue);
- **merge the subregions back** to their parent labels for the output segmentation,
  so the training target is unchanged.

The EM fit per region is sub-sampled to `em_max_fit_voxels` voxels for speed (the
full region is still assigned). Unlike the paper — which precomputes these maps
offline — this runs on the fly from the real image, so no preprocessing is needed.

```python
gen = SynthSegGenerator(em_label_completion=True).to("cuda")
image, label = gen(sparse_label_map, image=real_image)   # real_image drives the EM clustering
# or via the driver / config: set "em_label_completion": true and call synth(data, target)
```

> The EM path needs the real image: `SynthSegTransformsGPU(data, target)` and
> `RandomSynthSegGPU` (which reads the image being augmented) both supply it
> automatically; the bare `SynthSegGenerator.forward` needs `image=...`.

## Usage

### 1. Full end-to-end generator (faithful SynthSeg)

`SynthSegTransformsGPU` mirrors `AugTransformsGPU`: build from JSON, move to the
device, and call `transforms(data, target) → (image, target)`. The `data` tensor
is ignored; `target` is the label map.

```python
import importlib, torch
import auglab.configs as configs
from auglab.transforms.synthseg import SynthSegTransformsGPU

cfg = importlib.resources.files(configs) / "synthseg_params.json"
synth = SynthSegTransformsGPU(json_path=str(cfg)).to("cuda")

# data: (B, 1, D, H, W) image (ignored), target: (B, 1, D, H, W) label map
image, label = synth(data, target)   # image is fully synthetic, label is deformed/aligned
```

Or directly with the module API:

```python
from auglab.transforms.synthseg import SynthSegGenerator
gen = SynthSegGenerator(generation_labels=[0, 2, 3, 41, 42, ...],
                        n_neutral_labels=1, n_channels=1).to("cuda")
image, label = gen(label_map)         # label_map: (B, 1, D, H, W)
```

### 2. As an `ImageOnlyTransform` in an existing GPU pipeline

`RandomSynthSegGPU` replaces the image with a GMM synthesis of `params['seg']`
(intensity-only: GMM → bias → intensity → resolution). Put AugLab's geometric
transforms *before* it so the mask is deformed first and SynthSeg generates from
the deformed labels:

```python
from auglab.transforms.gpu.base import AugmentationSequentialCustom
from auglab.transforms.gpu.spatial import RandomAffine3DCustom
from auglab.transforms.synthseg import RandomSynthSegGPU

aug = AugmentationSequentialCustom(
    RandomAffine3DCustom(degrees=15, scale=[0.8, 1.2], p=1.0),
    RandomSynthSegGPU(generation_labels=None, n_channels=1,
                      bias_field_std=0.7, gamma_std=0.5, randomise_res=True, p=1.0),
    data_keys=["input", "mask"], same_on_batch=True,
)
image, seg = aug(image, seg)
```

> **Note (kornia 0.7.4 quirk):** when an `AugmentationSequentialCustom` is called
> with more than one data key, kornia detaches the returned **input image** to the
> CPU (the segmentation/mask stays on the GPU). This affects *every* AugLab GPU
> transform identically, not just SynthSeg — re-`.to(device)` the returned image
> if you need it back on the GPU. The full-pipeline `SynthSegTransformsGPU` driver
> (section 1) does **not** go through kornia's sequential and is unaffected.

### 3. Via a `SynthSeg` key in an `AugTransformsGPU` config

`AugTransformsGPU` recognises a top-level `"SynthSeg"` config block and appends a
`RandomSynthSegGPU` built from it (mapping `"probability"` → `p`). This lets any
harness that already constructs `AugTransformsGPU(json_path)` and calls
`augmentor(img, seg)` use SynthSeg with no code changes:

```jsonc
{ "SynthSeg": { "probability": 1.0, "n_channels": 1, "bias_field_std": 0.7,
                "gamma_std": 0.5, "randomise_res": true, "em_label_completion": false } }
```

Because this goes through an `ImageOnlyTransform`, it is **intensity-only**: the
image is synthesised and the segmentation is returned unchanged. The spatial keys
in the block (`flipping`, `scaling_bounds`, `rotation_bounds`, `nonlin_std`, ...)
are therefore inert in this path — add an `AffineTransform`/`FlipTransform` block
(which run *before* SynthSeg) for geometry, or use `SynthSegTransformsGPU`
(section 1) for the full pipeline with a deformed label map.

## Smoke test

Both modules are runnable and self-contained (no data files, CPU-friendly):

```bash
python -m auglab.transforms.synthseg.generator
python -m auglab.transforms.synthseg.transforms
```
