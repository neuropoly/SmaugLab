# SynthSeg generative augmentation

A faithful [torch] re-implementation of the **SynthSeg** "brain generator" as an
SmaugLab augmentation. Unlike every other transform in SmaugLab — which perturbs a
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

- **3D only** (5D `(B, C, D, H, W)` tensors), matching SmaugLab's GPU transforms.
- Affine transforms are applied **about the volume centre** (like SmaugLab's
  `RandomAffineGPU`), rather than the corner-origin used by neuron's
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

The driver reads **its own** schema: `SynthSegGenerator` parameters, either flat or
under a `"SynthSeg"` key, plus an optional top-level `"probability"`. That is not the
sectioned `GPU`/`CPU` schema `AugTransformsGPU` takes — see section 3 for that one.

```python
import json, torch
from pathlib import Path
from smauglab.transforms.synthseg import SynthSegTransformsGPU

Path("synthseg_params.json").write_text(json.dumps({
    "SynthSeg": {"probability": 1.0, "n_channels": 1, "bias_field_std": 0.7,
                 "gamma_std": 0.5, "randomise_res": True, "em_label_completion": False}
}))
synth = SynthSegTransformsGPU(json_path="synthseg_params.json").to("cuda")

# data: (B, 1, D, H, W) image (ignored), target: (B, 1, D, H, W) label map
image, label = synth(data, target)   # image is fully synthetic, label is deformed/aligned
```

`SynthSegGenerator`'s defaults are already the paper's values, so the faithful setup
needs no file at all; `params=` overrides whichever ones you want to change:

```python
synth = SynthSegTransformsGPU(params={"probability": 1.0, "bias_field_std": 0.7}).to("cuda")
```

Or directly with the module API:

```python
from smauglab.transforms.synthseg import SynthSegGenerator
gen = SynthSegGenerator(generation_labels=[0, 2, 3, 41, 42, ...],
                        n_neutral_labels=1, n_channels=1).to("cuda")
image, label = gen(label_map)         # label_map: (B, 1, D, H, W)
```

### 2. As an `ImageOnlyTransform` in an existing GPU pipeline

`RandomSynthSegGPU` replaces the image with a GMM synthesis of `params['seg']`
(intensity-only: GMM → bias → intensity → resolution). Put SmaugLab's geometric
transforms *before* it so the mask is deformed first and SynthSeg generates from
the deformed labels:

```python
from smauglab.transforms.gpu.base import AugmentationSequentialCustom
from smauglab.transforms.gpu.spatial import RandomAffineGPU
from smauglab.transforms.synthseg import RandomSynthSegGPU

aug = AugmentationSequentialCustom(
    RandomAffineGPU(degrees=15, scale=[0.8, 1.2], p=1.0),
    RandomSynthSegGPU(generation_labels=None, n_channels=1,
                      bias_field_std=0.7, gamma_std=0.5, randomise_res=True, p=1.0),
    data_keys=["input", "mask"], same_on_batch=True,
)
image, seg = aug(image, seg)
```

> **Note (kornia 0.7.4 quirk):** when an `AugmentationSequentialCustom` is called
> with more than one data key, kornia detaches the returned **input image** to the
> CPU (the segmentation/mask stays on the GPU). This affects *every* SmaugLab GPU
> transform identically, not just SynthSeg — re-`.to(device)` the returned image
> if you need it back on the GPU. The full-pipeline `SynthSegTransformsGPU` driver
> (section 1) does **not** go through kornia's sequential and is unaffected.

### 3. By naming `RandomSynthSegGPU` in an `AugTransformsGPU` config

`RandomSynthSegGPU` is a registered augmentation, so naming it in the `GPU` section
is all it takes. Any harness that already constructs `AugTransformsGPU(json_path)`
and calls `augmentor(img, seg)` then gets SynthSeg with no code changes:

```jsonc
{ "GPU": { "RandomSynthSegGPU": { "p": 1.0, "n_channels": 1, "bias_field_std": 0.7,
                                  "gamma_std": 0.5, "randomise_res": true,
                                  "em_label_completion": false } } }
```

The key is the class name and the probability parameter is `p`. A top-level
`"SynthSeg"` block with a `"probability"` inside it was the pre-registry spelling and
is rejected now, with both problems reported at once. `smauglab show RandomSynthSegGPU`
lists what the block accepts; the shipped `transform_params_paper-synthseg.json` is
this setup in full, alongside the rest of the paper's pipeline.

Because this goes through an `ImageOnlyTransform`, it is **intensity-only**: the
image is synthesised and the segmentation is returned unchanged. The transform forces
`apply_affine=False`, `apply_nonlinear=False`, `flipping=False` and
`output_shape=None` on the generator, so the spatial keys behave differently from
section 1 — `scaling_bounds`, `rotation_bounds` and `nonlin_std` are accepted but
inert, while `flipping` and `output_shape` are rejected rather than silently
overridden. For geometry, add a `RandomAffineGPU`/`RandomFlipTransformGPU` block
(which run *before* SynthSeg), or use `SynthSegTransformsGPU` (section 1) for the
full pipeline with a deformed label map.

## Smoke test

Both modules are checked by one self-contained run (no data files, CPU-friendly):

```bash
python -m smauglab.transforms.synthseg
```

The **package**, not the modules. `python -m smauglab.transforms.synthseg.transforms`
puts that file in `sys.modules` twice — once under its own name, because `__init__.py`
imports it, and once as `__main__` — and so executes it twice. Python warns about that
on its own account, and the augmentation registry rejects it outright, because
`@register` fires for a class that is already there. Running the package imports each
module exactly once.
