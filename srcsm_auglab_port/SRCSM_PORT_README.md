# SRCSM augmentation ported to AugLab / nnU-Net (PyTorch)

Reimplementation of the **augmentation operator only** from SRCSM
(Thaler et al., IEEE Access 2025; `github.com/imigraz/SRCSM_Domain_Generalization`,
GPL-3.0) as a plug-in augmentation-swap baseline on a common nnU-Net backbone.

This is a **faithful port, not their pipeline**. Their U-Net (`UnetAvgLinear3D`),
training framework, data loading, and source-matching are intentionally dropped —
nnU-Net supplies all of that. The point is to isolate the *augmentation mechanism*
so it sits on the identical backbone as the PALETTE runs.

## Files
- `sem_rand_conv_3d.py` — `RCNet` + `SemRandConv3D`, the operator (device/dtype-agnostic).
- `srcsm_transform.py`  — `SemRandConvTransform`, batchgenerators-style wrapper.
- `test_srcsm.py`, `test_transform.py` — self-checks.
- `gpu_smoke_test.py` — GPU test at real patch size (correctness, timing, fp16, mem).

## GPU validation
Validated on Killarney (Alliance Canada) on a single NVIDIA L40S, torch 2.12.1 /
CUDA 13.2, from the project venv (`.venv`), at the nnU-Net patch size
`160×128×160`:
- output stays on `cuda`, finite, gradient-free; fresh weights each call.
- per-call cost (per-label, B=2, L=5): **~39 ms**, peak GPU mem **~580 MB**.
- global (single RCNet) mode: ~4 ms/call.
- fp16 / autocast path: finite (nnU-Net trains under `autocast`).
- `SemRandConvTransform(device=None)` auto-selects `cuda`.

The operator is device- and dtype-agnostic (every tensor follows `x.device`/
`x.dtype`); the wrapper defaults to auto device selection. Run the GPU test with:
`python gpu_smoke_test.py` inside a GPU allocation.

## What the operator does (their spec, preserved)
- `RCNet`: 4× Conv3D, `num_filters_base=2`, last layer 1 filter; per-layer kernel
  ∈ {1,3}; `N(0,1)` weights **resampled every call, never trained**; leaky_relu(0.1);
  α∼U(0,1) blend `net(x)·α + x·(1−α)`; **per-sample Frobenius energy renorm** (1e-5 floor).
- `SemRandConv3D` per-label: one-hot GT labels over `range(num_labels)` **incl.
  background**; optional Gaussian-smoothed **soft** masks (k=5, σ=1); a **fresh
  RCNet per label**; composite `Σ rcnet_l(x)·mask_l` (masks not renormalised).
- Zero-output retry guard preserved.

## Load-bearing details that are easy to lose (all pinned in code)
1. Fresh random weights every call — sampled inside `forward`, not `nn.Parameter`s.
2. Frobenius renorm kept, **per sample** (their code assumed batch==1).
3. Soft per-label masks over all labels incl. background, not renormalised.
4. leaky_relu(0.1), random kernel {1,3}, zero-output retry.
5. Gradient-free (`torch.no_grad`, `requires_grad=False`).

## Two integration-time decisions (NOT in the operator — set these when wiring in)
1. **Disable nnU-Net's default intensity augmentations** (gamma / brightness /
   contrast / Gaussian noise+blur) for this arm, matching PALETTE's configuration.
   Otherwise you benchmark "SemRandConv + nnU-Net augs", not the operator. This is
   the single easiest way to make the comparison unfair in either direction.
2. **Normalisation.** The operator runs on nnU-Net's z-scored inputs, not their
   [-1,1]. The Frobenius renorm makes it scale-tolerant. State in the paper that
   (a) it runs on nnU-Net-normalised inputs and (b) the backbone is nnU-Net, so the
   result isolates the mechanism and does **not** reproduce their published numbers.

## Source matching — out of scope
Their `source_matching/` step computes an average source CDF and histogram-matches
the target at test time. That is a target-side adaptation, not a training
augmentation, so it is excluded from the main comparison. If a reviewer asks, run
it as a separate with/without ablation row rather than folding it in silently.

## Wiring into AugLab
Register `SemRandConvTransform(num_labels=<fg+1>, per_label=True, smoothing=True)`
in the intensity-augmentation slot (after spatial, before the network) — the same
slot GIN-3D occupies. The wrapper auto-detects nnU-Net v1 (`data`/`seg`, numpy,
batch) vs v2 (`image`/`segmentation`, torch, per-sample) layouts; override with
`image_key`/`seg_key` if your harness renames them. Single modality only (C==1).

## Fidelity statement for the manuscript
> We reimplemented the semantic-aware random-convolution operator of Thaler et al.
> in PyTorch from their public code, matching their hyperparameters (4-layer random
> Conv3D, kernel ∈ {1,3}, N(0,1) weights, leaky-ReLU 0.1, uniform blend, Frobenius
> energy renormalisation, per-label soft-mask compositing). We evaluate it as an
> augmentation swap on the identical nnU-Net backbone used for all other methods;
> their original U-Net and source-matching steps are therefore not part of this
> comparison, which isolates the augmentation mechanism.
