"""AugLab-style wrappers around the SynthSeg generative model.

Two entry points are provided:

* :class:`RandomSynthSegGPU` -- an :class:`ImageOnlyTransform` that *replaces*
  the image with a GMM-synthesised one derived from ``params['seg']``. It is
  intensity-only (no internal spatial deformation), so it composes with AugLab's
  existing geometric transforms (``RandomAffine3DCustom``, ``RandomFlipTransformGPU``,
  ...) inside an :class:`AugmentationSequentialCustom`: place those *before* it so
  the mask is deformed first and SynthSeg generates from the deformed labels,
  keeping image and label aligned. Drop it into a ``transform_params_gpu.json``
  pipeline like any other transform.

* :class:`SynthSegTransformsGPU` -- a config-driven top-level module mirroring
  :class:`auglab.transforms.gpu.transforms.AugTransformsGPU`. It runs the *full*
  SynthSeg pipeline (spatial deform + flip + GMM + bias + intensity + resolution)
  and returns ``(image, label)`` from ``forward(data, target)`` -- the calling
  convention used by the nnUNet trainer and ``train_monai.py``. This is the
  faithful end-to-end SynthSeg generator.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from kornia.core import Tensor

from auglab.transforms.gpu.base import ImageOnlyTransform
from auglab.transforms.synthseg.generator import SynthSegGenerator

# Keys understood from the JSON config / kwargs, forwarded to SynthSegGenerator.
_GENERATOR_KEYS = {
    "generation_labels", "output_labels", "n_neutral_labels", "generation_classes",
    "n_channels", "prior_distributions", "prior_means", "prior_stds",
    "flipping", "flip_axis", "scaling_bounds", "rotation_bounds", "shearing_bounds",
    "translation_bounds", "nonlin_std", "nonlin_scale", "svf_integration_steps",
    "bias_field_std", "bias_scale", "gamma_std", "clip", "normalise",
    "randomise_res", "max_res_iso", "max_res_aniso", "data_res", "thickness",
    "blur_range", "atlas_res", "output_shape",
    "em_label_completion", "em_n_foreground_clusters", "em_background_clusters_range",
    "em_background_label", "em_n_iters", "em_max_fit_voxels", "em_same_on_batch",
    "apply_affine", "apply_nonlinear", "apply_bias_field",
    "apply_intensity_augmentation", "apply_resolution",
}


def _filter_generator_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in _GENERATOR_KEYS}


class RandomSynthSegGPU(ImageOnlyTransform):
    """Replace the image with a SynthSeg GMM synthesis of ``params['seg']``.

    Intensity-only: GMM sampling -> bias field -> intensity augmentation ->
    resolution randomisation. Spatial deformation / flipping are disabled so that
    geometry stays consistent with the segmentation propagated by the surrounding
    :class:`AugmentationSequentialCustom` (use AugLab's geometric transforms for
    that, placed before this one).

    Args:
        apply_to_channel: image channels to overwrite with the synthesis.
        p: application probability (Kornia convention).
        Remaining kwargs are forwarded to :class:`SynthSegGenerator` (e.g.
        ``n_channels``, ``prior_means``, ``bias_field_std``, ``gamma_std``,
        ``randomise_res``, ``max_res_iso`` ...).
    """

    def __init__(
        self,
        apply_to_channel: Optional[List[int]] = None,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.apply_to_channel = apply_to_channel if apply_to_channel is not None else [0]
        gen_kwargs = _filter_generator_kwargs(kwargs)
        # Intensity-only: never deform/flip internally (geometry comes from the
        # surrounding sequential, which also transports the mask).
        gen_kwargs.update(apply_affine=False, apply_nonlinear=False, flipping=False, output_shape=None)
        self.generator = SynthSegGenerator(**gen_kwargs)

    @torch.no_grad()
    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        seg = params.get("seg", None)
        if seg is None:
            return input

        # Pass the real image so EM label completion (if enabled) can cluster it.
        synth, _ = self.generator(seg, image=input)  # (B, n_channels, D, H, W)
        synth = synth.to(device=input.device, dtype=input.dtype)

        for n, c in enumerate(self.apply_to_channel):
            if c < 0 or c >= input.shape[1]:
                continue
            src = n if n < synth.shape[1] else synth.shape[1] - 1
            x = synth[:, src]
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning nan: {self.__class__.__name__}", flush=True)
                continue
            input[:, c] = x
        return input


class SynthSegTransformsGPU(nn.Module):
    """Config-driven full SynthSeg brain generator.

    Mirrors :class:`AugTransformsGPU`: construct from a JSON path, move ``.to(device)``,
    then call ``transforms(data, target)`` to obtain ``(synthetic_image, label)``.
    The image input is ignored (SynthSeg synthesises purely from labels); ``target``
    supplies the label map (single-channel integer, one-hot, or ``(B, D, H, W)``).

    The JSON may either be a flat dict of :class:`SynthSegGenerator` parameters or
    wrap them under a ``"SynthSeg"`` key, optionally with a top-level
    ``"probability"`` controlling how often synthesis is applied (otherwise the
    original ``(data, target)`` is returned unchanged). ``probability`` defaults
    to ``1.0`` (always synthesise, as in the paper).
    """

    def __init__(self, json_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if params is None:
            if json_path is None:
                raise ValueError("Provide either json_path or params.")
            with open(os.path.join(json_path), "r") as f:
                config = json.load(f)
        else:
            config = params

        if "SynthSeg" in config:
            config = config["SynthSeg"]

        self.probability = float(config.get("probability", 1.0))
        self.return_onehot = bool(config.get("return_onehot", False))
        self.generator = SynthSegGenerator(**_filter_generator_kwargs(config))

    @torch.no_grad()
    def forward(self, data: Tensor, target: Tensor):
        if self.probability < 1.0 and float(torch.rand((), device=data.device)) >= self.probability:
            return data, target

        # ``data`` (the real image) is only consumed when em_label_completion is on.
        image, labels = self.generator(target, image=data)
        image = image.to(device=data.device, dtype=data.dtype)

        if self.return_onehot and target.dim() == 5 and target.shape[1] > 1:
            labels = self._to_onehot(labels, target.shape[1]).to(dtype=target.dtype)
        else:
            labels = labels.to(dtype=target.dtype)
        return image, labels

    @staticmethod
    def _to_onehot(labels: Tensor, n_channels: int) -> Tensor:
        # labels: (B, 1, D, H, W) with values in 1..n_channels for foreground.
        B, _, D, H, W = labels.shape
        onehot = torch.zeros(B, n_channels, D, H, W, device=labels.device)
        for c in range(n_channels):
            onehot[:, c] = (labels[:, 0] == (c + 1)).float()
        return onehot


if __name__ == "__main__":
    # Smoke test mirroring the generator's, exercising both wrappers.
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, D, H, W = 2, 40, 44, 42
    zz, yy, xx = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
    r = ((zz - D / 2) ** 2 + (yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    vol = torch.zeros(D, H, W, dtype=torch.long)
    vol[r < 15] = 1
    vol[r < 8] = 2
    labels = vol.view(1, 1, D, H, W).repeat(B, 1, 1, 1, 1)
    img = torch.randn(B, 1, D, H, W)

    # Full end-to-end driver
    driver = SynthSegTransformsGPU(params={"generation_labels": [0, 1, 2], "n_channels": 1}).to(device)
    out_img, out_lab = driver(img.to(device), labels.to(device))
    print("driver image", tuple(out_img.shape), "labels", tuple(out_lab.shape),
          torch.unique(out_lab).tolist())
    assert out_img.shape[0] == B and not torch.isnan(out_img).any()

    # Intensity-only ImageOnlyTransform
    t = RandomSynthSegGPU(generation_labels=[0, 1, 2], n_channels=1, p=1.0).to(device)
    out = t.apply_transform(img.clone().to(device), {"seg": labels.to(device)}, {})
    print("imageonly", tuple(out.shape), "range", (round(float(out.min()), 3), round(float(out.max()), 3)))
    assert out.shape == img.shape and not torch.isnan(out).any()
    print("OK")
