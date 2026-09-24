"""Per-anatomical-label overlay: fixed algorithm, tunable frequency and blend."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

#: A scalar blend, or a [lo, hi] range sampled per label per sample.
BlendSpec = float | Sequence[float]


class AnatomicalLabelOverlay(nn.Module):
    """PALETTE's per-label affine remap, with a tunable blend.

    Step 3 of `RandomPaletteGPU`: for each foreground label, with probability
    `label_remap_prob`, sample a fresh (mu, alpha) and remap that label's voxels
    on their own. This is what stops the synthesised contrast from being a pure
    function of intensity -- two structures that look identical in the input come
    out different because the segmentation says they are different.

    The one thing added over the original is `blend_strength`. `RandomPaletteGPU`
    overwrites the label's voxels outright, which is `blend_strength=1.0` here;
    anything lower mixes the remapped values back over the synthesised image, so
    the label boundary stops being a hard edge the network can memorise.

    The entry point is `remap_labels`, not `apply`: `nn.Module.apply` already means
    "run this function over every submodule", and shadowing it would break
    `.apply(fn)` on any pipeline containing one of these.

    Args:
        label_remap_prob: per-label, per-sample probability of remapping.
        min_label_voxels: minimum voxel count for a label to be eligible.
        label_classes: restrict the overlay to these class indices (None = every
            foreground class present in the batch).
        blend_strength: scalar in [0, 1] (1.0 = full overwrite), or a [lo, hi]
            range sampled per label per sample.
        alpha_magnitude_range: [lo, hi] for |alpha| in the remap.
    """

    def __init__(
        self,
        label_remap_prob: float = 0.5,
        min_label_voxels: int = 4,
        label_classes: list[int] | None = None,
        blend_strength: BlendSpec = 1.0,
        alpha_magnitude_range: Sequence[float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        self.label_remap_prob = float(label_remap_prob)
        self.min_label_voxels = int(min_label_voxels)
        self.label_classes = None if label_classes is None else list(label_classes)
        self.blend_strength = _normalise_blend(blend_strength)
        self.alpha_magnitude_range = tuple(alpha_magnitude_range)

    def remap_labels(
        self,
        synth: torch.Tensor,  # (B, N)
        labels: torch.Tensor,  # (B, 1, D, H, W) long
        shape: tuple[int, int, int],
    ) -> torch.Tensor:
        batch, n_voxels = synth.shape
        device = synth.device
        alpha_lo, alpha_hi = self.alpha_magnitude_range
        blend_lo, blend_hi = self.blend_strength
        depth, height, width = shape

        if labels.shape[2:] != (depth, height, width):
            labels = F.interpolate(labels.float(), size=(depth, height, width), mode="nearest").long()
        lbl = labels[:, 0].reshape(batch, n_voxels).clamp(min=0)

        classes = lbl.unique()
        classes = classes[classes > 0]
        if self.label_classes is not None:
            keep = torch.tensor(self.label_classes, device=device)
            classes = classes[torch.isin(classes, keep)]

        for c in classes:
            c_mask = (lbl == int(c.item())).float()
            c_cnt = c_mask.sum(dim=1, keepdim=True)

            applies = ((torch.rand(batch, 1, device=device) < self.label_remap_prob) & (c_cnt >= self.min_label_voxels)).float()
            if applies.sum() == 0:
                continue

            c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)

            mu_c = torch.rand(batch, 1, device=device)
            mag_c = torch.rand(batch, 1, device=device) * (alpha_hi - alpha_lo) + alpha_lo
            sign_c = (torch.rand(batch, 1, device=device) > 0.5).float() * 2 - 1
            alp_c = mag_c * sign_c

            if blend_lo == blend_hi:
                blend = torch.full((batch, 1), blend_lo, device=device)
            else:
                blend = torch.rand(batch, 1, device=device) * (blend_hi - blend_lo) + blend_lo

            new_vals = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)
            write_mask = c_mask * applies * blend
            synth = synth * (1.0 - write_mask) + new_vals * write_mask

        return synth


def _normalise_blend(spec: BlendSpec) -> tuple[float, float]:
    if isinstance(spec, (int, float)):
        return (float(spec), float(spec))
    lo, hi = float(spec[0]), float(spec[1])
    if lo > hi:
        raise ValueError(f"blend_strength range must be non-decreasing, got [{lo}, {hi}]")
    return (lo, hi)
