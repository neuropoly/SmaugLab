"""The composed PALETTE synthesis transform."""

from collections.abc import Iterable, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform
from smauglab.transforms.gpu.fromSeg import _gaussian_blur_3d, collapse_onehot_to_index
from smauglab.transforms.gpu.palette.base import (
    BlockContext,
    InitialPartitioner,
    RefinementPartitioner,
    signed_alpha_affine_remap,
)
from smauglab.transforms.gpu.palette.factory import make_initial, make_overlay, make_refinement
from smauglab.transforms.gpu.palette.overlay import AnatomicalLabelOverlay
from smauglab.transforms.rng import shared_choice

#: The composition `RandomPaletteGPU` hard-codes, which is what this defaults to.
DEFAULT_INITIAL: dict[str, Any] = {"type": "kmeans1d"}
DEFAULT_REFINEMENTS: list[dict[str, Any]] = [{"type": "voronoi"}]


@register(
    aug_id=AugId.PALETTE_COMPOSED,
    backend=Backend.GPU,
    group=AugType.TA,
    smoke_kwargs={"initial_partitioner": {"type": "em_gmm"}, "refinement_partitioners": []},
    summary="PALETTE synthesis composed from swappable partitioning blocks.",
)
class PaletteSynthesisGPU(ImageOnlyTransform):
    """PALETTE contrast synthesis composed from swappable partition blocks.

    Same pipeline as `RandomPaletteGPU` -- min-max normalise, partition, per-region
    affine remap, optional blur, per-label overlay, optional blur, foreground
    z-score -- but the partitioning is configuration rather than code. The initial
    partitioner builds a region map from the raw intensities; each refinement
    subdivides the map it is handed; the remap itself is fixed, because that is the
    part whose behaviour the augmentation is named after.

    At its defaults it is `RandomPaletteGPU`: 1-D K-means, then Voronoi, then the
    anatomical-label overlay. The point of the class is the compositions that are
    not that one -- EM/GMM instead of K-means, several refinements in a row, or no
    spatial subdivision at all.

    The three block parameters each accept either a config block (a dict with a
    `type`, as it is written in JSON) or an already-built instance, so the registry
    can build this straight from a config section and a test can hand it an object.

    Segmentation is read from `params['seg']`, injected by the pipeline. One-hot
    `[B, C_seg, D, H, W]` and index `[B, 1, D, H, W]` are both accepted; without it
    the overlay step is skipped.

    Args:
        initial_partitioner: `{"type": "kmeans1d" | "em_gmm", ...}`. None selects
            K-means, matching `RandomPaletteGPU`.
        refinement_partitioners: list of `{"type": "voronoi" | "em_gmm" |
            "identity", ...}`, applied in order. None selects a single Voronoi
            refinement, matching `RandomPaletteGPU`; `[]` means no refinement.
        overlay: `{"enabled": bool, ...}` for the per-anatomical-label remap. None
            selects the default overlay; `{"enabled": false}` turns it off.
        alpha_magnitude_range: [min, max] for |alpha| in the per-region remap.
        dark_threshold: voxels at or below this are treated as background.
        blur_sigmas_pre: Gaussian blur sigma options for the pass before the
            overlay (weighted toward 0 = no blur).
        blur_sigmas_post: the same, for the pass after it. Split in two because a
            blend below 1.0 wants the overlay's own edges softened, which a single
            shared draw cannot express.
        p: probability of applying the transform.
    """

    def __init__(
        self,
        initial_partitioner: InitialPartitioner | dict[str, Any] | None = None,
        refinement_partitioners: Sequence[RefinementPartitioner | dict[str, Any]] | None = None,
        overlay: AnatomicalLabelOverlay | dict[str, Any] | None = None,
        alpha_magnitude_range: Sequence[float] = (0.5, 2.0),
        dark_threshold: float = 0.01,
        blur_sigmas_pre: Sequence[float] = (0.0, 0.0, 0.0, 0.3, 0.5, 0.8),
        blur_sigmas_post: Sequence[float] = (0.0, 0.0, 0.0, 0.3, 0.5, 0.8),
        p: float = 1.0,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        specs = DEFAULT_REFINEMENTS if refinement_partitioners is None else refinement_partitioners
        self.initial = make_initial(DEFAULT_INITIAL if initial_partitioner is None else initial_partitioner)
        self.refinements = nn.ModuleList([make_refinement(spec, i) for i, spec in enumerate(specs)])
        self.overlay = make_overlay({} if overlay is None else overlay)
        self.alpha_magnitude_range = tuple(alpha_magnitude_range)
        self.dark_threshold = float(dark_threshold)
        self.blur_sigmas_pre = list(blur_sigmas_pre)
        self.blur_sigmas_post = list(blur_sigmas_post)

    @torch.no_grad()
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Any],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        seg_raw: torch.Tensor | None = params.get("seg")

        labels: torch.Tensor | None = None
        if seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] > 1:
            labels = collapse_onehot_to_index(seg_raw)
        elif seg_raw is not None and seg_raw.ndim == 5 and seg_raw.shape[1] == 1:
            labels = seg_raw.long()

        B, _C, D, H, W = input.shape
        N = D * H * W
        device = input.device
        eps = 1e-7

        # Min-max normalise channel-0 to [0, 1] per sample
        flat_all = input[:, 0].float().reshape(B, N)
        v_min = flat_all.min(dim=1).values.view(B, 1)
        v_max = flat_all.max(dim=1).values.view(B, 1)
        images_01 = ((flat_all - v_min) / (v_max - v_min + eps)).clamp(0, 1)

        flat_m_all = (images_01 > self.dark_threshold).float()  # foreground mask

        # Voxel coordinates (shared — same spatial dims for every sample)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(D, device=device, dtype=torch.float32),
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(N, 3)

        # ── Step 1: the block stack builds a partition; the remap is fixed ─────
        synth_list = []
        for i in range(B):
            ctx = BlockContext(
                image01=images_01[i],
                fg_mask=flat_m_all[i],
                coords=coords,
                shape=(D, H, W),
                device=device,
            )
            region_ids, n_regions = self.initial.partition(ctx)
            # ModuleList's iterator yields bare Modules, which loses `refine`.
            for refinement in cast(Iterable[RefinementPartitioner], self.refinements):
                region_ids, n_regions = refinement.refine(ctx, region_ids, n_regions)
            synth_list.append(signed_alpha_affine_remap(ctx.image01, ctx.fg_mask, region_ids, n_regions, self.alpha_magnitude_range))

        synth = torch.stack(synth_list)  # (B, N)

        # ── Step 2: optional blur, per-label overlay, optional blur ───────────
        sigma = shared_choice(self.blur_sigmas_pre) if self.blur_sigmas_pre else 0.0
        if sigma > 0.0:
            synth = _gaussian_blur_3d(synth.reshape(B, 1, D, H, W), sigma).reshape(B, N)

        if self.overlay is not None and labels is not None:
            synth = self.overlay.remap_labels(synth, labels, (D, H, W))

        sigma2 = shared_choice(self.blur_sigmas_post) if self.blur_sigmas_post else 0.0
        if sigma2 > 0.0:
            synth = _gaussian_blur_3d(synth.reshape(B, 1, D, H, W), sigma2).reshape(B, N)

        # ── Step 3: foreground z-score ────────────────────────────────────────
        b_cnt = flat_m_all.sum(dim=1, keepdim=True).clamp(min=1)
        b_mean = (synth * flat_m_all).sum(dim=1, keepdim=True) / b_cnt
        b_std = (((synth - b_mean) * flat_m_all).pow(2).sum(dim=1, keepdim=True) / b_cnt + eps).sqrt()
        synth_z = ((synth - b_mean) / b_std * flat_m_all).reshape(B, 1, D, H, W)

        out = input.clone()
        out[:, 0:1] = synth_z.to(input.dtype)
        return out
