"""What lies around the scan: shift the scan inside the patch, and synthesise what is outside it.

A sagittal spine slab is ~8 cm wide. nnU-Net centres it in a 128 mm patch and pads the rest
with 0 (seg -1 -> 0), so the network sees spine in every training window and nothing but a
flat zero slab beyond +-4 cm. On a whole-body axial scan it then finds vertebrae in muscle,
gluteals, fat and bowel: on NAKO VIBE most of its false-positive volume lies more than 5 cm
from the spine. `RandomFOVExtensionGPU` shows it windows where the spine sits at the edge or
is absent, surrounded by tissue that is labelled background.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from kornia.augmentation._3d.base import RigidAffineAugmentationBase3D
from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling
from torch import Tensor

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import segmentation_from
from smauglab.transforms.gpu.contrast import _foreground
from smauglab.transforms.synthseg.functional import _quadratic_log_likelihood_terms, gaussian_blur_3d, random_svf_field, warp_volume

# How the region map is carried into the empty part of the patch. Constants rather than
# parameters: they shape a texture that is always labelled background, and the screen that
# introduced the transform varied `shift` and `p`, not these.
_SCALE_RANGE = (0.75, 1.33)  # log-uniform, ~nnU-Net's 0.7-1.4
_EM_ITERS = 20  # SynthSeg's em_n_iters default
_EM_SAMPLES = 100_000  # SynthSeg's em_max_fit_voxels default
_SEAM = 1  # voxels of interpolated padding edge never used as a source
_CLOSE_ITERS = 8  # max-pool steps that close what the warped layers missed
_TEXTURE_SIGMA = 1.0  # correlation length of the within-region texture, in voxels
_RENDER_SIGMA = 0.5  # SynthSeg's blur at the atlas resolution, in voxels
_RENDER_BLUR_RANGE = 1.15  # SynthSeg's DynamicGaussianBlur jitter


def _dilate(mask: Tensor, radius: int) -> Tensor:
    """Binary dilation of a `[N, 1, D, H, W]` bool mask by a cube of half-width `radius`.

    Three 1-D max-pools rather than one dense `(2r+1)^3` pool: the same result for a cube.
    """
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    x = mask.float()
    for kernel in ((k, 1, 1), (1, k, 1), (1, 1, k)):
        x = F.max_pool3d(x, kernel, stride=1, padding=tuple(s // 2 for s in kernel))
    return x > 0


def _shift(volume: Tensor, offsets: list[list[int]], fill: float | bool) -> Tensor:
    """Translate each row of `[N, C, D, H, W]` by whole voxels; vacated voxels get `fill`.

    `offsets[b]` is the `(d, h, w)` displacement of row `b`. Integer and exact, so the
    image and the mask move identically and no label is interpolated.
    """
    out = torch.full_like(volume, fill)
    spatial = volume.shape[2:]
    for b, offset in enumerate(offsets):
        src, dst = [], []
        for size, o in zip(spatial, offset):
            if abs(o) >= size:
                break
            src.append(slice(max(0, -o), size - max(0, o)))
            dst.append(slice(max(0, o), size - max(0, -o)))
        else:
            out[b][(slice(None), *dst)] = volume[b][(slice(None), *src)]
    return out


def _em_regions(x: Tensor, weight: Tensor, n_regions: Tensor, k_max: int, eps: float = 1e-6) -> Tensor:
    """Fit a 1-D Gaussian mixture per row by EM and return each voxel's most likely component.

    The batched, weighted form of `synthseg.functional._em_gmm_1d`: every row is fitted at
    once, on `_EM_SAMPLES` voxels drawn uniformly and weighted by `weight`, so selecting the
    background costs no boolean indexing and no host synchronisation. Row `b` uses its first
    `n_regions[b]` of `k_max` components; the rest are masked out of the softmax.

    Args:
        x: `[N, V]` intensities, all voxels of the patch.
        weight: `[N, V]` 1.0 where a voxel may be fitted, else 0.0.
        n_regions: `[N]` long, components per row, at least 1.
        k_max: number of component slots.

    Returns:
        `[N, V]` long, the component index of every voxel.
    """
    n, v = x.shape
    idx = torch.randint(0, v, (n, _EM_SAMPLES), device=x.device)
    xs, ws = x.gather(1, idx), weight.gather(1, idx)

    comp = torch.arange(k_max, device=x.device)
    active = comp.view(1, -1) < n_regions.view(-1, 1)  # [N, K]
    total = ws.sum(1, keepdim=True).clamp_min(eps)
    shift = (ws * xs).sum(1, keepdim=True) / total
    centred = xs - shift

    big = torch.finfo(x.dtype).max
    lo = torch.where(ws > 0, centred, big).amin(1, keepdim=True)
    hi = torch.where(ws > 0, centred, -big).amax(1, keepdim=True)
    lo, hi = torch.where(total > eps, lo, 0.0), torch.where(total > eps, hi, 0.0)
    means = lo + (hi - lo) / (n_regions.view(-1, 1) - 1).clamp_min(1) * comp.view(1, -1)
    var = ((ws * centred * centred).sum(1, keepdim=True) / total).clamp_min(eps).expand(n, k_max)
    mix = active.float() / n_regions.view(-1, 1)

    design = torch.stack([centred * centred, centred, torch.ones_like(centred)], dim=2)  # [N, S, 3]
    for _ in range(_EM_ITERS):
        terms = _quadratic_log_likelihood_terms(means, var, mix, eps).permute(1, 0, 2)  # [N, 3, K]
        resp = torch.softmax((design @ terms).masked_fill(~active.unsqueeze(1), -math.inf), dim=2) * ws.unsqueeze(2)
        # Sums of r*x^2, r*x, r per component. A broadcast product rather than
        # `resp.mT @ design`: cuBLAS takes ~7.5 ms for that reduction shape, this ~0.04 ms.
        moments = (resp.unsqueeze(3) * design.unsqueeze(2)).sum(1)  # [N, K, 3]
        nk = moments[..., 2].clamp_min(eps)
        mix = torch.where(active, nk / total, 0.0)
        means = moments[..., 1] / nk
        var = (moments[..., 0] / nk - means * means).clamp_min(eps)

    terms = _quadratic_log_likelihood_terms(means, var, mix, eps).permute(1, 0, 2)
    c = (x - shift).unsqueeze(2)
    logits = torch.cat([c * c, c, torch.ones_like(c)], dim=2) @ terms  # [N, V, K]
    return logits.masked_fill(~active.unsqueeze(1), -math.inf).argmax(dim=2)


def _random_similarity(n: int, device: torch.device) -> Tensor:
    """`[n, 3, 3]` random rotation (uniform axis, angle in [-pi, pi]) x log-uniform scale x mirror."""
    axis = F.normalize(torch.randn(n, 3, device=device), dim=1)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
    kx = torch.zeros(n, 3, 3, device=device)
    kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    kx = kx - kx.transpose(1, 2)
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    sin, cos = angle.sin().view(n, 1, 1), angle.cos().view(n, 1, 1)
    rotation = eye + sin * kx + (1.0 - cos) * (kx @ kx)  # Rodrigues
    lo, hi = math.log(_SCALE_RANGE[0]), math.log(_SCALE_RANGE[1])
    scale = torch.exp(lo + torch.rand(n, 1, device=device) * (hi - lo))
    mirror = torch.where(torch.rand(n, 3, device=device) < 0.5, -1.0, 1.0)
    return rotation @ torch.diag_embed(scale * mirror)


def _extend_regions(regions: Tensor, todo: Tensor, layers: int) -> Tensor:
    """Write region labels into `todo` from warped copies of the labels already present.

    `regions` is `[N, 1, D, H, W]` long, `-1` where unlabelled. Each layer warps the
    current map -- including what earlier layers filled -- by a random rotation, scale,
    mirror and translation with reflection padding, and keeps the voxels of `todo` it
    lands a label on. Whatever is still unlabelled after the layers takes a neighbour's
    label by max-pool propagation, then region 0.
    """
    n = regions.shape[0]
    shape = regions.shape[2:]
    half = torch.tensor(shape, device=regions.device, dtype=torch.float32).view(1, 3, 1) / 2.0
    cur = regions.clone()
    for _ in range(layers):
        affine = torch.zeros(n, 4, 4, device=regions.device)
        affine[:, :3, :3] = _random_similarity(n, regions.device)
        affine[:, :3, 3:] = (torch.rand(n, 3, 1, device=regions.device) * 2.0 - 1.0) * half
        affine[:, 3, 3] = 1.0
        warped = warp_volume((cur + 1).float(), affine=affine, interp="nearest", padding_mode="reflection").round().long() - 1
        take = todo & (warped >= 0)
        cur = torch.where(take, warped, cur)
        todo = todo & ~take
    for _ in range(_CLOSE_ITERS):
        grown = F.max_pool3d((cur + 1).float(), 3, stride=1, padding=1).long() - 1
        take = todo & (grown >= 0)
        cur = torch.where(take, grown, cur)
        todo = todo & ~take
    return torch.where(todo, 0, cur)


def _render(image: Tensor, atlas: Tensor, layout: Tensor, k_max: int, eps: float = 1e-6) -> Tensor:
    """Draw every voxel of `layout` from its region's Gaussian, per channel, then blur as SynthSeg does.

    Means and variances are measured where `atlas` labels the patch itself (`-1` elsewhere),
    so the fill has the patch's own contrast; every intensity augmentation after this
    transform then acts on fill and anatomy alike. The within-region variation is spatially
    correlated noise with the region's standard deviation: SynthSeg's iid draws read as grain
    next to the smooth scan, a cue that alone would separate fill from anatomy.
    """
    n, c = image.shape[:2]
    # A comparison rather than `F.one_hot`, which reads the label range back to the host on CUDA.
    member = (atlas.flatten(1).unsqueeze(2) == torch.arange(k_max, device=atlas.device)).float()  # [N, V, K], -1 matches none
    flat = image.flatten(2)  # [N, C, V]
    count = member.sum(1).clamp_min(eps).unsqueeze(1)  # [N, 1, K]
    mean = (flat @ member) / count
    std = (((flat * flat) @ member) / count - mean * mean).clamp_min(0.0).sqrt()
    index = layout.flatten(1).clamp_min(0).unsqueeze(1).expand(n, c, -1)  # -1 lies outside the fill
    texture = gaussian_blur_3d(torch.randn_like(image), torch.full((3,), _TEXTURE_SIGMA, device=image.device)).flatten(2)
    texture = texture / texture.std(dim=2, keepdim=True).clamp_min(eps)
    drawn = mean.gather(2, index) + std.gather(2, index) * texture
    sigma = torch.full((3,), _RENDER_SIGMA, device=image.device)
    return gaussian_blur_3d(drawn.view_as(image), sigma, blur_range=_RENDER_BLUR_RANGE)


class FOVExtensionGenerator3D(RandomGeneratorBase):
    """Per row: the shift as a fraction of the patch along `(d, h, w)`, and the EM region count."""

    def __init__(self, shift: tuple[float, float, float], regions: tuple[int, int]) -> None:
        super().__init__()
        self.shift = shift
        self.regions = regions

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.shift_samplers = [
            UniformDistribution(
                torch.tensor(-s, device=device, dtype=dtype), torch.tensor(s, device=device, dtype=dtype), validate_args=False
            )
            for s in self.shift
        ]
        lo, hi = self.regions
        self.regions_sampler = UniformDistribution(
            torch.tensor(float(lo), device=device, dtype=dtype),
            torch.tensor(float(hi + 1), device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        n = batch_shape[0]
        shift = torch.stack([_adapted_rsampling((n,), s, same_on_batch) for s in self.shift_samplers], dim=1)
        regions = _adapted_rsampling((n,), self.regions_sampler, same_on_batch).floor().clamp(max=float(self.regions[1]))
        return {"shift": shift, "regions": regions}


@register(
    aug_id=AugId.FOV_EXTENSION,
    backend=Backend.GPU,
    group=AugType.GEO,
    smoke_kwargs={"shift": 0.5, "regions": (2, 4), "layers": 2},
)
class RandomFOVExtensionGPU(RigidAffineAugmentationBase3D):
    """Shift the scan inside the patch and fill everything outside it with EM-synthesised background.

    Per selected sample:

    1. Image and segmentation move by the same whole-voxel shift, up to `shift` of the patch
       per axis. At `shift=1.0` the spine can leave the window entirely.
    2. The *fill region* is every voxel that is not scan: the padding (every channel exactly
       `pad_value`, which is what nnU-Net pads with) plus whatever the shift vacated.
    3. The patch's own background -- label 0, more than `margin` voxels from any label, not
       on the padding's interpolated edge -- is split into a random 3 to 10 (`regions`)
       intensity regions by EM, as SynthSeg's label completion splits its background.
    4. The region map is post-processed: a 3x3x3 majority filter removes the EM speckle, and
       a random diffeomorphic field (SynthSeg's) deforms it.
    5. `layers` randomly rotated, scaled, mirrored and translated copies of it are composited
       into the fill region; any gap left takes a neighbour's region.
    6. Each fill voxel is drawn from its region's Gaussian, measured on the patch itself,
       with spatially correlated texture, and blurred at SynthSeg's atlas-resolution sigma.
       Its label is 0.

    Everything outside the fill region is the shifted input, bit for bit. A sample with no
    usable background still shifts and keeps the fill at `pad_value`, so image and mask agree.
    Without a segmentation (an image-only pipeline) the background is every non-padding voxel.

    **Row selection is always per sample**, whatever the pipeline's `same_on_batch` says.
    Under the paper config the trainer forces `same_on_batch=True` onto every child, which
    would make `p` one coin per batch and could empty both samples of a step of spine.

    Place it before every intensity transform, which `PIPELINE_ORDER` does, so the fill is
    augmented like real tissue. A label-to-image generator (SynthSeg, PALETTE synthesis)
    later in the pipeline regenerates the fill from label 0 and overwrites it.

    Args:
        shift: maximum shift as a fraction of the patch, one value or one per `(d, h, w)`
            axis, drawn uniformly in `[-shift, shift]`. 0 fills the existing padding only.
        regions: `(min, max)` number of EM regions, drawn per sample.
        margin: voxels between any label and the nearest background voxel used as a source.
        layers: warped copies of the region map composited into the fill region.
        pad_value: the intensity that marks padding.
    """

    def __init__(
        self,
        shift: float | Sequence[float] = 0.5,
        regions: Sequence[int] = (3, 10),
        margin: int = 4,
        layers: int = 6,
        pad_value: float = 0.0,
        p: float = 0.5,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=False, keepdim=keepdim)
        shifts = (float(shift),) * 3 if isinstance(shift, (int, float)) else tuple(float(s) for s in shift)
        if len(shifts) != 3 or not all(0.0 <= s <= 1.0 for s in shifts):
            raise ValueError(f"shift must be one value or three, each in [0, 1]. Got {shift!r}.")
        if len(regions) != 2 or not 1 <= int(regions[0]) <= int(regions[1]):
            raise ValueError(f"regions must be (min, max) with 1 <= min <= max. Got {regions!r}.")
        if int(margin) < 0 or int(layers) < 1:
            raise ValueError(f"margin must be >= 0 and layers >= 1. Got margin={margin!r}, layers={layers!r}.")
        self.regions = (int(regions[0]), int(regions[1]))
        self.margin = int(margin)
        self.layers = int(layers)
        self.pad_value = float(pad_value)
        self._param_generator = FOVExtensionGenerator3D(shifts, self.regions)

    @property  # type: ignore[override]
    def same_on_batch(self) -> bool:
        """Always False: see the class docstring. The setter ignores what a container writes."""
        return False

    @same_on_batch.setter
    def same_on_batch(self, value: bool) -> None:
        pass

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    @staticmethod
    def _offsets(params: dict[str, Tensor], spatial: Sequence[int]) -> list[list[int]]:
        size = torch.tensor(list(spatial), dtype=torch.float32)
        # One transfer for the whole batch, as in RandomCropTransformGPU.
        return (params["shift"].detach().float().cpu() * size).round().long().tolist()

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        offsets = self._offsets(params, input.shape[2:])
        # fp32 throughout: under the trainer's autocast `warp_volume`'s bmm and the EM
        # matmuls would run in half precision.
        with torch.autocast(device_type=input.device.type, enabled=False):
            image = input.float()
            n = image.shape[0]
            d, h, w = image.shape[2:]
            padding = (image == self.pad_value).all(dim=1, keepdim=True)
            seg = segmentation_from(params)
            labelled = torch.zeros_like(padding) if seg is None else _foreground(seg, dim=1).unsqueeze(1)

            image = _shift(image, offsets, self.pad_value)
            fill = _shift(padding, offsets, True)
            labelled = _shift(labelled, offsets, False)

            sources = ~(_dilate(labelled, self.margin) | _dilate(fill, _SEAM))
            k_max = self.regions[1]
            n_regions = params["regions"].to(image.device).long()
            em = _em_regions(image[:, 0].flatten(1), sources.flatten(1).float(), n_regions, k_max).view_as(padding)

            # Majority filter over the 3x3x3 neighbourhood, among source voxels only.
            votes = (em == torch.arange(k_max, device=em.device).view(1, -1, 1, 1, 1)).float() * sources
            atlas = F.avg_pool3d(votes, 3, stride=1, padding=1).argmax(dim=1, keepdim=True)
            atlas = torch.where(sources, atlas, -1)

            displacement = random_svf_field(n, (d, h, w), image.device)
            deformed = warp_volume((atlas + 1).float(), displacement=displacement, interp="nearest").round().long() - 1
            layout = _extend_regions(torch.where(fill, -1, deformed), fill, self.layers)

            rendered = _render(image, atlas, layout, k_max)
            has_sources = sources.flatten(1).any(dim=1).view(n, 1, 1, 1, 1)
            rendered = torch.where(has_sources, rendered, self.pad_value)
            out = torch.where(fill, rendered, image)
        return out.to(input.dtype)

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return _shift(input, self._offsets(params, input.shape[2:]), 0)
