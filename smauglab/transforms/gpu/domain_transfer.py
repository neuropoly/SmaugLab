"""
RandomDomainTransferGPU — domain-randomization augmentation.

Re-renders an image as a *different dataset/sequence* (e.g. a CT slice made to look like
spinegan-inphase MRI) using a precomputed per-class intensity-transfer LUT bank, so a segmentation
model sees its training data spanning many sequence appearances and becomes sequence-invariant.

With probability ``p`` (handled by the kornia base), each image is mapped from its known
``source_label`` cluster to a RANDOMLY chosen target cluster via class-wise 1-D LUTs gathered on the
GPU and blended by Gaussian-softened one-hot class weights (so class boundaries stay smooth).
The segmentation/anatomy is left unchanged — only intensities move.

The LUT bank (built offline by embeddaug/analysis/playground/build_transfer_bank.py) encodes the
best transfer per (source→target) pair: exact paired E[t|s] where the sequences are co-registered,
else mass-order GMM (captures e.g. CT-bright bone → MR-dark) on bone/disc + monotonic marginal
elsewhere. Background is transferred too.

Extra variability knobs (all default to the original single-target behaviour, so existing
configs are unchanged):
  * ``any_source`` — domain-randomisation mode: ignore the input's nominal ``source_label`` and
    draw from EVERY ``(X→Y)`` transfer in the bank (e.g. apply a ``t2w→fat`` curve to a CT). The
    input is percentile-normalised to ``[0,1]`` first, so any LUT is a valid bounded remap; this
    multiplies the appearance variety (~S·(T-1) curves instead of T) at the cost of realism.
  * ``blend_targets`` / ``blend_concentration`` — mix this many target LUTs per draw with
    random Dirichlet weights, turning the handful of discrete targets into a *continuum* of
    intermediate appearances (so no two augmentations repeat). Keep this small (2–3): blending
    across *all* targets regresses each LUT toward the mean and actually *reduces* spread.
  * ``p_class_mix`` — probability that each anatomical class independently draws its own
    target/blend (chimeric domains, e.g. CT-like background with T2-like bone).
  * ``bias_field_std`` / ``bias_scale`` — smooth multiplicative bias field on the output; this
    is the strongest contributor to per-sample appearance spread.
  * ``p_spatial_mix`` / ``spatial_mix_scale`` / ``spatial_mix_gain`` — blend two independent
    domain transfers across space via a smooth random field, so different regions of one patch
    look like different target sequences (a spatial effect that survives the z-score restore).
"""

import math
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Dirichlet
from torch.nn import functional as F

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform
from smauglab.transforms.kernels import gaussian_blur3d, random_bias_field3d

# Default transfer LUT bank (built by embeddaug/analysis/playground/build_transfer_bank.py).
# The transfer LUT bank is a multi-hundred-MB artefact built offline by
# embeddaug/analysis/playground/build_transfer_bank.py, so it is not shipped in the
# wheel. Point this env var at it; there is deliberately no baked-in default, because
# the previous one was an absolute path into a single machine's NAS home directory and
# silently made this transform unusable for everyone else.
BANK_PATH_ENV_VAR = "SMAUGLAB_DOMAIN_BANK"


def resolve_bank_path(bank_path: str | None = None) -> str:
    """Locate the domain-transfer LUT bank, explicit argument first, then the env var.

    Raises with the fix spelled out rather than letting np.load report a bare
    FileNotFoundError on a path the caller never chose.
    """
    resolved = bank_path or os.environ.get(BANK_PATH_ENV_VAR)
    if not resolved:
        raise FileNotFoundError(
            "RandomDomainTransferGPU needs a transfer LUT bank. Pass bank_path=..., set it in "
            f"the config, or export {BANK_PATH_ENV_VAR}=/path/to/domain_transfer_bank.npz "
            "(built by embeddaug/analysis/playground/build_transfer_bank.py)."
        )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Domain transfer bank not found at {resolved!r} (from {BANK_PATH_ENV_VAR} or bank_path).")
    return resolved


def _gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the 3 spatial dims of [N, C, D, H, W].

    Delegates to the shared implementation. That one pads with `reflect` rather than
    the `replicate` used here, and takes its radius from `ceil(3*sigma)` rather than
    `round`, so the kernel can be one tap wider -- see smauglab/transforms/kernels.py.
    """
    if sigma <= 0:
        return x
    return gaussian_blur3d(x, float(sigma))


def _random_bias_field3d(shape, std: float, scale: float, device, dtype) -> torch.Tensor:
    """Smooth positive multiplicative bias field over a ``[D, H, W]`` volume.

    Thin adapter over :func:`smauglab.transforms.kernels.random_bias_field3d`, which
    returns ``[batch, channels, D, H, W]``; this module wants the bare volume. The
    implementation used to be written out here as well -- line for line the same as
    ``synthseg/functional.py::bias_field`` -- under a comment saying it was "kept local
    so this module stays self-contained".
    """
    return random_bias_field3d(tuple(shape), std, scale, device, dtype)[0, 0]


def _random_smooth_field01(shape, scale: float, gain: float, device, dtype) -> torch.Tensor:
    """Smooth low-frequency random field in ``[0, 1]`` over a ``[D, H, W]`` volume.

    Same coarse-grid → trilinear-upsample idea as :func:`_random_bias_field3d`, but squashed
    through a sigmoid (with a random offset) so it bounds to ``[0, 1]`` and forms coherent
    soft regions — used to spatially interpolate two domain transfers. ``gain`` controls how
    distinct the regions are (higher → sharper boundaries); the ``U(-2, 2)`` offset varies the
    balance between the two domains per draw.
    """
    d, h, w = shape
    small = [max(2, math.ceil(s * scale)) for s in (d, h, w)]
    field = torch.randn(1, 1, *small, device=device, dtype=dtype)
    field = F.interpolate(field, size=(d, h, w), mode="trilinear", align_corners=True)[0, 0]
    offset = torch.rand((), device=device, dtype=dtype) * 4.0 - 2.0
    return torch.sigmoid(gain * field + offset)


@register(
    aug_id=AugId.DOMAIN_TRANSFER,
    backend=Backend.GPU,
    group=AugType.TA,
    external_asset=BANK_PATH_ENV_VAR,
    smoke_kwargs={"any_source": True},
)
class RandomDomainTransferGPU(ImageOnlyTransform):
    """Randomly transfer an image's appearance to another sequence/cluster (see module docstring)."""

    def __init__(
        self,
        bank_path: str | None = None,
        source_label: str | None = None,
        targets: list[str] | None = None,
        include_self: bool = True,
        any_source: bool = False,
        sigma: float = 2.0,
        apply_to_channel: list[int] | None = None,
        zscore_io: str = "auto",
        pct: float = 1.0,
        blend_targets: int = 1,
        blend_concentration: float = 1.0,
        p_class_mix: float = 0.0,
        bias_field_std: float = 0.0,
        bias_scale: float = 0.03,
        p_spatial_mix: float = 0.0,
        spatial_mix_scale: float = 0.03,
        spatial_mix_gain: float = 3.0,
        same_on_batch: bool = False,
        p: float = 0.2,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        if apply_to_channel is None:
            apply_to_channel = [0]
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        data = np.load(resolve_bank_path(bank_path), allow_pickle=True)
        self.labels: list[str] = [str(x) for x in data["labels"].tolist()]
        self.L = int(data["L"])
        self.num_classes = int(data["num_classes"])
        self.any_source = bool(any_source)
        self.source_label = source_label

        if self.any_source:
            # Domain-randomization mode: ignore the input's nominal source domain and draw from
            # EVERY (X→Y) transfer in the bank (e.g. apply spider_t2w→fat to a CT image). The
            # input is percentile-normalised to [0,1] first, so any LUT is a valid bounded remap;
            # this multiplies the appearance variety (≈ S·(T-1) curves instead of T).
            keys = []
            for k in data.files:
                if "__" not in k:
                    continue
                x, y = k.split("__", 1)
                if x not in self.labels or y not in self.labels:
                    continue
                if not include_self and x == y:  # identity transfers
                    continue
                if targets is not None and y not in targets:  # optional: restrict the target domain
                    continue
                keys.append(k)
            self.targets = keys
            if not self.targets:
                raise ValueError("no valid (source→target) pairs for domain transfer")
            bank = np.stack([data[k] for k in self.targets]).astype(np.float32)
        else:
            if source_label is None:
                raise ValueError("source_label is required (the cluster/sequence the data comes from)")
            if source_label not in self.labels:
                raise ValueError(f"source_label '{source_label}' not in bank labels {self.labels}")
            tgts = list(targets) if targets is not None else list(self.labels)
            if not include_self:
                tgts = [t for t in tgts if t != source_label]
            self.targets = [t for t in tgts if t in self.labels]
            if not self.targets:
                raise ValueError("no valid target clusters for domain transfer")
            # [T, num_classes, L] — LUTs source_label → each target
            bank = np.stack([data[f"{source_label}__{t}"] for t in self.targets]).astype(np.float32)
        self.register_buffer("lut_bank", torch.from_numpy(bank))
        self.sigma = float(sigma)
        self.apply_to_channel = apply_to_channel
        # The transfer LUTs live in percentile-scaled [0,1] space, but training feeds
        # z-score-normalized patches (nnUNetTrainerDAExtGPU). zscore_io controls the round-trip:
        #   "auto"  – treat as z-score iff values fall outside [0,1]
        #   "always"– always assume z-score input ; "never" – assume input already in [0,1]
        self.zscore_io = zscore_io
        self.pct = float(pct)
        # --- added variability (all default to the original single-target behaviour) ---
        # blend_targets>1 mixes that many target LUTs per draw (continuous domain blending);
        # p_class_mix lets each class draw its own target/blend (chimeric domains);
        # bias_field_std>0 adds a smooth multiplicative spatial inhomogeneity.
        self.blend_targets = int(blend_targets)
        self.blend_concentration = float(blend_concentration)
        self.p_class_mix = float(p_class_mix)
        self.bias_field_std = float(bias_field_std)
        self.bias_scale = float(bias_scale)
        # spatially-varying domain mixing: two domain transfers blended across space by a
        # smooth random field, so different regions look like different target sequences.
        self.p_spatial_mix = float(p_spatial_mix)
        self.spatial_mix_scale = float(spatial_mix_scale)
        self.spatial_mix_gain = float(spatial_mix_gain)

    def _sample_blended_luts(self, lut_bank: Tensor, n_seg_c: int, device) -> tuple[Tensor, bool]:
        """Build the per-class LUTs to use for one sample.

        Returns ``(lut_used, per_class)`` where ``lut_used`` is ``[n_draws, NC, L]`` with
        ``n_draws == n_seg_c`` if per-class hybridisation fired, else ``1``. Each draw is a
        random convex combination of ``K = blend_targets`` target LUTs (``Dirichlet`` weights).
        Defaults (``blend_targets=1, p_class_mix=0``) reproduce the original single random
        target pick exactly, including the RNG call, so seeded behaviour is unchanged.
        """
        T = lut_bank.shape[0]
        K = T if (self.blend_targets <= 0 or self.blend_targets > T) else self.blend_targets
        per_class = (self.p_class_mix > 0.0) and (float(torch.rand((), device=device)) < self.p_class_mix)

        if K == 1 and not per_class:  # fast path == original behaviour
            ti = int(torch.randint(T, (1,), device=device).item())
            return lut_bank[ti : ti + 1], False  # [1, NC, L]

        n_draws = n_seg_c if per_class else 1
        beta = torch.zeros(n_draws, T, device=device, dtype=lut_bank.dtype)
        for d in range(n_draws):
            idx = torch.randperm(T, device=device)[:K]
            if K == 1:
                wd = torch.ones(1, device=device, dtype=lut_bank.dtype)
            else:
                conc = torch.full((K,), self.blend_concentration, device=device, dtype=torch.float32)
                wd = Dirichlet(conc).sample().to(lut_bank.dtype)
            beta[d, idx] = wd
        lut_used = torch.einsum("dt,tcl->dcl", beta, lut_bank)  # [n_draws, NC, L]
        return lut_used, per_class

    @staticmethod
    def _accumulate(lut_used: Tensor, per_class: bool, w_b: Tensor, il: Tensor, ih: Tensor, xf: Tensor, n_seg_c: int) -> Tensor:
        """Class-weighted LUT interpolation ``Σ_c w_c · interp(LUT_c, x)`` → ``[D, H, W]``."""
        acc = torch.zeros_like(xf)
        for c in range(n_seg_c):
            lut_c = lut_used[c if per_class else 0, c]  # [L]
            acc += w_b[c] * (lut_c[il] * (1 - xf) + lut_c[ih] * xf)
        return acc

    def _is_zscore(self, x: Tensor) -> bool:
        if self.zscore_io == "always":
            return True
        if self.zscore_io == "never":
            return False
        return bool((x.min() < -1e-3) or (x.max() > 1.0 + 1e-3))

    def _to_unit(self, x: Tensor) -> Tensor:
        """Map a (z-scored) image into the LUT's [0,1] domain via percentile scaling (clip),
        matching how the bank's source histograms were normalised."""
        flat = x.reshape(-1).float()
        if flat.numel() > 1_000_000:  # cap for torch.quantile
            flat = flat[torch.linspace(0, flat.numel() - 1, 1_000_000, device=x.device).long()]
        lo = torch.quantile(flat, self.pct / 100.0)
        hi = torch.quantile(flat, 1.0 - self.pct / 100.0)
        return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)

    @torch.no_grad()
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        if "seg" not in params:
            return input
        seg = params["seg"]
        if seg.dim() != input.dim():  # accept [N, ...] integer seg → one-hot-ish
            if seg.dim() == input.dim() - 1:
                seg = seg.unsqueeze(1)
            else:
                return input
        if input.dim() != 5:  # this GPU pipeline is 3D: [N, C, D, H, W]
            return input

        out = input.clone()
        L, NC = self.L, self.num_classes
        # nn.Module.__getattr__ is typed to return Tensor | Module, so a registered
        # buffer reads back as the union however it was registered. Narrowed here
        # rather than at each _sample_blended_luts call below.
        lut_bank = self.lut_bank
        assert isinstance(lut_bank, Tensor)
        lut_bank = lut_bank.to(input.device)
        n_seg_c = min(seg.shape[1], NC)

        # soft per-class weights from Gaussian-blurred one-hot masks (Σ_c w_c ≈ 1)
        w = _gaussian_blur3d(seg[:, :n_seg_c].float(), self.sigma)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        N = input.shape[0]
        for ch in self.apply_to_channel:
            for b in range(N):
                x = input[b, ch]  # [D, H, W] — may be z-scored
                zmode = self._is_zscore(x)
                if zmode:  # remember scale, map into LUT's [0,1] domain
                    mu, sd = x.mean(), x.std().clamp_min(1e-6)
                    x01 = self._to_unit(x)
                else:
                    x01 = x.clamp(0.0, 1.0)

                xq = x01 * (L - 1)
                il = xq.floor().long().clamp(0, L - 1)
                ih = (il + 1).clamp(0, L - 1)
                xf = xq - il.float()

                # randomly chosen target LUT(s), optionally blended across targets and/or
                # hybridised per class (see _sample_blended_luts). With p_spatial_mix, two
                # independent domain transfers are blended across space by a smooth field, so
                # different regions look like different target sequences.
                spatial = (self.p_spatial_mix > 0.0) and (float(torch.rand((), device=input.device)) < self.p_spatial_mix)
                if spatial:
                    lutA, pcA = self._sample_blended_luts(lut_bank, n_seg_c, input.device)
                    lutB, pcB = self._sample_blended_luts(lut_bank, n_seg_c, input.device)
                    accA = self._accumulate(lutA, pcA, w[b], il, ih, xf, n_seg_c)
                    accB = self._accumulate(lutB, pcB, w[b], il, ih, xf, n_seg_c)
                    a = _random_smooth_field01(x01.shape, self.spatial_mix_scale, self.spatial_mix_gain, input.device, x01.dtype)
                    acc = (1.0 - a) * accA + a * accB
                else:
                    lut_used, per_class = self._sample_blended_luts(lut_bank, n_seg_c, input.device)
                    acc = self._accumulate(lut_used, per_class, w[b], il, ih, xf, n_seg_c)
                acc = acc.clamp(0.0, 1.0)

                if self.bias_field_std > 0.0:  # smooth multiplicative spatial inhomogeneity
                    field = _random_bias_field3d(acc.shape, self.bias_field_std, self.bias_scale, acc.device, acc.dtype)
                    acc = (acc * field).clamp(0.0, 1.0)

                if zmode:
                    # restore the input's mean/std (retain-stats): the affine rescale preserves the
                    # transferred distribution SHAPE while placing it back in the z-score scale the
                    # network trains on, so the augmented patch matches the rest of the batch.
                    acc = (acc - acc.mean()) / acc.std().clamp_min(1e-6) * sd + mu
                out[b, ch] = acc
        return out
