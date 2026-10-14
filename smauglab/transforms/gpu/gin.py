"""GIN: Global Intensity Non-linear augmentation.

An image pushed through a shallow convolutional network whose weights are drawn afresh on
every call, blended back towards the original by a per-sample `alpha`, and rescaled so the
result carries the same energy as the input. Because the network is random rather than
trained, the family of intensity mappings it realises is far wider than any fixed curve --
which is the point: it is a single-source domain-generalisation augmentation, meant to make
a model indifferent to the appearance of the scanner it was trained on.

Based on:
    Ouyang, C., Chen, C., Li, S., Li, Z., Qin, C., Bai, W., & Rueckert, D. (2022).
    Causality-inspired single-source domain generalization for medical image segmentation.
    IEEE Transactions on Medical Imaging, 42(4), 1095-1106. DOI 10.1109/TMI.2022.3224067

Ported from the authors' 3-D implementation (`models/imagefilter3d.py` in
https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization) and the
cleaner nnU-Net-oriented restatement of it in `dg_tta/gin.py`
(https://github.com/multimodallearning/DG-TTA). Both are MIT licensed.

Two deliberate departures from upstream, both argued at their call sites:

* **Reflect padding** rather than zeros, because a zero-padded 3x3x3 convolution darkens
  the patch border and an nnU-Net patch is an interior crop, not a whole image.
  `padding_mode="zeros"` restores upstream's behaviour exactly.
* **float32 as a floor**, because the Frobenius renormalisation overflows float16 often
  enough to matter and does so silently. A float64 input keeps its precision. See the
  comment in `apply_transform`.
"""

import contextlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from smauglab.registry import AugId, AugType, Backend, register
from smauglab.transforms.gpu.base import ImageOnlyTransform, autocast_active, segmentation_from
from smauglab.transforms.gpu.contrast import _channel_stats, _check_channel, _restore_stats, _select_and_check
from smauglab.transforms.rng import shared_rand

#: Upstream's guard against dividing by a vanishing norm, kept at its published value.
#: Also what makes an all-zero patch come out all-zero instead of NaN.
_NORM_EPS = 1e-5

#: The two output normalisations, and the two paddings. Spelled out so a config typo is
#: rejected at construction rather than silently falling through to the other branch.
_OUT_NORMS = ("frob", "none")
_PADDING_MODES = ("reflect", "zeros")


def _draw_kernel_sizes(kernel_sizes: Sequence[int], n_layers: int) -> list[int]:
    """One kernel size per layer, drawn from torch's RNG in a single call.

    `shared_choice` is the obvious helper and the one `RandomRandConvGPU` uses, but it
    ends in `float(...item())` -- a read back to the host -- so calling it once per layer
    costs `n_layers` stalls per forward. Bucketing one `shared_rand` draw costs one, and
    keeps the draw somewhere `torch.manual_seed` reaches and every DDP rank agrees on.
    """
    draw = shared_rand((n_layers,), torch.device("cpu"))
    # `shared_rand` is [0, 1), so the index is already in range; the clamp is belt and braces.
    index = (draw * len(kernel_sizes)).long().clamp_(max=len(kernel_sizes) - 1)
    return [int(kernel_sizes[i]) for i in index.tolist()]


def _gin_stack(
    work: Tensor,
    widths: Sequence[int],
    kernel_sizes: Sequence[int],
    negative_slope: float,
    padding_mode: str,
    same_on_batch: bool,
) -> Tensor:
    """Push a `[B, C, D, H, W]` volume through one freshly drawn conv cascade.

    `widths` is the channel count at every layer boundary, so it holds `n_layers + 1`
    entries and starts and ends at `C`; `kernel_sizes` holds the one size already drawn
    for each layer, not the set they were drawn from. Weights and biases are raw `N(0, 1)`
    with no fan-in scaling -- the paper's `N(0, I)`, and safe only because the caller
    renormalises the output; see the class docstring. Everything is drawn in `work.dtype`,
    which the caller has already floored at float32.

    The batch is folded into the group axis: with the volume viewed as
    `[1, B*Cin, D, H, W]`, a weight of `[B*Cout, Cin, k, k, k]` and `groups=B`, group `b`
    sees channels `[b*Cin : (b+1)*Cin]` and nothing else, so every sample gets its own
    independent network out of a single convolution.

    `same_on_batch` repeats one drawn network across the groups rather than switching to
    `groups=1`. Two reasons: the rows then come out bitwise identical, which is what the
    flag promises; and `groups=1` is the slower route for the shapes this actually runs.
    Measured on an idle A40 over a `[2, 1, 128, 128, 128]` batch with `k=3`, grouped
    against shared: `1 -> 2` channels 0.32 ms against 1.10 ms, `2 -> 2` 1.67 against 1.30,
    `2 -> 1` 1.69 against 7.33 -- so about 5.4 ms against 11.0 ms over the default
    cascade's four layers. `AugmentationSequentialCustom` forces `same_on_batch=True` onto
    every child, so that is the shipped path and not the exotic one.

    `kernels.depthwise_conv3d` cannot serve here: it is depthwise by construction
    (`groups = B*C`, one input channel per filter), and the interior layers map
    `interm -> interm` within a group.
    """
    batch = work.shape[0]
    spatial = work.shape[2:]
    x = work.reshape(1, batch * widths[0], *spatial)
    last = len(kernel_sizes) - 1

    for layer, k in enumerate(kernel_sizes):
        channels_in, channels_out = widths[layer], widths[layer + 1]
        if same_on_batch:
            weight = torch.randn((channels_out, channels_in, k, k, k), device=work.device, dtype=work.dtype)
            weight = weight.repeat(batch, 1, 1, 1, 1)
            bias = torch.randn((channels_out,), device=work.device, dtype=work.dtype).repeat(batch)
        else:
            weight = torch.randn((batch * channels_out, channels_in, k, k, k), device=work.device, dtype=work.dtype)
            bias = torch.randn((batch * channels_out,), device=work.device, dtype=work.dtype)

        if k > 1:
            # Skipped entirely for k == 1, where upstream's `padding=k//2` is 0 anyway and
            # the call would be a full-volume copy for nothing -- four of them per forward
            # at the default `kernel_sizes`.
            pad = [k // 2] * 6
            x = F.pad(x, pad, mode="reflect") if padding_mode == "reflect" else F.pad(x, pad)
        x = F.conv3d(x, weight, groups=batch)
        # One scalar per output plane, which is upstream's `[out_channel*nb, 1, 1, 1]`
        # written as the view it broadcasts to rather than left to right-alignment.
        x = x + bias.view(1, -1, 1, 1, 1)
        if layer != last:
            x = F.leaky_relu(x, negative_slope)

    return x.reshape(batch, widths[-1], *spatial)


@register(
    aug_id=AugId.GIN,
    backend=Backend.GPU,
    group=AugType.TA,
)
class RandomGINGPU(ImageOnlyTransform):
    """GIN: a cascade of randomly drawn convolutions, renormalised to the input's energy.

    If the image is a torch Tensor, it is expected to have [N, C, X, Y, Z] shape.

    The network is redrawn on every call, so there is no state and nothing to seed beyond
    torch's own RNG. Each layer's kernel size is drawn from `kernel_sizes`, its weights and
    bias from `N(0, 1)` with no fan-in scaling, and `leaky_relu` follows every layer but
    the last. The result is blended with the original at a per-sample `alpha` and then
    rescaled so its Frobenius norm matches the input's.

    That final rescaling is what makes the unscaled weights safe, and makes the output
    scale *exactly* independent of `n_layers`, `kernel_sizes` and `interm_channels`: the
    per-sample output RMS matches the input's to within 4.1e-07 relative for every
    combination of those. `RandomRandConvGPU` needs its `1/sqrt(k**3)` kernel scaling precisely because it
    has no such renormalisation. Note the invariant is on RMS and not on the standard
    deviation -- the biases and the leaky_relu move energy into the DC term, so the
    standard deviation still varies by a factor of about 1.3 across those settings.

    There is no `mix_prob` here, unlike its siblings: `alpha_range` already *is* the
    probability-weighted blend back towards the original, and a `mix_prob` on top would
    blend twice.

    Args:
        n_layers (int): Convolution layers in the cascade. Default is 4, the paper's value;
            the ablation there finds one or two too weak to imitate a domain shift and many
            more unrealistically aggressive. `n_layers=1` degenerates to a single linear
            `C -> C` convolution and ignores `interm_channels`.
        interm_channels (int): Width of the hidden layers. Default is 2, the paper's value.
        kernel_sizes (list of int): Odd kernel sizes to draw from, one draw per layer.
            Default is (1, 3); upstream warns against going large with several layers, and
            a size above the thinnest patch axis will make reflect padding raise.
        negative_slope (float): leaky_relu slope between layers. Default is 0.01, which is
            what upstream's bare `F.leaky_relu` uses.
        alpha_range (tuple of float): Range the blend weight is drawn from, where 0 keeps
            the original and 1 takes the network's output whole. Default is (0.0, 1.0),
            the paper's U(0, 1).
        out_norm (str): "frob" to rescale the output to the input's Frobenius norm, "none"
            to leave it alone. Default is "frob". Leaving it off makes the output scale
            track the configuration, so pair it with `retain_stats`.
        padding_mode (str): "reflect" or "zeros". Default is "reflect"; see the module
            docstring for why that differs from upstream.
        apply_to_channel (list of int): Channel indices the network maps. They are mapped
            *jointly* -- one network with `len(apply_to_channel)` inputs and outputs, so it
            mixes across channels as the paper intends, and the energy match is over the
            selected channels together rather than one at a time. Identical to a
            per-channel loop for single-channel data. Default is [0].
        retain_stats (bool): If True, restore the original per-sample mean and std
            afterwards. Default is False.
        same_on_batch (bool): Draw one network, one kernel size per layer and one alpha for
            the whole batch. Default is False.
        in_seg (float): Probability of restricting the result to inside the segmentation.
        out_seg (float): Probability of restricting the result to outside it.
        mix_in_out (bool): Apply to a random subset of the mask's classes rather than all.
        p (float): Probability of applying the transform. Default is 1.0.
        p_batch (float): Probability of applying the transform to the batch. Default is 1.0.
        keepdim (bool): Whether to keep the number of dimensions. Default is True.

    Returns:
        Tensor: The image with the selected channels replaced by the GIN output.

    """

    def __init__(
        self,
        n_layers: int = 4,
        interm_channels: int = 2,
        kernel_sizes: Sequence[int] = (1, 3),
        negative_slope: float = 0.01,
        alpha_range: tuple[float, float] = (0.0, 1.0),
        out_norm: str = "frob",
        padding_mode: str = "reflect",
        apply_to_channel: Sequence[int] = (0,),
        retain_stats: bool = False,
        same_on_batch: bool = False,
        in_seg: float = 0.0,
        out_seg: float = 0.0,
        mix_in_out: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if n_layers < 1:
            raise ValueError(f"n_layers must be at least 1, got {n_layers}. A zero-layer cascade is the identity.")
        if interm_channels < 1:
            raise ValueError(f"interm_channels must be at least 1, got {interm_channels}.")
        if len(kernel_sizes) == 0:
            raise ValueError("kernel_sizes must name at least one size to draw from.")
        for k in kernel_sizes:
            if k < 1 or k % 2 == 0:
                raise ValueError(
                    f"kernel_sizes must be odd and positive, got {list(kernel_sizes)}. An even size cannot keep the patch shape."
                )
        if out_norm not in _OUT_NORMS:
            raise ValueError(f"out_norm must be one of {_OUT_NORMS}, got {out_norm!r}.")
        if padding_mode not in _PADDING_MODES:
            raise ValueError(f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode!r}.")
        if alpha_range[0] > alpha_range[1]:
            raise ValueError(f"alpha_range must be (low, high), got {tuple(alpha_range)}.")
        # Harmless on every sibling, where a repeated channel is just a second application.
        # Here it changes the network: `len(apply_to_channel)` is the cascade's input width,
        # so (0, 0) builds a two-input net and then scatters both of its outputs into
        # channel 0, last write winning.
        if len(set(apply_to_channel)) != len(list(apply_to_channel)):
            raise ValueError(
                f"apply_to_channel must not repeat a channel, got {list(apply_to_channel)}. "
                "The channels are mapped jointly, so a duplicate changes the network's width."
            )
        self.n_layers = n_layers
        self.interm_channels = interm_channels
        self.kernel_sizes = kernel_sizes
        self.negative_slope = negative_slope
        self.alpha_range = alpha_range
        self.out_norm = out_norm
        self.padding_mode = padding_mode
        self.apply_to_channel = apply_to_channel
        self.retain_stats = retain_stats
        self.in_seg = in_seg
        self.out_seg = out_seg
        self.mix_in_out = mix_in_out

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        # A clone, not the caller's tensor: this writes channels back with
        # `input[:, c] = ...` and kornia passes the caller's own tensor through when
        # every sample applies.
        input = input.clone()
        seg_mask = segmentation_from(params)
        channels = [int(c) for c in self.apply_to_channel]
        for c in channels:
            # Before the gather, so a config typo names the key it came from rather than
            # raising a bare IndexError out of advanced indexing.
            _check_channel(self, c, input.shape[1])
        batch = input.shape[0]

        # float32 as a floor, for a failure that is otherwise invisible. The trainer runs
        # this pipeline inside `autocast`, so `F.conv3d` would hand back float16; the
        # kernels are raw N(0, 1) with no fan-in scaling, and `||mixed||_F` then passed
        # float16's 65504 in 23 of 150 draws at these defaults on a 1x128^3 patch, 48 of
        # 150 on 2x192^3, and 105 of 150 with kernel_sizes=(1, 3, 5, 7). The norm comes
        # back `inf`, `1 / (inf + eps)` is exactly 0, and the result is an all-zero patch
        # that `torch.isfinite(...).all()` reports as fine -- so neither
        # `_select_and_check` below nor the trainer's non-finite-loss guard ever sees it,
        # and the network trains on a blank volume.
        #
        # Guarded on `autocast_active()` rather than entered unconditionally: `autocast`
        # on mps complains that it is not implemented even with `enabled=False`, which is
        # the same reason train_step wraps its own region in a conditional.
        guard = torch.autocast(input.device.type, enabled=False) if autocast_active() else contextlib.nullcontext()
        with guard:
            # Advanced indexing already copies; the promotion is the other half of the
            # forcing, for a caller that hands over float16 storage. `promote_types`
            # rather than `.float()` so the floor is float32 and not the ceiling -- a
            # float64 input would otherwise be quietly halved in precision.
            work = input[:, channels].to(torch.promote_types(input.dtype, torch.float32))
            widths = [len(channels), *[self.interm_channels] * (self.n_layers - 1), len(channels)]
            net = _gin_stack(
                work,
                widths=widths,
                kernel_sizes=_draw_kernel_sizes(self.kernel_sizes, self.n_layers),
                negative_slope=self.negative_slope,
                padding_mode=self.padding_mode,
                same_on_batch=self.same_on_batch,
            )

            # One alpha per sample, shared across the selected channels. A single draw
            # under `same_on_batch`, which then broadcasts over the batch axis.
            low, high = self.alpha_range
            draws = torch.rand(1 if self.same_on_batch else batch, device=work.device, dtype=work.dtype)
            alpha = (draws * (high - low) + low).reshape(-1, 1, 1, 1, 1)
            mixed = alpha * net + (1.0 - alpha) * work

            if self.out_norm == "frob":
                # Upstream's `torch.norm(x.reshape(B, C, -1), dim=(-1, -2), p="fro")`
                # reduces over the channel axis as well as the spatial ones, so this is
                # one scalar per sample rather than one per channel -- deliberately, and
                # bitwise the same value as the reshape below. Channels therefore share a
                # scale factor, which couples modalities held at different scales when
                # `apply_to_channel` names more than one.
                energy_in = work.reshape(batch, -1).norm(dim=1).reshape(-1, 1, 1, 1, 1)
                energy_out = mixed.reshape(batch, -1).norm(dim=1).reshape(-1, 1, 1, 1, 1)
                # The factor is formed first. Upstream writes `mixed * (1 / (out + eps)) *
                # in`, which walks the volume twice and allocates a second copy of it; the
                # scale is one number per sample, so it costs nothing to combine.
                mixed = mixed * (energy_in / (energy_out + _NORM_EPS))

            out = mixed.to(input.dtype)
            for index, c in enumerate(channels):
                orig = input[:, c]
                x = out[:, index]
                if self.retain_stats:
                    x = _restore_stats(x, _channel_stats(orig))
                # A channel rejected here reverts on its own while the others keep the GIN
                # output, which gives up the joint energy match for that one call. That is
                # what every sibling does with a rejected channel, and a NaN volume is the
                # worse of the two outcomes.
                checked = _select_and_check(self, orig, x, seg_mask, " (gin)")
                if checked is None:
                    continue
                input[:, c] = checked

        return input
