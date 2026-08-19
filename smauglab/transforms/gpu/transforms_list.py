"""The random-order GPU pipelines, and the combinator they are built from.

`AugTransformsGPURandomOrder` and `AugTransformsGPURandomOrderTA` used to carry a
near-verbatim copy each of the ~300-line dispatch ladder in `transforms.py`, which
had already drifted from it (they passed a `crop=` argument no transform accepts,
and ordered SimulateLowRes differently). Both are now three lines: the only thing
that distinguishes them from `AugTransformsGPU` is how the registry's TA and GE
groups get bucketed, which `smauglab.transforms.build` handles.
"""

from typing import Any

import torch
from torch import Tensor, nn

from smauglab.transforms.build import PipelineMode
from smauglab.transforms.gpu.base import ImageOnlyTransform
from smauglab.transforms.gpu.transforms import AugTransformsGPU


class AugTransformsGPURandomOrder(AugTransformsGPU):
    """Geometry in order, then the TA and GE groups each shuffled in their own bucket."""

    mode = PipelineMode.RANDOM_ORDER


class AugTransformsGPURandomOrderTA(AugTransformsGPU):
    """Only the transfer augmentations are bucketed; everything else keeps its order."""

    mode = PipelineMode.RANDOM_ORDER_TA


class RandomChooseXTransformsGPU(ImageOnlyTransform):
    """Randomly choose X transforms to apply from a given list of ImageOnlyTransform transforms (GPU version).

    Args:
        transforms_list: List of initialized ImageOnlyTransform to choose from.
        num_transforms: Number of transforms to randomly select and apply.
        same_on_batch: apply the same transformation across the batch.
        p: probability for applying the X transforms to a batch. This param controls the augmentation
          probabilities batch-wise.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it to the batch
          form ``False``.

    """

    def __init__(
        self,
        transforms_list: list[ImageOnlyTransform],
        num_transforms: int = 1,
        same_on_batch: bool = False,
        p: float = 1.0,
        p_batch: float = 1.0,
        keepdim: bool = True,
        random_order: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if not isinstance(num_transforms, int) or num_transforms < 0:
            raise ValueError(f"num_transforms must be a non-negative int. Got {num_transforms!r}.")
        self.transforms_list = nn.ModuleList(transforms_list)
        self.num_transforms = num_transforms
        self.random_order = random_order

    def _apply_mix(self, x: Tensor, seg: Tensor | None) -> Tensor:
        if self.num_transforms == 0 or len(self.transforms_list) == 0:
            return x

        k = min(self.num_transforms, len(self.transforms_list))
        # sample without replacement
        if self.random_order:
            idx = torch.randperm(len(self.transforms_list), device=x.device)[:k]
        else:
            idx = torch.arange(len(self.transforms_list), device=x.device)[:k]

        child_params: dict[str, Tensor] = {}
        if seg is not None:
            child_params["seg"] = seg

        for j in idx.tolist():
            t = self.transforms_list[j]
            if torch.rand(1, device=x.device, dtype=x.dtype) > t.p:
                continue
            if not hasattr(t, "apply_transform"):
                raise TypeError(f"All transforms must implement apply_transform like ImageOnlyTransform. Got {type(t)}")
            # Most contrast transforms perform their random sampling inside
            # apply_transform, so an empty params dict is all they need. The ones with a
            # kornia `_param_generator` (the spatial transforms) read their draw out of
            # `params` instead, and calling apply_transform directly skips the
            # forward_parameters step that fills it -- they used to raise
            # "params must contain 'scale'" from inside a bucket. Sampling here keeps
            # the bucket usable for both kinds.
            t_params = child_params
            if getattr(t, "_param_generator", None) is not None:
                t_params = {**child_params, **t.forward_parameters(x.shape)}
            t_flags = getattr(t, "flags", {})
            x = t.apply_transform(x, t_params, t_flags, transform=None)
        return x

    @torch.no_grad()  # disable gradients for efficiency
    def apply_transform(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None) -> Tensor:
        seg = params.get("seg")

        if self.same_on_batch:
            return self._apply_mix(input, seg)

        batch_size = input.shape[0]
        # A clone, not `out = input`: the loop writes back through `out[i:i+1]`, so
        # without it the caller's batch is modified in place. Every sibling transform
        # in gpu/spatial.py clones.
        out = input.clone()
        for i in range(batch_size):
            xi = out[i : i + 1]
            seg_i = None
            seg_i = seg[i : i + 1] if seg is not None and isinstance(seg, torch.Tensor) and seg.shape[0] == batch_size else seg
            xi = self._apply_mix(xi, seg_i)
            out[i : i + 1] = xi
        return out
