"""Running a torchio transform over an (image, segmentation) pair.

`cpu/artifact.py` and `cpu/spatial.py` held eleven functions between them --
`aug_motion`, `aug_ghosting`, `aug_spike`, `aug_bias_field`, `aug_blur`, `aug_noise`,
`aug_swap`, `aug_flip`, `aug_affine`, `aug_elastic`, `aug_anisotropy` -- that were the
same seventeen lines each, differing only in which `tio.Random*` they constructed. Now
they are entries in a table and one `apply_tio` call.

That eleven-fold copy had already drifted once: the single-channel branch of
`aug_anisotropy` passed `axis=0` to `tio.LabelMap`, which nothing else did and which
torchio simply stores as unrecognised metadata. It was almost certainly meant to be
`tio.RandomAnisotropy(axes=...)`. Collapsing the copies drops it.
"""

from __future__ import annotations

import gc
import random
from collections.abc import Callable, Mapping
from typing import cast

import torch
import torchio as tio

#: A no-argument factory, so each call builds a freshly seeded torchio transform
#: rather than reusing one instance's sampling state across the run.
TransformFactory = Callable[[], tio.Transform]


def _image_data(subject: tio.Subject, key: str) -> torch.Tensor:
    """The tensor behind one of a Subject's images.

    `tio.Subject` is a dict subclass whose `__getitem__` is typed as returning `object`,
    so `subject[key].data` does not type-check. Attribute access (`subject.image`) is
    what the eleven functions this replaced used, and torchio does synthesise it -- but
    at runtime only, so no checker can see it either. One cast, explained once.
    """
    return cast(tio.Image, subject[key]).data


def apply_tio(transform: tio.Transform, img: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a torchio transform to an image/segmentation pair.

    Two input layouts, both from totalspineseg's augment.py:

    * `img` with two channels is the "step 2" layout -- channel 0 is the image and
      channel 1 an odd-disc segmentation. The second channel goes in as a `LabelMap`
      so torchio resamples it nearest-neighbour rather than interpolating labels.
    * Anything else is a plain image plus its segmentation.

    The explicit `del` and `gc.collect()` are inherited: torchio subjects hold the
    whole volume several times over and these run inside dataloader workers.
    """
    # Images come back through `_image_data`; see its docstring for why neither key
    # nor attribute access type-checks on its own.
    if img.shape[0] == 2:
        subject = transform(
            tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(img[0], dim=0)),
                discs=tio.LabelMap(tensor=torch.unsqueeze(img[1], dim=0)),
                seg=tio.LabelMap(tensor=seg),
            )
        )
        img_out = torch.cat((_image_data(subject, "image"), _image_data(subject, "discs")), dim=0)
        seg_out = _image_data(subject, "seg")
    else:
        subject = transform(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg)))
        img_out, seg_out = _image_data(subject, "image"), _image_data(subject, "seg")
    del subject
    gc.collect()  # Force garbage collection
    return img_out, seg_out


def apply_enabled(
    factories: Mapping[str, TransformFactory],
    img: torch.Tensor,
    seg: torch.Tensor,
    enabled: Mapping[str, bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply each enabled transform in `factories` order, chaining the result."""
    for name, factory in factories.items():
        if enabled.get(name):
            img, seg = apply_tio(factory(), img, seg)
    return img, seg


def select(flags: Mapping[str, bool], random_pick: bool) -> dict[str, bool]:
    """Which of the enabled flags to actually apply.

    With `random_pick`, exactly one of the enabled entries survives; otherwise every
    enabled entry does. Written once because `ArtifactTransform.get_parameters` and
    `SpatialCustomTransform.get_parameters` had the same five lines.
    """
    chosen = dict(flags)
    enabled = [name for name, on in flags.items() if on]
    if random_pick and enabled:
        keep = random.choice(enabled)
        chosen = {name: name == keep for name in flags}
    return chosen
