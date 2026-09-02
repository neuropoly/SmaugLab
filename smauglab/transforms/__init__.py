"""Augmentation transforms, split by execution backend.

`cpu` wraps batchgeneratorsv2 transforms for the dataloader worker; `gpu` wraps
kornia ones for the training step; `synthseg` holds the generative label-to-image
augmentation.

Importing this package is what populates `smauglab.registry`: every augmentation
class carries an `@register(...)` decorator, so the registry is complete once these
modules have been imported and empty before. `registry.load_all()` does exactly
this import, which is why lookups are correct without callers having to know which
module defines what.

Note this is deliberately NOT done from `smauglab/__init__.py`: it pulls in torch,
kornia and batchgeneratorsv2, and a bare `import smauglab` should not have to pay
for that.
"""

from smauglab.transforms.cpu import artifact, contrast, external, fromSeg, spatial  # noqa: F401
from smauglab.transforms.gpu import contrast as gpu_contrast  # noqa: F401
from smauglab.transforms.gpu import domain_transfer  # noqa: F401
from smauglab.transforms.gpu import fromSeg as gpu_fromSeg
from smauglab.transforms.gpu import spatial as gpu_spatial  # noqa: F401
from smauglab.transforms.synthseg import transforms as synthseg_transforms  # noqa: F401
