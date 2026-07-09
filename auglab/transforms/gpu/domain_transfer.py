"""Stub — RandomDomainTransferGPU not implemented in this installation."""
from auglab.transforms.gpu.base import ImageOnlyTransform


class RandomDomainTransferGPU(ImageOnlyTransform):
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "RandomDomainTransferGPU requires a domain bank and is not available in this installation."
        )

    def apply_transform(self, input, params, flags, transform=None):
        return input
