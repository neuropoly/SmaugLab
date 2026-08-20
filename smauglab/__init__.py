"""SmaugLab -- data augmentation strategies for MRI segmentation training.

Deliberately kept free of imports from `smauglab.transforms`. Pulling the
transforms in here would make a bare `import smauglab` drag in torch, kornia and
batchgeneratorsv2 (several seconds), which every console-script invocation would
then pay for. Import the subpackage you actually need:

    from smauglab.transforms.gpu.transforms import AugTransformsGPU
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smauglab")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
