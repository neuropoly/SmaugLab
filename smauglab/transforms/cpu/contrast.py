from typing import Union

import torch
import torch.nn.functional as F
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform

from smauglab.transforms.kernels import laplace_kernel, scharr_kernels


class InvertedGammaTransform(GammaTransform):
    """Gamma adjustment applied to the inverted image.

    batchgeneratorsv2 expresses this as `GammaTransform(p_invert_image=1)`, which the
    old config spelled as a `GammaTransform_invert` key -- a key with no class behind
    it. A real class keeps the config key 1:1 with a class on this backend too, and
    means the inversion cannot be requested two different ways.
    """

    def __init__(
        self,
        gamma: RandomScalar = (0.7, 1.5),
        synchronize_channels: bool = False,
        p_per_channel: float = 1,
        p_retain_stats: float = 1,
    ):
        super().__init__(
            gamma=gamma,
            p_invert_image=1,
            synchronize_channels=synchronize_channels,
            p_per_channel=p_per_channel,
            p_retain_stats=p_retain_stats,
        )


class _ConvBaseTransform(ImageOnlyTransform):
    """
    Applies a Laplace/Scharr filter to the image to highlight edges.

    Shared implementation. Configs address the per-kernel leaves below, which mirror
    the GPU split so both backends of one augmentation share an `aug_id`.

    Based on https://github.com/spinalcordtoolbox/disc-labeling-playground/blob/main/src/ply/models/transform.py
    """

    def __init__(self, kernel_type: str = "Laplace", absolute: bool = False, retain_stats: bool = False):
        super().__init__()
        if kernel_type not in ["Laplace", "Scharr"]:
            raise NotImplementedError('Currently only "Laplace" and "Scharr" are supported.')
        else:
            self.kernel_type = kernel_type
        self.absolute = absolute
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        # Scharr yields one kernel per spatial direction, Laplace a single kernel;
        # _apply_to_image dispatches on kernel_type to tell the two apart.
        kernel: Union[torch.Tensor, list[torch.Tensor]]
        spatial_dims = len(data_dict["image"].shape) - 1
        if spatial_dims not in (2, 3):
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")
        # Shared with the GPU backend. These tables used to be written out here and
        # again in gpu/contrast.py, which is how the 2-D Scharr x-kernel came to have
        # [-10, 0, -10] as its middle row on this side only -- summing to -20, so not
        # a gradient operator at all. See smauglab/transforms/kernels.py.
        kernel = laplace_kernel(spatial_dims) if self.kernel_type == "Laplace" else scharr_kernels(spatial_dims)

        return {"kernel_type": self.kernel_type, "kernel": kernel, "absolute": self.absolute, "retain_stats": self.retain_stats}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        """
        We expect (C, X, Y) or (C, X, Y, Z) shaped inputs for image and seg
        """
        for c in range(1):  # Works on the first channel only
            if params["retain_stats"]:
                orig_mean = torch.mean(img[c])
                orig_std = torch.std(img[c])
            img_ = img[c].unsqueeze(0).unsqueeze(0)  # adds temp batch and channel dim
            if params["kernel_type"] == "Laplace":
                tot_ = apply_filter(img_, params["kernel"])
            elif params["kernel_type"] == "Scharr":
                tot_ = torch.zeros_like(img_)
                for kernel in params["kernel"]:
                    if params["absolute"]:
                        tot_ += torch.abs(apply_filter(img_, kernel))
                    else:
                        tot_ += apply_filter(img_, kernel)
            img[c] = tot_[0, 0]
            if params["retain_stats"]:
                mean = torch.mean(img[c])
                std = torch.std(img[c])
                img[c] = (img[c] - mean) / torch.clamp(std, min=1e-7)
                img[c] = img[c] * orig_std + orig_mean  # return to original distribution
        return img


# One class per kernel, mirroring the GPU split so a config key names a class on
# either backend and `kernel_type` disappears from the config surface.


class LaplaceConvTransform(_ConvBaseTransform):
    """Laplacian edge enhancement."""

    def __init__(self, absolute: bool = False, retain_stats: bool = False):
        super().__init__(kernel_type="Laplace", absolute=absolute, retain_stats=retain_stats)


class ScharrConvTransform(_ConvBaseTransform):
    """Scharr gradient-magnitude edge filter."""

    def __init__(self, absolute: bool = True, retain_stats: bool = False):
        super().__init__(kernel_type="Scharr", absolute=absolute, retain_stats=retain_stats)


class HistogramEqualTransform(ImageOnlyTransform):
    """
    Update image intensity using histogram manipulations

    Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
    """

    def __init__(self, retain_stats: bool = False):
        super().__init__()
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        return {"retain_stats": self.retain_stats}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in range(1):  # Works on the first channel only
            if params["retain_stats"]:
                orig_mean = torch.mean(img[c])
                orig_std = torch.std(img[c])
            img_min, img_max = img[c].min(), img[c].max()

            # Flatten the image and compute the histogram
            img_flattened = img[c].flatten().to(torch.float32)
            hist, bins = torch.histogram(img_flattened, bins=256)

            # Compute bin edges
            bin_edges = torch.linspace(img_min, img_max, steps=257)  # 256 bins -> 257 edges

            # Compute the normalized cumulative distribution function (CDF)
            cdf = hist.cumsum(dim=0)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # Normalize to [0,1]
            cdf = cdf * (img_max - img_min) + img_min  # Scale back to image range

            # Perform histogram equalization
            indices = torch.searchsorted(bin_edges[:-1], img_flattened)
            img_eq = torch.index_select(cdf, dim=0, index=torch.clamp(indices, 0, 255))
            img[c] = img_eq.reshape(img[c].shape)

            if params["retain_stats"]:
                # Return to original distribution
                mean = torch.mean(img[c])
                std = torch.std(img[c])
                img[c] = (img[c] - mean) / torch.clamp(std, min=1e-7)
                img[c] = img[c] * orig_std + orig_mean
        return img


class _FunctionBaseTransform(ImageOnlyTransform):
    """
    Apply different functions to image pixels

    Based on https://github.com/neuropoly/totalspineseg/blob/main/totalspineseg/utils/augment.py
    """

    def __init__(self, function, retain_stats: bool = False):
        super().__init__()
        self.function = function
        self.retain_stats = retain_stats

    def get_parameters(self, **data_dict) -> dict:
        return {"function": self.function, "retain_stats": self.retain_stats}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in range(1):  # Works on the first channel only
            if params["retain_stats"]:
                orig_mean = torch.mean(img[c])
                orig_std = torch.std(img[c])

            # Normalize
            img[c] = (img[c] - img.min()) / (img.max() - img.min() + 0.00001)

            # Apply function
            img[c] = params["function"](img[c])

            if params["retain_stats"]:
                # Return to original distribution
                mean = torch.mean(img[c])
                std = torch.std(img[c])
                img[c] = (img[c] - mean) / torch.clamp(std, min=1e-7)
                img[c] = img[c] * orig_std + orig_mean
        return img


# One class per elementwise function; `function` is not expressible in JSON, so the
# old config had a single key that the builder fanned out over a hardcoded lambda
# list. Spelled out longhand rather than as torch.log1p / torch.sigmoid, which
# differ in the last ulp and would move the seeded determinism hashes.


def _log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log(1 + x)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


class _NamedFunctionTransform(_FunctionBaseTransform):
    """Shared constructor for the fixed-function leaves. Not registered itself."""

    #: Set by each leaf.
    function_impl: staticmethod

    def __init__(self, retain_stats: bool = False):
        super().__init__(function=type(self).function_impl, retain_stats=retain_stats)


class Log1pTransform(_NamedFunctionTransform):
    """Apply log(1 + x)."""

    function_impl = staticmethod(_log1p)


class SqrtTransform(_NamedFunctionTransform):
    """Apply sqrt(x)."""

    function_impl = staticmethod(torch.sqrt)


class SinTransform(_NamedFunctionTransform):
    """Apply sin(x)."""

    function_impl = staticmethod(torch.sin)


class ExpTransform(_NamedFunctionTransform):
    """Apply exp(x)."""

    function_impl = staticmethod(torch.exp)


class SigmoidTransform(_NamedFunctionTransform):
    """Apply the logistic sigmoid 1 / (1 + exp(-x))."""

    function_impl = staticmethod(_sigmoid)


def apply_filter(x: torch.Tensor, kernel: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Copied from https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/layers/simplelayers.py

    Filtering `x` with `kernel` independently for each batch and channel respectively.

    Args:
        x: the input image, must have shape (batch, channels, H[, W, D]).
        kernel: `kernel` must at least have the spatial shape (H_k[, W_k, D_k]).
            `kernel` shape must be broadcastable to the `batch` and `channels` dimensions of `x`.
        kwargs: keyword arguments passed to `conv*d()` functions.

    Returns:
        The filtered `x`.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import apply_filter
        >>> img = torch.rand(2, 5, 10, 10)  # batch_size 2, channels 5, 10x10 2D images
        >>> out = apply_filter(img, torch.rand(3, 3))   # spatial kernel
        >>> out = apply_filter(img, torch.rand(5, 3, 3))  # channel-wise kernels
        >>> out = apply_filter(img, torch.rand(2, 5, 3, 3))  # batch-, channel-wise kernels

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
    batch, chns, *spatials = x.shape
    n_spatial = len(spatials)
    if n_spatial > 3:
        raise NotImplementedError(f"Only spatial dimensions up to 3 are supported but got {n_spatial}.")
    k_size = len(kernel.shape)
    if k_size < n_spatial or k_size > n_spatial + 2:
        raise ValueError(f"kernel must have {n_spatial} ~ {n_spatial + 2} dimensions to match the input shape {x.shape}.")
    kernel = kernel.to(x)
    # broadcast kernel size to (batch chns, spatial_kernel_size)
    kernel = kernel.expand(batch, chns, *kernel.shape[(k_size - n_spatial) :])
    kernel = kernel.reshape(-1, 1, *kernel.shape[2:])  # group=1
    x = x.view(1, kernel.shape[0], *spatials)
    conv = [F.conv1d, F.conv2d, F.conv3d][n_spatial - 1]
    if "padding" not in kwargs:
        kwargs["padding"] = "same"

    if "stride" not in kwargs:
        kwargs["stride"] = 1
    output = conv(x, kernel, groups=kernel.shape[0], bias=None, **kwargs)
    return output.view(batch, chns, *output.shape[2:])


class ZscoreNormalization(ImageOnlyTransform):
    """
    Z-score normalization of image
    """

    def __init__(self) -> None:
        super().__init__()

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in range(1):
            mean = torch.mean(img[c])
            std = torch.std(img[c])
            img[c] = (img[c] - mean) / torch.clamp(std, min=1e-8)
        return img


# Temporary bridge for the CPU `if` ladder, which passes kernel_type from the config.
# Removed with that ladder; see the note in gpu/contrast.py.
ConvTransform = _ConvBaseTransform
FunctionTransform = _FunctionBaseTransform
