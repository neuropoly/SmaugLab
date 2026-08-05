from typing import Union

import torch
from torch.nn.functional import interpolate


class DownsampleSegForDSTransformCustom:
    """
    Custom deep supervision downsampling transform that handles batched tensors properly.
    Unlike the original DownsampleSegForDSTransform, this handles tensors with batch dimension.

    Input: [batch, channels, spatial_dims...]
    Output: List of [batch, channels, spatial_dims...] at different scales
    """

    def __init__(self, ds_scales: Union[list, tuple]):
        self.ds_scales = ds_scales

    def __call__(self, segmentation: torch.Tensor) -> list[torch.Tensor]:
        """
        Apply downsampling to segmentation tensor with batch dimension.

        Args:
            segmentation: [batch, channels, spatial_dims...] tensor

        Returns:
            List of downsampled tensors, each with shape [batch, channels, spatial_dims...]
        """
        results = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                # If single scale value, apply to all spatial dimensions
                s = [s] * (segmentation.ndim - 2)  # -2 for batch and channel dims
            else:
                assert len(s) == segmentation.ndim - 2, f"Scale length {len(s)} doesn't match spatial dims {segmentation.ndim - 2}"

            if all(i == 1 for i in s):
                # No downsampling needed
                results.append(segmentation)
            else:
                # Calculate new spatial shape
                spatial_shape = segmentation.shape[2:]  # Skip batch and channel dims
                new_shape = [round(i * j) for i, j in zip(spatial_shape, s)]

                # Store original dtype
                dtype = segmentation.dtype

                # Interpolate (convert to float for interpolation, then back to original dtype)
                downsampled = interpolate(segmentation.float(), size=new_shape, mode="nearest-exact").to(dtype)

                results.append(downsampled)

        return results
