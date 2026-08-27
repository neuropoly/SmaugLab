from collections.abc import Sequence
from typing import Any, Union

import torch
from torch.nn.functional import interpolate


def nnunet_tail_transforms(
    *,
    use_mask_for_norm: list[bool] | None = None,
    deep_supervision_scales: Union[list, tuple, None] = None,
    is_cascaded: bool = False,
    foreground_labels: Union[tuple[int, ...], list[int], None] = None,
    regions: list[Union[list[int], tuple[int, ...], int]] | None = None,
    ignore_label: int | None = None,
) -> list[Any]:
    """The nnU-Net transforms that follow SmaugLab's augmentations, in order.

    Five `get_training_transforms` methods across two modules ended with a
    character-identical copy of this -- intensity masking, the -1 label removal, the
    two cascade transforms, region conversion and deep-supervision downsampling.

    `deep_supervision_scales=None` skips the downsampling, which is how the GPU
    trainers had it: they carry the block commented out, because with GPU
    augmentations the mask is still being deformed after this point and the
    multi-scale targets have to be built from the augmented mask in `train_step`.

    `get_validation_transforms` deliberately does not use this. Its cascade branch
    adds only MoveSegAsOneHotToDataTransform, without the two RandomTransform
    wrappers, so it is a different sequence rather than another copy of this one.
    """
    from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
    from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
        RemoveRandomConnectedComponentFromOneHotEncodingTransform,
    )
    from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
    from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
    from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform
    from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
    from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

    transforms: list[Any] = []

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(
            MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            )
        )

    transforms.append(RemoveLabelTansform(-1, 0))

    # The following augmentations are related to special nnunet executions
    if is_cascaded:
        assert foreground_labels is not None, "We need foreground_labels for cascade augmentations"
        transforms.append(
            MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True)
        )
        transforms.append(
            RandomTransform(
                ApplyRandomBinaryOperatorTransform(channel_idx=list(range(-len(foreground_labels), 0)), strel_size=(1, 8), p_per_label=1),
                apply_probability=0.4,
            )
        )
        transforms.append(
            RandomTransform(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                    p_per_label=1,
                ),
                apply_probability=0.2,
            )
        )

    if regions is not None:
        # the ignore label must also be converted
        transforms.append(
            ConvertSegmentationToRegionsTransform(
                regions=[*list(regions), ignore_label] if ignore_label is not None else regions, channel_in_seg=0
            )
        )

    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    return transforms


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
        # Per-axis scale factors: either broadcast from a scalar or taken as given.
        s: Sequence[float]
        for ds_scale in self.ds_scales:
            if not isinstance(ds_scale, (tuple, list)):
                # If single scale value, apply to all spatial dimensions
                s = [ds_scale] * (segmentation.ndim - 2)  # -2 for batch and channel dims
            else:
                assert len(ds_scale) == segmentation.ndim - 2, (
                    f"Scale length {len(ds_scale)} doesn't match spatial dims {segmentation.ndim - 2}"
                )
                s = ds_scale

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
