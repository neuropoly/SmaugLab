import copy
import warnings
from collections.abc import Sequence
from typing import Any, Union

import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.augmentation._3d.base import AugmentationBase3D, RigidAffineAugmentationBase3D
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.image import ImageSequential
from kornia.augmentation.container.ops import (
    AugmentationSequentialOps,
    BoxSequentialOps,
    ClassSequentialOps,
    InputSequentialOps,
    KeypointSequentialOps,
    MaskSequentialOps,
    SequentialOpsInterface,
)
from kornia.augmentation.container.params import ParamItem
from kornia.augmentation.container.patch import PatchSequential
from kornia.augmentation.container.video import VideoSequential
from kornia.constants import DataKey, Resample
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from torch import Tensor
from torch.nn import Module

DataType = Union[Tensor, list[Tensor], Boxes, Keypoints]
SequenceDataType = Union[list[Tensor], list[list[Tensor]], list[Boxes], list[Keypoints]]
# Anything AugmentationSequentialCustom accepts as a stage. Every smauglab GPU
# transform qualifies, via ImageOnlyTransform or RigidAffineAugmentationBase3D.
# Narrower than nn.Module, which is what lets the `*transforms` splat type-check.
TransformType = Union[_AugmentationBase, ImageSequential]


class ImageOnlyTransform(RigidAffineAugmentationBase3D):
    r"""ImageOnlyTransform base class for customized image-only transformations.

    Args:
        p: probability for applying an augmentation. This param controls the augmentation probabilities
          element-wise for a batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    """

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def apply_non_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        # For the images where batch_prob == False.
        return input

    def apply_non_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return input

    def apply_non_transform_boxes(
        self, input: Boxes, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Boxes:
        return input

    def apply_transform_boxes(
        self, input: Boxes, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Boxes:
        return input

    def apply_non_transform_keypoint(
        self, input: Keypoints, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Keypoints:
        return input

    def apply_transform_keypoint(
        self, input: Keypoints, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Keypoints:
        return input

    def apply_non_transform_class(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return input

    def apply_transform_class(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Tensor | None = None
    ) -> Tensor:
        return input


class AugmentationSequentialCustom(AugmentationSequential):
    """Custom AugmentationSequential to handle masks augmentations."""

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: Union[Sequence[str], Sequence[int], Sequence[DataKey]] | None = (DataKey.INPUT,),
        same_on_batch: bool | None = None,
        keepdim: bool | None = None,
        random_apply: Union[int, bool, tuple[int, int]] = False,
        random_apply_weights: list[float] | None = None,
        transformation_matrix_mode: str = "silent",
        extra_args: dict[DataKey, dict[str, Any]] | None = None,
    ) -> None:
        self._transform_matrix: Tensor | None
        self._transform_matrices: list[Tensor | None] = []

        super().__init__(
            *args,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=random_apply,
            random_apply_weights=random_apply_weights,
        )

        self._parse_transformation_matrix_mode(transformation_matrix_mode)

        self._valid_ops_for_transform_computation: tuple[Any, ...] = (
            RigidAffineAugmentationBase2D,
            RigidAffineAugmentationBase3D,
            AugmentationSequential,
        )

        self.data_keys: list[DataKey] | None
        if data_keys is not None:
            self.data_keys = [DataKey.get(inp) for inp in data_keys]
        else:
            self.data_keys = data_keys

        if self.data_keys:
            if any(in_type not in DataKey for in_type in self.data_keys):
                raise AssertionError(f"`data_keys` must be in {DataKey}. Got {self.data_keys}.")

            if self.data_keys[0] != DataKey.INPUT:
                raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        self.transform_op = AugmentationSequentialOpsCustom(self.data_keys)

        self.contains_video_sequential: bool = False
        self.contains_3d_augmentation: bool = False
        for arg in args:
            if isinstance(arg, PatchSequential) and not arg.is_intensity_only():
                warnings.warn("Geometric transformation detected in PatchSeqeuntial, which would break bbox, mask.", stacklevel=1)
            if isinstance(arg, VideoSequential):
                self.contains_video_sequential = True
            # NOTE: only for images are supported for 3D.
            if isinstance(arg, AugmentationBase3D):
                self.contains_3d_augmentation = True
        self._transform_matrix = None
        self.extra_args = extra_args or {DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}}

    def transform_masks(self, input: Tensor, params: list[ParamItem], extra_args: dict[str, Any] | None = None) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = MaskSequentialOpsCustom.transform(input, module=module, param=param, extra_args=extra_args)
        return input


class MaskSequentialOpsCustom(MaskSequentialOps):
    @classmethod
    def transform(cls, input: Tensor, module: Module, param: ParamItem, extra_args: dict[str, Any] | None = None) -> Tensor:
        """Apply a transformation with respect to the parameters.

        Args:
            input: the input tensor.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}

        if isinstance(module, (K.GeometricAugmentationBase2D,)):
            input = module.transform_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=module.flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, (K.RigidAffineAugmentationBase3D,)):
            flags = module.flags | {"data_keys": [DataKey.MASK]}
            input = module.transform_masks(
                input,
                params=cls.get_instance_module_param(param),
                flags=flags,
                transform=module.transform_matrix,
                **extra_args,
            )

        elif isinstance(module, K.RandomTransplantation):
            input = module(input, params=cls.get_instance_module_param(param), data_keys=[DataKey.MASK], **extra_args)

        elif isinstance(module, (_AugmentationBase)):
            input = module.transform_masks(input, params=cls.get_instance_module_param(param), flags=module.flags, **extra_args)

        elif (isinstance(module, K.ImageSequential) and not module.is_intensity_only()) or isinstance(
            module, K.container.ImageSequentialBase
        ):
            input = module.transform_masks(input, params=cls.get_sequential_module_param(param), extra_args=extra_args)

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            input = MaskSequentialOps.transform(input, module=module.op, param=param, extra_args=extra_args)

        return input

    @classmethod
    def transform_list(
        cls, input: list[Tensor], module: Module, param: ParamItem, extra_args: dict[str, Any] | None = None
    ) -> list[Tensor]:
        """Apply a transformation with respect to the parameters.

        Args:
            input: list of input tensors.
            module: any torch Module but only kornia augmentation modules will count
                to apply transformations.
            param: the corresponding parameters to the module.
            extra_args: Optional dictionary of extra arguments with specific options for different input types.
        """
        if extra_args is None:
            extra_args = {}
        if isinstance(module, (K.GeometricAugmentationBase2D, K.RigidAffineAugmentationBase3D)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(inp, params=params_i, flags=module.flags, transform=module.transform_matrix, **extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (_AugmentationBase)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                params_i["batch_prob"] = params["batch_prob"][i]
                tfm_inp = module.transform_masks(inp, params=params_i, flags=module.flags, **extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif (isinstance(module, K.ImageSequential) and not module.is_intensity_only()) or isinstance(
            module, K.container.ImageSequentialBase
        ):
            tfm_input = []
            seq_params = cls.get_sequential_module_param(param)
            for inp in input:
                tfm_inp = module.transform_masks(inp, params=seq_params, extra_args=extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (K.auto.operations.OperationBase,)):
            raise NotImplementedError(
                "The support for list of masks under auto operations are not yet supported. You are welcome to file a PR in our repo."
            )
        return input


class AugmentationSequentialOpsCustom(AugmentationSequentialOps):
    def _get_op(self, data_key: DataKey) -> type[SequentialOpsInterface[Any]]:
        """Return the corresponding operation given a data key."""
        if data_key == DataKey.INPUT:
            return InputSequentialOps
        if data_key == DataKey.MASK:
            return MaskSequentialOpsCustom
        if data_key in {DataKey.BBOX, DataKey.BBOX_XYWH, DataKey.BBOX_XYXY}:
            return BoxSequentialOps
        if data_key == DataKey.KEYPOINTS:
            return KeypointSequentialOps
        if data_key == DataKey.CLASS:
            return ClassSequentialOps
        raise RuntimeError(f"Operation for `{data_key.name}` is not found.")

    def transform(
        self,
        *arg: DataType,
        module: Module,
        param: ParamItem,
        extra_args: dict[DataKey, dict[str, Any]],
        data_keys: Union[list[str], list[int], list[DataKey]] | None = None,
    ) -> Union[DataType, SequenceDataType]:
        _data_keys = self.preproc_datakeys(data_keys)

        if isinstance(module, K.RandomTransplantation):
            # For transforms which require the full input to calculate the parameters (e.g. RandomTransplantation)
            param = ParamItem(
                name=param.name,
                data=module.params_from_input(
                    *arg,  # type: ignore[arg-type]
                    data_keys=_data_keys,
                    params=param.data,  # type: ignore[arg-type]
                    extra_args=extra_args,
                ),
            )

        keys = [dk.name for dk in _data_keys]
        if "MASK" in keys:
            mask_index = keys.index("MASK")
            # kornia types ParamItem.data as dict | list[ParamItem] | None and the
            # inputs as the wider DataType, but the MASK entry of a leaf
            # augmentation is always a params dict holding a plain tensor. Asserted
            # rather than ignored so a violated assumption still fails loudly.
            mask = arg[mask_index]
            assert isinstance(param.data, dict)
            assert isinstance(mask, Tensor)
            param.data["seg"] = mask

        outputs = []
        for inp, dcate in zip(arg, _data_keys):
            op = self._get_op(dcate)
            extra_arg = extra_args.get(dcate, {})
            if dcate.name == "MASK" and isinstance(inp, list):
                outputs.append(MaskSequentialOpsCustom.transform_list(inp, module, param=param, extra_args=extra_arg))
            else:
                outputs.append(op.transform(inp, module, param=param, extra_args=extra_arg))
        if len(outputs) == 1 and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs
