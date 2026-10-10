import contextlib
import copy
import functools
import warnings
from collections.abc import Sequence
from typing import Any, Protocol, Union

import kornia.augmentation as K
import torch
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
from kornia.augmentation.utils import _adapted_rsampling, _adapted_sampling
from kornia.constants import DataKey, Resample
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from torch import Tensor
from torch.distributions import RelaxedBernoulli
from torch.nn import Module

try:  # private to kornia, and the only part of the faster parameter draw that needs it
    from kornia.augmentation.container.image import ImageSequentialBase, _get_new_batch_shape

    FAST_PARAMETER_DRAW = True
except ImportError:  # pragma: no cover -- a kornia that moved them; fall back to its own loop
    FAST_PARAMETER_DRAW = False

DataType = Union[Tensor, list[Tensor], Boxes, Keypoints]
SequenceDataType = Union[list[Tensor], list[list[Tensor]], list[Boxes], list[Keypoints]]
# Anything AugmentationSequentialCustom accepts as a stage. Every smauglab GPU
# transform qualifies, via ImageOnlyTransform or RigidAffineAugmentationBase3D.
# Narrower than nn.Module, which is what lets the `*transforms` splat type-check.
TransformType = Union[_AugmentationBase, ImageSequential]


@functools.lru_cache(maxsize=16)
def _identity_4x4(device: torch.device, dtype: torch.dtype) -> Tensor:
    """A single 4x4 identity, cached per device and dtype. Treat as read-only."""
    return torch.eye(4, device=device, dtype=dtype)


def identity_transform(module: AugmentationBase3D, input: Tensor) -> Tensor:
    """The batch of identity matrices kornia's `identity_matrix` would return.

    `kornia.eye_like` builds a fresh 4x4 with `torch.eye` and `repeat`s it over the
    batch: an allocation and two kernel launches per module, for a constant. The
    cached matrix is expanded instead, so no per-call storage is allocated at all.
    It is shared and must not be written into -- kornia only reads it, and composes
    with the out-of-place `index_put`.

    Falls back to the module's own method if it has overridden `identity_matrix`,
    which would mean it is not this matrix.
    """
    if type(module).identity_matrix is not AugmentationBase3D.identity_matrix:
        return module.identity_matrix(input)
    return _identity_4x4(input.device, input.dtype).expand(input.shape[0], 4, 4)


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
        # These transforms never move anything, so the matrix is always the
        # identity -- shared and cached rather than rebuilt per module. See
        # `identity_transform`.
        return identity_transform(self, input)

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

    def forward_parameters(self, batch_shape: torch.Size) -> list[ParamItem]:
        """kornia's own loop, with each leaf's draw routed through `_leaf_parameters`.

        Overridden here rather than on every transform class because the leaves are
        spread over five modules and the saving is the same from either side. The
        shape threading (`_get_new_batch_shape`) is kornia's and is kept, so a
        module that changes the batch shape -- none of this package's do -- still
        works. A kornia that no longer exposes those private helpers gets its own
        loop back.
        """
        if not FAST_PARAMETER_DRAW:
            return super().forward_parameters(batch_shape)

        params: list[ParamItem] = []
        for name, module in self.get_forward_sequence():
            if isinstance(module, (_AugmentationBase, K.MixAugmentationBaseV2, ImageSequentialBase)):
                param = ParamItem(name, _leaf_parameters(module, batch_shape))
            else:
                param = ParamItem(name, None)
            batch_shape = _get_new_batch_shape(param, batch_shape)
            params.append(param)
        return params

    def transform_masks(self, input: Tensor, params: list[ParamItem], extra_args: dict[str, Any] | None = None) -> Tensor:
        for param in params:
            module = self.get_submodule(param.name)
            input = MaskSequentialOpsCustom.transform(input, module=module, param=param, extra_args=extra_args)
        return input


def _leaves_masks_untouched(module: Module) -> bool:
    """Whether `module`'s mask pass is the identity, by construction.

    True when the module inherits both of `ImageOnlyTransform`'s mask hooks
    unchanged -- then `transform_masks` can only return what it was handed,
    whichever branch of the row selection it takes. Tested against the inherited
    functions rather than against `isinstance` alone so that a subclass which
    does override one of them keeps going through kornia.
    """
    if not isinstance(module, ImageOnlyTransform):
        return False
    cls = type(module)
    return (
        cls.apply_transform_mask is ImageOnlyTransform.apply_transform_mask
        and cls.apply_non_transform_mask is ImageOnlyTransform.apply_non_transform_mask
    )


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

        if _leaves_masks_untouched(module) and isinstance(input, Tensor) and input.dim() == 5:
            # Nothing to do, and `transform_masks` is not cheap: it deep-copies the params,
            # reshapes and validates the volume and resolves the row selection, all to hand
            # back the tensor it was given. Most of a pipeline is image-only.
            #
            # Restricted to an already-batched volume because that is the shape kornia's
            # `transform_tensor` / `_transform_output_shape` round-trip leaves untouched; a
            # 4-D mask would come back with a batch axis.
            return input

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
                # [i : i + 1], not [i]: indexing with a scalar gives a 0-dim
                # tensor, and kornia does `to_apply = batch_prob > 0.5` and then
                # branches on `to_apply.all()` / `.any()`. Those are the same value
                # for a 0-dim tensor, so the per-element branch was unreachable and
                # `in_tensor[to_apply]` would select along the wrong axis.
                params_i["batch_prob"] = params["batch_prob"][i : i + 1]
                tfm_inp = module.transform_masks(inp, params=params_i, flags=module.flags, transform=module.transform_matrix, **extra_args)
                tfm_input.append(tfm_inp)
            input = tfm_input

        elif isinstance(module, (_AugmentationBase)):
            tfm_input = []
            params = cls.get_instance_module_param(param)
            params_i = copy.deepcopy(params)
            for i, inp in enumerate(input):
                # [i : i + 1], not [i]: indexing with a scalar gives a 0-dim
                # tensor, and kornia does `to_apply = batch_prob > 0.5` and then
                # branches on `to_apply.all()` / `.any()`. Those are the same value
                # for a 0-dim tensor, so the per-element branch was unreachable and
                # `in_tensor[to_apply]` would select along the wrong axis.
                params_i["batch_prob"] = params["batch_prob"][i : i + 1]
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


def autocast_active() -> bool:
    """Whether any autocast region is open, without parsing the torch version.

    kornia's `is_autocast_enabled` asks the same question but routes through
    `torch_version_ge`, which re-parses `torch.__version__` with
    `packaging.version.Version` on every call -- 7.7 us a time, against 0.08 us for
    the two builtins, and a pipeline asks around a hundred times per batch.
    """
    if torch.is_autocast_enabled():
        return True
    try:
        return bool(torch.is_autocast_enabled("cpu"))
    except (TypeError, RuntimeError):  # pragma: no cover - older torch, CUDA-only query
        return False


def _can_apply_directly(module: Module, input: Any, extra_args: dict[str, Any] | None) -> bool:
    """Whether `_apply_leaf` may stand in for kornia's `forward` for this call.

    The conditions are exactly the ones under which the layers being skipped are
    identities:

    * one of this package's 3-D leaves, so `apply_func` is the 3-D one and the
      parameters are already generated;
    * an already-batched 5-D volume, so `transform_tensor` and
      `transform_output_tensor` both pass it through unchanged;
    * `batch_prob` present, which is what the row selection reads;
    * no `extra_args` to merge, and no autocast region whose output cast would be
      skipped.

    Anything else falls through to kornia. `unit_tests/test_container_fast_path.py`
    runs every registered transform down both routes and asserts the outputs are
    identical, so the duplication here is checked against whichever kornia version
    is installed rather than assumed.
    """
    return (
        isinstance(module, RigidAffineAugmentationBase3D)
        and isinstance(input, Tensor)
        and input.dim() == 5
        and not extra_args
        and not autocast_active()
    )


def _apply_leaf(module: RigidAffineAugmentationBase3D, input: Tensor, params: dict[str, Tensor]) -> Tensor:
    """kornia's `forward` -> `apply_func` -> `transform_inputs` chain, inlined.

    That chain costs more than most of the augmentations in a pipeline do: it
    deep-copies the parameter dict twice, reshapes and re-validates the volume
    three times, and asks kornia's autocast helper -- which parses the torch
    version string -- four times, all for a leaf whose work is one kernel. None of
    it varies with the data, and `_can_apply_directly` is what establishes that the
    skipped steps are no-ops for this call.

    The control flow below is kornia's, kept deliberately line-for-line
    recognisable against `_BasicAugmentationBase.forward`,
    `RigidAffineAugmentationBase3D.apply_func` and
    `_AugmentationBase.transform_inputs`.
    """
    module.validate_tensor(input)
    flags = module.flags
    # kornia stores the parameters it was handed on the module; `inverse()` and
    # anything introspecting a transform after the fact reads them from there.
    module._params = params

    batch_prob = params["batch_prob"]
    to_apply = batch_prob > 0.5
    applies_to_all = bool(to_apply.all())
    applies_to_none = not applies_to_all and not bool(to_apply.any())

    if applies_to_none:
        transform = identity_transform(module, input)
    elif applies_to_all:
        transform = module.compute_transformation(input, params=params, flags=flags)
    else:
        transform = identity_transform(module, input).index_put(
            (to_apply,), module.compute_transformation(input[to_apply], params=params, flags=flags)
        )
    module._transform_matrix = transform

    if applies_to_all:
        return module.apply_transform(input, params, flags, transform=transform)
    if applies_to_none:
        return module.apply_non_transform(input, params, flags, transform=transform)
    output = module.apply_non_transform(input, params, flags, transform=transform)
    applied = module.apply_transform(input[to_apply], params, flags, transform=transform[to_apply])
    return output.index_put((to_apply,), applied)


@functools.lru_cache(maxsize=16)
def _batch_shape_tensor(batch_shape: tuple[int, ...]) -> Tensor:
    """`forward_input_shape`, cached per input shape. Treat as read-only.

    kornia rebuilds this five-element `long` tensor inside `forward_parameters`,
    once per module per batch -- 8.5 us a time, 27 times a call here, for a value
    that is fixed for a training run. Its only reader is
    `_transform_input3d_by_shape`, which reads it.
    """
    return torch.tensor(batch_shape, dtype=torch.long)


class DrawsParameters(Protocol):
    """What `_leaf_parameters` needs off the module it is handed.

    A Protocol rather than `Module`, for the same reason `_RegionSelecting` in
    gpu/contrast.py is one: `forward_parameters` is not declared on `nn.Module`,
    so the attribute resolves through `Module.__getattr__`, which is typed as
    returning `Tensor | Module` -- and a `Tensor` is not callable. Naming the one
    method here is both what makes the call type-check and a statement of the
    helper's actual requirement.
    """

    def forward_parameters(self, batch_shape: torch.Size) -> Any: ...


def _leaf_parameters(module: DrawsParameters, batch_shape: torch.Size) -> Any:
    """`forward_parameters` for one leaf, with the constant parts lifted out.

    kornia's `__batch_prob_generator__` allocates a one-element tensor, sums it,
    compares the sum to 1 and broadcasts a multiply, all to decide something that
    `p_batch` already fixed -- and `p_batch` is 1 for every transform in this
    package's configs. When it is not 1, this defers to kornia unchanged.

    `unit_tests/test_container_fast_path.py` compares the dictionary this returns
    against kornia's, tensor by tensor, across a grid of `p` / `p_batch` /
    `same_on_batch` settings and seeds.
    """
    if not isinstance(module, _AugmentationBase) or module.p_batch != 1:
        return module.forward_parameters(batch_shape)

    batch = batch_shape[0]
    if module.p == 1:
        batch_prob = torch.ones(batch)
    elif module.p == 0:
        batch_prob = torch.zeros(batch)
    elif isinstance(module._p_gen, RelaxedBernoulli):
        batch_prob = _adapted_rsampling((batch,), module._p_gen, module.same_on_batch)
    else:
        batch_prob = _adapted_sampling((batch,), module._p_gen, module.same_on_batch)
    if batch_prob.dim() == 2:  # some samplers return a trailing singleton
        batch_prob = batch_prob[..., 0]

    selected = int((batch_prob > 0.5).sum().item())
    params = module.generate_parameters(torch.Size((selected, *batch_shape[1:])))
    params = {} if params is None else params
    params["batch_prob"] = batch_prob
    params["forward_input_shape"] = _batch_shape_tensor(tuple(batch_shape))
    return params


class InputSequentialOpsCustom(InputSequentialOps):
    """`InputSequentialOps` with the leaf call short-circuited where that is exact."""

    @classmethod
    def transform(cls, input: Tensor, module: Module, param: ParamItem, extra_args: dict[str, Any] | None = None) -> Tensor:
        if _can_apply_directly(module, input, extra_args):
            params = cls.get_instance_module_param(param)
            if "batch_prob" in params:
                return _apply_leaf(module, input, params)  # type: ignore[arg-type]
        return InputSequentialOps.transform(input, module, param, extra_args or {})


class AugmentationSequentialOpsCustom(AugmentationSequentialOps):
    def _get_op(self, data_key: DataKey) -> type[SequentialOpsInterface[Any]]:
        """Return the corresponding operation given a data key."""
        if data_key == DataKey.INPUT:
            return InputSequentialOpsCustom
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
            # kornia types ParamItem.data as dict | list[ParamItem] | None and the inputs
            # as the wider DataType, but a leaf augmentation's MASK entry is always a params
            # dict holding a plain tensor. Asserted rather than ignored, so it fails loudly.
            mask = arg[mask_index]
            assert isinstance(param.data, dict)
            assert isinstance(mask, Tensor)
            # A `SegmentationRef`, not the sliced tensor: see its docstring for the
            # 201 MB of copying that costs.
            param.data["seg"] = SegmentationRef(mask, param.data.get("batch_prob"))  # type: ignore[assignment]

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


class SegmentationRef:
    """The segmentation, carried through kornia's params dict without being copied.

    `params` is deep-copied on the way into every leaf call: kornia's
    `_process_kwargs_to_params_and_flags` runs `override_parameters`, which runs
    `deepcopy_dict`, which clones every value that is a `Tensor`. That happens
    once per data key per module -- three times per module for an image-and-mask
    pipeline -- and the segmentation is the only full-volume entry in the dict.
    Measured on the shipped `transform_params_gpu.json` with a
    `[2, 1, 128, 128, 128]` batch: **201 MB of label map copied per pipeline
    call**, to serve a dozen reads.

    Holding it behind an object instead of as a bare `Tensor` is what stops that:
    `deepcopy_dict` passes a non-Tensor through by reference. The row slicing that
    `_seg_for_applied_rows` performs then happens on first read rather than
    eagerly, which also skips it entirely for the modules -- most of them -- that
    never look at the segmentation.

    Read it with :func:`segmentation_from`, never by indexing `params` directly.
    """

    __slots__ = ("_batch_prob", "_mask", "_resolved")

    def __init__(self, mask: Tensor, batch_prob: Any = None) -> None:
        self._mask = mask
        self._batch_prob = batch_prob
        self._resolved: Tensor | None = None

    def tensor(self) -> Tensor:
        """The mask rows this module's `apply_transform` will be handed."""
        if self._resolved is None:
            self._resolved = _seg_for_applied_rows(self._mask, self._batch_prob)
        return self._resolved

    def __deepcopy__(self, memo: dict) -> "SegmentationRef":
        """Share the volumes rather than duplicate them.

        `MaskSequentialOpsCustom.transform_list` runs a real `copy.deepcopy` over
        the params, which would otherwise copy the segmentation after all the
        trouble taken to avoid it. Nothing mutates these tensors.
        """
        clone = SegmentationRef(self._mask, self._batch_prob)
        clone._resolved = self._resolved
        return clone


def segmentation_from(params: Any) -> Tensor | None:
    """The segmentation a transform should read, or None if it was not given one.

    Accepts the :class:`SegmentationRef` the pipeline injects and a bare tensor,
    which is what a direct `apply_transform(...)` call -- a test, a script, or
    `RandomChooseXTransformsGPU` handing a slice to its children -- passes.
    """
    value = (params or {}).get("seg")
    if value is None or isinstance(value, Tensor):
        return value
    if isinstance(value, SegmentationRef):
        return value.tensor()
    raise TypeError(f"params['seg'] must be a Tensor or a SegmentationRef, got {type(value).__name__}")


def _seg_for_applied_rows(mask: Tensor, batch_prob: Any) -> Tensor:
    """The mask rows `apply_transform` will actually be handed.

    kornia slices the image down to the samples that drew below `p`
    (`apply_transform(in_tensor[to_apply], params, ...)`) but passes `params`
    through untouched. The segmentation rides in `params`, so a consumer reading
    `params["seg"]` would pair a B-row mask with a B'-row image: a broadcast
    error for anything vectorised, and -- worse, because it is silent -- the
    wrong sample's segmentation for anything that loops over `input.shape[0]`.

    Slicing here rather than in each consumer keeps `seg[b]` aligned with
    `input[b]` everywhere, including the transforms that never noticed.
    """
    if not isinstance(batch_prob, Tensor) or batch_prob.numel() != mask.shape[0]:
        return mask
    to_apply = batch_prob > 0.5
    # The all-true case is the common one; indexing would copy the whole volume
    # for nothing. The all-false case never reaches apply_transform at all.
    if bool(to_apply.all()):
        return mask
    return mask[to_apply]


@contextlib.contextmanager
def record_applications(pipeline):
    """Record which transforms actually fire, in call order.

    A combined pass gives every augmentation its own probability, so any one output
    is the product of a random subset. The config only says what *could* have fired,
    which is no use at all when one draw in ten comes out destroyed and the question
    is which transform did it.

    Yields a list that fills as the pipeline runs. Each entry is the transform's
    class name, which tensor it ran on, whether that tensor changed, and its scalar
    parameters -- tensors are summarised rather than stored, because a spatial
    transform's parameters are the size of the batch.

    `apply_transform` is the hook because it is the one method every leaf
    augmentation implements and kornia only calls it for the elements it selected,
    including down the `RandomChooseXTransformsGPU` path, which dispatches to it
    directly rather than through `forward`.

    Two things in the output used to read as bugs and are not:

    * A geometric transform appears **twice**, once with `target: "image"` and once
      with `target: "mask"`. `AugmentationSequential` runs each module over the image
      and the mask in turn from the same sampled parameters, and the 3D geometric
      transforms implement `apply_transform_mask` by calling `self.apply_transform`
      -- which is this wrapper, since the patch is an instance attribute. One
      application per record; the identical `params` on the pair is the proof that
      image and mask moved together.
    * `RandomChooseXTransformsGPU` appears twice in `random_order` mode because the
      builder emits two of them, the transfer bucket and the general-enhancement one.
      `label` tells them apart.
    """
    fired: list[dict] = []
    patched = []

    for module in pipeline.modules():
        if not hasattr(module, "apply_transform") or module is pipeline:
            continue
        original = module.apply_transform

        def wrapper(input, params, flags, transform=None, _original=original, _module=module):
            # Most of these write through their input and return the same tensor
            # (`input[b, c] = x`), so comparing the result against `input` afterwards
            # compares it against itself. Keep a copy of what went in.
            before = input.clone()
            output = _original(input, params, flags, transform)
            changed = not (output.shape == before.shape and bool(torch.equal(output, before)))
            entry = {
                "transform": type(_module).__name__,
                "target": "mask" if DataKey.MASK in (flags or {}).get("data_keys", ()) else "image",
                "changed": changed,
                # The old name, kept so analyses written against it keep loading. It was
                # never true of the mask pass, which is exactly why `target` exists now.
                "changed_image": changed,
                "params": _scalar_params(params),
            }
            label = getattr(_module, "provenance_label", None)
            if label is not None:
                entry["label"] = label
            fired.append(entry)
            return output

        module.apply_transform = wrapper
        patched.append((module, original))

    try:
        yield fired
    finally:
        for module, original in patched:
            module.apply_transform = original


def _scalar_params(params) -> dict:
    """Keep what is readable; a spatial transform's parameters are batch-sized tensors."""
    out: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key == "seg":
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                out[key] = []
            elif value.numel() <= 8:
                out[key] = [round(float(v), 5) for v in value.flatten().tolist()]
            else:
                out[key] = f"<tensor {tuple(value.shape)}>"
        elif isinstance(value, (int, float, bool, str)):
            out[key] = value
    return out
