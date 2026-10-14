"""The SmaugLab nnU-Net trainer.

One class, because the config already says everything the three previous trainers encoded
between them. `nnUNetTrainerDAExt` built the CPU pipeline, `...GPU` built the GPU one plus
nnU-Net's SpatialTransform, and `...Hybrid` built both -- but a config is sectioned into
"CPU" and "GPU", so which sections are populated decides that on its own:

    transform_params.json         CPU: 19  GPU: 0   -> CPU-only, as ...DAExt did
    transform_params_gpu.json     CPU: 1   GPU: 26  -> GPU-only, as ...DAExtGPU did
    transform_params_hybrid.json  CPU: 19  GPU: 24  -> both, as ...DAExtHybrid did

The class keeps the name `nnUNetTrainerDAExtGPU` whatever the config contains, and that is
not cosmetic: nnU-Net writes the trainer class name into every checkpoint
(`checkpoint['trainer_name']`) and resolves the class from it at inference, so renaming it
would make several hundred trained models unloadable.
"""

import filecmp
import importlib
import math
import os
import shutil
import warnings
from typing import Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context, softmax_helper_dim1
from torch import autocast

from smauglab import configs
from smauglab.config import load_config
from smauglab.registry import AugType, Backend, InvalidConfigError, entries
from smauglab.trainers.utils import DownsampleSegForDSTransformCustom, nnunet_tail_transforms
from smauglab.transforms.build import SHARED_GROUPS, build_cpu_pipeline, build_gpu_stages
from smauglab.transforms.gpu.base import AugmentationSequentialCustom

#: Env var naming the config to train with.
CONFIG_ENV = "SMAUGLAB_PARAMS_JSON"

#: Previous name, still honoured so existing sweep scripts keep working. The three
#: old variables (_CPU_JSON, _GPU_JSON, _HYBRID_JSON) picked a trainer as much as a
#: file; only this one ever had an external caller.
LEGACY_CONFIG_ENV = "SMAUGLAB_PARAMS_GPU_JSON"

DEFAULT_CONFIG = "transform_params_gpu.json"

#: Detailed non-finite-loss reports per epoch before the guard falls back to counting.
#: One bad augmentation hits every batch of an epoch, and 250 identical lines is how a
#: warning worth reading becomes noise nobody reads.
NAN_REPORTS_PER_EPOCH = 3

#: Default floor for the AMP gradient scale. The scaler exists to lift small fp16
#: gradients out of the denormal range, so a scale below 1 only shrinks them -- it can
#: never help, and it is the first step of a ratchet that ends at zero. See
#: `_guard_grad_scale`.
DEFAULT_MIN_GRAD_SCALE = 1.0

#: Env var overriding that floor. It exists for the one case the default forecloses:
#: gradients that genuinely overflow fp16 at scale 1, where halving is the correct move
#: and the floor is in the way. Nothing observed so far needs it. See
#: `resolve_min_grad_scale`.
MIN_GRAD_SCALE_ENV = "SMAUGLAB_MIN_GRAD_SCALE"

#: `torch.amp.GradScaler`'s default `init_scale`. Only used to warn about a floor at or
#: above it, which would undo every backoff the scaler ever makes.
GRAD_SCALER_INIT_SCALE = 2.0**16
#: nnU-Net's own dice smoothing, from `nnUNetTrainer._build_loss`. Repeated here so the
#: supervised dice and the consistency dice that end up in the same sum are computed
#: identically; a different epsilon would make the two terms incomparable.
DICE_SMOOTH = 1e-5


def resolve_config_path() -> str:
    """Locate the config: new env var, then the deprecated one, then the default."""
    path = os.environ.get(CONFIG_ENV)
    if path:
        return path
    legacy = os.environ.get(LEGACY_CONFIG_ENV)
    if legacy:
        warnings.warn(
            f"{LEGACY_CONFIG_ENV} is deprecated; use {CONFIG_ENV}. The config's CPU and GPU "
            "sections now decide which augmentations run, so the name no longer implies a backend.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return str(importlib.resources.files(configs) / DEFAULT_CONFIG)


def resolve_min_grad_scale() -> float:
    """Read the gradient-scale floor from the env var, or return the default.

    Anything that is not a positive, finite number raises instead of falling back: a
    typo in a sweep script would otherwise train for days under a floor nobody chose,
    and a floor of zero or below is precisely the collapse `_guard_grad_scale` exists
    to prevent. Going below the default is allowed but warned about, because a scale
    under 1 cannot do the job the scaler is there for.
    """
    raw = os.environ.get(MIN_GRAD_SCALE_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MIN_GRAD_SCALE

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{MIN_GRAD_SCALE_ENV}={raw!r} is not a number") from exc

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{MIN_GRAD_SCALE_ENV}={raw!r} must be a positive, finite number. Zero or below leaves the "
            "scale free to ratchet to zero, the collapse the floor exists to prevent; `nan` never "
            "compares true, so the guard would never fire; `inf` would pin the scale past fp16."
        )

    if value < DEFAULT_MIN_GRAD_SCALE:
        warnings.warn(
            f"{MIN_GRAD_SCALE_ENV}={value} is below {DEFAULT_MIN_GRAD_SCALE}, where the scaler can only "
            "shrink gradients. That is useful only if the gradients genuinely overflow fp16 at scale 1; "
            "otherwise it buys underflow, and a floor deep in the denormals trains on all-zero gradients "
            "without ever reporting a non-finite step.",
            stacklevel=2,
        )
    elif value >= GRAD_SCALER_INIT_SCALE:
        warnings.warn(
            f"{MIN_GRAD_SCALE_ENV}={value} is at or above the scaler's initial scale "
            f"({GRAD_SCALER_INIT_SCALE:g}), so every backoff is undone immediately and an overflowing run "
            "will skip every step instead of adapting.",
            stacklevel=2,
        )
    return value


def _record_config(source: str, destination: str) -> None:
    """Write down which config this run was trained with.

    Overwriting unconditionally destroys the record whenever a run is resumed
    with a different SMAUGLAB_PARAMS_JSON: the earlier epochs were trained with
    the old config and nothing says so any more. The file is the only provenance
    a finished run carries, so a differing one is kept and the new one written
    beside it under a numbered name, with a warning naming both.
    """
    if not os.path.exists(destination):
        shutil.copy(source, destination)
        return

    if filecmp.cmp(source, destination, shallow=False):
        return

    index = 1
    root, extension = os.path.splitext(destination)
    while os.path.exists(f"{root}_{index}{extension}"):
        index += 1
    kept = f"{root}_{index}{extension}"
    shutil.copy(source, kept)
    warnings.warn(
        f"This run was previously trained with a different SmaugLab config. "
        f"{destination} is unchanged and describes the earlier epochs; the config now in use "
        f"was written to {kept}.",
        stacklevel=2,
    )


def _has_gpu_augmentations(config) -> bool:
    """Whether this config asks for anything on the GPU side."""
    return bool(config.names(Backend.GPU))


def _active(section: dict, names: list[str]) -> list[str]:
    """Of `names`, the ones the section enables with a non-zero probability."""
    return [name for name in names if isinstance(section.get(name), dict) and section[name].get("p", 1.0) != 0]


def _consistency_problems(config) -> list[str]:
    """Why this config cannot train with the consistency term. Empty means it can."""
    if not _has_gpu_augmentations(config):
        return [
            "loss.consistency needs the intensity augmentations in the GPU section: the reference "
            "image is the GPU pipeline's geometric stage, and with a CPU-only config the dataloader "
            "has already applied everything, so there is nothing left to compare against. Move the "
            "augmentations into a 'GPU' section, or set loss.consistency.enabled to false."
        ]
    return []


def _consistency_warnings(config) -> list[str]:
    """Things about this config that weaken the term without making it wrong."""
    warnings_out = []
    gpu = config.section(Backend.GPU)
    cpu = config.section(Backend.CPU)

    shared = [entry.name for entry in entries(backend=Backend.GPU) if entry.group in SHARED_GROUPS]
    if not _active(gpu, shared):
        warnings_out.append(
            f"consistency: no geometric augmentation is active in the GPU section "
            f"({', '.join(shared)}), so the shared stage is empty and the reference is the "
            f"dataloader's patch verbatim. The term still works -- nothing but the geometric "
            f"group touches the mask -- but this is usually a config mistake."
        )

    # CPU GEO is fine and must not be warned about: it runs before the batch reaches
    # train_step, so both branches inherit it identically. Only intensity work on the
    # CPU side makes the "unaugmented" reference a misnomer.
    intensity = [entry.name for entry in entries(backend=Backend.CPU) if entry.group in (AugType.GE, AugType.TA)]
    offenders = _active(cpu, intensity)
    if offenders:
        warnings_out.append(
            f"consistency: the CPU section applies intensity augmentations in the dataloader "
            f"({', '.join(offenders)}), so the reference is already augmented and the term compares "
            f"two differently augmented views rather than augmented against clean. Still a valid "
            f"objective, just not the one the name implies."
        )

    occluders = [entry.name for entry in entries(backend=Backend.GPU) if entry.group is AugType.OCC]
    for name in _active(gpu, occluders):
        probability = gpu[name].get("p", 1.0)
        warnings_out.append(
            f"consistency: {name} fires at p={probability} and is not applied to the reference "
            f"branch, so in those batches the term asks the network to agree with the reference "
            f"inside the occluded region, where the supervised target says background. The two "
            f"terms disagree there and nothing masks it; set its p to 0 to avoid it."
        )
    return warnings_out


class nnUNetTrainerDAExtGPU(nnUNetTrainer):
    """nnU-Net trainer driven entirely by a SmaugLab config.

    CPU-section augmentations run in the dataloader worker; GPU-section ones run on
    the batch in `train_step`.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        json_path = resolve_config_path()
        config = load_config(json_path)

        consistency = config.loss_options("consistency")
        split = bool(consistency["enabled"])
        if split:
            problems = _consistency_problems(config)
            if problems:
                raise InvalidConfigError(config.source, problems)

        # Built only when the config actually asks for GPU augmentations, so a
        # CPU-only config costs nothing per step and train_step stays a no-op.
        #
        # `self.transforms` is the augmented branch either way, so train_step's existing
        # path is untouched when the consistency term is off. `self.transforms_shared`
        # holds the geometric stage only when it is on; both containers get the mask,
        # because the segmentation-conditioned transfer augmentations read it through the
        # SegmentationRef that AugmentationSequentialCustom injects per call.
        self.transforms: AugmentationSequentialCustom | None = None
        self.transforms_shared: AugmentationSequentialCustom | None = None
        if _has_gpu_augmentations(config):
            shared, augmented = build_gpu_stages(
                config.section(Backend.GPU),
                mode=config.pipeline_mode(),
                options=config.pipeline_options("random_choose"),
                source=config.source,
                order_source=config.order_source(),
                split_shared=split,
            )

            def container(*stage) -> AugmentationSequentialCustom:
                return AugmentationSequentialCustom(
                    *stage,
                    data_keys=["input", "mask"],
                    # See AugTransformsGPU: True forces the flag onto every child and
                    # spends each transform's `p` once for the whole batch.
                    same_on_batch=True if config.same_on_batch() else None,
                ).to(self.device)

            if shared:
                self.transforms_shared = container(*shared)
            self.transforms = container(*augmented)

        # The criterion is built here rather than in `_build_loss`: it is a second term
        # beside `self.loss`, not a replacement for it, and everything it needs
        # (`label_manager`, `configuration_manager`, `is_ddp`) is already assigned by
        # nnUNetTrainer.__init__.
        self.consistency: MemoryEfficientSoftDiceLoss | None = None
        self.consistency_weight = 0.0
        self.consistency_detach = True
        self.consistency_nonlin = softmax_helper_dim1
        if split:
            self.consistency_weight = float(consistency["weight"])
            self.consistency_detach = bool(consistency["detach_reference"])
            # Regions mean a sigmoid head and a one-hot target, as in DC_and_BCE_loss;
            # otherwise softmax and do_bg=False. Mirrors `_build_loss` so the two dice
            # terms in `l` disagree about nothing.
            self.consistency_nonlin = torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1
            self.consistency = MemoryEfficientSoftDiceLoss(
                apply_nonlin=self.consistency_nonlin,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=self.label_manager.has_regions,
                smooth=DICE_SMOOTH,
                ddp=self.is_ddp,
            )

        print(f"Using SmaugLab transforms from: {json_path}")
        print(
            f"  CPU: {len(config.names(Backend.CPU))} augmentations, GPU: {len(config.names(Backend.GPU))}, mode: {config.pipeline_mode().value}"
        )
        if split:
            print(
                f"  loss.consistency: on (weight {self.consistency_weight}, "
                f"detach_reference {self.consistency_detach}), shared geometric stage: "
                f"{len(self.transforms_shared) if self.transforms_shared is not None else 0} transforms"
            )
            for warning in _consistency_warnings(config):
                print(f"  {warning}")
        else:
            print("  loss.consistency: off")

        _record_config(json_path, os.path.join(self.output_folder, "transform_params_used_for_training.json"))

        # A non-finite loss is otherwise invisible: `GradScaler` skips the step without a
        # word, so the run stops learning and the only symptom is `train_loss nan` in the
        # epoch line, naming neither the batch nor the cause. See `_report_nonfinite_loss`.
        self._nan_steps_this_epoch = 0
        self._nan_steps_total = 0
        self._scale_floor_hits_this_epoch = 0
        self._scale_floor_hits_total = 0
        # Resolved once: the floor is consulted on every step, and a run whose guard
        # changed halfway through because the environment did would be worse than no
        # override at all. A non-default value goes to stdout, which is what nnU-Net
        # captures into the training log -- the `warnings.warn` above only reaches
        # stderr, and the floor in force is provenance the finished run needs.
        self._min_grad_scale = resolve_min_grad_scale()
        if self._min_grad_scale != DEFAULT_MIN_GRAD_SCALE:
            note = (
                " (below the default, where the scaler can only shrink gradients)" if self._min_grad_scale < DEFAULT_MIN_GRAD_SCALE else ""
            )
            print(f"AMP gradient-scale floor set to {self._min_grad_scale} via {MIN_GRAD_SCALE_ENV}{note}")

        # Accumulated as a device tensor, not a float: reading the term back per step
        # would add a host sync train_step does not otherwise pay until its last line.
        self._consistency_sum: torch.Tensor | None = None
        self._consistency_steps = 0
        self._consistency_last: torch.Tensor | None = None

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, tuple[int, ...]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[list, tuple, None],
        mirror_axes: tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: list[bool] | None = None,
        is_cascaded: bool = False,
        foreground_labels: Union[tuple[int, ...], list[int]] | None = None,
        regions: list[Union[list[int], tuple[int, ...], int]] | None = None,
        ignore_label: int | None = None,
    ) -> BasicTransform:
        """Dataloader-side augmentations: whatever the config's CPU section names.

        A staticmethod because that is nnU-Net's contract, so it cannot reach the
        instance's parsed config and resolves the path itself. `load_config` is
        cached, so the file is still read and validated once.
        """
        transforms = []

        config = load_config(resolve_config_path())
        transforms.extend(
            build_cpu_pipeline(
                config.section(Backend.CPU),
                do_dummy_2d_data_aug=do_dummy_2d_data_aug,
                patch_size=patch_size,
                rotation=rotation_for_DA,
                source=config.source,
                order_source=config.order_source(),
            )
        )

        # Deep supervision has to come after whatever last deformed the mask. With GPU
        # augmentations that is train_step; without them nothing touches the mask after
        # this point, so it belongs here, where nnU-Net puts it. None tells the tail to skip.
        transforms.extend(
            nnunet_tail_transforms(
                use_mask_for_norm=use_mask_for_norm,
                deep_supervision_scales=None if _has_gpu_augmentations(config) else deep_supervision_scales,
                is_cascaded=is_cascaded,
                foreground_labels=foreground_labels,
                regions=regions,
                ignore_label=ignore_label,
            )
        )

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[list, tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[tuple[int, ...], list[int]] | None = None,
        regions: list[Union[list[int], tuple[int, ...], int]] | None = None,
        ignore_label: int | None = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(source_channel_idx=1, all_labels=foreground_labels, remove_channel_from_source=True)
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=[*list(regions), ignore_label] if ignore_label is not None else regions, channel_in_seg=0
                )
            )

        # transforms.append(ZscoreNormalization())

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        # Remove mirroring
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        # A tensor with GPU augmentations, a list without them. `get_training_transforms`
        # hands the dataloader `deep_supervision_scales=None` when the config has a GPU
        # section, because the mask is still to be augmented below and the downsampling has
        # to follow that, so the dataloader returns one full-resolution mask. A CPU-only
        # config keeps the downsampling in the dataloader, so `target` arrives as the list
        # of levels. Assuming the tensor raised AttributeError on the first batch of every
        # CPU-only run; upstream `validation_step` has always branched here.
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast only on CUDA: on cpu it is very slow, and mps complains that it is not
        # implemented even with `enabled=False`. Rationale verbatim in nnUNetTrainer.train_step.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # GPU augmentations run on the full-resolution data/target, then the
            # deep-supervision targets are built from the *augmented* mask. A CPU-only
            # config builds no GPU pipeline and the dataloader already produced them.
            reference = None
            if self.transforms_shared is not None:
                # The geometric stage, run once. Both branches see this output, so the
                # one segmentation that follows is correct for both of them.
                data, target = self.transforms_shared(data, target)
                # Not a clone. unit_tests/test_no_inplace_mutation.py pins that no
                # registered transform writes into the tensor it was handed, so the
                # second stage cannot reach back into this one, and a clone here would
                # cost a whole extra volume on every step.
                reference = data
            if self.transforms is not None:
                data, target = self.transforms(data, target)

                deep_supervision_scales = self._get_deep_supervision_scales()
                if deep_supervision_scales is not None:
                    ds_transform = DownsampleSegForDSTransformCustom(ds_scales=deep_supervision_scales)
                    target = ds_transform(target)

            output = self.network(data)
            # del data
            l = self.loss(output, target)
            if reference is not None and self.consistency is not None:
                l = l + self.consistency_weight * self._consistency_term(output, reference)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            # A non-finite loss needs no handling here: `unscale_` records `found_inf`,
            # `step` skips the update, and the next `zero_grad(set_to_none=True)` clears
            # the poisoned gradients. The weights are never touched.
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self._guard_grad_scale()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            # No scaler means CPU or MPS, where nothing skips the step for us -- and a
            # single NaN gradient applied here poisons every weight it touches, for the
            # rest of the run. `l` is already on the host, so the check is free.
            if bool(torch.isfinite(l)):
                self.optimizer.step()

        # One host transfer, shared by the guard and the return value; nnU-Net pays for it
        # here anyway, so a healthy step costs one `isfinite` on a scalar. Checking straight
        # after `self.loss(...)` would sync between forward and backward on every step.
        loss_cpu = l.detach().cpu()
        if not bool(torch.isfinite(loss_cpu)):
            self._report_nonfinite_loss(loss_cpu, data, target, output, batch.get("keys"))
        # nnU-Net's `on_train_epoch_end` reads only "loss", so the extra key is inert
        # upstream and the override below averages it. Absent when the term is off.
        if self._consistency_last is not None:
            return {"loss": loss_cpu.numpy(), "consistency": float(self._consistency_last)}
        return {"loss": loss_cpu.numpy()}

    def _guard_grad_scale(self) -> None:
        """Stop the AMP gradient scale ratcheting down to zero, which kills the run.

        `GradScaler` halves the scale on every step with a non-finite gradient and only
        doubles it back after `growth_interval` *consecutive* clean steps -- 2000 by
        default, which is eight epochs at nnU-Net's 250 iterations. The asymmetry means
        any non-finite rate above one step in 2000 drives the scale down monotonically,
        and nothing in torch floors it: from the initial 65536 it reaches denormals in
        ~150 halvings and then exactly 0.0.

        Zero is terminal, not merely small. `scale(loss)` gives 0, so backward produces
        zero gradients; `unscale_` then multiplies by `1/0 = inf` and 0*inf is NaN. The
        scaler finds the NaN, halves 0 to 0, and the run can never recover -- it keeps
        logging epochs having stopped learning entirely, and the denormal steps on the
        way down can push non-finite values into the weights before it gets there.

        Clamping at `self._min_grad_scale` keeps the run recoverable: bad steps are still
        skipped, and once the batches are clean again the scale grows back normally. A
        scale pinned to the floor is itself the diagnosis, so it is logged.
        """
        if self.grad_scaler.get_scale() >= self._min_grad_scale:
            return

        self._scale_floor_hits_this_epoch += 1
        self._scale_floor_hits_total += 1
        # `update(new_scale=...)` is the public way to set it, and resets the growth
        # tracker so the floor does not immediately get halved again by a stale count.
        self.grad_scaler.update(new_scale=self._min_grad_scale)

        if self._scale_floor_hits_this_epoch <= NAN_REPORTS_PER_EPOCH:
            self.print_to_log_file(
                f"AMP gradient scale fell below {self._min_grad_scale} and was reset to it "
                f"(epoch={self.current_epoch}, {self._scale_floor_hits_total} times since the run started). "
                "Steps are being skipped for non-finite gradients often enough that the scale cannot recover; "
                "look for the non-finite loss reports above for the cause."
            )
        elif self._scale_floor_hits_this_epoch == NAN_REPORTS_PER_EPOCH + 1:
            self.print_to_log_file("AMP gradient scale: further resets suppressed this epoch; see the epoch summary")

    def _consistency_term(self, output, reference: torch.Tensor) -> torch.Tensor:
        """Soft dice between the augmented and the reference prediction.

        Only the full-resolution head takes part. The supervised term spreads over the
        deep-supervision heads with the weights `nnUNetTrainer._build_loss` builds --
        including its `weights[-1] = 0` and the DDP-dependent `1e-6` variant of it --
        and re-deriving them here would copy a decision that belongs upstream. Head 0
        already carries most of the supervised weight, and it is the resolution the
        shared geometric stage is exact at.

        The reference is handed over as a *probability map*, not a label map:
        `MemoryEfficientSoftDiceLoss` takes its `x.shape == y.shape` branch and uses it
        as-is. Its sibling `SoftDiceLoss` cannot -- `get_tp_fp_fn_tn` does `~y_onehot`,
        which is a TypeError on a float tensor -- which is why this is the one of the
        two nnU-Net dice classes used here.

        Note that with a soft target the term does not bottom out at -1 when the two
        branches agree: its minimiser over the augmented prediction is the *hard* argmax
        of the reference, so it aligns the two branches and sharpens them at once. The
        logged number is therefore a dice against a soft target, not an agreement score.
        """
        criterion = self.consistency
        if criterion is None:  # pragma: no cover -- train_step guards the call
            raise RuntimeError("_consistency_term needs a criterion; loss.consistency is not enabled")

        with torch.no_grad() if self.consistency_detach else dummy_context():
            head = self.network(reference)
            head = head[0] if isinstance(head, (list, tuple)) else head
            probabilities = self.consistency_nonlin(head)

        prediction = output[0] if isinstance(output, (list, tuple)) else output
        term = criterion(prediction, probabilities)

        detached = term.detach()
        self._consistency_last = detached
        self._consistency_sum = detached if self._consistency_sum is None else self._consistency_sum + detached
        self._consistency_steps += 1
        return term

    def _report_nonfinite_loss(self, loss, data, target, output, keys) -> None:
        """Say *why* the loss went non-finite, in the training log.

        The verdict is the point. A batch that was already non-finite when it reached the
        network can only have come from the augmentation pipeline; a finite batch with a
        non-finite output is the network diverging; a finite output with a non-finite
        loss is the criterion. Three different bugs, three different fixes, and until now
        no line in the log distinguishing them -- or indeed mentioning any of them.

        Everything here scans whole volumes, which is why it runs only after the scalar
        check has already tripped.
        """
        self._nan_steps_this_epoch += 1
        self._nan_steps_total += 1
        if self._nan_steps_this_epoch > NAN_REPORTS_PER_EPOCH:
            if self._nan_steps_this_epoch == NAN_REPORTS_PER_EPOCH + 1:
                self.print_to_log_file("non-finite loss: further reports suppressed this epoch; see the epoch summary")
            return

        # `target` is a list once deep supervision has downsampled it, and a tensor
        # before that; `output` is a list of heads, highest resolution first.
        seg = target[0] if isinstance(target, (list, tuple)) else target
        head = output[0] if isinstance(output, (list, tuple)) else output

        bad = ~torch.isfinite(data)
        samples = torch.nonzero(bad.flatten(1).any(1)).flatten().tolist()
        if samples:
            verdict = "augmentation"
            detail = f"nonfinite_samples={samples} nonfinite_voxels={int(bad.sum())}"
        elif not bool(torch.isfinite(head).all()):
            dead = any(not bool(torch.isfinite(p).all()) for p in self.network.parameters())
            verdict, detail = "network", f"output_nonfinite=True weights_nonfinite={dead}"
        elif self._consistency_last is not None and not bool(torch.isfinite(self._consistency_last)):
            # Without this branch a NaN out of the reference forward is reported as
            # verdict=loss, which points the reader at nnU-Net's criterion -- the wrong file.
            verdict, detail = "consistency", "data and output are finite; the consistency term is not"
        else:
            verdict, detail = "loss", "data and output are finite"

        consistency = f" consistency={float(self._consistency_last)}" if self._consistency_last is not None else ""
        self.print_to_log_file(
            f"non-finite loss: value={float(loss)} verdict={verdict} epoch={self.current_epoch} "
            f"{detail}{consistency} target_fg_voxels={int((seg > 0).sum())} "
            f"keys={list(keys) if keys is not None else 'n/a'}"
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._nan_steps_this_epoch = 0
        self._scale_floor_hits_this_epoch = 0
        self._consistency_sum = None
        self._consistency_steps = 0

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        if self._nan_steps_this_epoch:
            # Printed here rather than in `on_epoch_end` so it sits beside the epoch's
            # training section, and still prints if validation later throws. It is also
            # the explanation for the `train_loss nan` two lines below it.
            self.print_to_log_file(
                f"non-finite loss in {self._nan_steps_this_epoch}/{self.num_iterations_per_epoch} batches "
                f"this epoch ({self._nan_steps_total} since the run started); "
                f"those steps did not update the weights"
            )
        if self._scale_floor_hits_this_epoch:
            self.print_to_log_file(
                f"AMP gradient scale hit the floor in {self._scale_floor_hits_this_epoch}/"
                f"{self.num_iterations_per_epoch} batches this epoch "
                f"({self._scale_floor_hits_total} since the run started); "
                f"without the floor the run would be unrecoverable by now"
            )
        if self._consistency_steps and self._consistency_sum is not None:
            # `nnUNetLogger.log` asserts the key already exists in its
            # `my_fantastic_logging` dict, so a new curve cannot be added without
            # patching upstream. Printed beside the epoch's training section instead,
            # like the non-finite-loss summary above -- and it is the only way to read
            # the epoch line, whose train_loss now carries both terms.
            mean = float(self._consistency_sum) / self._consistency_steps
            self.print_to_log_file(
                f"consistency: soft dice {-mean:.4f} over {self._consistency_steps} batches, "
                f"weight {self.consistency_weight}, contributing {self.consistency_weight * mean:+.4f} to train_loss"
            )
