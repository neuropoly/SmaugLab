"""Pull batches exactly as `nnUNetTrainerDAExtGPU` sees them, and re-apply a
substituted GPU config to them.

Every other way of looking at a SmaugLab augmentation builds the augmentor by hand
and feeds it a volume loaded from disk. That answers "what does this transform do to
an image", which is not the same question as "what does the network get". The
differences are not cosmetic: nnU-Net samples patches out of the *preprocessed*
store with foreground oversampling, runs its own `SpatialTransform` first, and --
this is the one that bites -- hands the GPU pipeline a **single-channel integer
label map**, not the one-hot mask the standalone scripts pass.

That last one changes behaviour. `RandomRedistributeSegGPU` takes its region count
from the mask's channel count (`masks = seg_b.bool(); R = masks.shape[0]`), and
`RandomSynthSegGPU` normalises per-class weights across channels. With one channel
both collapse to a single foreground region. Whether that is desirable is a separate
argument; it is what training does, so it is what this module reproduces.

So: mirror the trainer rather than re-deriving it. The CPU pipeline comes from the
real `get_dataloaders()`, the GPU pipeline from the real `__init__`, and the
application from the real `train_step`.

Nothing here is Dataset014-specific; the driver supplies the paths.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from smauglab.transforms.gpu.base import record_applications

#: The trainer's own env var (`smauglab.trainers.nnUNetTrainerDAExt.CONFIG_ENV`),
#: repeated rather than imported because importing that module pulls in nnunetv2,
#: which must not happen before `bind_nnunet_env` has run.
SMAUGLAB_ENV = "SMAUGLAB_PARAMS_JSON"


def bind_nnunet_env(*, raw: Path | str, preprocessed: Path | str, results: Path | str, n_proc_da: int = 0) -> None:
    """Point nnU-Net at a dataset, before anything imports it.

    `nnunetv2/paths.py` reads these three variables at *import* time and
    `nnUNetTrainer` does `from nnunetv2.paths import nnUNet_preprocessed,
    nnUNet_results`, binding the values into its own module namespace. Setting the
    environment afterwards therefore changes nothing, silently, and the trainer goes
    looking in whatever tree the shell happened to point at -- or worse, writes into
    it. Hence the guard: this is not a style preference, it is the only moment the
    assignment has any effect.

    `nnUNet_n_proc_DA` is different -- `get_allowed_n_proc_DA()` reads it at call
    time -- but it belongs with the others. Zero is deliberate: it makes
    `get_dataloaders()` build a `SingleThreadedAugmenter`, which runs in-process and
    so is reproducible from a seed, instead of the 40-worker non-deterministic one.
    """
    if "nnunetv2.paths" in sys.modules:
        raise RuntimeError(
            "nnunetv2 was imported before bind_nnunet_env(); the nnUNet_* paths are already "
            "bound and this call would do nothing. Move the call above the import."
        )
    os.environ["nnUNet_raw"] = str(raw)  # noqa: SIM112  (nnU-Net's own spelling)
    os.environ["nnUNet_preprocessed"] = str(preprocessed)  # noqa: SIM112  (nnU-Net's own spelling)
    os.environ["nnUNet_results"] = str(results)  # noqa: SIM112  (nnU-Net's own spelling)
    os.environ["nnUNet_n_proc_DA"] = str(n_proc_da)  # noqa: SIM112  (nnU-Net's own spelling)
    Path(results).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainerSpec:
    """Everything needed to rebuild the trainer, minus the augmentation config."""

    plans_path: Path
    dataset_json_path: Path
    configuration: str = "3d_fullres"
    fold: int = 0
    device: str = "cuda"


@dataclass
class CachedBatch:
    """One dataloader batch, held on the CPU.

    `data`/`target` are post-CPU-pipeline and pre-GPU-augmentation, i.e. exactly what
    `train_step` receives. `target` is `[B, 1, *patch]`; if it ever arrives as a list
    the deep-supervision downsampling has not been deferred and the config's GPU
    section is empty -- see `cache_train_batches`.
    """

    index: int
    data: torch.Tensor
    target: torch.Tensor
    keys: list[str]
    bboxes: list[list[list[int]]] = field(default_factory=list)

    def stems(self) -> list[str]:
        return [f"{key}_b{self.index}s{s}" for s, key in enumerate(self.keys)]


def make_trainer(spec: TrainerSpec, config_path: Path | str):
    """Build the real trainer around one SmaugLab config, without `initialize()`.

    `initialize()` also builds the network, the optimizer and the loss, none of which
    a dump needs, and it is where several hundred MB of ResEnc weights come from. The
    one thing it does that we *do* need is `_set_batch_size_and_oversample()` -- for a
    non-DDP run that is the single line `self.batch_size =
    self.configuration_manager.batch_size` -- so call that and nothing else.
    `get_dataloaders()` touches no other attribute that `__init__` left unset.
    """
    from smauglab.config import load_config
    from smauglab.trainers.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU

    os.environ[SMAUGLAB_ENV] = str(config_path)
    # Keyed on the path string. `write_temp_config` is content-addressed so distinct
    # configs already get distinct paths, but clearing costs nothing and removes the
    # failure mode where a caller reuses one filename and silently gets a stale parse.
    load_config.cache_clear()

    plans = json.loads(Path(spec.plans_path).read_text())
    dataset_json = json.loads(Path(spec.dataset_json_path).read_text())
    trainer = nnUNetTrainerDAExtGPU(plans, spec.configuration, spec.fold, dataset_json, torch.device(spec.device))
    trainer._set_batch_size_and_oversample()
    return trainer


def _instrument_bbox(loader) -> list:
    """Record the crop each sample came from; the dataloader does not return it."""
    recorded: list = []
    original = loader.get_bbox

    def wrapper(*args, **kwargs):
        lbs, ubs = original(*args, **kwargs)
        recorded.append([[int(v) for v in lbs], [int(v) for v in ubs]])
        return lbs, ubs

    loader.get_bbox = wrapper
    return recorded


def cache_train_batches(spec: TrainerSpec, base_config_path: Path | str, *, n_batches: int, seed: int) -> list[CachedBatch]:
    """Stage A: draw `n_batches` real training batches and keep them.

    Cached rather than re-drawn per augmentation for two reasons. Every augmentation
    then sees byte-identical input, which is the only way the resulting images can be
    compared against each other; and the preprocessed store is read once instead of
    thirty times.

    The base config must keep a non-empty GPU section even though this stage never
    applies it. `get_training_transforms` passes `deep_supervision_scales=None` to the
    tail *only* when the config has GPU augmentations; with an empty GPU section
    nnU-Net downsamples the mask in the dataloader and `target` comes back as a list
    of five tensors instead of one full-resolution mask. The assert below is what
    catches that.
    """
    trainer = make_trainer(spec, base_config_path)
    if trainer.transforms is None:
        raise RuntimeError(
            f"{base_config_path} has an empty GPU section. The dataloader would then return "
            "deep-supervision targets instead of one full-resolution mask, and the cache would "
            "be the wrong shape for every downstream pass."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    mt_train, _mt_val = trainer.get_dataloaders()
    # get_dataloaders() burns one batch from each side to spin the workers up; that
    # happens before this point, so the recorder only ever sees batches we keep.
    boxes = _instrument_bbox(mt_train.data_loader)

    patch_size = tuple(trainer.configuration_manager.patch_size)
    batches: list[CachedBatch] = []
    for index in range(n_batches):
        seen = len(boxes)
        raw = next(mt_train)
        data, target = raw["data"], raw["target"]

        assert isinstance(target, torch.Tensor), (
            "target came back as a list, so deep-supervision downsampling ran in the dataloader. "
            "The config's GPU section is empty -- see the docstring."
        )
        assert data.shape == (trainer.batch_size, 1, *patch_size), f"unexpected data shape {tuple(data.shape)}"
        assert target.shape == (trainer.batch_size, 1, *patch_size), f"unexpected target shape {tuple(target.shape)}"
        assert not torch.is_floating_point(target), f"target should be integral, got {target.dtype}"

        batches.append(
            CachedBatch(
                index=index,
                data=data.detach().cpu().clone(),
                target=target.detach().cpu().clone(),
                keys=list(raw["keys"]),
                bboxes=boxes[seen:],
            )
        )
        print(f"  batch {index}: {raw['keys']}", flush=True)

    del trainer
    torch.cuda.empty_cache()
    return batches


def save_batches(batches: Sequence[CachedBatch], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([vars(b) for b in batches], path)
    return path


def load_batches(path: Path | str) -> list[CachedBatch]:
    return [CachedBatch(**entry) for entry in torch.load(path, weights_only=False)]


def apply_gpu_config(
    trainer, batch: CachedBatch, *, seed: int, verify_ds: bool = False, trace: list | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The augmentation half of `train_step`, verbatim, on one cached batch.

    `.clone()` is load-bearing: several transforms write through their input
    (`RandomRedistributeSegGPU` ends with `input[b, c] = x`), so without it the cache
    would carry the previous augmentation's output into the next one.

    Deep supervision is not applied to what gets written. The first `ds_scale` is
    `[1, 1, 1]`, so level 0 of what the network sees *is* this full-resolution mask;
    the remaining four are downsampled copies and dumping them would be noise.
    `verify_ds` runs the transform anyway and checks the shapes, which is the part
    worth knowing is still true.

    Pass a list as `trace` to have it filled with the transforms that actually fired
    (see `record_applications`). For a single-augmentation pass the directory name
    already says what ran; for a combined pass it is the only record.
    """
    from torch import autocast

    from smauglab.trainers.utils import DownsampleSegForDSTransformCustom

    device = trainer.device
    torch.manual_seed(seed)

    data = batch.data.clone().to(device, non_blocking=True)
    target = batch.target.clone().to(device, non_blocking=True)

    recorder = record_applications(trainer.transforms) if trace is not None else contextlib.nullcontext([])
    with autocast(device.type, enabled=True) if device.type == "cuda" else _nullcontext():
        with recorder as fired:
            data, target = trainer.transforms(data, target)
        if trace is not None:
            trace.extend(fired)

        if verify_ds:
            scales = trainer._get_deep_supervision_scales()
            if scales is not None:
                levels = DownsampleSegForDSTransformCustom(ds_scales=scales)(target)
                expected = [tuple(round(s * a) for s, a in zip(scale, target.shape[2:])) for scale in scales]
                got = [tuple(level.shape[2:]) for level in levels]
                assert got == expected, f"deep supervision shapes {got} != {expected}"

    # autocast leaves the image in half precision, and any geometric resampling of the
    # mask comes back as float; both have to be put back before they hit a NIfTI.
    return data.detach().float().cpu(), target.detach().round().to(torch.int16).cpu()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def write_pair(image: np.ndarray, seg: np.ndarray, out_dir: Path, stem: str, spacing: Sequence[float]) -> None:
    """One image/mask pair in nnU-Net's own naming.

    The affine is a plain diagonal of the plans spacing: these volumes live in
    nnU-Net's internal axis order, after `transpose_forward`, which is the space the
    network works in and the space the metrics need. It is not the source scan's
    anatomical orientation and does not claim to be.
    """
    import nibabel as nib

    out_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([*[float(s) for s in spacing], 1.0])
    nib.save(nib.Nifti1Image(np.ascontiguousarray(image, dtype=np.float32), affine), str(out_dir / f"{stem}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(np.ascontiguousarray(seg, dtype=np.int16), affine), str(out_dir / f"{stem}.nii.gz"))


def write_batch(
    data: torch.Tensor, target: torch.Tensor, batch: CachedBatch, out_dir: Path, spacing: Sequence[float], draw: int | None = None
) -> int:
    """Write every sample of one (possibly augmented) batch. Returns how many."""
    suffix = "" if draw is None else f"_i{draw}"
    for sample, stem in enumerate(batch.stems()):
        write_pair(
            data[sample, 0].numpy(),
            target[sample, 0].numpy(),
            out_dir,
            f"{stem}{suffix}",
            spacing,
        )
    return len(batch.keys)
