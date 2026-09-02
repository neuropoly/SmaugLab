"""Rewrite a pre-registry config into the current format.

Config keys used to be loosely-related labels: `ScharrTransform` built a
`RandomConvTransformGPU(kernel_type="Scharr")`, `SynthSeg` built a
`RandomSynthSegGPU`, and `nnUNetSpatialTransform` built nothing at all in the GPU
pipeline. A key is now the class name, exactly, and a parameter is a constructor
argument, exactly.

Nothing here is consulted when *loading* a config -- old spellings do not work, by
design. This module exists only to rewrite files, so that configs sitting in old
experiment output folders can be brought forward instead of silently rotting. A
test asserts that no legacy key in `LEGACY_KEYS` resolves through the registry.

Usage:
    smauglab migrate OLD.json -o NEW.json
    smauglab migrate --check smauglab/configs/*.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from smauglab import registry
from smauglab.registry import Backend

# --- key mapping -----------------------------------------------------------------

#: Legacy GPU config key -> current class name.
LEGACY_GPU_KEYS: dict[str, str] = {
    "FlipTransform": "RandomFlipTransformGPU",
    "AffineTransform": "RandomAffineGPU",
    "SynthSeg": "RandomSynthSegGPU",
    "RandomPALETTETransform": "RandomPaletteGPU",
    # The class was renamed twice before the registry existed; both spellings appear
    # in configs_paul and both mean what is now RandomPaletteGPU.
    "ImageContrastV26_6_2GPUTransform": "RandomPaletteGPU",
    "RandomDomainTransferGPU": "RandomDomainTransferGPU",
    "DomainTransferTransform": "RandomDomainTransferGPU",
    "InverseTransform": "RandomInverseGPU",
    "HistogramEqualizationTransform": "RandomHistogramEqualizationGPU",
    "RedistributeSegTransform": "RandomRedistributeSegGPU",
    "ScharrTransform": "RandomScharrGPU",
    "UnsharpMaskTransform": "RandomUnsharpMaskGPU",
    "RandomConvTransform": "RandomRandConvGPU",
    "ClampTransform": "RandomClampGPU",
    "GaussianNoiseTransform": "RandomGaussianNoiseGPU",
    "GaussianBlurTransform": "RandomGaussianBlurGPU",
    "BrightnessTransform": "RandomBrightnessGPU",
    "GammaTransform": "RandomGammaGPU",
    "InvGammaTransform": "RandomInvGammaGPU",
    "ContrastTransform": "RandomContrastGPU",
    "SimulateLowResTransform": "RandomLowResTransformGPU",
    "AcqTransform": "RandomAcqTransformGPU",
    "CropTransform": "RandomCropTransformGPU",
    "BiasFieldTransform": "RandomBiasFieldGPU",
    "ZscoreNormalizationTransform": "ZscoreNormalizationGPU",
}

#: Legacy CPU config key -> current class name. Most already matched their class.
LEGACY_CPU_KEYS: dict[str, str] = {
    "HistogramEqualTransform": "HistogramEqualTransform",
    "RedistributeTransform": "RedistributeTransform",
    "ShapeTransform": "ShapeTransform",
    "ArtifactTransform": "ArtifactTransform",
    "SpatialCustomTransform": "SpatialCustomTransform",
    "SpatialTransform": "SpatialTransform",
    "GaussianNoiseTransform": "GaussianNoiseTransform",
    "GaussianBlurTransform": "GaussianBlurTransform",
    "MultiplicativeBrightnessTransform": "MultiplicativeBrightnessTransform",
    "ContrastTransform": "ContrastTransform",
    "SimulateLowResolutionTransform": "SimulateLowResolutionTransform",
    "GammaTransform": "GammaTransform",
    "GammaTransform_invert": "InvertedGammaTransform",
}

#: `FunctionTransform` was one key that the builder fanned out over a hardcoded
#: lambda list, so it becomes five blocks -- one per function, in ladder order.
FUNCTION_LEAVES = {
    Backend.GPU: ["RandomLog1pGPU", "RandomSqrtGPU", "RandomSinGPU", "RandomExpGPU", "RandomSigmoidGPU"],
    Backend.CPU: ["Log1pTransform", "SqrtTransform", "SinTransform", "ExpTransform", "SigmoidTransform"],
}

#: `ConvTransform` carried its kernel in a parameter; the kernel now picks the class.
CPU_CONV_BY_KERNEL = {"Laplace": "LaplaceConvTransform", "Scharr": "ScharrConvTransform"}

#: Which config parameters the old builder actually forwarded, per legacy key.
#:
#: Anything a config set outside these sets was silently discarded -- the builder
#: simply never read it, and `**kwargs` swallowed the rest. Carrying such a value
#: forward would *activate* a setting that has never had any effect, so the migrator
#: drops it and says so. `transform_params_gpu_inoutseg.json` really does set
#: `mix_prob` on `GaussianBlurTransform`, which the builder never passed on.
#:
#: Derived mechanically from `_build_transforms` while it still existed.
FORWARDED_GPU_PARAMS: dict[str, set[str]] = {
    "FlipTransform": {"flip_axis", "keepdim", "probability", "same_on_batch"},
    "AffineTransform": {"degrees", "probability", "resample", "scale", "shear", "translate"},
    "SynthSeg": {"probability"},
    "RandomPALETTETransform": {
        "alpha_magnitude_range",
        "blur_sigmas",
        "c_choices",
        "dark_threshold",
        "label_classes",
        "label_remap_prob",
        "min_label_voxels",
        "n_kmeans_subsample",
        "probability",
        "s_choices",
        "skip_parcellation_prob",
        "skip_sub_parc_prob",
    },
    "RandomDomainTransferGPU": {
        "any_source",
        "apply_to_channel",
        "bank_path",
        "bias_field_std",
        "bias_scale",
        "blend_concentration",
        "blend_targets",
        "include_self",
        "p_class_mix",
        "p_spatial_mix",
        "pct",
        "probability",
        "same_on_batch",
        "sigma",
        "source_label",
        "spatial_mix_gain",
        "spatial_mix_scale",
        "targets",
        "zscore_io",
    },
    "InverseTransform": {"in_seg", "mix_in_out", "mix_prob", "out_seg", "probability", "retain_stats"},
    "HistogramEqualizationTransform": {"in_seg", "mix_in_out", "mix_prob", "out_seg", "probability", "retain_stats"},
    "RedistributeSegTransform": {"dilation_iterations_range", "in_seg", "probability", "retain_stats", "std_noise_range"},
    "ScharrTransform": {"absolute", "in_seg", "mix_in_out", "mix_prob", "out_seg", "probability", "retain_stats"},
    "UnsharpMaskTransform": {"in_seg", "mix_in_out", "mix_prob", "out_seg", "probability", "sigma", "unsharp_amount"},
    "RandomConvTransform": {"in_seg", "kernel_sizes", "mix_in_out", "mix_prob", "out_seg", "probability", "retain_stats"},
    "ClampTransform": {"in_seg", "max_clamp_amount", "mix_in_out", "out_seg", "probability", "retain_stats"},
    "GaussianNoiseTransform": {"in_seg", "mean", "mix_in_out", "out_seg", "probability", "std"},
    "GaussianBlurTransform": {"in_seg", "mix_in_out", "out_seg", "probability", "sigma"},
    "BrightnessTransform": {"brightness_range", "in_seg", "mix_in_out", "out_seg", "probability"},
    "GammaTransform": {"gamma_range", "in_seg", "mix_in_out", "out_seg", "probability", "retain_stats"},
    "InvGammaTransform": {"gamma_range", "in_seg", "mix_in_out", "out_seg", "probability", "retain_stats"},
    "ContrastTransform": {"contrast_range", "in_seg", "mix_in_out", "out_seg", "probability", "retain_stats"},
    "FunctionTransform": {"in_seg", "mix_in_out", "out_seg", "probability", "retain_stats"},
    "SimulateLowResTransform": {"probability", "same_on_batch", "scale"},
    "AcqTransform": {"probability", "same_on_batch", "scale"},
    "CropTransform": {"crop", "pos", "probability", "same_on_batch"},
    "BiasFieldTransform": {"coefficients", "in_seg", "mix_in_out", "out_seg", "probability", "retain_stats"},
    "ZscoreNormalizationTransform": {"probability"},
}

FORWARDED_CPU_PARAMS: dict[str, set[str]] = {
    "ConvTransform": {"absolute", "kernel_type", "probability", "retain_stats"},
    "FunctionTransform": {"probability", "retain_stats"},
    "HistogramEqualTransform": {"probability", "retain_stats"},
    "RedistributeTransform": {"in_seg", "probability", "retain_stats"},
    "ShapeTransform": {"ignore_axes", "probability", "shape_min"},
    "ArtifactTransform": {"bias_field", "blur", "ghosting", "motion", "noise", "probability", "random_pick", "spike", "swap"},
    "SpatialCustomTransform": {"affine", "anisotropy", "elastic", "flip", "probability", "random_pick"},
    "SpatialTransform": {
        "bg_style_seg_sampling",
        "p_elastic_deform",
        "p_rotation",
        "p_scaling",
        "p_synchronize_scaling_across_axes",
        "patch_center_dist_from_border",
        "random_crop",
        "scaling",
    },
    "GaussianNoiseTransform": {"noise_variance", "p_per_channel", "probability", "synchronize_channels"},
    "GaussianBlurTransform": {"benchmark", "blur_sigma", "p_per_channel", "probability", "synchronize_axes", "synchronize_channels"},
    "MultiplicativeBrightnessTransform": {"multiplier_range", "p_per_channel", "probability", "synchronize_channels"},
    "ContrastTransform": {"contrast_range", "p_per_channel", "preserve_range", "probability", "synchronize_channels"},
    "SimulateLowResolutionTransform": {
        "allowed_channels",
        "ignore_axes",
        "p_per_channel",
        "probability",
        "scale",
        "synchronize_axes",
        "synchronize_channels",
    },
    "GammaTransform_invert": {"gamma", "p_per_channel", "p_retain_stats", "probability", "synchronize_channels"},
    "GammaTransform": {"gamma", "p_invert_image", "p_per_channel", "p_retain_stats", "probability", "synchronize_channels"},
}

#: SynthSeg is the exception: the builder forwarded every key except `probability`
#: straight to SynthSegGenerator, so its accepted set is the generator's signature.
FORWARD_EVERYTHING = {"SynthSeg"}

#: Renamed parameters.
PARAM_RENAMES = {"probability": "p", "shear": "shears"}

#: Values the old builder hardcoded, which no config carried and no class defaults to.
#: Without these the migrated config would quietly change behaviour -- nnU-Net's
#: SpatialTransform defaults `mode_seg` to "bilinear", which would interpolate
#: segmentation labels into fractional values instead of resampling them nearest.
LADDER_CONSTANTS: dict[tuple[str, str], dict[str, Any]] = {
    ("CPU", "SpatialTransform"): {"mode_seg": "nearest"},
}

#: Keys whose transform no longer exists at all. The classes were deleted or only
#: ever lived on a branch, so these blocks have been silently doing nothing.
DEAD_KEYS = {
    "ImageContrastGPUTransform": "the class was removed in 0d6270d; its num_bins parameter maps to nothing that still exists",
    "PaletteSynthesisTransform": "only ever existed on the palette-refactor branch",
}

#: Every legacy spelling, for the test that proves none of them still resolve.
LEGACY_KEYS = set(LEGACY_GPU_KEYS) | set(LEGACY_CPU_KEYS) | set(DEAD_KEYS) | {"FunctionTransform", "ConvTransform"}

RESERVED_SECTIONS = ("GPU", "CPU", "MONAI", "pipeline")


class MigrationError(Exception):
    """The config cannot be migrated without a human deciding something."""


#: Parameter names that only ever appear in a pre-registry config.
_LEGACY_PARAM_MARKERS = frozenset({"probability", "shear", "kernel_type"})


def is_current(payload: dict) -> bool:
    """True if the document is already in the current format.

    Decided per document, not per key, because most CPU legacy keys *are* also
    current class names (`RedistributeTransform` names a class and named one
    before). Keying off that coincidence made the migrator skip the
    forwarded-parameter filter on legacy CPU configs and leak parameters the old
    builder never passed on.
    """
    if not any(section in payload for section in ("GPU", "CPU")):
        return False  # flat legacy layout, backend never stated
    for backend in (Backend.GPU, Backend.CPU):
        section = payload.get(backend.value)
        if not isinstance(section, dict):
            continue
        known = set(registry.names(backend))
        for key, block in section.items():
            if key.startswith("_"):
                continue
            if key not in known:
                return False
            if isinstance(block, dict) and _LEGACY_PARAM_MARKERS & set(block):
                return False
    return True


def _looks_like_gpu(section: dict) -> bool:
    """A flat legacy config carries no section marker, so infer from its keys."""
    gpu_only = {"FlipTransform", "AffineTransform", "ScharrTransform", "ZscoreNormalizationTransform", "AcqTransform"}
    cpu_only = {"SpatialCustomTransform", "ArtifactTransform", "ShapeTransform", "GammaTransform_invert"}
    return len(gpu_only & set(section)) >= len(cpu_only & set(section))


def _migrate_block(new_name: str, params: dict, backend: Backend, source: str, legacy_key: str | None = None) -> dict[str, Any]:
    """Rename parameters, and drop the ones that never reached the transform."""
    entry = registry.get(new_name, backend)
    accepted = set(registry.accepted_params(entry))
    forwarded = (FORWARDED_GPU_PARAMS if backend is Backend.GPU else FORWARDED_CPU_PARAMS).get(legacy_key or "")

    out: dict[str, Any] = {}
    for key, value in params.items():
        name = PARAM_RENAMES.get(key, key)
        if name in entry.context_params:
            # Supplied by the trainer at runtime; a config value was always ignored.
            continue
        if forwarded is not None and legacy_key not in FORWARD_EVERYTHING and key not in forwarded:
            # The old builder never read this key, so it has never had an effect.
            # Keeping it would switch on a setting that has always been dormant.
            continue
        if name not in accepted:
            # Silently swallowed by **kwargs before, so dropping it changes nothing.
            continue
        out[name] = value
    if backend is Backend.GPU and "p" not in out:
        # The old builder defaulted an absent probability to 0 (the block was there
        # but off). Spelling it out keeps that meaning now that an absent block is
        # what "not in the pipeline" means.
        out["p"] = 0
    # Values the builder hardcoded rather than reading from the config.
    out.update(LADDER_CONSTANTS.get((backend.value, new_name), {}))
    _ = source
    return out


def _migrate_section(section: dict, backend: Backend, source: str) -> tuple[dict, list[str]]:
    """Return the migrated section plus a list of human-readable notes."""
    keymap = LEGACY_GPU_KEYS if backend is Backend.GPU else LEGACY_CPU_KEYS
    forwarded_table = FORWARDED_GPU_PARAMS if backend is Backend.GPU else FORWARDED_CPU_PARAMS
    migrated: dict[str, Any] = {}
    notes: list[str] = []

    # The CPU builder read `retain_stats` from the *top level* of the config and
    # applied it to every block that takes one, rather than reading it per block.
    # Push it down so each transform states its own value.
    shared_retain_stats = section.get("retain_stats") if backend is Backend.CPU else None
    if shared_retain_stats is not None:
        notes.append(f"pushed top-level retain_stats={shared_retain_stats} into the blocks that read it")

    for key, value in section.items():
        if key.startswith("_") or key in RESERVED_SECTIONS:
            continue
        if key in ("Comment1", "Comment2"):
            migrated[f"_comment{key[-1]}"] = value
            continue
        if key in DEAD_KEYS:
            notes.append(f"dropped {key!r}: {DEAD_KEYS[key]}")
            continue
        if key in ("mirror_axes", "retain_stats", "nnUNetSpatialTransform", "RandomChooseXTransforms"):
            continue  # relocated by the caller
        if not isinstance(value, dict):
            notes.append(f"dropped non-block key {key!r}")
            continue

        block = value
        if shared_retain_stats is not None and "retain_stats" in forwarded_table.get(key, ()):
            block = {**value, "retain_stats": shared_retain_stats}

        if key == "FunctionTransform":
            for leaf in FUNCTION_LEAVES[backend]:
                migrated[leaf] = _migrate_block(leaf, block, backend, source, legacy_key=key)
            notes.append(f"expanded FunctionTransform into {len(FUNCTION_LEAVES[backend])} blocks")
            continue

        if key == "ConvTransform" and backend is Backend.CPU:
            kernel = block.get("kernel_type", "Scharr")
            if kernel not in CPU_CONV_BY_KERNEL:
                raise MigrationError(f"{source}: ConvTransform has unsupported kernel_type {kernel!r}")
            leaf = CPU_CONV_BY_KERNEL[kernel]
            migrated[leaf] = _migrate_block(leaf, block, backend, source, legacy_key=key)
            notes.append(f"ConvTransform(kernel_type={kernel!r}) -> {leaf}")
            continue

        new_name = keymap.get(key)
        if new_name is None:
            raise MigrationError(
                f"{source}: no migration rule for {backend.value} key {key!r}. "
                "Add it to LEGACY_GPU_KEYS/LEGACY_CPU_KEYS, or delete the block if the transform is gone."
            )
        migrated[new_name] = _migrate_block(new_name, block, backend, source, legacy_key=key)

    # Emit in registry order so the file reads in the order the pipeline runs.
    order = {name: i for i, name in enumerate(registry.names(backend))}
    ordered = {k: migrated[k] for k in sorted(migrated, key=lambda n: (n.startswith("_") is False, order.get(n, 10_000), n))}
    return ordered, notes


def migrate(payload: dict, source: str = "<config>") -> tuple[dict, list[str]]:
    """Migrate a whole config document. Returns (new payload, notes)."""
    payload = copy.deepcopy(payload)
    out: dict[str, Any] = {}
    notes: list[str] = []

    if is_current(payload):
        # Nothing to rewrite; only normalise section order so migrating twice
        # produces the same bytes as migrating once.
        return _ordered(payload), notes

    sections: dict[Backend, dict] = {}
    if "GPU" in payload or "CPU" in payload:
        if isinstance(payload.get("GPU"), dict):
            sections[Backend.GPU] = payload["GPU"]
        if isinstance(payload.get("CPU"), dict):
            sections[Backend.CPU] = payload["CPU"]
    else:
        sections[Backend.GPU if _looks_like_gpu(payload) else Backend.CPU] = payload

    out.update({key: value for key, value in payload.items() if key.startswith("_")})

    for backend, section in sections.items():
        migrated, section_notes = _migrate_section(section, backend, source)
        notes.extend(section_notes)

        # nnUNetSpatialTransform is not a GPU augmentation -- AugTransformsGPU never
        # read it. It configures nnU-Net's own CPU-side SpatialTransform, which the
        # trainer used to fetch by re-opening the JSON a second time.
        spatial = section.get("nnUNetSpatialTransform")
        if isinstance(spatial, dict):
            target = out.setdefault(Backend.CPU.value, {})
            target["SpatialTransform"] = _migrate_block("SpatialTransform", spatial, Backend.CPU, source, legacy_key="SpatialTransform")
            notes.append("moved nnUNetSpatialTransform -> CPU.SpatialTransform")

        if backend is Backend.CPU:
            axes = section.get("mirror_axes")
            if axes:
                migrated["MirrorTransform"] = {"allowed_axes": axes}
                notes.append("moved mirror_axes -> CPU.MirrorTransform.allowed_axes")

        choose = section.get("RandomChooseXTransforms")
        if isinstance(choose, dict):
            out.setdefault("pipeline", {})["random_choose"] = choose
            notes.append("moved RandomChooseXTransforms -> pipeline.random_choose")

        if migrated:
            existing = out.get(backend.value, {})
            out[backend.value] = {**migrated, **existing}

    return _ordered(out), notes


def _ordered(payload: dict) -> dict:
    """Fixed section order, so migrating twice gives the same bytes as once.

    Without this the order depends on whether a relocation
    (nnUNetSpatialTransform -> CPU.SpatialTransform) created the CPU section first.
    """
    order = ["GPU", "CPU", "MONAI", "pipeline"]
    out = {k: payload[k] for k in payload if k.startswith("_")}
    out.update({k: payload[k] for k in order if k in payload})
    out.update({k: v for k, v in payload.items() if k not in out})
    return out


def needs_migration(payload: dict) -> bool:
    """True if the document is not already in the current format."""
    try:
        migrated, _ = migrate(payload)
    except MigrationError:
        return True
    return migrated != payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smauglab migrate", description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="write here instead of alongside the input")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--check", action="store_true", help="exit 1 if any config would change")
    args = parser.parse_args(argv)

    if args.output and len(args.configs) > 1:
        parser.error("-o takes a single input config")

    stale = 0
    for path in args.configs:
        payload = json.loads(path.read_text())
        try:
            migrated, notes = migrate(payload, source=path.name)
        except MigrationError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 2

        text = json.dumps(migrated, indent=4) + "\n"
        if args.check:
            if path.read_text() != text:
                stale += 1
                print(f"needs migration: {path}")
                for note in notes:
                    print(f"    {note}")
            continue

        target = args.output or (path if args.in_place else path.with_suffix(".migrated.json"))
        target.write_text(text)
        print(f"{path} -> {target}")
        for note in notes:
            print(f"    {note}")

    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
