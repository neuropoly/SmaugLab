"""auglab_srcsm_predict: predict with an nnUNetTrainerSRCSM model, SRCSM-style.

SRCSM (Thaler et al., IEEE Access 2025) is the training augmentation PLUS a test-time source
matching step: every test image is histogram-matched to the average intensity CDF of the training
images before prediction. This command chains the two:

  1. source CDF  average CDF of the training images ($nnUNet_raw/<Dataset>/imagesTr/*_0000.nii.gz by
                 default), computed once and cached as srcsm_source_cdf.json in the model folder
  2. match       every input image -> a matched copy (same header/affine, only voxel values change)
  3. predict     nnUNetv2_predict on the matched copies (-tr nnUNetTrainerSRCSM, --disable_tta: the
                 trainer trains without mirroring)

Steps 1-2 run srcsm_auglab_port/source_matching.py (python -m srcsm_auglab_port.source_matching),
which can also be used on its own. --no-source-matching skips 1-2 (plain SRC, the authors' ablation).

Example:
  auglab_srcsm_predict -i imagesTs -o preds -d 101 -c 3d_fullres -f 0 1 2
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", required=True, help="input folder, nnU-Net naming (<case>_0000.nii.gz)")
    ap.add_argument("-o", required=True, help="output folder for the segmentations")
    ap.add_argument("-d", required=True, help="dataset name or id the model was trained on")
    ap.add_argument("-c", default="3d_fullres", help="configuration (default 3d_fullres)")
    ap.add_argument("-p", default="nnUNetPlans", help="plans identifier (default nnUNetPlans)")
    ap.add_argument("-tr", default="nnUNetTrainerSRCSM", help="trainer (default nnUNetTrainerSRCSM)")
    ap.add_argument("-f", nargs="+", default=["0", "1", "2", "3", "4"], help="folds to ensemble (default 0 1 2 3 4)")
    ap.add_argument("-chk", default="checkpoint_best.pth", help="checkpoint (default checkpoint_best.pth)")
    ap.add_argument("-device", default="cuda", help="cuda, cpu or mps (default cuda)")
    ap.add_argument("--modality", choices=("mr", "ct"), default=None,
                    help="intensity branch for the training images (default: ct if the dataset's channel 0 is CT, else mr)")
    ap.add_argument("--input-modality", choices=("mr", "ct"), default=None,
                    help="intensity branch for the inputs, if they differ from the training images (default: --modality)")
    ap.add_argument("--source-images", default=None,
                    help="training images for the source CDF (default $nnUNet_raw/<Dataset>/imagesTr)")
    ap.add_argument("--exclude-cases", default=None,
                    help="JSON list of case ids to leave out of the source CDF (test cases kept inside imagesTr)")
    ap.add_argument("--matched-dir", default=None, help="keep the matched inputs here (default: a temporary folder)")
    ap.add_argument("--workers", type=int, default=4, help="processes for the matching (default 4)")
    ap.add_argument("--no-source-matching", action="store_true", help="predict on the raw inputs (SRC only)")
    args, extra = ap.parse_known_args()   # unknown flags are passed on to nnUNetv2_predict

    from nnunetv2.paths import nnUNet_raw
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    dataset = maybe_convert_to_dataset_name(args.d)
    model_dir = Path(get_output_folder(dataset, args.tr, args.p, args.c))
    if not model_dir.is_dir():
        sys.exit(f"model folder not found: {model_dir}")
    predict = ["nnUNetv2_predict", "-d", dataset, "-c", args.c, "-p", args.p, "-tr", args.tr,
               "-f", *args.f, "-chk", args.chk, "-device", args.device, "--disable_tta", "-o", args.o, *extra]

    if args.no_source_matching:
        _run(predict + ["-i", args.i])
        return

    inputs = sorted(Path(args.i).glob("*.nii.gz"))
    if not inputs:
        sys.exit(f"no .nii.gz in {args.i}")
    if any(not f.name.endswith("_0000.nii.gz") for f in inputs):
        sys.exit("source matching supports single-channel models only (every input must be <case>_0000.nii.gz)")

    modality = args.modality
    if modality is None:
        ch0 = json.loads((model_dir / "dataset.json").read_text())["channel_names"]["0"]
        modality = "ct" if ch0.strip().upper() == "CT" else "mr"
    source = Path(args.source_images or Path(nnUNet_raw) / dataset / "imagesTr").resolve()

    # 1. source CDF (cached in the model folder; rebuilt when the source, modality or exclusions change)
    key = {"source_dir": str(source), "modality": modality,
           "exclude_cases": sorted(json.loads(Path(args.exclude_cases).read_text())) if args.exclude_cases else []}
    cdf = model_dir / "srcsm_source_cdf.json"
    if not os.access(model_dir, os.W_OK):
        cdf = Path(args.o) / "srcsm_source_cdf.json"
    key_file = cdf.with_suffix(".key.json")
    if not (cdf.is_file() and key_file.is_file() and json.loads(key_file.read_text()) == key):
        cmd = [sys.executable, "-m", "srcsm_auglab_port.source_matching", "cdf",
               "--images", source, "--modality", modality, "--out", cdf]
        if args.exclude_cases:
            cmd += ["--exclude-cases", args.exclude_cases]
        _run(cmd)
        key_file.write_text(json.dumps(key, indent=1))
    else:
        print(f"using cached source CDF {cdf}")

    # 2. match, 3. predict
    with tempfile.TemporaryDirectory(prefix="srcsm_matched_") as tmp:
        matched = Path(args.matched_dir or tmp)
        _run([sys.executable, "-m", "srcsm_auglab_port.source_matching", "match", "--cdf", cdf,
              "--in-dir", args.i, "--out-dir", matched, "--modality", args.input_modality or modality,
              "--workers", args.workers,
              "--overwrite"])
        _run(predict + ["-i", matched])


if __name__ == "__main__":
    main()
