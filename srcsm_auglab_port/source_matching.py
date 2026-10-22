#!/usr/bin/env python3
"""
SRCSM's test-time SOURCE MATCHING (Thaler et al., IEEE Access 2025), re-implemented from the authors'
code (github.com/imigraz/SRCSM_Domain_Generalization, bin/source_matching/utils/{image,dataset,
histogram_matching}.py) so the SRCSM baseline can be scored with its full published pipeline:
every test image is histogram-matched to the AVERAGE intensity CDF of the training (source) images
before prediction.

Faithful to the authors' code:
  * per-image preprocessing: MR -> clip at the 100th percentile (a no-op), scale [0, max] -> [0, 2048];
    CT -> clip to a fixed HU window; then round to int.
  * mask = voxels that are neither the image minimum nor the image maximum.
  * average CDF = CDF of the SUMMED in-mask histograms of all source images.
  * mapping: the image minimum -> the source overall minimum; every other in-mask value v with own
    CDF y -> the largest source value whose average CDF is <= y (their bisect_left / previous-key rule;
    identical here because the average CDF is strictly increasing); voxels whose value is not a key
    (the image maximum) snap to the nearest key (ties to the upper key, as in their code).
  * output intensities are the source's (preprocessed) values.
One deliberate deviation: the CT window is [-1024, 3071] (the full 12-bit HU range) instead of their
abdominal [-1023, 1024], because ToothFairy2's CBCT training data (and teeth/bone in CT) extend to
~3000 and nnU-Net's CTNormalization of those models expects that range.
Second deviation: MR intensities are clipped at 0 first. Resampled MR volumes carry small negative
interpolation values (Open-MS T1w: down to -126 with 70% zero background); without the clip the image
minimum is one of those voxels, the zero background stays inside SRCSM's mask and is mapped to a tissue
intensity (verified on Open-MS T1w: background -> 595 on a 0-2048 scale). With it, the background is
the image minimum and is excluded exactly as in the authors' (non-negative) data.

Geometry: the output keeps the input's header/affine exactly; only the voxel values change.

Usage:
  srcsm_source_matching.py cdf   --images <dir> --modality mr|ct --out <cdf.json> [--pattern '*_0000.nii.gz'] [--exclude-cases <json>]
  srcsm_source_matching.py match --cdf <cdf.json> --in-dir <dir> --out-dir <dir> --modality mr|ct [--workers N]
  srcsm_source_matching.py qc    --cdf <cdf.json> --in-dir <dir> --out-dir <dir> --modality mr|ct
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

CT_WINDOW = (-1024, 3071)
MR_MAX = 2048


def preprocess(arr: np.ndarray, modality: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if modality == "ct":
        a = np.clip(a, *CT_WINDOW)
    else:
        a = np.clip(a, 0, None)            # deviation 2: MR has no negative intensities (see header)
        mx = a.max()                       # 100th-percentile clip = no-op; scale [0, max] -> [0, 2048]
        a = a * (MR_MAX / mx) if mx > 0 else a
    return np.round(a).astype(np.int32)


def in_mask_hist(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mn, mx = a.min(), a.max()
    v = a[(a != mn) & (a != mx)]
    keys, counts = np.unique(v, return_counts=True)
    return keys, counts


def cmd_cdf(args) -> None:
    files = sorted(Path(args.images).glob(args.pattern))
    if args.exclude_cases:     # held-out test cases that sit in imagesTr (BraTS keeps its 70 test cases there)
        excl = set(json.loads(Path(args.exclude_cases).read_text()))
        n0 = len(files)
        files = [f for f in files if f.name[:-len("_0000.nii.gz")] not in excl]
        print(f"excluded {n0 - len(files)} held-out cases listed in {args.exclude_cases}")
    if not files:
        sys.exit(f"no images matching {args.pattern} in {args.images}")
    all_k, all_c, mins, maxs = [], [], [], []
    for f in files:
        a = preprocess(nib.load(str(f)).get_fdata(dtype=np.float32), args.modality)
        k, c = in_mask_hist(a)
        all_k.append(k); all_c.append(c); mins.append(int(a.min())); maxs.append(int(a.max()))
    keys, inv = np.unique(np.concatenate(all_k), return_inverse=True)
    counts = np.bincount(inv, weights=np.concatenate(all_c).astype(np.float64))
    cdf = np.cumsum(counts) / counts.sum()
    out = {"modality": args.modality, "n_images": len(files), "source_dir": str(Path(args.images).resolve()),
           "overall_min_val": int(min(mins)), "overall_max_val": int(max(maxs)),
           "keys": keys.astype(int).tolist(), "cdf": cdf.tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"average CDF from {len(files)} images ({args.modality}) -> {args.out}  "
          f"[{out['overall_min_val']}, {out['overall_max_val']}], {len(keys)} keys")


def match_array(arr: np.ndarray, modality: str, dst: dict) -> np.ndarray:
    a = preprocess(arr, modality)
    k, c = in_mask_hist(a)
    src_keys = np.concatenate([[a.min()], k]).astype(np.int64)
    src_cdf = np.concatenate([[-1.0], np.cumsum(c) / c.sum()]) if c.size else np.array([-1.0])
    dst_keys = np.concatenate([[dst["overall_min_val"]], np.asarray(dst["keys"])]).astype(np.float64)
    dst_cdf = np.concatenate([[-1.0], np.asarray(dst["cdf"])])
    idx = np.searchsorted(dst_cdf, src_cdf, side="right") - 1        # largest dst CDF <= src CDF
    lut = dst_keys[np.clip(idx, 0, None)]
    # every voxel -> nearest src key (ties -> upper key), then through the lookup table
    v = a.ravel()
    j = np.clip(np.searchsorted(src_keys, v, side="left"), 0, src_keys.size - 1)
    jm = np.clip(j - 1, 0, None)
    use_lower = np.abs(src_keys[jm] - v) < np.abs(src_keys[j] - v)
    j = np.where(use_lower, jm, j)
    return lut[j].reshape(a.shape).astype(np.float32)


def _match_one(job):
    src, dst_path, modality, dst = job
    img = nib.load(str(src))
    out = match_array(img.get_fdata(dtype=np.float32), modality, dst)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr["scl_slope"], hdr["scl_inter"] = 1.0, 0.0
    nib.save(nib.Nifti1Image(out, img.affine, hdr), str(dst_path))
    return src.name


def cmd_match(args) -> None:
    dst = json.loads(Path(args.cdf).read_text())
    ind, outd = Path(args.in_dir), Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    files = sorted(ind.glob("*.nii.gz"))
    if not files:
        sys.exit(f"no .nii.gz in {ind}")
    jobs = [(f, outd / f.name, args.modality, dst) for f in files if args.overwrite or not (outd / f.name).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_match_one, jobs):
            pass
    n_out = len(list(outd.glob("*.nii.gz")))
    if n_out != len(files):
        sys.exit(f"!! {ind}: {len(files)} inputs but {n_out} matched outputs")
    print(f"matched {len(jobs)} new ({n_out}/{len(files)} total) {ind.name} -> {outd}")


def _ks(a_keys, a_cdf, b_keys, b_cdf):
    grid = np.union1d(a_keys, b_keys)
    fa = np.concatenate([[0.0], a_cdf])[np.searchsorted(a_keys, grid, side="right")]
    fb = np.concatenate([[0.0], b_cdf])[np.searchsorted(b_keys, grid, side="right")]
    return float(np.abs(fa - fb).max())


def cmd_qc(args) -> None:
    """KS distance of each image's in-mask CDF to the source average CDF, before vs after matching
    (computed in the source's preprocessed space; matched outputs are already in it), and a
    geometry check (same shape + affine)."""
    dst = json.loads(Path(args.cdf).read_text())
    dk, dc = np.asarray(dst["keys"]), np.asarray(dst["cdf"])
    rows = []
    for f in sorted(Path(args.in_dir).glob("*.nii.gz")):
        g = Path(args.out_dir) / f.name
        i0, i1 = nib.load(str(f)), nib.load(str(g))
        geom = i0.shape == i1.shape and np.allclose(i0.affine, i1.affine, atol=1e-4)
        a0 = preprocess(i0.get_fdata(dtype=np.float32), args.modality)
        a1 = np.round(i1.get_fdata(dtype=np.float32)).astype(np.int64)
        k0, c0 = in_mask_hist(a0); k1, c1 = in_mask_hist(a1)
        ks0 = _ks(k0, np.cumsum(c0) / c0.sum(), dk, dc) if c0.size else float("nan")
        ks1 = _ks(k1, np.cumsum(c1) / c1.sum(), dk, dc) if c1.size else float("nan")
        rows.append((f.name, geom, ks0, ks1))
    ks0 = np.array([r[2] for r in rows]); ks1 = np.array([r[3] for r in rows])
    print(f"{Path(args.in_dir).name}: n={len(rows)} geometry_ok={all(r[1] for r in rows)} "
          f"KS before median {np.nanmedian(ks0):.3f} -> after {np.nanmedian(ks1):.3f} (max after {np.nanmax(ks1):.3f})")
    if not all(r[1] for r in rows) or not np.nanmedian(ks1) < np.nanmedian(ks0):
        sys.exit("!! QC FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cdf"); p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    p.add_argument("--modality", choices=("mr", "ct"), required=True); p.add_argument("--pattern", default="*_0000.nii.gz")
    p.add_argument("--exclude-cases", default=None, help="JSON list of case ids to leave out of the source CDF")
    p = sub.add_parser("match"); p.add_argument("--cdf", required=True); p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--modality", choices=("mr", "ct"), required=True)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("qc"); p.add_argument("--cdf", required=True); p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--modality", choices=("mr", "ct"), required=True)
    args = ap.parse_args()
    {"cdf": cmd_cdf, "match": cmd_match, "qc": cmd_qc}[args.cmd](args)


if __name__ == "__main__":
    main()
