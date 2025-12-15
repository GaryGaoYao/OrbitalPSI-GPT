#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ct_to_stl.py

Run nnU-Net v2 prediction for a single CT NIfTI (.nii / .nii.gz),
then convert selected label masks into an STL surface via marching cubes.

Label convention (your model):
  0 = background
  1 = orbital bone
  2 = globe / eyeball volume

Examples:
  # Export bone surface STL
  python ct_to_stl.py -i D:/data/case01.nii.gz --export bone

  # Export globe surface STL
  python ct_to_stl.py -i D:/data/case01.nii.gz --export globe

  # Export both
  python ct_to_stl.py -i D:/data/case01.nii.gz --export both

Notes:
  - nnU-Net v2 expects channel-suffixed inputs: caseXX_0000.nii.gz
  - This script writes to a temp folder for prediction I/O
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import trimesh


# -----------------------------
# Defaults (edit for your release)
# -----------------------------
% DEFAULT_DATASET = "Dataset112_DentalSegmentator"
DEFAULT_DATASET = "Dataset378_Orbital_Seg"
DEFAULT_CONFIG  = "3d_fullres"
DEFAULT_TRAINER = "nnUNetTrainer"
DEFAULT_PLANS   = "nnUNetPlans"
DEFAULT_FOLD    = "0"

# Your label mapping
LABEL_MAP = {
    "bone": [1],
    "globe": [2],
    "both": [1, 2],
}


# -----------------------------
# Utilities
# -----------------------------
def setup_logger(verbosity: int = 0) -> None:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def has_cuda() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def check_executable(name: str) -> None:
    from shutil import which
    if which(name) is None:
        raise RuntimeError(
            f"Cannot find executable '{name}' in PATH.\n"
            f"Install nnU-Net v2 and ensure nnUNetv2_predict is available."
        )

def ensure_single_channel_3d(
    nii_path: Path,
    out_dir: Path,
    case_id: str = "case01",
    dtype: np.dtype = np.float32
) -> Path:
    """Ensure input is 3D single-channel. If 4D, take first channel/timepoint."""
    img = nib.load(str(nii_path))
    data = img.get_fdata()

    if data.ndim == 4:
        logging.warning("Input is 4D. Taking data[..., 0] as the first channel.")
        data = data[..., 0]
        img = nib.Nifti1Image(data.astype(dtype), img.affine, img.header)

    if data.ndim != 3:
        raise RuntimeError(f"Input must be 3D volume. Got ndim={data.ndim}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = out_dir / f"{case_id}_0000.nii.gz"
    nib.save(img, str(fixed))
    return fixed

def run_nnunet_predict(
    in_dir: Path,
    out_dir: Path,
    dataset: str,
    config: str,
    trainer: str,
    plans: str,
    fold: str,
    device: str,
    nnunet_results: Optional[Path] = None,
) -> None:
    check_executable("nnUNetv2_predict")

    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-d", dataset,
        "-c", config,
        "-tr", trainer,
        "-p", plans,
        "-f", fold,
        "-device", device,
    ]

    env = os.environ.copy()
    if nnunet_results is not None:
        env["nnUNet_results"] = str(nnunet_results)

    logging.info("Running nnU-Net v2 prediction:")
    logging.info("  " + " ".join(cmd))
    if nnunet_results is not None:
        logging.info(f"Using nnUNet_results={nnunet_results}")

    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"nnUNetv2_predict failed with return code {r.returncode}.")

def pick_single_prediction(out_dir: Path) -> Path:
    segs = sorted(out_dir.glob("*.nii.gz"))
    if not segs:
        raise RuntimeError(f"No nnU-Net outputs found in: {out_dir}")
    if len(segs) > 1:
        logging.warning(f"Multiple outputs found ({len(segs)}). Using: {segs[0].name}")
    return segs[0]

def nii_labels_to_stl(
    seg_nii: Path,
    stl_path: Path,
    labels: Sequence[int],
    level: float = 0.5,
) -> None:
    img = nib.load(str(seg_nii))
    data = img.get_fdata()
    aff = img.affine

    labels = tuple(int(x) for x in labels)

    mask = np.zeros(data.shape, dtype=bool)
    for lb in labels:
        mask |= (data == lb)

    voxels = int(mask.sum())
    if voxels == 0:
        raise RuntimeError(
            f"Segmentation mask is empty for labels={labels}. "
            f"Check label mapping or model output."
        )

    logging.info(f"Mask voxels (labels={labels}): {voxels}")

    verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=level)

    # voxel -> world via affine
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_w = (verts_h @ aff.T)[:, :3]

    mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, process=False)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(stl_path))
    logging.info(f"STL saved: {stl_path}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run nnU-Net v2 prediction on a CT NIfTI and export bone/globe labels to STL."
    )
    p.add_argument("--input", "-i", required=True, help="Input CT NIfTI (.nii or .nii.gz).")
    p.add_argument("--outdir", "-o", default=None, help="Output folder (default: same directory as input).")
    p.add_argument("--case-id", default="case01", help="Case ID used for nnU-Net naming (default: case01).")

    # nnU-Net params
    p.add_argument("--dataset", default=DEFAULT_DATASET, help=f"nnU-Net dataset (default: {DEFAULT_DATASET}).")
    p.add_argument("--config",  default=DEFAULT_CONFIG,  help=f"nnU-Net config (default: {DEFAULT_CONFIG}).")
    p.add_argument("--trainer", default=DEFAULT_TRAINER, help=f"nnU-Net trainer (default: {DEFAULT_TRAINER}).")
    p.add_argument("--plans",   default=DEFAULT_PLANS,   help=f"nnU-Net plans (default: {DEFAULT_PLANS}).")
    p.add_argument("--fold",    default=DEFAULT_FOLD,    help=f"nnU-Net fold (default: {DEFAULT_FOLD}).")

    # weights location
    p.add_argument(
        "--nnunet-results",
        default=None,
        help="Path to nnUNet_results (where trained weights reside). "
             "If omitted, uses environment variable nnUNet_results."
    )

    # export selection
    p.add_argument(
        "--export",
        choices=["bone", "globe", "both"],
        default="bone",
        help="Which structure(s) to export to STL (default: bone)."
    )
    p.add_argument("--suffix", default=None, help="Output STL suffix (default depends on --export).")
    p.add_argument("--level", type=float, default=0.5, help="Marching cubes level (default: 0.5).")

    # device & logging
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "cuda:0", "cuda:1"], help="Force device.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Verbose logging (-v or -vv).")

    return p.parse_args()

def main() -> int:
    args = parse_args()
    setup_logger(args.verbose)

    in_file = Path(args.input).expanduser().resolve()
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else in_file.parent

    # device selection
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if has_cuda() else "cpu"

    # parse nnUNet_results
    nnunet_results = Path(args.nnunet_results).expanduser().resolve() if args.nnunet_results else None

    # decide suffix / output name
    export_key = args.export
    labels: List[int] = LABEL_MAP[export_key]

    if args.suffix is None:
        suffix = f"_{export_key}"
    else:
        suffix = args.suffix

    name = in_file.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = Path(name).stem

    stl_file = out_dir / f"{stem}{suffix}.stl"

    logging.info(f"Input:  {in_file}")
    logging.info(f"Outdir: {out_dir}")
    logging.info(f"Device: {device}")
    logging.info(f"Export: {export_key} -> labels={labels}")
    if nnunet_results is None:
        logging.info("nnUNet_results: (from environment, if set)")
    else:
        logging.info(f"nnUNet_results: {nnunet_results}")

    with tempfile.TemporaryDirectory(prefix="nnunet_pred_") as tmp:
        tmp = Path(tmp)
        tmp_in = tmp / "in"
        tmp_out = tmp / "out"

        ensure_single_channel_3d(in_file, tmp_in, case_id=args.case_id)

        run_nnunet_predict(
            in_dir=tmp_in,
            out_dir=tmp_out,
            dataset=args.dataset,
            config=args.config,
            trainer=args.trainer,
            plans=args.plans,
            fold=args.fold,
            device=device,
            nnunet_results=nnunet_results,
        )

        pred_nii = pick_single_prediction(tmp_out)
        nii_labels_to_stl(
            seg_nii=pred_nii,
            stl_path=stl_file,
            labels=labels,
            level=args.level,
        )

    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        logging.error(str(e))
        raise
