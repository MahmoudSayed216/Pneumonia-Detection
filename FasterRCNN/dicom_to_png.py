"""
convert_dcm_to_png.py — Convert only the DICOM files referenced in a CSV to 16-bit PNGs.

Why 16-bit?
    Standard 8-bit PNG maps the full pixel range to 256 levels, which loses
    fine-grained intensity detail that matters in medical imaging. 16-bit PNG
    preserves 65 536 levels, retaining the full dynamic range of the X-ray
    after normalisation.

Usage:
    python convert_dcm_to_png.py \
        --csv     resampled_data.csv \
        --dcm_dir /path/to/dicoms \
        --out_dir /path/to/pngs

    # Parallelise across 8 CPU cores:
    python convert_dcm_to_png.py \
        --csv     resampled_data.csv \
        --dcm_dir /path/to/dicoms \
        --out_dir /path/to/pngs \
        --workers 8
"""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pydicom
import pydicom.pixel_data_handlers.util as pdu
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Single-file conversion
# ---------------------------------------------------------------------------

def convert_one(args_tuple):
    """
    Convert a single DICOM file to a 16-bit PNG.
    Packed into a tuple for multiprocessing.Pool compatibility.
    Returns (patient_id, error_message_or_None).
    """
    dcm_path, out_dir = args_tuple
    dcm_path = Path(dcm_path)
    out_path = Path(out_dir) / dcm_path.with_suffix(".png").name

    # Skip if already converted — safe to re-run on interrupted jobs
    if out_path.exists():
        return (dcm_path.stem, None)

    try:
        ds  = pydicom.dcmread(str(dcm_path))
        arr = pdu.apply_modality_lut(ds.pixel_array, ds).astype(np.float32)

        # Per-image min-max normalisation to [0, 1]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

        arr_16 = (arr * 65535).astype(np.uint16)
        Image.fromarray(arr_16, mode="I;16").save(out_path)

        return (dcm_path.stem, None)

    except Exception as e:
        return (dcm_path.stem, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dcm_dir = Path(args.dcm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Read CSV and get the patient IDs we actually need ---
    df          = pd.read_csv(args.csv)
    patient_ids = set(df["patientId"].unique())
    print(f"CSV contains {len(patient_ids)} unique patient IDs to convert.")

    # --- Match IDs to DICOM files in dcm_dir ---
    dcm_files = [
        dcm_dir / f"{pid}.dcm"
        for pid in patient_ids
    ]

    # Warn about any IDs whose file is missing
    missing = [p for p in dcm_files if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} DICOM file(s) not found in {dcm_dir}:")
        for p in missing[:10]:   # cap output at 10 lines
            print(f"  {p.name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    dcm_files = [p for p in dcm_files if p.exists()]
    if not dcm_files:
        print("No matching DICOM files found. Aborting.")
        return

    print(f"Converting {len(dcm_files)} files → {out_dir}")

    n_workers = args.workers or cpu_count()
    tasks     = [(str(p), str(out_dir)) for p in dcm_files]

    errors = []
    with Pool(processes=n_workers) as pool:
        for pid, err in tqdm(
            pool.imap_unordered(convert_one, tasks),
            total=len(tasks),
            desc="Converting",
            unit="file",
        ):
            if err:
                errors.append((pid, err))

    print(f"\nDone. {len(tasks) - len(errors)}/{len(tasks)} files converted successfully.")
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for pid, err in errors:
            print(f"  {pid}: {err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV-referenced DICOM X-rays to 16-bit PNG"
    )
    parser.add_argument("--csv",     "--csv",                  required=True,
                        help="CSV file with a 'patientId' column (e.g. resampled_data.csv)")
    parser.add_argument("--dcm_dir", "--dcm-dir", dest="dcm_dir", required=True,
                        help="Directory containing the source .dcm files")
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True,
                        help="Directory to write the output .png files")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker count (default: all CPU cores)")
    main(parser.parse_args())