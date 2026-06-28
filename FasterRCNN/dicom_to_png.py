"""
convert_dcm_to_png.py — Convert a directory of DICOM chest X-rays to 16-bit PNGs.

Why 16-bit?
    Standard 8-bit PNG maps the full pixel range to 256 levels, which loses
    fine-grained intensity detail that matters in medical imaging. 16-bit PNG
    preserves 65 536 levels, retaining the full dynamic range of the X-ray
    after normalisation.

Usage:
    python convert_dcm_to_png.py --dcm_dir data/dicoms --out_dir data/pngs

    # Parallelise across 8 CPU cores (recommended for large datasets):
    python convert_dcm_to_png.py --dcm_dir data/dicoms --out_dir data/pngs --workers 8
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
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
    Designed to be called from a multiprocessing Pool, so all args are packed
    into a single tuple.

    Returns (stem, error_message_or_None).
    """
    dcm_path, out_dir = args_tuple

    out_path = Path(out_dir) / Path(dcm_path).with_suffix(".png").name

    # Skip if already converted (allows resuming interrupted runs)
    if out_path.exists():
        return (dcm_path.stem, None)

    try:
        ds  = pydicom.dcmread(dcm_path)
        arr = pdu.apply_modality_lut(ds.pixel_array, ds).astype(np.float32)

        # Per-image min-max normalisation → [0, 1]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

        # Scale to 16-bit range and save
        arr_16 = (arr * 65535).astype(np.uint16)
        Image.fromarray(arr_16, mode="I;16").save(out_path)

        return (Path(dcm_path).stem, None)

    except Exception as e:
        return (Path(dcm_path).stem, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dcm_dir = Path(args.dcm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(dcm_dir.glob("*.dcm"))
    if not dcm_files:
        print(f"No .dcm files found in {dcm_dir}")
        return

    print(f"Found {len(dcm_files)} DICOM files → saving PNGs to {out_dir}")

    n_workers = args.workers or cpu_count()
    tasks = [(str(p), str(out_dir)) for p in dcm_files]

    errors = []
    with Pool(processes=n_workers) as pool:
        for stem, err in tqdm(
            pool.imap_unordered(convert_one, tasks),
            total=len(tasks),
            desc="Converting",
            unit="file",
        ):
            if err:
                errors.append((stem, err))

    print(f"\nDone. {len(tasks) - len(errors)}/{len(tasks)} files converted successfully.")
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for stem, err in errors:
            print(f"  {stem}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM X-rays to 16-bit PNG")
    parser.add_argument("--dcm_dir",  required=True, help="Directory containing .dcm files")
    parser.add_argument("--out_dir",  required=True, help="Directory to write .png files")
    parser.add_argument("--workers",  type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    main(parser.parse_args())