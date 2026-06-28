"""
Script 1 — DICOM → PNG converter
---------------------------------
Converts a directory of .dcm files to PNG images, normalising
the pixel array to uint8 so standard vision libraries can read them.

Usage:
    python 01_convert_dicom.py \
        --dicom_dir  /data/dicom \
        --output_dir /data/images \
        [--workers 8]

Output layout (mirrors the input tree):
    /data/images/<patientId>.png
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
def convert_one(dcm_path: Path, output_dir: Path) -> str:
    """
    Convert a single DICOM file to a normalised 8-bit PNG.
    Returns the output path as a string (or raises on failure).
    """
    try:
        ds = pydicom.dcmread(str(dcm_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Handle multi-frame DICOMs — take the middle frame
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[pixel_array.shape[0] // 2]

        # Apply DICOM window / VOI LUT if present, then normalise to [0, 255]
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) \
                else float(ds.WindowCenter[0])
            width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) \
                else float(ds.WindowWidth[0])
            lo = center - width / 2.0
            hi = center + width / 2.0
            pixel_array = np.clip(pixel_array, lo, hi)
            pixel_array = (pixel_array - lo) / (hi - lo) * 255.0
        else:
            pmin, pmax = pixel_array.min(), pixel_array.max()
            if pmax > pmin:
                pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
            else:
                pixel_array = np.zeros_like(pixel_array)

        img = Image.fromarray(pixel_array.astype(np.uint8))

        # Preserve relative path so the output tree mirrors the input tree
        rel = dcm_path.stem          # patientId (filename without extension)
        out_path = output_dir / f"{rel}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return str(out_path)

    except Exception as exc:
        return f"ERROR {dcm_path}: {exc}"


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG")
    parser.add_argument("--dicom_dir",  required=True, help="Root directory containing .dcm files")
    parser.add_argument("--output_dir", required=True, help="Where to save PNG files")
    parser.add_argument("--workers",    type=int, default=4, help="Parallel workers (default 4)")
    args = parser.parse_args()

    dicom_dir  = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(dicom_dir.rglob("*.dcm"))
    if not dcm_files:
        print(f"[!] No .dcm files found under {dicom_dir}")
        return

    print(f"[*] Found {len(dcm_files)} DICOM files — converting with {args.workers} workers …")

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, p, output_dir): p for p in dcm_files}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="file"):
            result = fut.result()
            if result.startswith("ERROR"):
                errors.append(result)

    print(f"[✓] Done. {len(dcm_files) - len(errors)}/{len(dcm_files)} converted successfully.")
    if errors:
        print(f"[!] {len(errors)} errors:")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()