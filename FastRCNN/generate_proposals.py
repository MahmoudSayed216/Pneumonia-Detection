"""
generate_proposals.py — Pre-compute selective search proposals for all images.

Run once before training:
    python generate_proposals.py \
        --csv      data/annotations.csv \
        --img_dir  data/pngs \
        --out_dir  data/proposals \
        --workers  4
"""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_proposals(args_tuple):
    img_path, out_dir, max_proposals = args_tuple
    img_path = Path(img_path)
    out_path = Path(out_dir) / f"{img_path.stem}.npy"

    if out_path.exists():
        return (img_path.stem, None)

    try:
        # IMREAD_UNCHANGED reads 16-bit correctly, then convert to 8-bit for selective search
        img_16 = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_16 is None:
            return (img_path.stem, "cv2 failed to read image")

        # Normalize to 8-bit BGR for selective search
        img_8 = (img_16 / 256).astype(np.uint8)
        if len(img_8.shape) == 2:          # grayscale → BGR
            img_8 = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR)

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img_8)
        ss.switchToSelectiveSearchFast()

        rects = ss.process()[:max_proposals]
        boxes = np.array(
            [[x, y, x + w, y + h] for x, y, w, h in rects],
            dtype=np.float32,
        )
        np.save(out_path, boxes)
        return (img_path.stem, None)

    except Exception as e:
        return (img_path.stem, str(e))


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df  = pd.read_csv(args.csv)
    ids = df["patientId"].unique().tolist()

    tasks = [
        (str(Path(args.img_dir) / f"{pid}.png"), str(out_dir), args.max_proposals)
        for pid in ids
        if (Path(args.img_dir) / f"{pid}.png").exists()
    ]
    print(f"Generating proposals for {len(tasks)} images → {out_dir}")

    errors = []
    with Pool(processes=args.workers or cpu_count()) as pool:
        for pid, err in tqdm(pool.imap_unordered(compute_proposals, tasks), total=len(tasks)):
            if err:
                errors.append((pid, err))

    print(f"\nDone. {len(tasks) - len(errors)}/{len(tasks)} succeeded.")
    if errors:
        for pid, err in errors:
            print(f"  {pid}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           required=True)
    parser.add_argument("--img_dir",       required=True)
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--max_proposals", type=int, default=2000)
    parser.add_argument("--workers",       type=int, default=None)
    main(parser.parse_args())