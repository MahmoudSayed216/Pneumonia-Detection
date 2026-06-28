"""
Script 2 — Pre-compute Selective Search proposals
--------------------------------------------------
Runs OpenCV Selective Search over every image whose patientId appears
in the CSV, applies per-image NMS, and saves results to a compact HDF5.

Usage:
    python 02_precompute_proposals.py \
        --image_dir  /data/images \
        --output_h5  /data/proposals.h5 \
        --csv_path   /data/labels.csv \
        [--mode fast|quality] \
        [--max_proposals 2000] \
        [--nms_thresh 0.7] \
        [--workers 4]

HDF5 layout:
    /proposals/<patientId>    float32 array  shape (N, 4)  [x1,y1,x2,y2]

Requires:
    pip install opencv-contrib-python   (NOT the plain opencv-python)
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------- #
def nms(boxes: np.ndarray, thresh: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0.0, xx2 - xx1 + 1)
        h   = np.maximum(0.0, yy2 - yy1 + 1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][iou <= thresh]

    return np.array(keep, dtype=np.int64)


def compute_proposals_for_image(
    img_path: Path,
    mode: str,
    max_proposals: int,
    nms_thresh: float,
) -> tuple:
    patient_id = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()   # [x, y, w, h]
    if len(rects) == 0:
        return patient_id, np.zeros((0, 4), dtype=np.float32)

    boxes = np.array(rects, dtype=np.float32)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]   # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]   # y2

    keep  = nms(boxes, nms_thresh)
    boxes = boxes[keep]

    if len(boxes) > max_proposals:
        boxes = boxes[:max_proposals]

    return patient_id, boxes


def worker_fn(args):
    img_path, mode, max_proposals, nms_thresh = args
    return compute_proposals_for_image(img_path, mode, max_proposals, nms_thresh)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Pre-compute Selective Search proposals")
    parser.add_argument("--image_dir",     required=True,           help="Directory of PNG images")
    parser.add_argument("--output_h5",     required=True,           help="Output HDF5 file path")
    parser.add_argument("--csv_path",      required=True,           help="Labels CSV — only these patientIds are processed")
    parser.add_argument("--mode",          default="fast",          choices=["fast", "quality"])
    parser.add_argument("--max_proposals", type=int,  default=2000)
    parser.add_argument("--nms_thresh",    type=float, default=0.7)
    parser.add_argument("--workers",       type=int,  default=4)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)

    # Only process patient IDs that exist in the CSV
    valid_ids = set(pd.read_csv(args.csv_path)["patientId"].unique().tolist())
    print(f"[*] CSV contains {len(valid_ids)} unique patient IDs")

    all_imgs  = sorted(image_dir.rglob("*.png")) + sorted(image_dir.rglob("*.jpg"))
    img_paths = [p for p in all_imgs if p.stem in valid_ids]

    if not img_paths:
        print(f"[!] No matching images found under {image_dir}")
        print(f"    Total images scanned: {len(all_imgs)} — none matched the CSV patient IDs")
        return

    print(f"[*] Matched {len(img_paths)}/{len(all_imgs)} images to CSV "
          f"(mode={args.mode}, max={args.max_proposals}, workers={args.workers}) …")

    work_args = [(p, args.mode, args.max_proposals, args.nms_thresh) for p in img_paths]

    results, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker_fn, a): a[0] for a in work_args}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            try:
                patient_id, boxes = fut.result()
                results[patient_id] = boxes
            except Exception as exc:
                errors.append(f"{futures[fut]}: {exc}")

    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), "w") as f:
        grp = f.create_group("proposals")
        for pid, boxes in results.items():
            grp.create_dataset(pid, data=boxes, compression="gzip")

    print(f"[✓] Saved proposals for {len(results)} images → {out_path}")
    if errors:
        print(f"[!] {len(errors)} errors:")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()