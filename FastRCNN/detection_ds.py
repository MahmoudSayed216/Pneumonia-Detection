"""
Script 3 — Fast R-CNN Dataset
-------------------------------
PyTorch Dataset that ties together:
  • The PNG images converted from DICOM (Script 1)
  • The ground-truth CSV  (patientId, x, y, width, height, Target)
  • The pre-computed Selective Search proposals (Script 2, HDF5)

For each sample the Dataset returns a dict that the Fast R-CNN model
(Script 4) expects:
  {
    "image"       : FloatTensor  (3, H, W)   — normalised image
    "proposals"   : FloatTensor  (P, 4)      — [x1, y1, x2, y2] Selective Search boxes
    "boxes"       : FloatTensor  (G, 4)      — ground-truth boxes  [x1, y1, x2, y2]
    "labels"      : LongTensor   (G,)        — 1 for pneumonia, 0 background
    "patient_id"  : str
  }

Usage (standalone sanity-check):
    python 03_dataset.py \
        --csv_path   /data/labels.csv \
        --image_dir  /data/images \
        --proposals  /data/proposals.h5

The collate_fn at the bottom of this file handles variable-length proposals
and ground-truth boxes for DataLoader batching.
"""

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


# --------------------------------------------------------------------------- #
#  Augmentation / normalisation pipelines
# --------------------------------------------------------------------------- #

# ImageNet stats are used even for greyscale X-rays because the pretrained
# ResNet backbone expects 3-channel input with these statistics.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool = True, image_size: int = 800) -> T.Compose:
    """
    Returns a torchvision transform pipeline.
    Images are resized so the shorter side == image_size, then we apply
    random horizontal flip (train-only) and ImageNet normalisation.
    """
    ops = [T.Resize(image_size)]
    if train:
        ops += [T.RandomHorizontalFlip(p=0.5)]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(ops)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class PneumoniaDetectionDataset(Dataset):
    """
    Pneumonia detection dataset for Fast R-CNN training.

    Parameters
    ----------
    csv_path      : path to the ground-truth CSV
    image_dir     : directory containing PNG images named <patientId>.png
    proposals_h5  : HDF5 file produced by Script 2
    split         : list of patient IDs in this split (train / val)
    train         : whether to apply training-time augmentation
    image_size    : shorter side to resize to
    max_proposals : maximum Selective Search proposals to keep per image
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        proposals_h5: str,
        split: Optional[list] = None,
        train: bool = True,
        image_size: int = 800,
        max_proposals: int = 2000,
    ):
        self.image_dir     = Path(image_dir)
        self.train         = train
        self.max_proposals = max_proposals
        self.transform     = build_transforms(train, image_size)

        # ---- load CSV -------------------------------------------------------
        df = pd.read_csv(csv_path)

        # Rows with Target == 0 have no bounding box → only keep the patientId
        # Group so each patient appears once
        pos = df[df["Target"] == 1].copy()
        neg_ids = df[df["Target"] == 0]["patientId"].unique().tolist()

        # Build a mapping  patientId → list of [x1,y1,x2,y2] boxes
        self.gt: dict[str, np.ndarray] = {}

        for pid, grp in pos.groupby("patientId"):
            boxes = grp[["x", "y", "width", "height"]].values.astype(np.float32)
            # convert [x, y, w, h] → [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            self.gt[pid] = boxes

        for pid in neg_ids:
            if pid not in self.gt:
                self.gt[pid] = np.zeros((0, 4), dtype=np.float32)

        # Patient list
        all_ids = list(self.gt.keys())
        if split is not None:
            all_ids = [p for p in all_ids if p in set(split)]
        self.patient_ids = sorted(all_ids)

        # ---- open the proposals HDF5 in lazy mode ---------------------------
        self._h5_path = proposals_h5
        self._h5: Optional[h5py.File] = None  # opened lazily per worker

    # ------------------------------------------------------------------ #
    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.patient_ids[idx]

        # ---- image ----------------------------------------------------------
        img_path = self.image_dir / f"{pid}.png"
        img = Image.open(img_path).convert("RGB")   # greyscale DICOM → RGB
        img_tensor = self.transform(img)             # (3, H, W) float32

        # ---- ground-truth boxes & labels ------------------------------------
        gt_boxes = self.gt[pid].copy()              # (G, 4)  x1y1x2y2
        gt_labels = np.ones(len(gt_boxes), dtype=np.int64) if len(gt_boxes) > 0 \
                    else np.zeros(0, dtype=np.int64)

        # ---- proposals from HDF5 --------------------------------------------
        self._open_h5()
        if pid in self._h5["proposals"]:
            proposals = self._h5["proposals"][pid][:].astype(np.float32)
        else:
            # Fall back: use GT boxes as proposals (degenerate case)
            proposals = gt_boxes.copy() if len(gt_boxes) > 0 \
                        else np.zeros((1, 4), dtype=np.float32)

        # Limit number of proposals
        if len(proposals) > self.max_proposals:
            proposals = proposals[:self.max_proposals]

        return {
            "image":      img_tensor,
            "proposals":  torch.from_numpy(proposals),
            "boxes":      torch.from_numpy(gt_boxes),
            "labels":     torch.from_numpy(gt_labels),
            "patient_id": pid,
        }

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  Collate function
# --------------------------------------------------------------------------- #

def fast_rcnn_collate(batch: list) -> dict:
    """
    Stacks images into a single tensor; keeps proposals / boxes / labels
    as lists because they differ in length across the batch.
    """
    images      = torch.stack([b["image"]     for b in batch], dim=0)
    proposals   = [b["proposals"]  for b in batch]
    boxes       = [b["boxes"]      for b in batch]
    labels      = [b["labels"]     for b in batch]
    patient_ids = [b["patient_id"] for b in batch]

    return {
        "images":      images,
        "proposals":   proposals,
        "boxes":       boxes,
        "labels":      labels,
        "patient_ids": patient_ids,
    }


# --------------------------------------------------------------------------- #
#  Utility: train / val split
# --------------------------------------------------------------------------- #

def make_splits(
    csv_path: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Returns (train_ids, val_ids) — stratified by Target to keep class balance.
    """
    df = pd.read_csv(csv_path)
    all_ids = df["patientId"].unique().tolist()

    pos_ids = df[df["Target"] == 1]["patientId"].unique().tolist()
    neg_ids = [p for p in all_ids if p not in set(pos_ids)]

    rng = np.random.default_rng(seed)

    def split_list(lst):
        n_val = max(1, int(len(lst) * val_fraction))
        idx   = rng.permutation(len(lst))
        return [lst[i] for i in idx[n_val:]], [lst[i] for i in idx[:n_val]]

    pos_train, pos_val = split_list(pos_ids)
    neg_train, neg_val = split_list(neg_ids)

    return pos_train + neg_train, pos_val + neg_val


# --------------------------------------------------------------------------- #
#  Sanity check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",  required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--proposals", required=True)
    args = parser.parse_args()

    train_ids, val_ids = make_splits(args.csv_path)
    print(f"[*] Train: {len(train_ids)}  Val: {len(val_ids)}")

    ds = PneumoniaDetectionDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        proposals_h5=args.proposals,
        split=train_ids,
        train=True,
    )
    print(f"[*] Dataset length: {len(ds)}")

    sample = ds[0]
    print(f"[*] Sample patient_id : {sample['patient_id']}")
    print(f"    image shape        : {sample['image'].shape}")
    print(f"    proposals shape    : {sample['proposals'].shape}")
    print(f"    GT boxes shape     : {sample['boxes'].shape}")
    print(f"    GT labels          : {sample['labels']}")

    loader = DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=fast_rcnn_collate, num_workers=0
    )
    batch = next(iter(loader))
    print(f"[*] Batch images shape : {batch['images'].shape}")
    print("[✓] Dataset OK")