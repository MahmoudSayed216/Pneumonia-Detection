import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class DetectionDataset(Dataset):
    """
    Dataset for Faster R-CNN pneumonia detection from chest X-rays.

    CSV schema: patient_id, x, y, width, height, Target
        - patient_id    : filename stem (without extension) matching the image file
        - x, y          : top-left corner of the bounding box (pixels)
        - width, height : box dimensions (pixels)
        - Target        : integer class label (1 = pneumonia).
                          NaN in any box column means the image is healthy
                          (no pneumonia) — the model still sees it, but with
                          an empty box list so it learns to suppress proposals.

    Args:
        csv_path   : path to the annotations CSV
        img_dir    : directory that contains the images
        split      : "train", "val", or "all"
        train_frac : fraction of patients used for training (e.g. 0.8 -> 80% train,
                     20% val). The split is performed on the SORTED list of unique
                     patient IDs so the same patient never appears in both sets,
                     preventing label leakage from repeated IDs.
        transforms : optional callable applied to (image, Target) pairs
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split: str = "train",
        train_frac: float = 0.8,
        transforms=None,
    ):
        assert split in ("train", "val", "all"), \
            f"split must be 'train', 'val', or 'all', got '{split}'"
        assert 0.0 < train_frac < 1.0, \
            f"train_frac must be in (0, 1), got {train_frac}"

        self.img_dir    = img_dir
        self.transforms = transforms

        df = pd.read_csv(csv_path)

        all_pids = sorted(df["patientId"].unique().tolist())
        cutoff   = int(len(all_pids) * train_frac)

        if split == "train":
            selected = all_pids[:cutoff]
        elif split == "val":
            selected = all_pids[cutoff:]
        else:
            selected = all_pids

        self.patient_ids = selected

        self.annotations = {
            pid: df[df["patientId"] == pid].reset_index(drop=True)
            for pid in self.patient_ids
        }

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        # 16-bit grayscale needs to be scaled down to 8-bit before RGB conversion
        if img.mode in ("I;16", "I"):
            img = img.point(lambda x: x / 256).convert("L")
        img = img.convert("RGB")
        return TF.to_tensor(img)

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        img_path   = os.path.join(self.img_dir, f"{patient_id}.png")

        image = self._load_image(img_path)

        ann   = self.annotations[patient_id]
        valid = ann.dropna(subset=["x", "y", "width", "height"])

        if len(valid) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
            area   = torch.zeros((0,),   dtype=torch.float32)
        else:
            boxes = torch.tensor(
                [
                    [row.x, row.y, row.x + row.width, row.y + row.height]
                    for row in valid.itertuples()
                ],
                dtype=torch.float32,
            )
            labels = torch.tensor(valid["Target"].values, dtype=torch.int64)
            area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros(len(labels), dtype=torch.int64)

        Target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  iscrowd,
        }

        if self.transforms is not None:
            image, Target = self.transforms(image, Target)

        return image, Target


def collate_fn(batch):
    """
    Faster R-CNN expects a list of (image, Target) tuples, NOT a stacked
    tensor, because images can differ in size and box count.
    """
    return tuple(zip(*batch))