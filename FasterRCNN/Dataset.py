import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# DICOM loader (module-level so it can be reused independently)
# ---------------------------------------------------------------------------

def _load_dicom(path: str) -> torch.Tensor:
    # Read file and apply stored rescale slope/intercept (RescaleSlope /
    # RescaleIntercept) so values are in consistent radiological units.
    ds  = pydicom.dcmread(path)
    arr = pydicom.pixel_data_handlers.util.apply_modality_lut(
        ds.pixel_array, ds
    ).astype(np.float32)

    # Per-image min-max normalisation to [0, 1].
    # Each chest X-ray has its own brightness range; normalising per-image
    # is standard practice and avoids dataset-wide outliers dominating.
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)   # degenerate image — all pixels identical

    # [H, W] -> [3, H, W]: replicate the grayscale channel so the
    # ImageNet-pretrained ResNet-50 backbone receives the expected 3-channel input.
    tensor = torch.from_numpy(arr).unsqueeze(0)
    tensor = tensor.expand(3, -1, -1).contiguous()
    return tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DetectionDataset(Dataset):
    """
    Dataset for Faster R-CNN pneumonia detection from chest X-rays.

    CSV schema: patient_id, x, y, width, height, target
        - patient_id    : filename stem (without extension) matching the image file
        - x, y          : top-left corner of the bounding box (pixels)
        - width, height : box dimensions (pixels)
        - target        : integer class label (1 = pneumonia).
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
        transforms : optional callable applied to (image, target) pairs
        img_ext    : image file extension, defaults to ".dcm"; pass ".png" for PNGs
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split: str = "train",
        train_frac: float = 0.8,
        transforms=None,
        img_ext: str = ".dcm",
    ):
        assert split in ("train", "val", "all"), \
            f"split must be 'train', 'val', or 'all', got '{split}'"
        assert 0.0 < train_frac < 1.0, \
            f"train_frac must be in (0, 1), got {train_frac}"

        self.img_dir    = img_dir
        self.transforms = transforms
        self.img_ext    = img_ext

        df = pd.read_csv(csv_path)

        # Sort unique patient IDs for a deterministic, reproducible split.
        # Sorting on UUID-style IDs distributes patients uniformly since
        # UUIDs have no temporal or alphabetical ordering bias.
        all_pids = sorted(df["patient_id"].unique().tolist())
        cutoff   = int(len(all_pids) * train_frac)

        if split == "train":
            selected = all_pids[:cutoff]
        elif split == "val":
            selected = all_pids[cutoff:]
        else:
            selected = all_pids

        self.patient_ids = selected

        # Build annotation lookup only for patients in this split
        self.annotations = {
            pid: df[df["patient_id"] == pid].reset_index(drop=True)
            for pid in self.patient_ids
        }

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_image(self, path: str) -> torch.Tensor:
        """Return a float32 tensor [3, H, W] in [0, 1] for any supported format."""
        if path.lower().endswith(".dcm"):
            return _load_dicom(path)
        # Fallback for PNG / JPEG
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]
        img_path   = os.path.join(self.img_dir, f"{patient_id}{self.img_ext}")

        # --- Image ---
        image = self._load_image(img_path)   # float32 tensor [3, H, W] in [0, 1]

        # --- Annotations ---
        ann = self.annotations[patient_id]

        # A patient is healthy when the box columns are NaN.
        # Keep only rows with valid box coordinates.
        valid      = ann.dropna(subset=["x", "y", "width", "height"])
        is_healthy = len(valid) == 0

        if is_healthy:
            # Negative sample: Faster R-CNN handles empty targets correctly.
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
            area   = torch.zeros((0,),   dtype=torch.float32)
        else:
            # Convert (x, y, w, h) -> (x_min, y_min, x_max, y_max)
            boxes = torch.tensor(
                [
                    [row.x, row.y, row.x + row.width, row.y + row.height]
                    for row in valid.itertuples()
                ],
                dtype=torch.float32,
            )
            labels = torch.tensor(valid["target"].values, dtype=torch.int64)
            area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # iscrowd=0 -> evaluator treats every instance individually
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Faster R-CNN expects a list of (image, target) tuples, NOT a stacked
    tensor, because images can differ in size and box count.
    """
    return tuple(zip(*batch))