import pandas as pd
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def get_pneumonia_dicts(csv_path: str, img_dir: str, patient_ids: list) -> list:
    df = pd.read_csv(csv_path)
    df = df[df["patientId"].isin(patient_ids)]

    dataset = []
    for idx, pid in enumerate(patient_ids):
        img_path = f"{img_dir}/{pid}.png"

        with Image.open(img_path) as img:
            w, h = img.size

        record = {
            "file_name": img_path,
            "image_id":  idx,
            "height":    h,
            "width":     w,
        }

        rows  = df[df["patientId"] == pid]
        valid = rows.dropna(subset=["x", "y", "width", "height"])

        annos = []
        for row in valid.itertuples():
            annos.append({
                "bbox":        [row.x, row.y, row.width, row.height],
                "bbox_mode":   BoxMode.XYWH_ABS,
                "category_id": 0,       # pneumonia = 0 (Detectron2 is 0-indexed)
                "iscrowd":     0,
            })

        record["annotations"] = annos
        dataset.append(record)

    return dataset


def register_pneumonia(csv_path: str, img_dir: str, train_frac: float = 0.8):
    all_pids = sorted(pd.read_csv(csv_path)["patientId"].unique().tolist())
    cutoff   = int(len(all_pids) * train_frac)

    train_ids = all_pids[:cutoff]
    val_ids   = all_pids[cutoff:]

    for split, pids in [("train", train_ids), ("val", val_ids)]:
        name = f"pneumonia_{split}"

        # Guard against re-registration if imported multiple times
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

        DatasetCatalog.register(
            name,
            lambda p=pids: get_pneumonia_dicts(csv_path, img_dir, p)
        )
        MetadataCatalog.get(name).set(thing_classes=["pneumonia"])

    print(f"Registered: {len(train_ids)} train / {len(val_ids)} val patients")
    return train_ids, val_ids