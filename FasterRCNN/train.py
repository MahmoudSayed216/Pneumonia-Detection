"""
train.py — Fine-tune a Faster R-CNN (ResNet-50 FPN backbone) on a custom dataset.

Usage:
    python train.py \
        --csv     data/annotations.csv \
        --img_dir data/images \
        --img_ext .png \
        --num_classes 2 \
        --epochs 10 \
        --batch_size 4 \
        --lr 0.005 \
        --output checkpoints/model_final.pth
"""

import argparse
import os
import time

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import DetectionDataset, collate_fn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Load a Faster R-CNN pretrained on COCO and replace its box predictor
    so the output head matches `num_classes` (including the background class).

    Args:
        num_classes : number of foreground classes + 1 (background)
        pretrained  : start from COCO weights when True
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Swap the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch: int):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(
                f"  [Epoch {epoch}  {i + 1}/{n_batches}]  "
                f"loss: {losses.item():.4f}  "
                f"(cls={loss_dict['loss_classifier'].item():.3f}, "
                f"box={loss_dict['loss_box_reg'].item():.3f}, "
                f"obj={loss_dict['loss_objectness'].item():.3f}, "
                f"rpn_box={loss_dict['loss_rpn_box_reg'].item():.3f})"
            )

    return total_loss / n_batches


@torch.inference_mode()
def evaluate(model, loader, device):
    """
    Runs inference on the validation set and returns:
        - map_50   : mAP at IoU threshold 0.50
        - map_custom : mean mAP averaged over IoU thresholds 0.40 → 0.75
                       in steps of 0.05  (i.e. 0.40, 0.45, 0.50, 0.55,
                       0.60, 0.65, 0.70, 0.75)

    torchmetrics' MeanAveragePrecision only supports the fixed COCO range
    (0.50:0.95), so map_custom is computed manually: we instantiate one
    metric per threshold and average the results.
    """
    model.eval()

    # --- mAP @ 0.50 (standard VOC metric) ---
    metric_50 = MeanAveragePrecision(iou_thresholds=[0.50], iou_type="bbox")

    # --- mAP @ 0.40:0.75 step 0.05 ---
    custom_thresholds = list(np.arange(0.40, 0.76, 0.05).round(2))
    metric_custom = MeanAveragePrecision(iou_thresholds=custom_thresholds, iou_type="bbox")

    for images, targets in loader:
        images = [img.to(device) for img in images]

        outputs = model(images)   # list of dicts: boxes, labels, scores

        # torchmetrics expects CPU tensors
        preds = [
            {
                "boxes":  o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu(),
            }
            for o in outputs
        ]
        gts = [
            {
                "boxes":  t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            }
            for t in targets
        ]

        metric_50.update(preds, gts)
        metric_custom.update(preds, gts)

    result_50     = metric_50.compute()
    result_custom = metric_custom.compute()

    return {
        "map_50":     result_50["map"].item(),
        "map_custom": result_custom["map"].item(),   # mean over 0.40:0.75
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset & splits ---
    full_dataset = DetectionDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_ext=args.img_ext,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Dataset: {n_total} images  →  train={n_train}, val={n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # --- Model ---
    # num_classes = number of foreground categories + 1 (background)
    model = build_model(num_classes=args.num_classes, pretrained=args.pretrained)
    model.to(device)

    # --- Optimizer & scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    # --- Training loop ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_map50 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        metrics    = evaluate(model, val_loader, device)
        lr_scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"mAP@50={metrics['map_50']:.4f}  "
            f"mAP@[.40:.75]={metrics['map_custom']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save the best checkpoint (tracked by mAP@50)
        if metrics["map_50"] > best_map50:
            best_map50 = metrics["map_50"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "map_50":     metrics["map_50"],
                    "map_custom": metrics["map_custom"],
                },
                args.output,
            )
            print(f"  ✓ Saved best model (mAP@50={best_map50:.4f}) → {args.output}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN")

    # Data
    parser.add_argument("--csv",         required=True,  help="Path to annotations CSV")
    parser.add_argument("--img_dir",     required=True,  help="Directory containing images")
    parser.add_argument("--img_ext",     default=".png", help="Image file extension")
    parser.add_argument("--val_split",   type=float, default=0.2,
                        help="Fraction of data used for validation")
    parser.add_argument("--num_workers", type=int,   default=4)

    # Model
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of foreground classes (background is added automatically)")
    parser.add_argument("--pretrained",  action="store_true", default=True,
                        help="Initialize from COCO pretrained weights")

    # Training
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int,   default=3,
                        help="Decay LR every N epochs")
    parser.add_argument("--lr_gamma",     type=float, default=0.1)

    # Output
    parser.add_argument("--output", default="checkpoints/model_best.pth",
                        help="Where to save the best checkpoint")

    main(parser.parse_args())