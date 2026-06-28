"""
train.py — Fine-tune Faster R-CNN on 2x T4 GPUs using DistributedDataParallel.

Launch with torchrun (NOT python directly):
    torchrun --nproc_per_node=2 train.py \
        --csv       data/annotations.csv \
        --img_dir   data/pngs \
        --img_ext   .png \
        --num_classes 2 \
        --epochs    10 \
        --batch_size 4 \
        --lr        0.01 \
        --output    checkpoints/model_best.pth

Notes:
  - --batch_size is PER GPU. With 2 GPUs the effective batch size is 2x.
  - --lr should be scaled linearly with the number of GPUs (linear scaling rule).
    Single-GPU default was 0.005 → 2-GPU default is 0.01.
  - Checkpointing and printing only happen on rank 0 to avoid duplicate output.
  - Evaluation runs on rank 0 only (val set is small; no need to shard it).
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import DetectionDataset, collate_fn


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialise the process group. torchrun sets the env vars automatically."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Load Faster R-CNN pretrained on COCO and replace the box-predictor head
    to match num_classes (foreground classes + 1 background).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch: int, rank: int):
    model.train()
    # Tell the sampler which epoch this is so shuffling differs each epoch
    loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    n_batches  = len(loader)

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if is_main(rank) and (i + 1) % max(1, n_batches // 5) == 0:
            print(
                f"  [Epoch {epoch}  {i + 1}/{n_batches}]  "
                f"loss: {losses.item():.4f}  "
                f"(cls={loss_dict['loss_classifier'].item():.3f}, "
                f"box={loss_dict['loss_box_reg'].item():.3f}, "
                f"obj={loss_dict['loss_objectness'].item():.3f}, "
                f"rpn_box={loss_dict['loss_rpn_box_reg'].item():.3f})"
            )

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Evaluation  (rank 0 only)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, loader, device):
    """
    Compute mAP@50 and mAP@[0.40:0.75 step 0.05] on the validation set.
    Called on rank 0 only; the raw (un-wrapped) model is passed in.
    """
    model.eval()

    metric_50     = MeanAveragePrecision(iou_thresholds=[0.50], iou_type="bbox")
    custom_thresholds = list(np.arange(0.40, 0.76, 0.05).round(2))
    metric_custom = MeanAveragePrecision(iou_thresholds=custom_thresholds, iou_type="bbox")

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        outputs = model(images)

        preds = [
            {"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu()}
            for o in outputs
        ]
        gts = [
            {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
            for t in targets
        ]

        metric_50.update(preds, gts)
        metric_custom.update(preds, gts)

    return {
        "map_50":     metric_50.compute()["map"].item(),
        "map_custom": metric_custom.compute()["map"].item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if is_main(rank):
        print(f"DDP: {dist.get_world_size()} GPUs  |  device: {device}")

    # --- Dataset & splits ---
    # Split is done inside DetectionDataset on sorted unique patient IDs so
    # the same patient is never in both train and val (no leakage).
    train_ds = DetectionDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,
        split="train",
        train_frac=args.train_frac,
    )
    val_ds = DetectionDataset(
        csv_path=args.csv,
        img_dir=args.img_dir,

        split="val",
        train_frac=args.train_frac,
    )

    if is_main(rank):
        print(f"Dataset split (train_frac={args.train_frac}): "
              f"train={len(train_ds)} patients, val={len(val_ds)} patients")

    # DistributedSampler shards the training set across GPUs so each GPU sees
    # a unique, non-overlapping subset of samples every epoch.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,       # per-GPU batch size
        sampler=train_sampler,            # replaces shuffle=True
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Validation only runs on rank 0 — no need to shard it
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --- Model ---
    model = build_model(num_classes=args.num_classes, pretrained=args.pretrained)
    model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    # --- Optimizer & scheduler ---
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    # --- Training loop ---
    if is_main(rank):
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    best_map50 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(ddp_model, optimizer, train_loader, device, epoch, rank)

        # Synchronise all ranks before evaluation
        dist.barrier()

        # Evaluate on rank 0 using the unwrapped model
        if is_main(rank):
            metrics = evaluate(ddp_model.module, val_loader, device)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"mAP@50={metrics['map_50']:.4f}  "
                f"mAP@[.40:.75]={metrics['map_custom']:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"time={elapsed:.1f}s"
            )

            if metrics["map_50"] > best_map50:
                best_map50 = metrics["map_50"]
                # Save ddp_model.module (the unwrapped weights) so the
                # checkpoint can be loaded without DDP at inference time.
                torch.save(
                    {
                        "epoch":                epoch,
                        "model_state_dict":     ddp_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "map_50":               metrics["map_50"],
                        "map_custom":           metrics["map_custom"],
                    },
                    args.output,
                )
                print(f"  ✓ Saved best model (mAP@50={best_map50:.4f}) → {args.output}")

        lr_scheduler.step()
        dist.barrier()   # keep all ranks in step before next epoch

    if is_main(rank):
        print("Training complete.")

    cleanup_ddp()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN (2x T4, DDP)")

    # Data
    parser.add_argument("--csv",         required=True,  help="Path to annotations CSV")
    parser.add_argument("--img_dir",     required=True,  help="Directory containing images")
    parser.add_argument("--img_ext",     default=".png", help="Image file extension")
    parser.add_argument("--train_frac",  type=float, default=0.8,
                        help="Fraction of patients used for training (rest go to val)")
    parser.add_argument("--num_workers", type=int,   default=4,
                        help="DataLoader workers per GPU")

    # Model
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of foreground classes (background added automatically)")
    parser.add_argument("--pretrained",  action="store_true", default=True)

    # Training
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=4,
                        help="Per-GPU batch size (effective = batch_size × num_GPUs)")
    parser.add_argument("--lr",           type=float, default=0.01,
                        help="Learning rate (pre-scaled for 2 GPUs; single-GPU default is 0.005)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int,   default=3)
    parser.add_argument("--lr_gamma",     type=float, default=0.1)

    # Output
    parser.add_argument("--output", default="checkpoints/model_best.pth")

    main(parser.parse_args())