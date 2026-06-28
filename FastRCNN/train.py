"""
Script 4 — Fast R-CNN: model definition + training loop
---------------------------------------------------------
Multi-GPU : torch.nn.DataParallel  (2× T4 out of the box)
Metrics   : mAP@0.50  and  mAP@[0.40:0.05:0.75]  via torchmetrics
            computed on the validation set after every epoch and logged
            to stdout + training_history.json

Architecture
    Backbone  : ResNet-50 pretrained on ImageNet
    Freeze    : stem (conv1) + layer1
    Unfreeze  : layer2, layer3, layer4 + head
    Head      : RoI Pool 7×7 → FC(4096) → FC(4096) → cls + bbox heads

Hyperparameters (15-20 epochs)
    Optimiser : SGD  momentum=0.9  weight_decay=1e-4  nesterov=True
    LR        : 0.005  →  0.0005 (epoch 7)  →  0.00005 (epoch 14)
    Warmup    : 1 epoch linear ramp

Usage:
    # single-node 2-GPU (DataParallel) — just run normally, CUDA_VISIBLE_DEVICES handles the rest
    CUDA_VISIBLE_DEVICES=0,1 python 04_train.py \
        --csv_path   /data/labels.csv \
        --image_dir  /data/images \
        --proposals  /data/proposals.h5 \
        --output_dir /data/checkpoints \
        --batch_size 4          # 2 per GPU  ← recommended for 2×T4
        [--epochs 20]
        [--resume /data/checkpoints/epoch_010.pth]

Install:
    pip install torch torchvision torchmetrics
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import RoIPool, nms, box_convert
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from dataset import (
    PneumoniaDetectionDataset,
    fast_rcnn_collate,
    make_splits,
)


# =========================================================================== #
#  Box utilities
# =========================================================================== #

def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. boxes in [x1,y1,x2,y2]."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    ix1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def encode_boxes(proposals: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Fast R-CNN box encoding → [dx, dy, dw, dh]."""
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gx = gt_boxes[:, 0] + 0.5 * gw
    gy = gt_boxes[:, 1] + 0.5 * gh

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw.clamp(min=1e-6))
    dh = torch.log(gh / ph.clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_boxes. proposals [x1,y1,x2,y2], deltas [dx,dy,dw,dh]."""
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph

    gx = deltas[:, 0] * pw + px
    gy = deltas[:, 1] * ph + py
    gw = torch.exp(deltas[:, 2].clamp(max=4.0)) * pw
    gh = torch.exp(deltas[:, 3].clamp(max=4.0)) * ph

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh
    return torch.stack([x1, y1, x2, y2], dim=1)


def label_proposals(
    proposals: torch.Tensor,
    gt_boxes:  torch.Tensor,
    gt_labels: torch.Tensor,
    fg_iou_thresh: float = 0.5,
    bg_iou_hi:     float = 0.5,
    bg_iou_lo:     float = 0.1,
    num_samples:   int   = 128,
    fg_fraction:   float = 0.25,
) -> tuple:
    """
    Sample 128 RoIs per image: 25 % fg (IoU ≥ 0.5), 75 % bg (0.1 ≤ IoU < 0.5).
    Returns (sampled_proposals, sampled_labels, sampled_targets).
    """
    device = proposals.device

    if gt_boxes.numel() == 0:
        n   = min(num_samples, len(proposals))
        idx = torch.randperm(len(proposals), device=device)[:n]
        return (
            proposals[idx],
            torch.zeros(n, dtype=torch.long, device=device),
            torch.zeros(n, 4, device=device),
        )

    iou = box_iou(proposals, gt_boxes)        # (P, G)
    max_iou, best_gt = iou.max(dim=1)

    fg_mask = max_iou >= fg_iou_thresh
    bg_mask = (max_iou >= bg_iou_lo) & (max_iou < bg_iou_hi)

    n_fg    = min(int(num_samples * fg_fraction), fg_mask.sum().item())
    fg_idx  = fg_mask.nonzero(as_tuple=False).squeeze(1)
    fg_idx  = fg_idx[torch.randperm(len(fg_idx), device=device)[:n_fg]]

    n_bg    = num_samples - n_fg
    bg_idx  = bg_mask.nonzero(as_tuple=False).squeeze(1)
    bg_idx  = bg_idx[torch.randperm(len(bg_idx), device=device)[:n_bg]]

    sampled_idx   = torch.cat([fg_idx, bg_idx])
    sampled_props = proposals[sampled_idx]

    sampled_lbls  = torch.zeros(len(sampled_idx), dtype=torch.long, device=device)
    sampled_lbls[:n_fg] = gt_labels[best_gt[fg_idx]]

    sampled_tgts  = torch.zeros(len(sampled_idx), 4, device=device)
    if n_fg > 0:
        sampled_tgts[:n_fg] = encode_boxes(
            sampled_props[:n_fg], gt_boxes[best_gt[fg_idx]]
        )

    return sampled_props, sampled_lbls, sampled_tgts


# =========================================================================== #
#  Model
# =========================================================================== #

class FastRCNN(nn.Module):
    """
    Fast R-CNN — ResNet-50 backbone, RoI Pool head.

    Freeze strategy
    ---------------
    Frozen   : stem (conv1+bn1) + layer1  → universal low-level features
    Trainable: layer2, layer3, layer4, head
    BN in frozen stages stays in eval mode throughout (freeze_bn()).
    """

    def __init__(
        self,
        num_classes:       int   = 2,           # 0=bg, 1=pneumonia
        roi_pool_size:     int   = 7,
        roi_spatial_scale: float = 1.0 / 16.0,  # ResNet-50 stride at layer4
    ):
        super().__init__()
        self.num_classes = num_classes

        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4

        # Freeze stem + layer1
        for m in [self.stem, self.layer1]:
            for p in m.parameters():
                p.requires_grad = False

        self.roi_pool = RoIPool(
            output_size=(roi_pool_size, roi_pool_size),
            spatial_scale=roi_spatial_scale,
        )

        in_feat = 2048 * roi_pool_size * roi_pool_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feat, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096,    4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
        )
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def freeze_bn(self):
        for m in [self.stem, self.layer1]:
            for mod in m.modules():
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    mod.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.freeze_bn()
        return self

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x                               # (B, 2048, H/32, W/32)

    def forward(
        self,
        images:    torch.Tensor,               # (B, 3, H, W)
        proposals: list[torch.Tensor],         # list of (P_i, 4)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.extract_features(images)   # (B, 2048, h, w)

        roi_list = []
        for i, props in enumerate(proposals):
            idx_col = torch.full((len(props), 1), i,
                                 dtype=props.dtype, device=props.device)
            roi_list.append(torch.cat([idx_col, props], dim=1))
        rois = torch.cat(roi_list, dim=0)      # (total_P, 5)

        pooled = self.roi_pool(fmap, rois)     # (total_P, 2048, 7, 7)
        feat   = self.head(pooled)             # (total_P, 4096)

        return self.cls_score(feat), self.bbox_pred(feat)


# =========================================================================== #
#  Loss
# =========================================================================== #

def fast_rcnn_loss(
    cls_logits:  torch.Tensor,
    bbox_deltas: torch.Tensor,
    labels:      torch.Tensor,
    box_targets: torch.Tensor,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_loss = nn.functional.cross_entropy(cls_logits, labels)

    fg_mask = labels > 0
    if fg_mask.sum() == 0:
        loc_loss = cls_logits.sum() * 0.0
    else:
        fg_deltas  = bbox_deltas[fg_mask]
        fg_targets = box_targets[fg_mask]
        fg_labels  = labels[fg_mask]

        col_start = fg_labels * 4
        idxs = torch.stack([col_start + k for k in range(4)], dim=1)
        pred_boxes = fg_deltas.gather(1, idxs)

        loc_loss = nn.functional.smooth_l1_loss(pred_boxes, fg_targets, beta=1.0)

    return cls_loss + lam * loc_loss, cls_loss, loc_loss


# =========================================================================== #
#  LR scheduler: linear warmup → StepLR
# =========================================================================== #

class LinearWarmupThenStep:
    def __init__(self, optimizer, base_lr, warmup_epochs, step_size, gamma):
        self.optimizer      = optimizer
        self.base_lr        = base_lr
        self.warmup_epochs  = warmup_epochs
        self.step_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.step_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# =========================================================================== #
#  mAP evaluation
# =========================================================================== #

def decode_predictions(
    cls_logits:  torch.Tensor,     # (N, C)
    bbox_deltas: torch.Tensor,     # (N, C*4)
    proposals:   torch.Tensor,     # (N, 4)
    score_thresh: float = 0.05,
    nms_thresh:   float = 0.5,
    num_classes:  int   = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode raw model outputs for one image into (boxes, scores, labels)
    after score thresholding and per-class NMS.
    """
    scores_all = torch.softmax(cls_logits, dim=1)  # (N, C)

    all_boxes, all_scores, all_labels = [], [], []

    for cls_id in range(1, num_classes):            # skip background
        cls_scores = scores_all[:, cls_id]          # (N,)
        keep_mask  = cls_scores > score_thresh
        if keep_mask.sum() == 0:
            continue

        cls_scores = cls_scores[keep_mask]
        cls_props  = proposals[keep_mask]
        cls_deltas = bbox_deltas[keep_mask, cls_id * 4 : cls_id * 4 + 4]
        cls_boxes  = decode_boxes(cls_props, cls_deltas)

        keep = nms(cls_boxes, cls_scores, nms_thresh)
        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores[keep])
        all_labels.append(torch.full((len(keep),), cls_id,
                                     dtype=torch.long, device=cls_logits.device))

    if not all_boxes:
        dev = cls_logits.device
        return (torch.zeros(0, 4, device=dev),
                torch.zeros(0,    device=dev),
                torch.zeros(0,    dtype=torch.long, device=dev))

    return (torch.cat(all_boxes),
            torch.cat(all_scores),
            torch.cat(all_labels))


@torch.no_grad()
def evaluate_map(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int,
) -> dict:
    """
    Run inference on the validation set and compute:
      • mAP@0.50
      • mAP@[0.40:0.05:0.75]  (7 thresholds)
    using torchmetrics.detection.MeanAveragePrecision.

    Returns dict with keys  map_50, map_40_75.
    """
    # iou_thresholds for the COCO-style range variant
    iou_thresholds_range = [round(t, 2) for t in
                            torch.arange(0.40, 0.76, 0.05).tolist()]

    metric_50    = MeanAveragePrecision(iou_thresholds=[0.50],
                                        box_format="xyxy").to(device)
    metric_range = MeanAveragePrecision(iou_thresholds=iou_thresholds_range,
                                        box_format="xyxy").to(device)

    # unwrap DataParallel to call extract_features / forward on a single module
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()

    proposal_offset = 0  # running offset into the concatenated RoI output

    for batch in tqdm(loader, desc="mAP eval", leave=False):
        images    = batch["images"].to(device)
        proposals = [p.to(device) for p in batch["proposals"]]
        gt_boxes  = [b.to(device) for b in batch["boxes"]]
        gt_labels = [l.to(device) for l in batch["labels"]]

        # We do NOT sample here — pass all proposals for inference
        cls_logits, bbox_deltas = raw_model(images, proposals)

        # Split concatenated outputs back per image
        sizes = [len(p) for p in proposals]
        cls_splits   = cls_logits.split(sizes)
        delta_splits = bbox_deltas.split(sizes)

        preds, targets = [], []
        for i, (cls_l, bbox_d, props, gtb, gtl) in enumerate(
            zip(cls_splits, delta_splits, proposals, gt_boxes, gt_labels)
        ):
            boxes, scores, labels = decode_predictions(
                cls_l, bbox_d, props,
                score_thresh=0.05, nms_thresh=0.5,
                num_classes=num_classes,
            )
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
            targets.append({
                "boxes":  gtb,
                "labels": gtl if gtl.numel() > 0
                          else torch.zeros(0, dtype=torch.long, device=device),
            })

        metric_50.update(preds, targets)
        metric_range.update(preds, targets)

    raw_model.train()

    res_50    = metric_50.compute()
    res_range = metric_range.compute()

    return {
        "map_50":    res_50["map"].item(),
        "map_40_75": res_range["map"].item(),
    }


# =========================================================================== #
#  Training loop
# =========================================================================== #

def run_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    train:     bool,
) -> dict:
    model.train(train)
    total_loss = cls_sum = loc_sum = 0.0
    n_batches  = 0

    phase = "Train" if train else "Val  "

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc=phase, leave=False):
            images    = batch["images"].to(device)
            proposals = [p.to(device) for p in batch["proposals"]]
            gt_boxes  = [b.to(device) for b in batch["boxes"]]
            gt_labels = [l.to(device) for l in batch["labels"]]

            # Sample proposals per image
            sampled_props, sampled_lbls, sampled_tgts = [], [], []
            for props, boxes, lbls in zip(proposals, gt_boxes, gt_labels):
                sp, sl, st = label_proposals(props, boxes, lbls)
                sampled_props.append(sp)
                sampled_lbls.append(sl)
                sampled_tgts.append(st)

            cat_lbls = torch.cat(sampled_lbls)
            cat_tgts = torch.cat(sampled_tgts)

            cls_logits, bbox_deltas = model(images, sampled_props)
            loss, cls_l, loc_l = fast_rcnn_loss(
                cls_logits, bbox_deltas, cat_lbls, cat_tgts
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Clip on the underlying module's params (DataParallel-safe)
                raw = model.module if isinstance(model, nn.DataParallel) else model
                nn.utils.clip_grad_norm_(raw.parameters(), max_norm=10.0)
                optimizer.step()

            total_loss += loss.item()
            cls_sum    += cls_l.item()
            loc_sum    += loc_l.item()
            n_batches  += 1

    n = max(n_batches, 1)
    return {
        "loss":     total_loss / n,
        "cls_loss": cls_sum    / n,
        "loc_loss": loc_sum    / n,
    }


# =========================================================================== #
#  Entry point
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="Train Fast R-CNN — 2×T4 ready")
    p.add_argument("--csv_path",    required=True)
    p.add_argument("--image_dir",   required=True)
    p.add_argument("--proposals",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="Total batch size across all GPUs (recommend 4 for 2×T4 = 2/gpu)")
    p.add_argument("--lr",           type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--step_size",    type=int,   default=7)
    p.add_argument("--gamma",        type=float, default=0.1)
    p.add_argument("--warmup",       type=int,   default=1)
    p.add_argument("--val_frac",     type=float, default=0.15)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_classes",  type=int,   default=2)
    p.add_argument("--resume",       type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus    = torch.cuda.device_count()
    print(f"[*] Device : {device}  |  GPUs visible: {n_gpus}")

    # ---- Data ---------------------------------------------------------------
    train_ids, val_ids = make_splits(args.csv_path, args.val_frac, args.seed)
    print(f"[*] Train: {len(train_ids)}  Val: {len(val_ids)}")

    train_ds = PneumoniaDetectionDataset(
        args.csv_path, args.image_dir, args.proposals,
        split=train_ids, train=True,
    )
    val_ds = PneumoniaDetectionDataset(
        args.csv_path, args.image_dir, args.proposals,
        split=val_ids, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=fast_rcnn_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=fast_rcnn_collate,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model — wrap in DataParallel if multiple GPUs ----------------------
    model = FastRCNN(num_classes=args.num_classes).to(device)

    if n_gpus > 1:
        print(f"[*] Wrapping model in DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    total     = sum(p.numel() for p in raw_model.parameters())
    frozen    = sum(p.numel() for p in raw_model.parameters() if not p.requires_grad)
    print(f"[*] Parameters — total: {total:,}  frozen: {frozen:,}  "
          f"trainable: {total - frozen:,}")

    # ---- Optimiser & Scheduler ----------------------------------------------
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = LinearWarmupThenStep(
        optimizer, args.lr, args.warmup, args.step_size, args.gamma
    )

    # ---- Resume -------------------------------------------------------------
    start_epoch    = 0
    best_map50     = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_map50  = ckpt.get("best_map50", 0.0)
        print(f"[*] Resumed from epoch {ckpt['epoch']}  best mAP@50={best_map50:.4f}")

    # ---- Output dir ---------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    # =========================================================================
    #  Main loop
    # =========================================================================
    SEP = "─" * 90

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # -- Train ------------------------------------------------------------
        train_m = run_one_epoch(model, train_loader, optimizer, device, train=True)

        # -- Validation loss --------------------------------------------------
        val_m = run_one_epoch(model, val_loader, None, device, train=False)

        # -- mAP (val set) ----------------------------------------------------
        map_m = evaluate_map(model, val_loader, device, args.num_classes)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        # -- Per-epoch log (stdout) -------------------------------------------
        print(SEP)
        print(
            f"Epoch {epoch+1:>3}/{args.epochs}  |  "
            f"lr={current_lr:.2e}  |  elapsed={elapsed:.0f}s"
        )
        print(
            f"  LOSS   train={train_m['loss']:.4f}"
            f"  (cls={train_m['cls_loss']:.4f} loc={train_m['loc_loss']:.4f})"
            f"   val={val_m['loss']:.4f}"
            f"  (cls={val_m['cls_loss']:.4f} loc={val_m['loc_loss']:.4f})"
        )
        print(
            f"  mAP    @0.50={map_m['map_50']:.4f}"
            f"   @[.40:.75]={map_m['map_40_75']:.4f}"
        )

        # -- History record ---------------------------------------------------
        record = {
            "epoch":   epoch + 1,
            "lr":      current_lr,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
            **map_m,
        }
        history.append(record)

        # -- Checkpoint (every epoch, resume-able) ----------------------------
        torch.save({
            "epoch":      epoch,
            "model":      raw_model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_map50": best_map50,
            "args":       vars(args),
        }, output_dir / f"epoch_{epoch+1:03d}.pth")

        # -- Best model (tracked by mAP@0.50) --------------------------------
        if map_m["map_50"] > best_map50:
            best_map50 = map_m["map_50"]
            torch.save(raw_model.state_dict(), output_dir / "best_model.pth")
            print(f"  ★  New best mAP@0.50={best_map50:.4f}  → saved best_model.pth")

    print(SEP)

    # -- Write history JSON ---------------------------------------------------
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[✓] Training complete.  Best mAP@0.50 = {best_map50:.4f}")
    print(f"    Outputs → {output_dir}")


if __name__ == "__main__":
    main()