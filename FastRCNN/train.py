"""
train_fast_rcnn.py — Fine-tune Fast R-CNN on chest X-rays.

Launch with:
    python train_fast_rcnn.py \
        --csv          data/annotations.csv \
        --img_dir      data/pngs \
        --proposal_dir data/proposals \
        --num_classes  2 \
        --epochs       10 \
        --output       checkpoints/fast_rcnn_best.pth
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align, box_iou

from detection_ds import DetectionDataset, collate_fn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FastRCNN(nn.Module):
    """
    Fast R-CNN with a ResNet-50 backbone (up to layer3, stride 16).
    Proposals are fed in externally (e.g. from selective search).
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # Stride 16 after layer3: conv1(2) * maxpool(2) * layer2(2) * layer3(2) = 16
        self.body = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )
        self.spatial_scale = 1 / 16.0
        self.roi_size      = 7

        # layer3 outputs 1024 channels
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),          nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cls_score  = nn.Linear(1024, num_classes)
        self.bbox_pred  = nn.Linear(1024, num_classes * 4)
        self.num_classes = num_classes

    def extract_roi_features(self, images, proposals):
        device    = images[0].device
        img_batch = torch.stack(images)
        features  = self.body(img_batch)

        # roi_align expects [K, 5]: (batch_idx, x1, y1, x2, y2)
        box_list = []
        for b, props in enumerate(proposals):
            idx = torch.full((len(props), 1), b, dtype=torch.float32, device=device)
            box_list.append(torch.cat([idx, props.float()], dim=1))
        all_boxes = torch.cat(box_list, dim=0)

        rois = roi_align(
            features, all_boxes,
            output_size=self.roi_size,
            spatial_scale=self.spatial_scale,
            aligned=True,
        )
        return self.head(rois)

    def forward(self, images, proposals, targets=None):
        """
        images    : list of [3, H, W] float tensors
        proposals : list of [N, 4] tensors (x1, y1, x2, y2)
        targets   : list of {boxes, labels} dicts — required during training
        """
        device  = images[0].device
        pooled  = self.extract_roi_features(images, proposals)
        logits  = self.cls_score(pooled)
        deltas  = self.bbox_pred(pooled)

        if targets is None:
            return logits, deltas

        # --- Match proposals to GT and build training targets ---
        all_labels   = []
        all_reg_tgts = []
        all_pos_mask = []

        for props, tgt in zip(proposals, targets):
            gt_boxes  = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device)

            if len(gt_boxes) == 0:
                lbl      = torch.zeros(len(props), dtype=torch.long, device=device)
                reg_tgts = torch.zeros((len(props), 4), device=device)
                pos_mask = torch.zeros(len(props), dtype=torch.bool, device=device)
            else:
                iou             = box_iou(props.float(), gt_boxes.float())
                max_iou, best   = iou.max(dim=1)

                lbl = torch.zeros(len(props), dtype=torch.long, device=device)
                pos_mask = max_iou >= 0.5
                lbl[pos_mask] = gt_labels[best[pos_mask]]
                lbl[(max_iou >= 0.1) & ~pos_mask] = -1   # ignore ambiguous

                reg_tgts = _encode_boxes(props.float(), gt_boxes[best].float())

            all_labels.append(lbl)
            all_reg_tgts.append(reg_tgts)
            all_pos_mask.append(pos_mask)

        all_labels   = torch.cat(all_labels)
        all_reg_tgts = torch.cat(all_reg_tgts)
        all_pos_mask = torch.cat(all_pos_mask)

        # Classification loss — skip ignored labels (-1)
        valid    = all_labels >= 0
        loss_cls = F.cross_entropy(logits[valid], all_labels[valid])

        # Regression loss — positives only, GT class channel
        pos_idx = all_pos_mask.nonzero(as_tuple=True)[0]
        if len(pos_idx) > 0:
            pos_labels  = all_labels[pos_idx]
            pred_deltas = deltas[pos_idx].view(-1, self.num_classes, 4)
            pred_deltas = pred_deltas[torch.arange(len(pos_idx)), pos_labels]
            loss_reg    = F.smooth_l1_loss(pred_deltas, all_reg_tgts[pos_idx], beta=1.0)
        else:
            loss_reg = deltas.sum() * 0.0

        return {"loss_cls": loss_cls, "loss_reg": loss_reg}


def _encode_boxes(proposals, gt_boxes):
    """Encode GT boxes as (dx, dy, dw, dh) deltas relative to proposals."""
    pw = proposals[:, 2] - proposals[:, 0]; px = proposals[:, 0] + 0.5 * pw
    ph = proposals[:, 3] - proposals[:, 1]; py = proposals[:, 1] + 0.5 * ph
    gw = gt_boxes[:, 2]  - gt_boxes[:, 0];  gx = gt_boxes[:, 0]  + 0.5 * gw
    gh = gt_boxes[:, 3]  - gt_boxes[:, 1];  gy = gt_boxes[:, 1]  + 0.5 * gh
    return torch.stack([(gx-px)/pw, (gy-py)/ph,
                        torch.log(gw/pw), torch.log(gh/ph)], dim=1)


def _decode_boxes(proposals, deltas, num_classes):
    """Decode predicted deltas back into boxes (used at inference)."""
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pw = proposals[:, 2] - proposals[:, 0]; px = proposals[:, 0] + 0.5 * pw
    ph = proposals[:, 3] - proposals[:, 1]; py = proposals[:, 1] + 0.5 * ph
    cx = dx * pw + px;  cy = dy * ph + py
    w  = torch.exp(dw) * pw;  h = torch.exp(dh) * ph
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)


# ---------------------------------------------------------------------------
# Proposal loader (wraps DetectionDataset to also return proposals)
# ---------------------------------------------------------------------------

class FastRCNNDataset(DetectionDataset):
    """
    Extends DetectionDataset to load pre-computed selective search proposals.
    Proposals are stored as {patient_id}.npy in proposal_dir.
    """

    def __init__(self, proposal_dir: str, max_proposals: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.proposal_dir  = proposal_dir
        self.max_proposals = max_proposals

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        patient_id    = self.patient_ids[idx]

        prop_path = os.path.join(self.proposal_dir, f"{patient_id}.npy")
        if os.path.exists(prop_path):
            props = torch.from_numpy(np.load(prop_path))[:self.max_proposals]
        else:
            # Fallback: use the full image as one proposal
            _, H, W = image.shape
            props = torch.tensor([[0, 0, W, H]], dtype=torch.float32)

        return image, props, target


def collate_fast_rcnn(batch):
    images, proposals, targets = zip(*batch)
    return list(images), list(proposals), list(targets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, (images, proposals, targets) in enumerate(loader):
        images    = [img.to(device) for img in images]
        proposals = [p.to(device)   for p  in proposals]
        targets   = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, proposals, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(
                f"  [Epoch {epoch}  {i+1}/{n_batches}]  "
                f"loss: {losses.item():.4f}  "
                f"(cls={loss_dict['loss_cls'].item():.3f}, "
                f"reg={loss_dict['loss_reg'].item():.3f})"
            )

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, loader, device, score_thresh=0.05, nms_thresh=0.5):
    from torchvision.ops import nms

    model.eval()
    metric_50     = MeanAveragePrecision(iou_thresholds=[0.50], iou_type="bbox", sync_on_compute=False)
    custom_thresh = list(np.arange(0.40, 0.76, 0.05).round(2))
    metric_custom = MeanAveragePrecision(iou_thresholds=custom_thresh,  iou_type="bbox", sync_on_compute=False)

    for images, proposals, targets in loader:
        images    = [img.to(device) for img in images]
        proposals = [p.to(device)   for p  in proposals]

        logits, deltas = model(images, proposals)

        scores  = F.softmax(logits, dim=1)
        n_props = [len(p) for p in proposals]

        preds, gts = [], []
        offset = 0
        for b, n in enumerate(n_props):
            s = scores[offset:offset+n]             # [N, num_classes]
            d = deltas[offset:offset+n]             # [N, num_classes*4]
            p = proposals[b]

            # Take the foreground class (class 1 for binary)
            fg_scores = s[:, 1]
            fg_deltas = d.view(-1, model.num_classes, 4)[:, 1, :]
            boxes_out = _decode_boxes(p, fg_deltas, model.num_classes)

            # Threshold + NMS
            keep = fg_scores > score_thresh
            boxes_out, fg_scores = boxes_out[keep], fg_scores[keep]
            if len(boxes_out) > 0:
                keep = nms(boxes_out, fg_scores, nms_thresh)
                boxes_out, fg_scores = boxes_out[keep], fg_scores[keep]

            preds.append({
                "boxes":  boxes_out.cpu(),
                "scores": fg_scores.cpu(),
                "labels": torch.ones(len(boxes_out), dtype=torch.long),
            })
            gts.append({
                "boxes":  targets[b]["boxes"],
                "labels": targets[b]["labels"],
            })
            offset += n

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_kwargs = dict(
        csv_path=args.csv,
        img_dir=args.img_dir,
        train_frac=args.train_frac,
        proposal_dir=args.proposal_dir,
        max_proposals=args.max_proposals,
    )
    train_ds = FastRCNNDataset(split="train", **ds_kwargs)
    val_ds   = FastRCNNDataset(split="val",   **ds_kwargs)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fast_rcnn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fast_rcnn)

    model = FastRCNN(num_classes=args.num_classes, pretrained=args.pretrained)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_map50 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        metrics    = evaluate(eval_model, val_loader, device)
        elapsed    = time.time() - t0

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
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     eval_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "map_50":               metrics["map_50"],
                "map_custom":           metrics["map_custom"],
            }, args.output)
            print(f"  ✓ Saved best model (mAP@50={best_map50:.4f}) → {args.output}")

        scheduler.step()

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Fast R-CNN")

    parser.add_argument("--csv",           required=True)
    parser.add_argument("--img_dir",       required=True)
    parser.add_argument("--proposal_dir",  required=True,  help="Dir with .npy proposal files")
    parser.add_argument("--train_frac",    type=float, default=0.8)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--num_classes",   type=int,   required=True)
    parser.add_argument("--pretrained",    action="store_true", default=True)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--max_proposals", type=int,   default=300)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--output",        default="checkpoints/fast_rcnn_best.pth")

    main(parser.parse_args())