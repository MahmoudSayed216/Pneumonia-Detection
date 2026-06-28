"""
train.py — Fine-tune Fast R-CNN with Detectron2.

Usage:
    python train.py \
        --csv         data/annotations.csv \
        --img_dir     data/pngs \
        --output      checkpoints/ \
        --epochs      15 \
        --batch_size  4 \
        --lr          0.001
"""

import argparse
import os

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from register_dataset import register_pneumonia


# ---------------------------------------------------------------------------
# Evaluator hook so validation mAP is printed after each epoch
# ---------------------------------------------------------------------------

class TrainerWithEval(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # --- Register datasets ---
    register_pneumonia(
        csv_path=args.csv,
        img_dir=args.img_dir,
        train_frac=args.train_frac,
    )

    # --- Config ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
    ))

    # Datasets
    cfg.DATASETS.TRAIN = ("pneumonia_train",)
    cfg.DATASETS.TEST  = ("pneumonia_val",)

    # Pretrained weights from COCO
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml"
    )

    # Solver
    # MAX_ITER = epochs * (num_train_patients / batch_size)
    # We compute it approximately; Detectron2 works in iterations not epochs
    cfg.SOLVER.IMS_PER_BATCH    = args.batch_size
    cfg.SOLVER.BASE_LR          = args.lr
    cfg.SOLVER.MAX_ITER         = args.max_iter
    cfg.SOLVER.STEPS            = (
        int(args.max_iter * 0.6),
        int(args.max_iter * 0.8),
    )                             # LR drops at 60% and 80% of training
    cfg.SOLVER.GAMMA            = 0.1
    cfg.SOLVER.WARMUP_ITERS     = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period

    # Evaluation period (in iterations)
    cfg.TEST.EVAL_PERIOD = args.eval_period

    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1   # pneumonia only
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # DataLoader
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # Output
    cfg.OUTPUT_DIR = args.output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"Output dir  : {cfg.OUTPUT_DIR}")
    print(f"Max iters   : {cfg.SOLVER.MAX_ITER}")
    print(f"LR          : {cfg.SOLVER.BASE_LR}")
    print(f"Batch size  : {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Eval every  : {cfg.TEST.EVAL_PERIOD} iters")

    # --- Train ---
    trainer = TrainerWithEval(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # --- Final evaluation on val set ---
    print("\nRunning final evaluation on val set...")
    evaluator  = COCOEvaluator("pneumonia_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "pneumonia_val")
    results    = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Fast R-CNN (Detectron2)")

    parser.add_argument("--csv",              required=True)
    parser.add_argument("--img_dir",          required=True)
    parser.add_argument("--output",           default="checkpoints/")
    parser.add_argument("--train_frac",       type=float, default=0.8)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--lr",               type=float, default=0.001)
    parser.add_argument("--max_iter",         type=int,   default=5000,
                        help="Total training iterations. Rule of thumb: "
                             "epochs * (num_patients * 0.8 / batch_size)")
    parser.add_argument("--eval_period",      type=int,   default=500,
                        help="Run validation every N iterations")
    parser.add_argument("--checkpoint_period",type=int,   default=500,
                        help="Save checkpoint every N iterations")

    main(parser.parse_args())