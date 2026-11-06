"""Compare prediction files with the ground-truth labels and print mAP, precision and recall per class."""
import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from ultralytics.utils.metrics import ap_per_class

parser = argparse.ArgumentParser()
parser.add_argument("--preds", required=True, help="folder with prediction .txt files (YOLO format + confidence)")
parser.add_argument("--images", default="data/taco_yolo/test.txt")
parser.add_argument("--labels", default="data/taco_yolo/labels")
parser.add_argument("--config", default="configs/classes.yaml")
parser.add_argument("--conf", type=float, default=None, help="also show precision/recall at this exact threshold")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
names = {i: c["name"] for i, c in enumerate(cfg["classes"])}

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # 0.5, 0.55, ....., 0.95 (same as YOLO)

def read_boxes(path, with_conf):
    """Read a YOLO txt file -> (classes, boxes as x1y1x2y2, confidence)."""
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [line.split() for line in f if line.strip()]
    data = np.array(rows, dtype=float).reshape(-1, 6 if with_conf else 5)
    cls = data[:, 0].astype(int)
    cx, cy, w, h = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    conf = data[:, 5] if with_conf else None
    return cls, boxes, conf


def box_iou(a, b):
    """IoU between every box in a (M, 4) and every box in b (N, 4) -> (M, N)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None)*np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def match(pred_cls, gt_cls, iou):
    """For each prediction (sorted by confidence highest first) and each IoU threshold
    True if it hits a not-yet-matched ground-truth box of the same class. Same rule as YOLO's validator."""
    correct = np.zeros((len(pred_cls), len(IOU_THRESHOLDS)), dtype=bool)
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :]) #wrong class = no match
    taken = np.zeros((len(gt_cls), len(IOU_THRESHOLDS)), dtype=bool)
    for j in range(len(pred_cls)):
        available = np.where(taken, 0, iou[:, j, None]) # IoU with each free GT box, per threshold
        k = available.argmax(0)                         # best free GT box, per threshold
        correct[j] = available[k, range(len(IOU_THRESHOLDS))] >= IOU_THRESHOLDS
        taken[k, range(len(IOU_THRESHOLDS))] |= correct[j]
    return correct

with open(args.images) as f:
    stems = [os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip()]

all_correct, all_conf, all_pred_cls, all_gt_cls = [], [], [], []
for stem in stems:
    gt_cls, gt_boxes, _ = read_boxes(os.path.join(args.labels, stem + ".txt"), with_conf=False)
    pred_cls, pred_boxes, conf = read_boxes(os.path.join(args.preds, stem + ".txt"), with_conf=True)
    order = np.argsort(-conf)
    pred_cls, pred_boxes, conf = pred_cls[order], pred_boxes[order], conf[order]

    all_correct.append(match(pred_cls, gt_cls, box_iou(gt_boxes, pred_boxes)))
    all_conf.append(conf)
    all_pred_cls.append(pred_cls)
    all_gt_cls.append(gt_cls)

correct = np.concatenate(all_correct)
conf = np.concatenate(all_conf)
pred_cls = np.concatenate(all_pred_cls)
gt_cls = np.concatenate(all_gt_cls)

# Same function YOLO uses, so the numbers are directly comparable with YOLO's val results
_, _, p, r, _, ap, classes, *_ = ap_per_class(
    correct, conf, pred_cls, gt_cls, plot=True, save_dir=Path(args.preds), names=names
)

print(f"{len(stems)} images, {len(gt_cls)} ground-truth boxes, {len(conf)} predictions\n")
print(f"{'class':12s} {'GT':>5s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>9s}  (P/R at the best-F1 threshold)")
for i, c in enumerate(classes):
    n = (gt_cls == c).sum()
    print(f"{names[c]:12s} {n:5d} {p[i]:6.3f} {r[i]:6.3f} {ap[i, 0]:6.3f} {ap[i].mean():9.3f}")
print(f"{'all':12s} {len(gt_cls):5d} {p.mean():6.3f} {r.mean():6.3f} {ap[:, 0].mean():6.3f} {ap.mean():9.3f}")

if args.conf is not None:
    print(f"\nAt confidence >= {args.conf} (IoU 0.5):")
    keep = conf >= args.conf
    for c in classes:
        tp = (correct[keep, 0] & (pred_cls[keep] == c)).sum()
        n_pred = (pred_cls[keep] == c).sum()
        n_gt = (gt_cls == c).sum()
        print(f"{names[c]:12s} precision {tp / max(n_pred, 1):.3f} recall {tp /max(n_gt, 1):.3f} ({n_pred} boxes)")
