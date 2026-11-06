"""Stage2: let CLIP decide the class of every Grounding DINO box. Reads one prediction folder, writes another."""
import argparse
import os

import torch
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

from clip_classifier import ClipClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--preds", required=True, help="folder with Grounding DINO predictions")
parser.add_argument("--out", required=True, help="folder for the re-classified predictions")
parser.add_argument("--images", default="data/taco_yolo/test.txt", help="same image list used for the predictions")
parser.add_argument("--config", default="configs/classes.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

classifier = ClipClassifier(cfg["clip_model"], [c["clip"] for c in cfg["classes"]], cfg["clip_other"])

list_dir = os.path.dirname(args.images)
with open(args.images) as f :
    image_paths = [os.path.join(list_dir, line.strip()) for line in f if line.strip()]

os.makedirs(args.out, exist_ok=True)
for path in tqdm(image_paths):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(args.preds, stem + ".txt")) as f:
        rows = [[float(v) for v in line.split()] for line in f if line.strip()]

    lines = []
    if rows:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        W, H = image.size
        # YOLO format (normalised cx, cy, w, h) -> pixel corners x1, y1, x2, y2
        boxes = torch.tensor([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
                              for _, cx, cy, w, h, _ in rows])
        clip_ids, clip_probs = classifier.classify(image, boxes, cfg["clip_pad"])

        for row, cls, prob in zip(rows, clip_ids.tolist(), clip_probs.tolist()):
            if cls < 0:  # CLIP says "other": not litter we care about
                continue
            _, cx, cy, w, h, score = row
            # New class from CLIP; new score: both models must be confident
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score * prob:.4f}")

    with open(os.path.join(args.out, stem + ".txt"), "w") as f:
        f.write("\n".join(lines))