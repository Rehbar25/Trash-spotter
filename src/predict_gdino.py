"""Run Grounding DINO on a list of images and save predictions in YOLO format, plus a confidence coloumn."""
import argparse
import os

import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

from detector import GroundingDinoDetector

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="data/taco_yolo/test.txt", help="text file listing image paths")
parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
parser.add_argument("--out", required=True, help="folder for the prediction .txt files")
parser.add_argument("--config", default="configs/classes.yaml")
parser.add_argument("--min-score", type=float, default=0.1, help="keep low scores so thresholds can be tuned later")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# Every prompt phrase, and the class id it belongs to (None = catch prompt, discarded)
phrases, phrase_class = [], []
for class_id, c in enumerate(cfg["classes"]):
    for p in c["prompts"]:
        phrases.append(p)
        phrase_class.append(class_id)
for p in cfg["catch_prompts"]:
    phrases.append(p)
    phrase_class.append(None)

detector = GroundingDinoDetector(args.model, phrases)

# Paths in the list are relative to the list file ("./images/x.jpg")
list_dir = os.path.dirname(args.images)
with open(args.images) as f:
    image_paths = [os.path.join(list_dir, line.strip()) for line in f if line.strip()]

os.makedirs(args.out, exist_ok=True)
for path in tqdm(image_paths):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    W, H = image.size
    boxes, scores, phrases_ids = detector.detect(image, args.min_score, cfg["nms_iou"])

    lines = []
    for (x1, y1, x2, y2), score, pid in zip(boxes.tolist(), scores.tolist(), phrases_ids.tolist()):
        cls = phrase_class[pid]
        if cls is None: #catch prompt
            continue
        # Same format as the YOLO label files with the confidence added at the end
        lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f} {score:.4f}")

    stem = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(args.out, stem + ".txt"), "w") as f:
        f.write("\n".join(lines))