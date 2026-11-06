"""Convert TACO (COCO format, 60 categories) into a YOLO dataset with our classes."""
import json
import os
from collections import defaultdict

import yaml 
from PIL import Image, ImageOps
from tqdm import tqdm

TACO_DIR = "data/taco/data"
OUT_DIR = "data/taco_yolo"
CONFIG_PATH = "configs/classes.yaml"
MAX_SIDE = 1600 # shrink images so the longest side is at most this

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
with open(os.path.join(TACO_DIR, "annotations.json")) as f:
    taco = json.load(f)

# TACO category name -> our class id
name_to_class = {}
for class_id, c in enumerate(cfg["classes"]):
    for taco_name in c["taco"]:
        name_to_class[taco_name] = class_id

# Catch typos in the config:every name must be a real TACO category
taco_names = {cat["name"] for cat in taco["categories"]}
unknown = set(name_to_class) - taco_names
assert not unknown, f"Not TACO categories (check spelling): {unknown}"

# TACO category id -> our class id, only for categories we use
taco_id_to_class = {cat["id"]: name_to_class[cat["name"]] for cat in taco["categories"] if cat["name"] in name_to_class}

# Group the annotations we keep by image
anns_by_image = defaultdict(list)
for ann in taco["annotations"]:
    if ann["category_id"] in taco_id_to_class:
        anns_by_image[ann["image_id"]].append(ann)

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)
counts = [0]*len(cfg["classes"])
images_with_objects = 0

for im in tqdm(taco["images"]):
    # "batch_1/000006.jpg" -> "batch_1_000006" (batches reuse file names)
    stem = os.path.splitext(im["file_name"].replace("/", "_"))[0]
    W, H = im["width"], im["height"]

    # Annotations were made on the upright image, so apply the EXIF rotation
    image = ImageOps.exif_transpose(Image.open(os.path.join(TACO_DIR, im["file_name"]))).convert("RGB")
    assert image.size == (W, H), f"Size mismatch for {im['file_name']}"
    image.thumbnail((MAX_SIDE, MAX_SIDE)) #keeps aspect ratio only ever shrinks
    image.save(os.path.join(OUT_DIR, "images", stem + ".jpg"), quality=90)

    lines = []
    for ann in anns_by_image.get(im["id"], []):
        x,y,w,h = ann["bbox"] #COCO: top-left x, top-left y, width, height (pixels)
        cls = taco_id_to_class[ann["category_id"]]
        # YOLO: class center_x center_y width height, all as fractions of the image size
        lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y+h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        counts[cls] += 1
    if lines:
        images_with_objects += 1

        # Images with none of our classes get an empty file: YOLO treats them as background
    with open(os.path.join(OUT_DIR, "labels", stem + ".txt"), "w") as f:
        f.write("\n".join(lines))

print("\nObjects per class:")
for c, n in zip(cfg["classes"], counts):
    print(f" {c['name']:12s} {n}")
print(f"Images with at least one object: {images_with_objects} / {len(taco['images'])}")

