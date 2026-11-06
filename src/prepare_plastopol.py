"""PlastOPol -> resized images (max 1600 px) + one-class YOLO labels + official train/test lists.
Download first (CC BY 4.0): https://zenodo.org/records/5829156 -> unzip so data/plastopol/ has
images.zip and annotations/annotations/*.json. Images are read straight from images.zip (saves 7.7 GB)."""
import io
import json
import os
import zipfile

from PIL import Image, ImageOps

SRC = "data/plastopol"
OUT = "data/plastopol_yolo"
MAX_SIDE = 1600

os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
zf = zipfile.ZipFile(f"{SRC}/images.zip")

for split, ann_file in [("train", "plastopol_train_val.json"), ("test", "plastopol_test.json")]:
    with open(f"{SRC}/annotations/annotations/{ann_file}") as f:
        d = json.load(f)
    boxes = {}
    for a in d["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a["bbox"])  # COCO: x, y, w, h in pixels

    listed = []
    for im in d["images"]:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(zf.read("images/" + im["file_name"])))).convert("RGB")
        assert img.size == (im["width"], im["height"]), f"size mismatch: {im['file_name']}"
        W, H = img.size
        stem = "plastopol_" + os.path.splitext(im["file_name"])[0]

        # One class ("litter"): these human labels are only used for the out-of-domain TEST, never for training
        lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                 for x, y, w, h in boxes.get(im["id"], [])]
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(f"{OUT}/images/{stem}.jpg", quality=90)
        with open(f"{OUT}/labels/{stem}.txt", "w") as f:
            f.write("\n".join(lines))
        listed.append(f"./images/{stem}.jpg")

    with open(f"{OUT}/{split}.txt", "w") as f:
        f.write("\n".join(listed) + "\n")
    print(f"{split}: {len(listed)} images")
