"""Resize the unlabeled TACO 'unofficial' images to max 1600 px. They become the pool that Grounding DINO auto-labels."""
import argparse 
import os

from PIL import Image, ImageOps
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/taco/data/unofficial")
parser.add_argument("--out", default="data/unlabeled/images")
parser.add_argument("--max-side", type=int, default=1600)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
files = sorted(f for f in os.listdir(args.src) if f.lower().endswith((".jpg", "jpeg", ".png")))

broken = []
for name in tqdm(files):
    out_path = os.path.join(args.out, "unofficial_" + os.path.splitext(name)[0] + ".jpg")
    if os.path.exists(out_path): #already done (lets you stop and resume)
        continue
    try:
        image = ImageOps.exif_transpose(Image.open(os.path.join(args.src, name))).convert("RGB")
    except Exception as e: #half-downloaded or corrupt file
        broken.append((name, str(e)))
        continue
    image.thumbnail((args.max_side, args.max_side)) #keeps aspect ratio, only shrinks
    image.save(out_path, quality=90)

print(f"{len(files) - len(broken)} images saved to {args.out}, {len(broken)} broken")

for name, err in broken:
    print("  broken:", name, err)