"""Turn auto-label predictions into a YOLO training set: images (hardlinked) + label files + an image list."""
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--images", required=True, help="image list used for the predictions")
parser.add_argument("--preds", required=True, help="folder with the CLIP-reclassified predictions")
parser.add_argument("--name", required=True, help="name of the image list to write, e.g. pool")
parser.add_argument("--conf", type=float, default=0.2, help="only boxes at or above this confidence become labels")
parser.add_argument("--out", default="data/auto_yolo")
parser.add_argument("--skip-empty", action="store_true",
                    help="leave out images with no auto-labels (use when every image is known to contain litter)")
args = parser.parse_args()

os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

list_dir = os.path.dirname(args.images)
with open(args.images) as f:
    image_paths = [os.path.join(list_dir, line.strip()) for line in f if line.strip()]

n_boxes, n_empty, listed = 0, 0, []
for path in image_paths:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]

    # Image: a hardlink is a second name for the same file, so it takes no extra disk space
    dst = os.path.join(args.out, "images", name)
    if not os.path.exists(dst):
        try:
            os.link(path, dst)
        except OSError:  # different disk: hardlinks are impossible, copy instead
            shutil.copy(path, dst)

    # Label: keep confident boxes, drop the confidence column -> plain YOLO format
    with open(os.path.join(args.preds, stem + ".txt")) as f:
        rows = [line.split() for line in f if line.strip()]
    labels = [" ".join(r[:5]) for r in rows if float(r[5]) >= args.conf]
    if args.skip_empty and not labels:
        continue  # in a litter-only dataset, "teacher found nothing" is a miss, not a true background image

    with open(os.path.join(args.out, "labels", stem + ".txt"), "w") as f:
        f.write("\n".join(labels))

    n_boxes += len(labels)
    n_empty += len(labels) == 0
    listed.append(f"./images/{name}")

with open(os.path.join(args.out, args.name + ".txt"), "w") as f:
    f.write("\n".join(listed) + "\n")

print(f"{args.name}: {len(listed)} images, {n_boxes} boxes, {n_empty} images without boxes (kept as background)")
