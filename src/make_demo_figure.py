"""Run the detector on a few images and tile the results into one picture for the README."""
import argparse
import os

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="runs/train/auto_v2_26s/weights/best.pt")
parser.add_argument("--images", nargs="+", required=True)
parser.add_argument("--out", default="docs/demo.jpg")
parser.add_argument("--conf", type=float, default=0.3)
parser.add_argument("--tile", type=int, nargs=2, default=[480, 360], help="tile width and height in pixels")
parser.add_argument("--cols", type=int, default=2)
args = parser.parse_args()

model = YOLO(args.weights)
tiles = []
for path in args.images:
    image = cv2.imread(path)
    result = model.predict(image, conf=args.conf, verbose=False)[0]
    # Draw boxes with line width / font scaled to the photo size, so labels stay readable after shrinking
    s = max(image.shape[:2])
    plotted = result.plot(line_width=max(2, s // 250), font_size=max(12, s // 40))
    # Scale so the tile is fully covered, then cut the center (all tiles get the same size)
    tw, th = args.tile
    h, w = plotted.shape[:2]
    k = max(tw / w, th / h)
    small = cv2.resize(plotted, (round(w * k), round(h * k)))
    y0, x0 = (small.shape[0] - th) // 2, (small.shape[1] - tw) // 2
    tiles.append(small[y0:y0 + th, x0:x0 + tw])

assert len(tiles) % args.cols == 0, "number of images must be a multiple of --cols"
rows = [cv2.hconcat(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
cv2.imwrite(args.out, cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("saved", args.out)
