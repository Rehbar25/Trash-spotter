"""Compare PyTorch vs TensorRT FP16 vs TensorRT INT8: accuracy on the test set and speed."""
import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", nargs="+", required=True, help="model files to compare (.pt / .engine)")
parser.add_argument("--data", default="data/taco_yolo/dataset.yaml")
parser.add_argument("--image", default="data/raw/img5.jpg", help="image used for the speed test")
parser.add_argument("--runs", type=int, default=300, help="timed predictions per model")
parser.add_argument("--out", default="runs/benchmark")
args = parser.parse_args()

image = cv2.imread(args.image)
rows = []
for w in args.weights:
    model = YOLO(w, task="detect")
    name = os.path.basename(w)

    # Accuracy: same test set and settings as before. batch=1 because the engines were built for batch 1
    r = model.val(data=args.data, batch=1, imgsz=640, half=w.endswith(".pt"), plots=False, verbose=False,
                  project=os.path.abspath(args.out), name=name, exist_ok=True)

    # Speed: warm up first (GPU clocks, memory allocation), then time many single-image predictions
    for _ in range(20):
        model.predict(image, imgsz=640, verbose=False)
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        model.predict(image, imgsz=640, verbose=False)
        times.append(time.perf_counter() - t)
    ms = np.median(times) * 1000  # median: robust to the odd slow run

    size = os.path.getsize(w) / 1e6
    rows.append((name, size, r.box.map50, r.box.map, ms, 1000 / ms))

print(f"\n{'model':22s} {'size MB':>8s} {'mAP50':>7s} {'mAP50-95':>9s} {'ms/img':>7s} {'FPS':>6s}")
for name, size, m50, m, ms, fps in rows:
    print(f"{name:22s} {size:8.1f} {m50:7.3f} {m:9.3f} {ms:7.2f} {fps:6.0f}")
print("\nms/img = full predict() (resize + model + postprocess), median of", args.runs, "runs")
