"""Split the converted TACO dataset into fixed train and test sets."""
import os
import random
import yaml

DATA_DIR = "data/taco_yolo"
CONFIG_PATH = "configs/classes.yaml"
TEST_FRACTION = 1 / 3
SEED = 42 #fixed seed = the same split every time the script runs

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# sort first: os.listdir order can differ between machines, and the shuffle must start from the same order
images = sorted(os.listdir(os.path.join(DATA_DIR, "images")))
random.Random(SEED).shuffle(images)
n_test = round(len(images)*TEST_FRACTION)
splits = {"test": images[:n_test], "train": images[n_test:]}

# Ultralytics accepts a text file listing image paths; "./" means relative to that file
for name, files in splits.items():
    with open(os.path.join(DATA_DIR, f"{name}.txt"), "w") as f:
        f.write("\n".join(f"./images/{x}" for x in files) + "\n")
    print(f"{name}: {len(files)} images")

# Dataset file for Ultralytics, with class names taken from our config (single source of truth)
dataset = {
    "path": os.path.abspath(DATA_DIR),
    "train": "train.txt",
    "val": "test.txt",
    "names": {i: c["name"] for i, c in enumerate(cfg["classes"])},
}
with open (os.path.join(DATA_DIR, "dataset.yaml"), "w") as f:
    yaml.safe_dump(dataset, f, sort_keys=False)
print(f"Wrote {DATA_DIR}/dataset.yaml")