import torch
import os
import sys
import yaml

from PIL import Image, ImageDraw, ImageFont, ImageOps
from detector import GroundingDinoDetector

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CONFIG_PATH = "configs/classes.yaml"
IMAGE_PATH = sys.argv[1]
OUTPUT_PATH = os.path.join("runs", "gdino_" + os.path.basename(IMAGE_PATH))

# building the prompt from the config instead of hardcoding it.
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

class_phrases = [p for c in cfg["classes"] for p in c["prompts"]]
phrases = class_phrases + cfg["catch_prompts"]

detector = GroundingDinoDetector(MODEL_ID, phrases)
print(f"Prompt: {detector.prompt}")

# exif_transpose applies the phone's rotation flag so the image is upright


image = ImageOps.exif_transpose(Image.open(IMAGE_PATH)).convert("RGB")
boxes, scores, phrase_ids = detector.detect(image, cfg["box_threshold"], cfg["nms_iou"])

font = ImageFont.load_default(size=image.width // 40)
draw = ImageDraw.Draw(image)

for box, score, pid in zip(boxes, scores, phrase_ids):
    phrase = phrases[pid]
    is_catch = pid >= len(class_phrases) # catch prompts come after the class prompts 
    color = "gray" if is_catch else "red"
    x1, y1, x2, y2 = box.tolist()
    draw.rectangle([x1, y1, x2, y2], outline = color, width=image.width // 300)
    draw.text((x1, y1 - font.size - 5), f"{phrase} {score:.2f}", fill=color, font=font)
    tag = " (catch, discarded)" if is_catch else ""
    print(f"{phrase:15s} {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]{tag}")

image.save(OUTPUT_PATH)
print(f"\n{len(boxes)} detections, saved to {OUTPUT_PATH}")
print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")