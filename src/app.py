"""TrashSpotter demo: detect litter in an image, or track and count litter in a video."""
import argparse
import os
import tempfile
from collections import Counter, defaultdict

import cv2
import gradio as gr
import imageio
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="runs/train/auto_26s/weights/best.pt")
parser.add_argument("--device", default="0", help="GPU id, or 'cpu'")
parser.add_argument("--max-seconds", type=int, default=30, help="only process the start of long videos")
parser.add_argument("--min-frames", type=int, default=5, help="a track must be seen this often to be counted")
args = parser.parse_args()

model = YOLO(args.weights)
NAMES = model.names  # {0: 'bottle', 1: 'can', ...}, stored inside the weights


def count_table(counts):
    """{class_id: count} -> rows for the table, every class shown (also the zeros)."""
    return [[NAMES[i], counts.get(i, 0)] for i in NAMES]


def detect_image(image, conf):
    if image is None:
        return None, []
    # Gradio gives RGB, but Ultralytics expects numpy images in BGR (OpenCV order)
    result = model.predict(image[:, :, ::-1], conf=conf, device=args.device, verbose=False)[0]
    counts = defaultdict(int)
    for c in result.boxes.cls.int().tolist():
        counts[c] += 1
    annotated = result.plot()[:, :, ::-1]  # plot() returns BGR (OpenCV), Gradio wants RGB
    return annotated, count_table(counts)


def track_video(video_path, conf):
    if video_path is None:
        return None, []
    tracker_model = YOLO(args.weights)  # fresh model = fresh tracker, so IDs from the last video don't carry over
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * args.max_seconds)

    out_path = os.path.join(tempfile.mkdtemp(), "tracked.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=1)  # H.264 plays in browsers

    hits = defaultdict(int)          # track id -> number of frames it was seen in
    votes = defaultdict(Counter)     # track id -> how often each class was predicted for it
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        # persist=True: keep the tracker between calls, so the same object keeps the same ID
        result = tracker_model.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml",
                                     device=args.device, verbose=False)[0]
        if result.boxes.id is not None:
            for tid, c in zip(result.boxes.id.int().tolist(), result.boxes.cls.int().tolist()):
                hits[tid] += 1
                votes[tid][c] += 1
        writer.append_data(result.plot()[:, :, ::-1])  # BGR -> RGB
    cap.release()
    writer.close()

    # Each object is counted once. Short flickering tracks are ignored, and the class is the majority vote
    counts = defaultdict(int)
    for tid, n in hits.items():
        if n >= args.min_frames:
            counts[votes[tid].most_common(1)[0][0]] += 1
    return out_path, count_table(counts)


with gr.Blocks(title="TrashSpotter") as demo:
    gr.Markdown("# TrashSpotter\nLitter detector trained only on auto-labels (Grounding DINO + CLIP → YOLO26s).")
    with gr.Tab("Image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload a photo")
            img_out = gr.Image(label="Detections")
        img_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
        img_table = gr.Dataframe(headers=["class", "count"], label="Objects found")
        gr.Button("Detect").click(detect_image, [img_in, img_conf], [img_out, img_table])
    with gr.Tab("Video"):
        with gr.Row():
            vid_in = gr.Video(label=f"Upload a video (first {args.max_seconds} s are used)")
            vid_out = gr.Video(label="Tracked")
        vid_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
        vid_table = gr.Dataframe(headers=["class", "unique objects"], label="Litter counted (each object once)")
        gr.Button("Track and count").click(track_video, [vid_in, vid_conf], [vid_out, vid_table])

if __name__ == "__main__":
    demo.launch()
