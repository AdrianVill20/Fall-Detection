"""
extract_keypoints.py
Uses new MediaPipe API (0.10.x compatible with Python 3.13)
Usage: python preprocessing/extract_keypoints.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os
import urllib.request

MODEL_PATH   = "pose_landmarker.task"
DATA_DIR     = "data/raw_videos"
OUTPUT_CSV   = "data/keypoints.csv"
SAMPLE_EVERY = 3
MODEL_URL    = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose model (~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def extract_features(landmarks):
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    def pt(idx):
        return (landmarks[idx].x, landmarks[idx].y)

    left_knee_angle  = compute_angle(pt(23), pt(25), pt(27))
    right_knee_angle = compute_angle(pt(24), pt(26), pt(28))
    left_hip_angle   = compute_angle(pt(11), pt(23), pt(25))
    right_hip_angle  = compute_angle(pt(12), pt(24), pt(26))

    nose_y  = landmarks[0].y
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    height_ratio = abs(nose_y - ankle_y) if abs(nose_y - ankle_y) > 1e-6 else 1.0

    return coords + [left_knee_angle, right_knee_angle,
                     left_hip_angle, right_hip_angle, height_ratio]


def process_videos(label_name, label_value, rows, detector):
    folder = os.path.join(DATA_DIR, label_name)
    if not os.path.exists(folder):
        print(f"[WARN] Folder not found: {folder}")
        return

    videos = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".avi", ".mp4", ".mov"))
    ])

    if not videos:
        print(f"[WARN] No videos found in {folder}")
        return

    for fname in videos:
        path = os.path.join(folder, fname)
        cap = cv2.VideoCapture(path)
        frame_count = 0
        extracted = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % SAMPLE_EVERY != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                features = extract_features(result.pose_landmarks[0])
                rows.append(features + [label_value])
                extracted += 1

        cap.release()
        print(f"  [{label_name}] {fname}: {extracted} samples")


def main():
    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    rows = []
    print("Processing FALL videos...")
    process_videos("fall", 1, rows, detector)
    print("\nProcessing NO_FALL (ADL) videos...")
    process_videos("no_fall", 0, rows, detector)

    detector.close()

    if not rows:
        print("\n[ERROR] No data extracted.")
        print("Check: data/raw_videos/fall/*.avi and data/raw_videos/no_fall/*.avi")
        return

    col_names = []
    for i in range(33):
        col_names += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
    col_names += ["left_knee_angle", "right_knee_angle",
                  "left_hip_angle", "right_hip_angle", "height_ratio", "label"]

    df = pd.DataFrame(rows, columns=col_names)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! {len(df)} total samples -> {OUTPUT_CSV}")
    print(f"  Fall:    {len(df[df.label==1])}")
    print(f"  No-fall: {len(df[df.label==0])}")


if __name__ == "__main__":
    main()
