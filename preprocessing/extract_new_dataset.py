"""
extract_new_dataset.py
Extracts keypoints from the new video dataset and merges with existing keypoints.csv

New dataset structure:
  D:/VideoDataset/
    Fall/
      SBJ_01_LOC1/
        ACT1_L_1/
          *.mp4
    ADL/
      SBJ_01_LOC3/
        ACT13_B_1/
          *.mp4

Usage: python preprocessing/extract_new_dataset.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pose_utils import extract_features

# ── Config ────────────────────────────────────────────────────────────────────
NEW_FALL_DIR  = r"D:\VideoDataset\Fall"
NEW_ADL_DIR   = r"D:\VideoDataset\ADL"
EXISTING_CSV  = "data/keypoints.csv"
OUTPUT_CSV    = "data/keypoints.csv"
POSE_MODEL    = "pose_landmarker.task"
SAMPLE_EVERY  = 3


def get_mp4(folder):
    """Find the first mp4 in a folder."""
    for f in os.listdir(folder):
        if f.lower().endswith(".mp4"):
            return os.path.join(folder, f)
    return None


def process_dataset(root_dir, label_value, rows, detector):
    """Walk root_dir -> subject folders -> activity folders -> mp4."""
    if not os.path.exists(root_dir):
        print(f"[WARN] Not found: {root_dir}")
        return

    label_name = "fall" if label_value == 1 else "no_fall"
    total = 0

    for subject in sorted(os.listdir(root_dir)):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path) or subject.startswith("."):
            continue

        for activity in sorted(os.listdir(subj_path)):
            act_path = os.path.join(subj_path, activity)
            if not os.path.isdir(act_path):
                continue

            mp4 = get_mp4(act_path)
            if mp4 is None:
                continue

            cap = cv2.VideoCapture(mp4)
            frame_count = 0
            extracted   = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % SAMPLE_EVERY != 0:
                    continue

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = detector.detect(mp_image)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    feats = extract_features(result.pose_landmarks[0])
                    rows.append(feats + [label_value])
                    extracted += 1

            cap.release()
            total += extracted
            print(f"  [{label_name}] {subject}/{activity}: {extracted} samples")

    print(f"  Total new {label_name} samples: {total}\n")


def main():
    base     = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    opts     = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.PoseLandmarker.create_from_options(opts)

    new_rows = []
    print("Extracting FALL videos from new dataset...")
    process_dataset(NEW_FALL_DIR, 1, new_rows, detector)

    print("Extracting ADL (no_fall) videos from new dataset...")
    process_dataset(NEW_ADL_DIR, 0, new_rows, detector)

    detector.close()

    if not new_rows:
        print("[ERROR] No new samples extracted. Check your dataset paths.")
        return

    col_names = []
    for i in range(33):
        col_names += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
    col_names += ["left_knee_angle", "right_knee_angle",
                  "left_hip_angle", "right_hip_angle", "height_ratio", "label"]

    new_df = pd.DataFrame(new_rows, columns=col_names)

    if os.path.exists(EXISTING_CSV):
        print(f"Loading existing data from {EXISTING_CSV}...")
        old_df = pd.read_csv(EXISTING_CSV)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged.dropna(inplace=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Old samples: {len(old_df)}")
    else:
        merged = new_df

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMerged dataset saved to {OUTPUT_CSV}")
    print(f"  Total:   {len(merged)}")
    print(f"  Fall:    {len(merged[merged.label==1])}")
    print(f"  No-fall: {len(merged[merged.label==0])}")
    print("\nNow run: python model/train.py")


if __name__ == "__main__":
    main()
