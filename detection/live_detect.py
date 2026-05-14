"""
live_detect.py
Live fall detection with velocity-based fall check.
- Tracks how fast body keypoints move between frames
- Sudden rapid movement = likely fall
- Works for forward, backward, sideways falls
Usage: python detection/live_detect.py
Press Q to quit.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import joblib
import time
import threading
import numpy as np
from collections import deque
from utils.pose_utils import extract_features

MODEL_PATH   = "model/fall_model.joblib"
POSE_MODEL   = "pose_landmarker.task"
CONFIRM_SECS = 3
CAMERA_INDEX = 1

REQUIRED_LANDMARKS   = [23, 24, 25, 26, 27, 28]
VISIBILITY_THRESHOLD = 0.35

# Velocity thresholds
# Tracks average movement of key landmarks between frames
# Normal movement: slow, controlled
# Fall: sudden spike in movement
VELOCITY_WINDOW    = 5     # frames to average velocity over
VELOCITY_THRESHOLD = 0.03  # normalized units per frame; tune if needed

# Key landmarks to track for velocity (hips + shoulders + nose)
VELOCITY_LANDMARKS = [0, 11, 12, 23, 24]

SKELETON_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def is_full_body_visible(landmarks):
    for idx in REQUIRED_LANDMARKS:
        if landmarks[idx].visibility < VISIBILITY_THRESHOLD:
            return False
    return True


def get_key_positions(landmarks):
    """Get (x, y) of key landmarks as numpy array."""
    return np.array([[landmarks[i].x, landmarks[i].y]
                     for i in VELOCITY_LANDMARKS])


def compute_velocity(pos_history):
    """Average movement magnitude over recent frames."""
    if len(pos_history) < 2:
        return 0.0
    diffs = []
    positions = list(pos_history)
    for i in range(1, len(positions)):
        diff = np.linalg.norm(positions[i] - positions[i-1], axis=1).mean()
        diffs.append(diff)
    return np.mean(diffs)


def draw_skeleton(frame, landmarks, w, h):
    for a, b in SKELETON_CONNECTIONS:
        ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
        bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)


def play_beep():
    try:
        import sounddevice as sd
        sr = 44100
        t  = np.linspace(0, 10.0, int(sr * 10.0), False)
        w  = 0.5 * np.sin(2 * np.pi * 1000 * t)
        sd.play(w.astype(np.float32), sr)
        sd.wait()
    except Exception as e:
        print(f"[Sound error] {e}")


def main():
    clf = joblib.load(MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. Try changing CAMERA_INDEX.")
        return

    fall_start    = None
    alert_active  = False
    sound_playing = False
    pos_history   = deque(maxlen=VELOCITY_WINDOW)

    # Track whether a sudden movement was recently detected
    sudden_move_detected = False
    sudden_move_time     = 0

    print("Live detection running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        prediction   = 0
        body_visible = False
        velocity     = 0.0

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame, landmarks, w, h)

            if is_full_body_visible(landmarks):
                body_visible = True

                # Update velocity tracking
                key_pos = get_key_positions(landmarks)
                pos_history.append(key_pos)
                velocity = compute_velocity(pos_history)

                # Detect sudden movement spike
                now = time.time()
                if velocity > VELOCITY_THRESHOLD:
                    sudden_move_detected = True
                    sudden_move_time     = now

                # Sudden move expires after 2 seconds
                if now - sudden_move_time > 2.0:
                    sudden_move_detected = False

                # Get model prediction
                features   = extract_features(landmarks)
                model_pred = clf.predict([features])[0]

                # Fall = model says fall AND there was a recent sudden movement
                prediction = 1 if (model_pred == 1 and sudden_move_detected) else 0

        now = time.time()

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            cv2.putText(frame, "No person detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
            fall_start    = None
            alert_active  = False
            sound_playing = False
            pos_history.clear()
            sudden_move_detected = False

        elif not body_visible:
            cv2.putText(frame, "Unable to detect - show full body",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            fall_start    = None
            alert_active  = False
            sound_playing = False

        elif prediction == 1:
            if fall_start is None:
                fall_start = now
            elapsed = now - fall_start
            if elapsed >= CONFIRM_SECS:
                alert_active = True
            else:
                remaining = CONFIRM_SECS - elapsed
                cv2.putText(frame, f"Fall? Confirming... {remaining:.1f}s",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            fall_start    = None
            alert_active  = False
            sound_playing = False
            cv2.putText(frame, "Status: Normal",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

        # Show velocity on screen for tuning
        vel_color = (0, 0, 255) if velocity > VELOCITY_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Velocity: {velocity:.4f}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 1)

        if alert_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "!! FALL DETECTED - EMERGENCY !!",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            if not sound_playing:
                threading.Thread(target=play_beep, daemon=True).start()
                sound_playing = True

        cv2.imshow("Fall Detection - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()