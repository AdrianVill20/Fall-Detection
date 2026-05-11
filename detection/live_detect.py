"""
live_detect.py
Live fall detection with:
- Skeleton dot overlay
- Partial body detection check
- Rule-based geometry check on top of model prediction
- Sound alert
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
from utils.pose_utils import extract_features

MODEL_PATH   = "model/fall_model.joblib"
POSE_MODEL   = "pose_landmarker.task"
CONFIRM_SECS = 3
CAMERA_INDEX = 1

REQUIRED_LANDMARKS   = [23, 24, 25, 26, 27, 28]
VISIBILITY_THRESHOLD = 0.5

# Fall geometry thresholds
# height_ratio = |nose_y - ankle_y|
# When standing: nose is far above ankles → large ratio (e.g. 0.6-0.9)
# When fallen:   nose is near ankles → small ratio (e.g. 0.0-0.3)
FALL_HEIGHT_RATIO_MAX = 0.35   # must be below this to count as fall

# hip_shoulder_diff = |hip_y - shoulder_y|
# When standing: hips are well below shoulders → large value
# When fallen:   hips and shoulders at similar height → small value
FALL_HIP_SHOULDER_MAX = 0.20   # must be below this to count as fall

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


def is_fallen_by_geometry(landmarks):
    """
    Returns True only if body geometry strongly suggests a fall.
    This runs ON TOP of the model to reduce false positives.
    """
    nose_y      = landmarks[0].y
    ankle_y     = (landmarks[27].y + landmarks[28].y) / 2
    hip_y       = (landmarks[23].y + landmarks[24].y) / 2
    shoulder_y  = (landmarks[11].y + landmarks[12].y) / 2

    height_ratio      = abs(nose_y - ankle_y)
    hip_shoulder_diff = abs(hip_y - shoulder_y)

    fallen = (height_ratio < FALL_HEIGHT_RATIO_MAX and
              hip_shoulder_diff < FALL_HIP_SHOULDER_MAX)
    return fallen


def draw_skeleton(frame, landmarks, w, h):
    for a, b in SKELETON_CONNECTIONS:
        ax = int(landmarks[a].x * w)
        ay = int(landmarks[a].y * h)
        bx = int(landmarks[b].x * w)
        by = int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 2)
    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)


def play_beep():
    try:
        import sounddevice as sd
        duration = 3.0
        freq = 1000
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        sd.play(wave.astype(np.float32), sample_rate)
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
    print("Live detection running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        prediction    = 0
        body_visible  = False

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame, landmarks, w, h)

            if is_full_body_visible(landmarks):
                body_visible = True
                features     = extract_features(landmarks)
                model_pred   = clf.predict([features])[0]
                geo_pred     = is_fallen_by_geometry(landmarks)

                # BOTH model AND geometry must agree it's a fall
                prediction = 1 if (model_pred == 1 and geo_pred) else 0
            else:
                body_visible = False

        now = time.time()

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            cv2.putText(frame, "No person detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
            fall_start    = None
            alert_active  = False
            sound_playing = False

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