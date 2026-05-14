"""
video_detect.py
Fall detection on a video file using new MediaPipe API (0.10.x)
Usage: python detection/video_detect.py path/to/video.mp4
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
from utils.pose_utils import extract_features

MODEL_PATH   = "model/fall_model.joblib"
POSE_MODEL   = "pose_landmarker.task"
CONFIRM_SECS = 3


def main():
    if len(sys.argv) < 2:
        print("Usage: python detection/video_detect.py <video_path>")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return

    clf = joblib.load(MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    confirm_frames = int(CONFIRM_SECS * fps)
    fall_frame_count = 0
    alert_active = False
    frame_num = 0

    print(f"Processing: {video_path} | Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Process every 2nd frame for speed on slow machines
        if frame_num % 2 != 0:
            continue

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        prediction = 0
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            features   = extract_features(result.pose_landmarks[0])
            prediction = clf.predict([features])[0]

        if prediction == 1:
            fall_frame_count += 1
            if fall_frame_count >= confirm_frames:
                alert_active = True
            else:
                remaining = (confirm_frames - fall_frame_count) / fps
                cv2.putText(frame, f"Fall? Confirming... {remaining:.1f}s",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            fall_frame_count = 0
            alert_active     = False

        if alert_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "!! FALL DETECTED - EMERGENCY !!",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        else:
            status = "Fall" if prediction == 1 else "Normal"
            color  = (0, 0, 255) if prediction == 1 else (0, 200, 0)
            cv2.putText(frame, f"Status: {status}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Fall Detection - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()