"""
pose_utils.py
Shared helpers using new MediaPipe API (0.10.x)
"""

import numpy as np


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
