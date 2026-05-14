# Scope and Limitations — Fall Detection System

## Scope

The Fall Detection System is designed to identify when a person falls to the ground using computer vision and machine learning. Its scope covers the following:

### Functional Scope

1. **Fall Detection via Pose Estimation**
   - Uses MediaPipe Pose Landmarker to detect 33 body landmarks from a video feed (webcam or recorded video)
   - Extracts 104 numerical features per frame: 99 raw landmark coordinates (x, y, z) + 4 joint angles (left/right knee, left/right hip) + 1 height ratio
   - Classifies each frame as "fall" or "no fall" using a Random Forest classifier

2. **Dual-Confirmation Mechanism**
   - Combines ML model prediction with a geometric rule-based check (height ratio + hip-shoulder difference)
   - Both must agree "fall" before an alert is considered, reducing false positives

3. **Confirmation Timer**
   - Requires the fall state to persist for a configurable duration (default 3 seconds) before triggering the alarm
   - Prevents brief false alarms from quick bending or crouching

4. **Real-Time Alerting**
   - Displays a red overlay and banner on the video feed
   - Plays an audible alarm (1 kHz tone) via the system speaker
   - Logs each confirmed fall event with a timestamp

5. **User Interface**
   - Desktop GUI built with Tkinter
   - Buttons: Live Camera, Open Video File, Stop
   - Configurable settings: camera index, confirmation duration
   - Live statistics: frames processed, falls detected, current mode
   - Skeleton overlay drawn on detected persons

6. **Offline Video Processing**
   - Can process pre-recorded video files (.mp4, .avi, .mov)
   - Makes detection accessible when live camera is unavailable

7. **Dataset Pipeline**
   - Extract keypoints from raw video footage for training
   - Support for multiple dataset folder structures
   - Data augmentation (noisy copies) to handle class imbalance during training

8. **Velocity-Based Detection** (standalone live_detect.py only)
   - Tracks landmark movement speed between frames
   - Flags sudden rapid movements as potential falls
   - Reduces false alarms from the ML model alone

### Intended Use Case

- Indoor monitoring with a fixed camera (e.g., bedroom, living room, nursing home room)
- Single-person scenes with the full body visible in frame
- Side-angle camera placement (not top-down or directly front-facing)
- Elderly care / assisted living environments where falls need immediate notification

---

## Limitations

### 1. Forward Falls Are Not Detected (Geometry Check Issue)

**File:** `app.py`, lines 65–73 and 348

The system requires **both** the ML model and the geometric rule to agree before triggering a fall alert:

```python
prediction = 1 if (model_pred == 1 and geo_pred) else 0
```

The geometry check (`is_fallen_by_geometry`) considers a person fallen only when:
- `height_ratio` (nose-to-ankle vertical distance) < 0.55
- `hip_shoulder_diff` (hip-to-shoulder vertical distance) < 0.30

When a person falls **forward** (face-down or face-first):
- The nose-to-ankle vertical distance remains **large** — the body is stretched out vertically from the camera's perspective
- The hip-to-shoulder vertical separation may also remain **above 0.30**
- The geometry check returns `False`, overriding the ML model's prediction

**Result:** The person is detected as standing upright or in a normal state, even though a fall has occurred.

### 2. Backward Falls May Also Be Missed

Similarly, falls where the person falls backward (lying on their back) present:
- Large vertical spread between head and feet
- Significant hip-shoulder vertical separation
- The same geometry check failure as forward falls

### 3. Camera Angle Dependency

The system assumes a **side-angle** camera view. If the camera is:
- **Top-down** (ceiling-mounted): The height ratio and hip-shoulder difference become unreliable since the body appears compressed along both axes
- **Front-facing** (directly in front): Forward falls appear as a person approaching the ground, but the geometry check may not trigger until the person is fully prone
- **Too far or too close**: Landmark visibility decreases, causing the system to skip detection

### 4. Full-Body Visibility Requirement

**File:** `app.py`, lines 29, 58–62

The system requires landmarks 23–28 (both hips, both knees, both ankles) to have visibility > 0.5. This means:
- Person must be fully in frame from head to toe
- Person cannot be behind furniture, blankets, or other occluding objects
- Person cannot be lying on a bed or sofa with body parts hidden
- Low-light conditions may reduce landmark confidence below the threshold

### 5. Single-Person Detection

- MediaPipe detects multiple people but the system only processes the **first detected person** (`result.pose_landmarks[0]`)
- In a room with multiple people, the system may track the wrong person or get confused

### 6. No Distinction Between Intentional and Unintentional Falls

- Sitting down quickly, lying on the floor intentionally, or crouching may be misclassified as falls
- The confirmation timer helps but does not differentiate intent

### 7. Frame Processing Lag

- System skips every other frame for performance (line 325)
- On slower machines, the effective frame rate may drop further
- Fast falls between processed frames could be missed

### 8. Training Data Limitations

- The Random Forest model is only as good as the training videos
- The dataset contains 60 fall videos (30 from 2 camera angles) and 40 ADL (non-fall) videos — this is relatively small
- Falls in the dataset may not represent all real-world fall types (e.g., trip-and-fall, slip-and-fall, fainting)
- Different body types, clothing, lighting conditions, and room layouts are not exhaustively covered

### 9. No Post-Fall Status Detection

- Once a fall is detected and the alarm sounds, the system does not check whether:
  - The person has gotten back up
  - The person remains on the ground (in need of help)
  - Another person has arrived to assist
- The alert persists only while the person is in a "fallen" pose; if they stand up, the alert stops

### 10. Sound Alert Limitations

- The alarm plays through the system's default speaker using `sounddevice`
- If the system has no speakers, is muted, or volume is low, the alert may not be heard
- No integration with SMS, email, push notifications, or calling services

### 11. No Privacy Protections

- Video frames are not encrypted or anonymized
- The system does not delete or restrict access to recorded data
- Designed for local processing only, but no security measures are implemented

---

## Future Improvements (Suggested)

- **Remove or relax the geometry check** for forward/backward falls, or use the ML model alone
- **Train on forward-fall videos** to improve ML model recognition
- **Add camera calibration** to handle different viewing angles
- **Multi-person tracking** to follow the right person
- **Post-fall monitoring** to detect if a person remains on the ground
- **Push notifications** via email, SMS, or messaging apps
- **Edge device deployment** (Raspberry Pi, Jetson Nano) for standalone operation
