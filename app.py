"""
app.py
Fall Detection System - Main UI
Usage: python app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sys
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import joblib
import time
import numpy as np
from PIL import Image, ImageTk

sys.path.append(os.path.dirname(__file__))
from utils.pose_utils import extract_features

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH   = "model/fall_model.joblib"
POSE_MODEL   = "pose_landmarker.task"
CONFIRM_SECS = 3

REQUIRED_LANDMARKS   = [23, 24, 25, 26, 27, 28]
VISIBILITY_THRESHOLD = 0.5

FALL_HEIGHT_RATIO_MAX = 0.55
FALL_HIP_SHOULDER_MAX = 0.30

SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

# ── Colors & Fonts ───────────────────────────────────────────────────────────
BG      = "#0a0e1a"
PANEL   = "#111827"
CARD    = "#1a2235"
ACCENT  = "#00d4ff"
ACCENT2 = "#ff3f6c"
SUCCESS = "#00e676"
WARNING = "#ffab00"
TEXT    = "#e8eaf6"
SUBTEXT = "#7986cb"
FONT_SUB  = ("Courier New", 10)
FONT_BTN  = ("Courier New", 11, "bold")
FONT_STAT = ("Courier New", 13, "bold")
FONT_LOG  = ("Courier New", 9)


# ── Helpers ──────────────────────────────────────────────────────────────────
def is_full_body_visible(landmarks):
    for idx in REQUIRED_LANDMARKS:
        if landmarks[idx].visibility < VISIBILITY_THRESHOLD:
            return False
    return True


def is_fallen_by_geometry(landmarks):
    nose_y     = landmarks[0].y
    ankle_y    = (landmarks[27].y + landmarks[28].y) / 2
    hip_y      = (landmarks[23].y + landmarks[24].y) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    height_ratio      = abs(nose_y - ankle_y)
    hip_shoulder_diff = abs(hip_y - shoulder_y)
    return (height_ratio < FALL_HEIGHT_RATIO_MAX and
            hip_shoulder_diff < FALL_HIP_SHOULDER_MAX)


def draw_skeleton(frame, landmarks, w, h):
    for a, b in SKELETON_CONNECTIONS:
        ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
        bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 212, 255), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 63, 108), -1)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)


def play_beep():
    try:
        import sounddevice as sd
        sr = 44100
        t  = np.linspace(0, 3.0, int(sr * 3.0), False)
        w  = 0.5 * np.sin(2 * np.pi * 1000 * t)
        sd.play(w.astype(np.float32), sr)
        sd.wait()
    except Exception:
        pass


# ── Main App ─────────────────────────────────────────────────────────────────
class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall Detection System")
        self.root.configure(bg=BG)
        self.root.geometry("1100x720")
        self.root.resizable(False, False)

        self.clf      = None
        self.detector = None
        self.cap      = None
        self.running  = False

        self.fall_start    = None
        self.alert_active  = False
        self.sound_playing = False
        self.fall_count    = 0
        self.frame_count   = 0

        self._load_model()
        self._build_ui()

    def _load_model(self):
        try:
            self.clf = joblib.load(MODEL_PATH)
            base = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
            opts = vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.PoseLandmarker.create_from_options(opts)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")

    def _build_ui(self):
        # ── Left panel ──
        left = tk.Frame(self.root, bg=PANEL, width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Title
        tf = tk.Frame(left, bg=PANEL)
        tf.pack(pady=(30,5), padx=20, fill="x")
        tk.Label(tf, text="⬡ FALL", font=("Courier New",26,"bold"),
                 bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(tf, text="DETECTION", font=("Courier New",26,"bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w")
        tk.Label(tf, text="SYSTEM v1.0", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w")

        tk.Frame(left, bg=ACCENT, height=2).pack(fill="x", padx=20, pady=15)

        # Status
        sc = tk.Frame(left, bg=CARD)
        sc.pack(fill="x", padx=20, pady=5)
        tk.Label(sc, text="SYSTEM STATUS", font=FONT_SUB,
                 bg=CARD, fg=SUBTEXT).pack(anchor="w", padx=15, pady=(12,2))
        self.status_label = tk.Label(sc, text="● IDLE",
                                     font=FONT_STAT, bg=CARD, fg=WARNING)
        self.status_label.pack(anchor="w", padx=15, pady=(0,12))

        # Stats
        stc = tk.Frame(left, bg=CARD)
        stc.pack(fill="x", padx=20, pady=5)
        tk.Label(stc, text="STATISTICS", font=FONT_SUB,
                 bg=CARD, fg=SUBTEXT).pack(anchor="w", padx=15, pady=(12,4))

        for label, attr, color, default in [
            ("Frames processed", "lbl_frames", TEXT,    "0"),
            ("Falls detected",   "lbl_falls",  ACCENT2, "0"),
            ("Mode",             "lbl_mode",   TEXT,    "—"),
        ]:
            row = tk.Frame(stc, bg=CARD)
            row.pack(fill="x", padx=15, pady=2)
            tk.Label(row, text=label, font=FONT_SUB, bg=CARD, fg=SUBTEXT).pack(side="left")
            lbl = tk.Label(row, text=default, font=FONT_SUB, bg=CARD, fg=color)
            lbl.pack(side="right")
            setattr(self, attr, lbl)
        tk.Frame(stc, bg=CARD, height=8).pack()

        tk.Frame(left, bg=CARD, height=1).pack(fill="x", padx=20, pady=10)

        # Buttons
        btn_cfg = dict(font=FONT_BTN, relief="flat", bd=0,
                       cursor="hand2", pady=12, padx=10)

        self.btn_live = tk.Button(left, text="▶  LIVE CAMERA",
                                  bg=ACCENT, fg=BG, activebackground="#00b8d9",
                                  command=self._start_live, **btn_cfg)
        self.btn_live.pack(fill="x", padx=20, pady=5)

        self.btn_video = tk.Button(left, text="📁  OPEN VIDEO FILE",
                                   bg=CARD, fg=ACCENT, activebackground="#1e2d45",
                                   command=self._start_video, **btn_cfg)
        self.btn_video.pack(fill="x", padx=20, pady=5)

        self.btn_stop = tk.Button(left, text="■  STOP",
                                  bg=ACCENT2, fg="white", activebackground="#cc2244",
                                  command=self._stop, state="disabled", **btn_cfg)
        self.btn_stop.pack(fill="x", padx=20, pady=5)

        # Settings
        for label, var_name, default, lo, hi, inc in [
            ("Camera Index:",    "cam_var",  1,   0, 5,  1),
            ("Confirm seconds:", "conf_var", 3.0, 1, 10, 0.5),
        ]:
            row = tk.Frame(left, bg=PANEL)
            row.pack(fill="x", padx=20, pady=3)
            tk.Label(row, text=label, font=FONT_SUB,
                     bg=PANEL, fg=SUBTEXT).pack(side="left")
            var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
            setattr(self, var_name, var)
            tk.Spinbox(row, from_=lo, to=hi, increment=inc, textvariable=var,
                       width=4, font=FONT_SUB, bg=CARD, fg=TEXT,
                       buttonbackground=CARD, relief="flat").pack(side="right")

        # Log
        tk.Frame(left, bg=ACCENT, height=1).pack(fill="x", padx=20, pady=12)
        tk.Label(left, text="EVENT LOG", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=20)
        self.log_box = tk.Text(left, height=8, bg=CARD, fg=SUBTEXT,
                               font=FONT_LOG, relief="flat", state="disabled",
                               wrap="word", bd=0)
        self.log_box.pack(fill="both", padx=20, pady=5, expand=True)

        # ── Right panel ──
        right = tk.Frame(self.root, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        feed_frame = tk.Frame(right, bg=BG)
        feed_frame.pack(fill="both", expand=True, padx=20, pady=(20,5))

        self.canvas = tk.Canvas(feed_frame, bg="#0d1117",
                                highlightthickness=1, highlightbackground=ACCENT)
        self.canvas.pack(fill="both", expand=True)

        self.placeholder = tk.Label(
            self.canvas,
            text="NO FEED\n\nPress LIVE CAMERA or\nOPEN VIDEO FILE to begin",
            font=("Courier New", 14), bg="#0d1117", fg=SUBTEXT)
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.alert_banner = tk.Label(
            right, text="⚠  FALL DETECTED — EMERGENCY  ⚠",
            font=("Courier New", 14, "bold"),
            bg=ACCENT2, fg="white", pady=10)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _start_live(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(int(self.cam_var.get()))
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {int(self.cam_var.get())}")
            return
        self._reset_stats()
        self.running = True
        self.lbl_mode.config(text="Live")
        self._set_buttons(running=True)
        self._log("Live camera started.")
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _start_video(self):
        if self.running:
            return
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return
        self._reset_stats()
        self.running = True
        self.lbl_mode.config(text="Video")
        self._set_buttons(running=True)
        self._log(f"Video loaded: {os.path.basename(path)}")
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _stop(self):
        self.running = False
        self._set_buttons(running=False)
        self.root.after(0, self._hide_alert)
        self.status_label.config(text="● IDLE", fg=WARNING)
        self._log("Detection stopped.")

    def _reset_stats(self):
        self.fall_start    = None
        self.alert_active  = False
        self.sound_playing = False
        self.fall_count    = 0
        self.frame_count   = 0
        self.lbl_frames.config(text="0")
        self.lbl_falls.config(text="0")

    def _set_buttons(self, running):
        self.btn_live.config( state="disabled" if running else "normal")
        self.btn_video.config(state="disabled" if running else "normal")
        self.btn_stop.config( state="normal"   if running else "disabled")

    def _detection_loop(self):
        confirm_secs = float(self.conf_var.get())
        frame_num    = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self._log("Feed ended.")
                self.root.after(0, lambda: self._set_buttons(running=False))
                break

            frame_num        += 1
            self.frame_count += 1

            # Skip every other frame for speed
            if frame_num % 2 != 0:
                continue

            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = self.detector.detect(mp_image)

            prediction   = 0
            body_visible = False
            status_msg   = "No person detected"
            status_color = (128, 128, 128)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks, w, h)

                if is_full_body_visible(landmarks):
                    body_visible = True
                    features     = extract_features(landmarks)
                    model_pred   = self.clf.predict([features])[0]
                    confidence   = self.clf.predict_proba([features])[0][1] * 100
                    geo_pred     = is_fallen_by_geometry(landmarks)
                    prediction   = 1 if (model_pred == 1 and geo_pred) else 0
                    status_msg   = f"Normal  |  Conf: {confidence:.0f}%"
                    status_color = (0, 230, 118)
                else:
                    status_msg   = "Unable to detect - show full body"
                    status_color = (0, 165, 255)

            now = time.time()

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                self.fall_start    = None
                self.alert_active  = False
                self.sound_playing = False

            elif not body_visible:
                self.fall_start    = None
                self.alert_active  = False
                self.sound_playing = False

            elif prediction == 1:
                if self.fall_start is None:
                    self.fall_start = now
                elapsed = now - self.fall_start
                if elapsed >= confirm_secs:
                    if not self.alert_active:
                        self.fall_count += 1
                        fc = self.fall_count
                        self.root.after(0, lambda f=fc: self.lbl_falls.config(text=str(f)))
                        self._log(f"FALL #{self.fall_count} confirmed!")
                    self.alert_active = True
                    status_msg   = "!! FALL DETECTED !!"
                    status_color = (0, 0, 255)
                else:
                    remaining    = confirm_secs - elapsed
                    status_msg   = f"Fall? Confirming... {remaining:.1f}s"
                    status_color = (0, 165, 255)
            else:
                self.fall_start    = None
                self.alert_active  = False
                self.sound_playing = False

            cv2.putText(frame, status_msg, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

            if self.alert_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                if not self.sound_playing:
                    threading.Thread(target=play_beep, daemon=True).start()
                    self.sound_playing = True
                self.root.after(0, self._show_alert)
            else:
                self.root.after(0, self._hide_alert)

            fc = self.frame_count
            self.root.after(0, lambda f=fc: self.lbl_frames.config(text=str(f)))

            if self.alert_active:
                self.root.after(0, lambda: self.status_label.config(
                    text="● FALL DETECTED", fg=ACCENT2))
            elif body_visible and prediction == 0:
                self.root.after(0, lambda: self.status_label.config(
                    text="● MONITORING", fg=SUCCESS))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text="● ACTIVE", fg=ACCENT))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.root.after(0, lambda f=frame_rgb: self._update_canvas(f))

        if self.cap:
            self.cap.release()

    def _update_canvas(self, frame_rgb):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        img   = Image.fromarray(frame_rgb)
        img   = img.resize((cw, ch), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        try:
            self.placeholder.place_forget()
        except Exception:
            pass

    def _show_alert(self):
        self.alert_banner.pack(side="bottom", fill="x")

    def _hide_alert(self):
        self.alert_banner.pack_forget()


if __name__ == "__main__":
    root = tk.Tk()
    app  = FallDetectionApp(root)
    root.mainloop()