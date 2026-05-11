"""
app.py
Fall Detection System - Main UI
Usage: python app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import subprocess
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
FALL_HEIGHT_RATIO_MAX  = 0.35
FALL_HIP_SHOULDER_MAX  = 0.20

SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

# ── Colors & Fonts ───────────────────────────────────────────────────────────
BG        = "#0a0e1a"
PANEL     = "#111827"
CARD      = "#1a2235"
ACCENT    = "#00d4ff"
ACCENT2   = "#ff3f6c"
SUCCESS   = "#00e676"
WARNING   = "#ffab00"
TEXT      = "#e8eaf6"
SUBTEXT   = "#7986cb"
FONT_HEAD = ("Courier New", 22, "bold")
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
    height_ratio     = abs(nose_y - ankle_y)
    hip_shoulder_diff = abs(hip_y - shoulder_y)
    return height_ratio < FALL_HEIGHT_RATIO_MAX and hip_shoulder_diff < FALL_HIP_SHOULDER_MAX


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
        t  = np.linspace(0, 2.0, int(sr * 2.0), False)
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
        self.mode     = None   # "live" or "video"

        self.fall_start    = None
        self.alert_active  = False
        self.sound_playing = False
        self.fall_count    = 0
        self.frame_count   = 0
        self.status_text   = "Idle"

        self._load_model()
        self._build_ui()

    # ── Model ────────────────────────────────────────────────────────────────
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

    # ── UI Layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left panel ──
        left = tk.Frame(self.root, bg=PANEL, width=300)
        left.pack(side="left", fill="y", padx=0, pady=0)
        left.pack_propagate(False)

        # Logo / Title
        title_frame = tk.Frame(left, bg=PANEL)
        title_frame.pack(pady=(30, 5), padx=20, fill="x")
        tk.Label(title_frame, text="⬡ FALL", font=("Courier New", 26, "bold"),
                 bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(title_frame, text="DETECTION", font=("Courier New", 26, "bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w")
        tk.Label(title_frame, text="SYSTEM v1.0",
                 font=FONT_SUB, bg=PANEL, fg=SUBTEXT).pack(anchor="w")

        # Divider
        tk.Frame(left, bg=ACCENT, height=2).pack(fill="x", padx=20, pady=15)

        # Status card
        stat_card = tk.Frame(left, bg=CARD, bd=0)
        stat_card.pack(fill="x", padx=20, pady=5)
        tk.Label(stat_card, text="SYSTEM STATUS", font=FONT_SUB,
                 bg=CARD, fg=SUBTEXT).pack(anchor="w", padx=15, pady=(12,2))
        self.status_label = tk.Label(stat_card, text="● IDLE",
                                     font=FONT_STAT, bg=CARD, fg=WARNING)
        self.status_label.pack(anchor="w", padx=15, pady=(0,12))

        # Stats
        stats_card = tk.Frame(left, bg=CARD)
        stats_card.pack(fill="x", padx=20, pady=5)
        tk.Label(stats_card, text="STATISTICS", font=FONT_SUB,
                 bg=CARD, fg=SUBTEXT).pack(anchor="w", padx=15, pady=(12,4))

        row1 = tk.Frame(stats_card, bg=CARD)
        row1.pack(fill="x", padx=15, pady=2)
        tk.Label(row1, text="Frames processed", font=FONT_SUB, bg=CARD, fg=SUBTEXT).pack(side="left")
        self.lbl_frames = tk.Label(row1, text="0", font=FONT_SUB, bg=CARD, fg=TEXT)
        self.lbl_frames.pack(side="right")

        row2 = tk.Frame(stats_card, bg=CARD)
        row2.pack(fill="x", padx=15, pady=2)
        tk.Label(row2, text="Falls detected", font=FONT_SUB, bg=CARD, fg=SUBTEXT).pack(side="left")
        self.lbl_falls = tk.Label(row2, text="0", font=FONT_SUB, bg=CARD, fg=ACCENT2)
        self.lbl_falls.pack(side="right")

        row3 = tk.Frame(stats_card, bg=CARD)
        row3.pack(fill="x", padx=15, pady=(2,12))
        tk.Label(row3, text="Mode", font=FONT_SUB, bg=CARD, fg=SUBTEXT).pack(side="left")
        self.lbl_mode = tk.Label(row3, text="—", font=FONT_SUB, bg=CARD, fg=TEXT)
        self.lbl_mode.pack(side="right")

        # Divider
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

        # Camera index
        cam_frame = tk.Frame(left, bg=PANEL)
        cam_frame.pack(fill="x", padx=20, pady=(10,0))
        tk.Label(cam_frame, text="Camera Index:", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(side="left")
        self.cam_var = tk.IntVar(value=1)
        tk.Spinbox(cam_frame, from_=0, to=5, textvariable=self.cam_var,
                   width=3, font=FONT_SUB, bg=CARD, fg=TEXT,
                   buttonbackground=CARD, relief="flat").pack(side="right")

        # Confirm seconds
        conf_frame = tk.Frame(left, bg=PANEL)
        conf_frame.pack(fill="x", padx=20, pady=4)
        tk.Label(conf_frame, text="Confirm seconds:", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(side="left")
        self.conf_var = tk.DoubleVar(value=3.0)
        tk.Spinbox(conf_frame, from_=1.0, to=10.0, increment=0.5,
                   textvariable=self.conf_var, width=4, font=FONT_SUB,
                   bg=CARD, fg=TEXT, buttonbackground=CARD, relief="flat").pack(side="right")

        # Log
        tk.Frame(left, bg=ACCENT, height=1).pack(fill="x", padx=20, pady=12)
        tk.Label(left, text="EVENT LOG", font=FONT_SUB, bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=20)
        self.log_box = tk.Text(left, height=8, bg=CARD, fg=SUBTEXT,
                               font=FONT_LOG, relief="flat", state="disabled",
                               wrap="word", bd=0)
        self.log_box.pack(fill="both", padx=20, pady=5, expand=True)

        # ── Right panel (video feed) ──
        right = tk.Frame(self.root, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        # Feed area
        feed_frame = tk.Frame(right, bg=BG)
        feed_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(feed_frame, bg="#0d1117", highlightthickness=1,
                                highlightbackground=ACCENT)
        self.canvas.pack(fill="both", expand=True)

        # Placeholder
        self.placeholder = tk.Label(self.canvas,
                                    text="NO FEED\n\nPress LIVE CAMERA or\nOPEN VIDEO FILE to begin",
                                    font=("Courier New", 14), bg="#0d1117", fg=SUBTEXT)
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Alert banner (hidden by default)
        self.alert_banner = tk.Label(right,
                                     text="⚠  FALL DETECTED — EMERGENCY  ⚠",
                                     font=("Courier New", 14, "bold"),
                                     bg=ACCENT2, fg="white", pady=10)

    # ── Logging ──────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ── Button actions ───────────────────────────────────────────────────────
    def _start_live(self):
        if self.running:
            return
        self.mode = "live"
        self.cap  = cv2.VideoCapture(self.cam_var.get())
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {self.cam_var.get()}")
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
        self.mode = "video"
        self.cap  = cv2.VideoCapture(path)
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
        self._hide_alert()
        self.status_label.config(text="● IDLE", fg=WARNING)
        self._log("Detection stopped.")

    def _reset_stats(self):
        self.fall_start   = None
        self.alert_active = False
        self.sound_playing = False
        self.fall_count   = 0
        self.frame_count  = 0
        self.lbl_frames.config(text="0")
        self.lbl_falls.config(text="0")

    def _set_buttons(self, running):
        state_on  = "disabled" if running else "normal"
        state_off = "normal"   if running else "disabled"
        self.btn_live.config(state=state_on)
        self.btn_video.config(state=state_on)
        self.btn_stop.config(state=state_off)

    # ── Detection loop ───────────────────────────────────────────────────────
    def _detection_loop(self):
        confirm_secs = self.conf_var.get()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self._log("Feed ended.")
                self.root.after(0, lambda: self._set_buttons(running=False))
                break

            self.frame_count += 1
            h, w = frame.shape[:2]

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
                    geo_pred     = is_fallen_by_geometry(landmarks)
                    prediction   = 1 if (model_pred == 1 and geo_pred) else 0
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
                        self.root.after(0, lambda: self.lbl_falls.config(
                            text=str(self.fall_count)))
                        self._log(f"FALL #{self.fall_count} confirmed!")
                    self.alert_active = True
                    status_msg   = "!! FALL DETECTED !!"
                    status_color = (0, 0, 255)
                else:
                    remaining    = confirm_secs - elapsed
                    status_msg   = f"Confirming... {remaining:.1f}s"
                    status_color = (0, 165, 255)
            else:
                self.fall_start    = None
                self.alert_active  = False
                self.sound_playing = False
                status_msg   = "Status: Normal"
                status_color = (0, 230, 118)

            # Draw status on frame
            cv2.putText(frame, status_msg, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

            # Alert overlay on frame
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

            # Update stats label
            fc = self.frame_count
            self.root.after(0, lambda f=fc: self.lbl_frames.config(text=str(f)))

            # Update status dot
            if self.alert_active:
                self.root.after(0, lambda: self.status_label.config(
                    text="● FALL DETECTED", fg=ACCENT2))
            elif body_visible and prediction == 0:
                self.root.after(0, lambda: self.status_label.config(
                    text="● MONITORING", fg=SUCCESS))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text="● ACTIVE", fg=ACCENT))

            # Send frame to canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.root.after(0, lambda f=frame_rgb: self._update_canvas(f))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

        if self.cap:
            self.cap.release()

    # ── Canvas update ─────────────────────────────────────────────────────────
    def _update_canvas(self, frame_rgb):
        if not self.running and self.frame_count == 0:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        img  = Image.fromarray(frame_rgb)
        img  = img.resize((cw, ch), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        if self.placeholder.winfo_exists():
            self.placeholder.place_forget()

    # ── Alert banner ─────────────────────────────────────────────────────────
    def _show_alert(self):
        self.alert_banner.pack(side="bottom", fill="x")

    def _hide_alert(self):
        self.alert_banner.pack_forget()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Installing Pillow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk

    root = tk.Tk()
    app  = FallDetectionApp(root)
    root.mainloop()
