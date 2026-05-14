"""
app.py
Fall Detection System - Main UI with Records Viewer
Usage: python app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
import csv
import subprocess

sys.path.append(os.path.dirname(__file__))
from utils.pose_utils import extract_features

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH    = "model/fall_model.joblib"
POSE_MODEL    = "pose_landmarker.task"
CONFIRM_SECS  = 3
RECORDS_DIR   = "fall_records"
LOG_FILE      = os.path.join(RECORDS_DIR, "fall_log.csv")

REQUIRED_LANDMARKS    = [23, 24, 25, 26, 27, 28]
VISIBILITY_THRESHOLD  = 0.5
FALL_HEIGHT_RATIO_MAX = 0.55
FALL_HIP_SHOULDER_MAX = 0.30
PRE_FALL_SECS         = 3
POST_FALL_SECS        = 5

SKELETON_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

# ── Colors ───────────────────────────────────────────────────────────────────
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
def ensure_dirs():
    os.makedirs(RECORDS_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["Fall #", "Date", "Time", "Mode", "Screenshot", "Video Clip"])


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
    return (abs(nose_y - ankle_y)    < FALL_HEIGHT_RATIO_MAX and
            abs(hip_y - shoulder_y)  < FALL_HIP_SHOULDER_MAX)


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


# ════════════════════════════════════════════════════════════════════════════
# RECORDS WINDOW
# ════════════════════════════════════════════════════════════════════════════
class RecordsWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Fall Records")
        self.configure(bg=BG)
        self.geometry("900x600")
        self.resizable(True, True)

        self.selected_row  = None
        self.preview_imgtk = None

        self._build_ui()
        self._load_records()

    def _build_ui(self):
        # ── Top bar ──
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x", padx=0, pady=0)
        tk.Label(top, text="  📋  FALL RECORDS", font=("Courier New", 14, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left", pady=12, padx=10)
        tk.Button(top, text="🔄  Refresh", font=FONT_SUB, bg=CARD, fg=ACCENT,
                  relief="flat", cursor="hand2", padx=10, pady=6,
                  command=self._load_records).pack(side="right", padx=10, pady=8)
        tk.Button(top, text="🗑  Delete Selected", font=FONT_SUB, bg=CARD, fg=ACCENT2,
                  relief="flat", cursor="hand2", padx=10, pady=6,
                  command=self._delete_selected).pack(side="right", padx=5, pady=8)

        # ── Main area ──
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Left — table
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Records.Treeview",
                        background=CARD, foreground=TEXT,
                        rowheight=32, fieldbackground=CARD,
                        font=("Courier New", 9))
        style.configure("Records.Treeview.Heading",
                        background=PANEL, foreground=ACCENT,
                        font=("Courier New", 9, "bold"))
        style.map("Records.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        cols = ("Fall #", "Date", "Time", "Mode", "Screenshot", "Clip")
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 style="Records.Treeview")
        for col, w in zip(cols, [55, 90, 75, 55, 200, 200]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double_click)

        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Right — preview panel
        right = tk.Frame(main, bg=PANEL, width=260)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="PREVIEW", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(pady=(15, 5))

        self.preview_canvas = tk.Canvas(right, bg="#0d1117", width=240, height=180,
                                        highlightthickness=1, highlightbackground=ACCENT)
        self.preview_canvas.pack(padx=10)
        self.no_preview = tk.Label(self.preview_canvas, text="Select a record\nto preview",
                                   font=FONT_SUB, bg="#0d1117", fg=SUBTEXT)
        self.no_preview.place(relx=0.5, rely=0.5, anchor="center")

        tk.Frame(right, bg=CARD, height=1).pack(fill="x", padx=10, pady=10)

        tk.Label(right, text="DETAILS", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=15)
        self.detail_box = tk.Text(right, height=8, bg=CARD, fg=TEXT,
                                  font=FONT_LOG, relief="flat", state="disabled",
                                  wrap="word", bd=0)
        self.detail_box.pack(fill="x", padx=10, pady=5)

        tk.Frame(right, bg=CARD, height=1).pack(fill="x", padx=10, pady=5)

        tk.Button(right, text="▶  Play Video Clip",
                  font=FONT_BTN, bg=ACCENT, fg=BG, relief="flat",
                  cursor="hand2", pady=8,
                  command=self._play_clip).pack(fill="x", padx=10, pady=3)

        tk.Button(right, text="🖼  Open Screenshot",
                  font=FONT_BTN, bg=CARD, fg=ACCENT, relief="flat",
                  cursor="hand2", pady=8,
                  command=self._open_screenshot).pack(fill="x", padx=10, pady=3)

    def _load_records(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        if not os.path.exists(LOG_FILE):
            return
        with open(LOG_FILE, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for i, row in enumerate(reader):
                tag = "even" if i % 2 == 0 else "odd"
                self.tree.insert("", "end", values=row, tags=(tag,))
        self.tree.tag_configure("odd", background="#151f30")

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        self.selected_row = self.tree.item(sel[0])["values"]
        self._show_preview()
        self._show_details()

    def _on_double_click(self, event):
        # Double click on preview canvas opens screenshot
        self._play_clip()

    def _show_preview(self):
        if not self.selected_row:
            return
        shot_name = self.selected_row[4]
        shot_path = os.path.join(RECORDS_DIR, shot_name)
        if not os.path.exists(shot_path):
            return
        img = Image.open(shot_path)
        img = img.resize((240, 180), Image.LANCZOS)
        self.preview_imgtk = ImageTk.PhotoImage(img)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_imgtk)
        try:
            self.no_preview.place_forget()
        except Exception:
            pass

    def _show_details(self):
        if not self.selected_row:
            return
        r = self.selected_row
        txt = (f"Fall #:      {r[0]}\n"
               f"Date:        {r[1]}\n"
               f"Time:        {r[2]}\n"
               f"Mode:        {r[3]}\n"
               f"Screenshot:  {r[4]}\n"
               f"Clip:        {r[5]}\n")
        self.detail_box.config(state="normal")
        self.detail_box.delete("1.0", "end")
        self.detail_box.insert("end", txt)
        self.detail_box.config(state="disabled")

    def _play_clip(self):
        if not self.selected_row:
            messagebox.showinfo("Info", "Select a record first.")
            return
        clip_path = os.path.join(RECORDS_DIR, self.selected_row[5])
        if not os.path.exists(clip_path):
            messagebox.showerror("Error", f"Clip not found:\n{clip_path}")
            return
        # Open with system default video player
        os.startfile(clip_path)

    def _open_screenshot(self):
        if not self.selected_row:
            messagebox.showinfo("Info", "Select a record first.")
            return
        shot_name = str(self.selected_row[4]).strip()
        shot_path = os.path.abspath(os.path.join(RECORDS_DIR, shot_name))
        if not shot_name or not os.path.exists(shot_path):
            messagebox.showerror("Error", f"File not found:\n{shot_path}")
            return
        os.startfile(shot_path)

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Select a record to delete.")
            return
        row = self.tree.item(sel[0])["values"]
        if not messagebox.askyesno("Confirm", f"Delete Fall #{row[0]} record?"):
            return

        # Delete files
        for fname in [row[4], row[5]]:
            path = os.path.join(RECORDS_DIR, fname)
            if os.path.exists(path):
                os.remove(path)

        # Remove from CSV
        rows = []
        with open(LOG_FILE, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if str(r[0]) != str(row[0])]
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        self._load_records()
        self.detail_box.config(state="normal")
        self.detail_box.delete("1.0", "end")
        self.detail_box.config(state="disabled")
        self.preview_canvas.delete("all")
        self.no_preview.place(relx=0.5, rely=0.5, anchor="center")


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
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
        self.mode     = None

        self.fall_start    = None
        self.alert_active  = False
        self.sound_playing = False
        self.fall_count    = 0
        self.frame_count   = 0

        self.fps_estimate     = 15
        self.frame_buffer     = []
        self.recording_post   = False
        self.post_fall_frames = []
        self.post_fall_end    = 0
        self.current_clip_path  = None
        self.current_fall_num   = 0
        self.current_date_str   = ""
        self.current_time_str   = ""
        self.current_shot_path  = ""

        ensure_dirs()
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
        left = tk.Frame(self.root, bg=PANEL, width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tf = tk.Frame(left, bg=PANEL)
        tf.pack(pady=(30,5), padx=20, fill="x")
        tk.Label(tf, text="⬡ FALL", font=("Courier New",26,"bold"),
                 bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(tf, text="DETECTION", font=("Courier New",26,"bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w")
        tk.Label(tf, text="SYSTEM v1.0", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w")

        tk.Frame(left, bg=ACCENT, height=2).pack(fill="x", padx=20, pady=15)

        sc = tk.Frame(left, bg=CARD)
        sc.pack(fill="x", padx=20, pady=5)
        tk.Label(sc, text="SYSTEM STATUS", font=FONT_SUB,
                 bg=CARD, fg=SUBTEXT).pack(anchor="w", padx=15, pady=(12,2))
        self.status_label = tk.Label(sc, text="● IDLE",
                                     font=FONT_STAT, bg=CARD, fg=WARNING)
        self.status_label.pack(anchor="w", padx=15, pady=(0,12))

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

        btn_cfg = dict(font=FONT_BTN, relief="flat", bd=0,
                       cursor="hand2", pady=12, padx=10)

        self.btn_live = tk.Button(left, text="▶  LIVE CAMERA",
                                  bg=ACCENT, fg=BG, activebackground="#00b8d9",
                                  command=self._start_live, **btn_cfg)
        self.btn_live.pack(fill="x", padx=20, pady=4)

        self.btn_video = tk.Button(left, text="📁  OPEN VIDEO FILE",
                                   bg=CARD, fg=ACCENT, activebackground="#1e2d45",
                                   command=self._start_video, **btn_cfg)
        self.btn_video.pack(fill="x", padx=20, pady=4)

        self.btn_stop = tk.Button(left, text="■  STOP",
                                  bg=ACCENT2, fg="white", activebackground="#cc2244",
                                  command=self._stop, state="disabled", **btn_cfg)
        self.btn_stop.pack(fill="x", padx=20, pady=4)

        tk.Button(left, text="📋  VIEW RECORDS",
                  bg=CARD, fg=SUCCESS, activebackground="#1e2d45",
                  command=self._open_records_window, **btn_cfg).pack(fill="x", padx=20, pady=4)

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

        tk.Frame(left, bg=ACCENT, height=1).pack(fill="x", padx=20, pady=10)
        tk.Label(left, text="EVENT LOG", font=FONT_SUB,
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=20)
        self.log_box = tk.Text(left, height=7, bg=CARD, fg=SUBTEXT,
                               font=FONT_LOG, relief="flat", state="disabled",
                               wrap="word", bd=0)
        self.log_box.pack(fill="both", padx=20, pady=5, expand=True)

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
        self.mode = "Live"
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
        self.mode = "Video"
        self._reset_stats()
        self.running = True
        self.lbl_mode.config(text="Video")
        self._set_buttons(running=True)
        self._log(f"Video: {os.path.basename(path)}")
        threading.Thread(target=self._detection_loop, daemon=True).start()

    def _stop(self):
        self.running = False
        self._set_buttons(running=False)
        self.root.after(0, self._hide_alert)
        self.status_label.config(text="● IDLE", fg=WARNING)
        self._log("Detection stopped.")

    def _open_records_window(self):
        RecordsWindow(self.root)

    def _reset_stats(self):
        self.fall_start       = None
        self.alert_active     = False
        self.sound_playing    = False
        self.fall_count       = 0
        self.frame_count      = 0
        self.frame_buffer     = []
        self.recording_post   = False
        self.finalizing_clip  = False
        self.post_fall_frames = []
        self.lbl_frames.config(text="0")
        self.lbl_falls.config(text="0")

    def _set_buttons(self, running):
        self.btn_live.config( state="disabled" if running else "normal")
        self.btn_video.config(state="disabled" if running else "normal")
        self.btn_stop.config( state="normal"   if running else "disabled")

    def _save_fall_record(self, fall_num, ts_str, date_str, time_str, screenshot_frame):
        base      = f"fall_{fall_num:03d}_{ts_str}"
        shot_path = os.path.join(RECORDS_DIR, f"{base}.jpg")
        snap      = screenshot_frame.copy()
        h         = snap.shape[0]
        cv2.putText(snap, f"FALL DETECTED  {date_str} {time_str}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        cv2.imwrite(shot_path, snap)

        self.current_clip_path = os.path.join(RECORDS_DIR, f"{base}_clip.mp4")
        self.current_fall_num  = fall_num
        self.current_date_str  = date_str
        self.current_time_str  = time_str
        self.current_shot_path = shot_path

        self.recording_post   = True
        self.post_fall_frames = list(self.frame_buffer)
        self.post_fall_end    = time.time() + POST_FALL_SECS

        self._log(f"📸 Screenshot saved: {base}.jpg")
        self._log(f"🎥 Recording {POST_FALL_SECS}s clip...")

    def _finalize_clip(self):
        if not self.post_fall_frames:
            return
        h, w = self.post_fall_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(self.current_clip_path, fourcc,
                                 max(self.fps_estimate, 10), (w, h))
        for f in self.post_fall_frames:
            out.write(f)
        out.release()

        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                self.current_fall_num,
                self.current_date_str,
                self.current_time_str,
                self.mode,
                os.path.basename(self.current_shot_path),
                os.path.basename(self.current_clip_path)
            ])

        self._log(f"✅ Clip saved: {os.path.basename(self.current_clip_path)}")
        self.recording_post   = False
        self.finalizing_clip  = False
        self.post_fall_frames = []

    def _detection_loop(self):
        confirm_secs    = float(self.conf_var.get())
        frame_num       = 0
        t_start         = time.time()
        saved_this_fall = False

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self._log("Feed ended.")
                if self.recording_post:
                    self._finalize_clip()
                self.root.after(0, lambda: self._set_buttons(running=False))
                break

            frame_num        += 1
            self.frame_count += 1

            elapsed_total = time.time() - t_start
            if elapsed_total > 0:
                self.fps_estimate = max(5, int(frame_num / elapsed_total))

            self.frame_buffer.append(frame.copy())
            max_buf = int((PRE_FALL_SECS + POST_FALL_SECS) * self.fps_estimate)
            if len(self.frame_buffer) > max_buf:
                self.frame_buffer.pop(0)

            if self.recording_post and not self.finalizing_clip:
                self.post_fall_frames.append(frame.copy())
                if time.time() >= self.post_fall_end:
                    self.finalizing_clip = True
                    threading.Thread(target=self._finalize_clip, daemon=True).start()

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
                saved_this_fall    = False

            elif not body_visible:
                self.fall_start    = None
                self.alert_active  = False
                self.sound_playing = False
                saved_this_fall    = False

            elif prediction == 1:
                if self.fall_start is None:
                    self.fall_start = now
                    saved_this_fall = False
                elapsed = now - self.fall_start
                if elapsed >= confirm_secs:
                    if not self.alert_active:
                        self.fall_count += 1
                        fc = self.fall_count
                        self.root.after(0, lambda f=fc: self.lbl_falls.config(text=str(f)))
                        self._log(f"⚠ FALL #{self.fall_count} confirmed!")
                        if not saved_this_fall:
                            saved_this_fall = True
                            ts_now   = time.localtime()
                            ts_str   = time.strftime("%Y%m%d_%H%M%S", ts_now)
                            d_str    = time.strftime("%Y-%m-%d", ts_now)
                            t_str    = time.strftime("%H:%M:%S", ts_now)
                            fn       = self.fall_count
                            snap     = frame.copy()
                            threading.Thread(
                                target=self._save_fall_record,
                                args=(fn, ts_str, d_str, t_str, snap),
                                daemon=True).start()
                    self.alert_active = True
                    status_msg   = f"!! FALL DETECTED !!  {time.strftime('%H:%M:%S')}"
                    status_color = (0, 0, 255)
                else:
                    remaining    = confirm_secs - elapsed
                    status_msg   = f"Fall? Confirming... {remaining:.1f}s"
                    status_color = (0, 165, 255)
            else:
                self.fall_start    = None
                self.alert_active  = False
                self.sound_playing = False
                saved_this_fall    = False

            cv2.putText(frame, status_msg, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
            cv2.putText(frame, time.strftime("%Y-%m-%d  %H:%M:%S"),
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

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