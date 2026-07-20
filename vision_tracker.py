"""
╔══════════════════════════════════════════════════════════════════╗
║           REAL-TIME FACE MESH + HAND TRACKING                   ║
║           Powered by MediaPipe Tasks API & OpenCV               ║
╚══════════════════════════════════════════════════════════════════╝

Author  : <Murat Can Nergiz>
GitHub  : https://github.com/<codcreater1>
License : MIT

Features:
  ✦ 478-point Face Mesh (iris landmarks included)
  ✦ Dual hand tracking with 21 landmarks per hand
  ✦ Real-time FPS counter
  ✦ Live finger gesture recognition (peace, fist, thumbs-up, open hand,
    OK sign, rock on)
  ✦ Hold-to-trigger actions: hold OK Sign for 1.2s to auto-capture a
    screenshot (per-hand progress bar shown in the HUD)
  ✦ Pinch-to-zoom: thumb-index distance drives live digital zoom
  ✦ Swipe detection: fast open-hand left/right motion fires a
    next/prev slide action
  ✦ Blink detection via Eye Aspect Ratio (EAR)
  ✦ On-screen HUD with landmark counts
  ✦ Screenshot capture (press S)
  ✦ Mirror / flip toggle (press M)
  ✦ Overlay opacity control (press +/-)
  ✦ Auto model download on first run

Requirements:
  pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time
import math
from collections import deque
from datetime import datetime

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Config:
    # Model paths & URLs
    FACE_MODEL      = "face_landmarker.task"
    HAND_MODEL      = "hand_landmarker.task"
    FACE_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                       "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    HAND_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                       "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

    # Detection thresholds
    MIN_DETECTION_CONF  = 0.55
    MIN_PRESENCE_CONF   = 0.55
    MIN_TRACKING_CONF   = 0.50
    MAX_FACES           = 2
    MAX_HANDS           = 2

    # Visual style
    FACE_DOT_COLOR      = (0,   230, 140)   # green-mint
    FACE_CONTOUR_COLOR  = (255, 220,  60)   # amber
    FACE_IRIS_COLOR     = (100, 200, 255)   # sky blue
    HAND_R_DOT         = (0,   160, 255)   # orange
    HAND_R_LINE        = (0,   220, 255)   # yellow
    HAND_L_DOT         = (220,  60, 255)   # violet
    HAND_L_LINE        = (180, 120, 255)   # lavender
    HUD_BG             = (10,   10,  20)   # near-black
    HUD_ACCENT         = (0,   230, 140)   # green-mint
    TEXT_PRIMARY       = (230, 230, 230)
    TEXT_DIM           = (120, 120, 120)

    # FPS smoothing window
    FPS_WINDOW          = 30

    # Screenshot output folder
    SCREENSHOT_DIR      = "screenshots"

    # Gesture recognition — pinch / hold-to-trigger
    OK_PINCH_PX          = 40     # thumb-tip↔index-tip distance (px) that counts as an "OK" pinch
    GESTURE_HOLD_SECONDS = 1.2    # how long a gesture must be held to fire its action
    GESTURE_COOLDOWN     = 2.0    # min seconds between auto-triggered actions

    # Pinch-to-zoom (thumb-index distance, normalized by hand size)
    PINCH_ZOOM_MIN_RATIO = 0.25   # fully pinched  -> zoom = ZOOM_MIN
    PINCH_ZOOM_MAX_RATIO = 1.4    # fully spread   -> zoom = ZOOM_MAX
    ZOOM_MIN             = 1.0
    ZOOM_MAX             = 2.5
    ZOOM_SMOOTHING       = 0.25   # 0..1, higher = snappier

    # Swipe detection (wrist horizontal motion)
    SWIPE_HISTORY_SEC    = 0.5    # window used to measure swipe velocity
    SWIPE_MIN_DIST_RATIO = 0.18   # min horizontal travel (fraction of frame width)
    SWIPE_COOLDOWN       = 0.8    # min seconds between swipe triggers
    SWIPE_DISPLAY_SEC    = 1.0    # how long the swipe label stays on HUD


# ═══════════════════════════════════════════════════════════════
# HAND LANDMARK INDICES  (MediaPipe convention)
# ═══════════════════════════════════════════════════════════════

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Fingertip indices: thumb, index, middle, ring, pinky
FINGERTIPS  = [4, 8, 12, 16, 20]
# Second joint (PIP) indices
FINGER_PIPS = [3, 6, 10, 14, 18]

# ═══════════════════════════════════════════════════════════════
# FACE LANDMARK INDICES (subset for contour & iris)
# ═══════════════════════════════════════════════════════════════

FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,397,365,
    379,378,400,377,152,148,176,149,150,136,172,58,132,93,
    234,127,162,21,54,103,67,109,10
]

LEFT_EYE_CONTOUR  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
RIGHT_EYE_CONTOUR = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]

# Iris centres (index 468=left iris, 473=right iris in 478-point model)
LEFT_IRIS   = [468, 469, 470, 471, 472]
RIGHT_IRIS  = [473, 474, 475, 476, 477]

# EAR landmarks: [p1,p2,p3,p4,p5,p6] for each eye
LEFT_EAR_PTS  = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_PTS = [362, 385, 387, 263, 373, 380]


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def download_model(url: str, path: str) -> None:
    """Download a MediaPipe .task model file if not already present."""
    if not os.path.exists(path):
        print(f"  [↓] Downloading {path} …")
        urllib.request.urlretrieve(url, path)
        print(f"  [✓] Saved: {path}")
    else:
        print(f"  [✓] Model found: {path}")


def to_pixel(lm, w: int, h: int) -> tuple[int, int]:
    """Convert a normalized landmark to pixel coordinates."""
    return int(lm.x * w), int(lm.y * h)


def distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x,y) pixel points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, indices: list, w: int, h: int) -> float:
    """
    Eye Aspect Ratio (EAR) for blink detection.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = [to_pixel(landmarks[i], w, h) for i in indices]
    A = distance(pts[1], pts[5])
    B = distance(pts[2], pts[4])
    C = distance(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def draw_polyline(frame, pts: list, color: tuple, thickness: int = 1,
                  closed: bool = False) -> None:
    """Draw a polyline through a list of (x,y) pixel points."""
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)
    if closed and len(pts) > 1:
        cv2.line(frame, pts[-1], pts[0], color, thickness, cv2.LINE_AA)


def blend_overlay(frame, overlay, alpha: float) -> np.ndarray:
    """Alpha-blend an overlay onto frame."""
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def apply_digital_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    """Crop the center of the frame and resize back up = digital zoom."""
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom), int(h / zoom)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    cropped = frame[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════════════════════
# PINCH-TO-ZOOM
# ═══════════════════════════════════════════════════════════════

def pinch_zoom_factor(landmarks, w: int, h: int, cfg: Config) -> float:
    """
    Map thumb-tip↔index-tip distance (normalized by palm size, so it
    works at any distance from the camera) to a target zoom level.
    """
    pts = [to_pixel(lm, w, h) for lm in landmarks]
    pinch_dist = distance(pts[FINGERTIPS[0]], pts[FINGERTIPS[1]])
    palm_size  = distance(pts[0], pts[9]) + 1e-6   # wrist -> middle_mcp
    ratio = pinch_dist / palm_size

    ratio = max(cfg.PINCH_ZOOM_MIN_RATIO,
                min(cfg.PINCH_ZOOM_MAX_RATIO, ratio))
    t = (ratio - cfg.PINCH_ZOOM_MIN_RATIO) / (
        cfg.PINCH_ZOOM_MAX_RATIO - cfg.PINCH_ZOOM_MIN_RATIO)
    return cfg.ZOOM_MIN + t * (cfg.ZOOM_MAX - cfg.ZOOM_MIN)


# ═══════════════════════════════════════════════════════════════
# SWIPE DETECTION
# ═══════════════════════════════════════════════════════════════

class SwipeTracker:
    """Tracks wrist x-position over a short time window per hand side
    and reports a 'Left'/'Right' swipe once travel/velocity thresholds
    are crossed."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = {}       # side -> deque[(t, x_norm)]
        self.last_trigger = 0.0

    def update(self, side: str, wrist_x_norm: float, now: float):
        hist = self.history.setdefault(side, deque())
        hist.append((now, wrist_x_norm))
        while hist and now - hist[0][0] > self.cfg.SWIPE_HISTORY_SEC:
            hist.popleft()

    def detect(self, now: float) -> str | None:
        """Returns 'Left', 'Right', or None. Fires at most once per cooldown."""
        if now - self.last_trigger < self.cfg.SWIPE_COOLDOWN:
            return None
        for side, hist in self.history.items():
            if len(hist) < 2:
                continue
            t0, x0 = hist[0]
            t1, x1 = hist[-1]
            travel = x1 - x0
            if abs(travel) >= self.cfg.SWIPE_MIN_DIST_RATIO:
                self.last_trigger = now
                hist.clear()
                return "Right" if travel > 0 else "Left"
        return None

    def drop_missing(self, active_sides: set) -> None:
        for side in list(self.history.keys()):
            if side not in active_sides:
                del self.history[side]


# ═══════════════════════════════════════════════════════════════
# GESTURE RECOGNITION
# ═══════════════════════════════════════════════════════════════

def fingers_up(landmarks, w: int, h: int) -> list[bool]:
    """
    Return a list of 5 booleans: [thumb, index, middle, ring, pinky].
    True = finger is extended.
    """
    up = []
    pts = [to_pixel(landmarks[i], w, h) for i in range(21)]

    # Thumb: compare x position (left/right depends on handedness; use y as fallback)
    thumb_tip  = pts[FINGERTIPS[0]]
    thumb_pip  = pts[FINGER_PIPS[0]]
    up.append(thumb_tip[0] < thumb_pip[0] or thumb_tip[1] < thumb_pip[1])

    # Other four fingers: tip y < pip y  →  finger extended (image coords)
    for tip_idx, pip_idx in zip(FINGERTIPS[1:], FINGER_PIPS[1:]):
        up.append(pts[tip_idx][1] < pts[pip_idx][1])
    return up


def middle_ring_pinky_up(up: list[bool]) -> bool:
    """True if middle/ring/pinky are extended (used to keep OK-sign gesture
    and plain pinch-to-zoom from overlapping)."""
    _, _, middle, ring, pinky = up
    return middle and ring and pinky


def is_ok_pinch(landmarks, w: int, h: int, threshold_px: int) -> bool:
    """True when thumb-tip and index-tip are close enough to form an 'OK' circle."""
    pts = [to_pixel(lm, w, h) for lm in landmarks]
    return distance(pts[FINGERTIPS[0]], pts[FINGERTIPS[1]]) < threshold_px


def classify_gesture(up: list[bool], pinch: bool = False) -> str:
    """Map finger states (+ optional thumb/index pinch) to a named gesture."""
    thumb, index, middle, ring, pinky = up
    total = sum(up)

    if pinch and middle and ring and pinky:
        return "👌  OK Sign"
    if index and pinky and not middle and not ring:
        return "🤟  Rock On"
    if total == 0:
        return "✊  Fist"
    if total == 5:
        return "🖐  Open Hand"
    if index and middle and not ring and not pinky and not thumb:
        return "✌  Peace"
    if thumb and not index and not middle and not ring and not pinky:
        return "👍  Thumbs Up"
    if index and not middle and not ring and not pinky:
        return "☝  Pointing"
    if pinky and not index and not middle and not ring:
        return "🤙  Pinky"
    return f"   Custom ({total} fingers)"


# ═══════════════════════════════════════════════════════════════
# DRAWING — HANDS
# ═══════════════════════════════════════════════════════════════

def draw_hand(frame, landmarks, dot_color: tuple, line_color: tuple,
              w: int, h: int, alpha_overlay=None, alpha: float = 0.8) -> None:
    """Draw hand skeleton with optional semi-transparent fill."""
    pts = [to_pixel(lm, w, h) for lm in landmarks]

    # Connection lines
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 2, cv2.LINE_AA)

    # Landmark dots (larger at fingertips)
    for idx, (px, py) in enumerate(pts):
        r = 7 if idx in FINGERTIPS else 4
        cv2.circle(frame, (px, py), r, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r, (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# DRAWING — FACE
# ═══════════════════════════════════════════════════════════════

def draw_face(frame, landmarks, w: int, h: int, cfg: Config) -> None:
    """Draw face mesh: all 478 dots, oval contour, eye contours, iris rings."""
    pts = [to_pixel(lm, w, h) for lm in landmarks]

    # --- All mesh dots (tiny) ---
    for px, py in pts[:468]:
        cv2.circle(frame, (px, py), 1, cfg.FACE_DOT_COLOR, -1, cv2.LINE_AA)

    # --- Face oval ---
    oval_pts = [pts[i] for i in FACE_OVAL if i < len(pts)]
    draw_polyline(frame, oval_pts, cfg.FACE_CONTOUR_COLOR, thickness=2)

    # --- Eye contours ---
    left_eye_pts  = [pts[i] for i in LEFT_EYE_CONTOUR  if i < len(pts)]
    right_eye_pts = [pts[i] for i in RIGHT_EYE_CONTOUR if i < len(pts)]
    draw_polyline(frame, left_eye_pts,  cfg.FACE_CONTOUR_COLOR, thickness=1, closed=True)
    draw_polyline(frame, right_eye_pts, cfg.FACE_CONTOUR_COLOR, thickness=1, closed=True)

    # --- Iris rings (only if 478-point model) ---
    if len(pts) >= 478:
        for iris_group in (LEFT_IRIS, RIGHT_IRIS):
            iris_pts = [pts[i] for i in iris_group]
            cx = int(np.mean([p[0] for p in iris_pts]))
            cy = int(np.mean([p[1] for p in iris_pts]))
            radius = int(distance(iris_pts[0], (cx, cy))) + 2
            cv2.circle(frame, (cx, cy), radius, cfg.FACE_IRIS_COLOR, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2,      cfg.FACE_IRIS_COLOR, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# HUD (Heads-Up Display)
# ═══════════════════════════════════════════════════════════════

def draw_hud(frame, fps: float, face_count: int, hand_data: list,
             blink_left: bool, blink_right: bool,
             overlay_alpha: float, mirror: bool, cfg: Config) -> None:
    """
    Render a sleek semi-transparent HUD panel in the top-left corner.
    hand_data = list of (handedness_str, gesture_str)
    """
    h_frame, w_frame = frame.shape[:2]
    panel_w, panel_h = 260, 180 + len(hand_data) * 32
    panel_h = max(panel_h, 180)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                  cfg.HUD_BG, -1)
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                  cfg.HUD_ACCENT, 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    y0     = 30
    dy     = 22
    pad    = 18

    def put(text, y, color=None, scale=0.52, bold=False):
        c = color or cfg.TEXT_PRIMARY
        t = 2 if bold else 1
        cv2.putText(frame, text, (pad, y), font, scale, c, t, cv2.LINE_AA)

    # Title
    put("▸ VISION TRACKER", y0, cfg.HUD_ACCENT, scale=0.58, bold=True)

    # FPS
    fps_color = (0, 220, 80) if fps >= 20 else (0, 140, 255) if fps >= 10 else (0, 60, 220)
    put(f"FPS      {fps:5.1f}", y0 + dy, fps_color)

    # Face
    put(f"Faces    {face_count}", y0 + 2 * dy)
    blink_txt = ""
    if face_count:
        bl = "BLINK" if blink_left  else "open"
        br = "BLINK" if blink_right else "open"
        blink_txt = f"L:{bl}  R:{br}"
    put(blink_txt, y0 + 3 * dy, cfg.TEXT_DIM, scale=0.44)

    # Hands
    put(f"Hands    {len(hand_data)}", y0 + 4 * dy)
    for i, (side, gesture, hold_ratio) in enumerate(hand_data):
        color = cfg.HAND_R_DOT if side == "Right" else cfg.HAND_L_DOT
        row_y = y0 + (5 + i) * dy
        put(f"  {side[:1]}: {gesture}", row_y, color, scale=0.46)
        if hold_ratio > 0:
            bar_x, bar_w, bar_y = pad + 150, 70, row_y - 8
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6),
                          cfg.TEXT_DIM, 1, cv2.LINE_AA)
            fill_w = int(bar_w * hold_ratio)
            fill_c = (0, 220, 80) if hold_ratio >= 1.0 else cfg.HUD_ACCENT
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 6),
                              fill_c, -1, cv2.LINE_AA)

    # Settings row
    y_settings = 8 + panel_h - 14
    settings = f"alpha:{overlay_alpha:.1f}  mirror:{'ON' if mirror else 'OFF'}"
    put(settings, y_settings, cfg.TEXT_DIM, scale=0.40)


def draw_zoom_swipe_hud(frame, zoom: float, swipe_label: str, w_frame: int, cfg: Config) -> None:
    """Top-right indicator for pinch-zoom level and last swipe action."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_w = 190
    x0 = w_frame - box_w - 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 8), (x0 + box_w, 66), cfg.HUD_BG, -1)
    cv2.rectangle(overlay, (x0, 8), (x0 + box_w, 66), cfg.HUD_ACCENT, 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, f"Zoom: {zoom:.2f}x", (x0 + 10, 30), font, 0.5,
                cfg.TEXT_PRIMARY, 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w = x0 + 10, 38, box_w - 20
    fill = int(bar_w * (zoom - cfg.ZOOM_MIN) / (cfg.ZOOM_MAX - cfg.ZOOM_MIN + 1e-6))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), cfg.TEXT_DIM, 1, cv2.LINE_AA)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 6), cfg.HUD_ACCENT, -1, cv2.LINE_AA)

    swipe_text = f"Swipe: {swipe_label}" if swipe_label else "Swipe: —"
    cv2.putText(frame, swipe_text, (x0 + 10, 58), font, 0.5,
                cfg.HUD_ACCENT if swipe_label else cfg.TEXT_DIM, 1, cv2.LINE_AA)


def draw_keybinds(frame, cfg: Config) -> None:
    """Render key-binding cheatsheet in the bottom-left corner."""
    keys = [
        ("Q", "Quit"),
        ("S", "Screenshot"),
        ("M", "Mirror"),
        ("+", "Overlay +"),
        ("-", "Overlay -"),
    ]
    h_frame = frame.shape[0]
    font    = cv2.FONT_HERSHEY_SIMPLEX
    pad_x, pad_y = 12, h_frame - len(keys) * 20 - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (6, pad_y - 8),
                  (160, h_frame - 6), cfg.HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (k, desc) in enumerate(keys):
        y = pad_y + i * 20
        cv2.putText(frame, f"[{k}]", (10, y), font, 0.42,
                    cfg.HUD_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (44, y), font, 0.42,
                    cfg.TEXT_PRIMARY, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════

def build_detectors(cfg: Config):
    """Download models (if needed) and initialise both landmarkers."""
    print("\n[MediaPipe] Preparing models …")
    download_model(cfg.FACE_MODEL_URL, cfg.FACE_MODEL)
    download_model(cfg.HAND_MODEL_URL, cfg.HAND_MODEL)

    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.FACE_MODEL),
            num_faces=cfg.MAX_FACES,
            min_face_detection_confidence=cfg.MIN_DETECTION_CONF,
            min_face_presence_confidence=cfg.MIN_PRESENCE_CONF,
            min_tracking_confidence=cfg.MIN_TRACKING_CONF,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )

    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=cfg.HAND_MODEL),
            num_hands=cfg.MAX_HANDS,
            min_hand_detection_confidence=cfg.MIN_DETECTION_CONF,
            min_hand_presence_confidence=cfg.MIN_PRESENCE_CONF,
            min_tracking_confidence=cfg.MIN_TRACKING_CONF,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
    )

    print("[MediaPipe] Detectors ready.\n")
    return face_det, hand_det


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    cfg = Config()
    os.makedirs(cfg.SCREENSHOT_DIR, exist_ok=True)

    face_det, hand_det = build_detectors(cfg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check your device index.")

    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx   = 0

    # State
    fps_times       = deque(maxlen=cfg.FPS_WINDOW)
    overlay_alpha   = 0.85   # mesh overlay opacity
    mirror          = True   # flip frame horizontally
    gesture_hold    = {}     # side -> {"gesture": str, "start": float}
    last_action_ts  = 0.0    # perf_counter of last auto-triggered action

    zoom_current    = cfg.ZOOM_MIN   # smoothed digital zoom level
    swipe_tracker   = SwipeTracker(cfg)
    swipe_label     = ""
    swipe_label_ts  = 0.0

    print("=" * 60)
    print("  Real-Time Vision Tracker — running")
    print("  Press Q to quit, S to screenshot, M to mirror")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Frame grab failed — exiting.")
            break

        frame_idx += 1
        timestamp_ms = int(frame_idx * (1000.0 / native_fps))

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ── Convert for MediaPipe ──────────────────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Run detectors ──────────────────────────────────────
        face_result = face_det.detect_for_video(mp_img, timestamp_ms)
        hand_result = hand_det.detect_for_video(mp_img, timestamp_ms)

        # ── Create drawing canvas (overlay) ───────────────────
        canvas = frame.copy()

        # ── Draw FACES ─────────────────────────────────────────
        blink_left = blink_right = False
        for face_lms in face_result.face_landmarks:
            draw_face(canvas, face_lms, w, h, cfg)
            # EAR blink detection (requires ≥ 6 landmarks per eye)
            try:
                ear_l = eye_aspect_ratio(face_lms, LEFT_EAR_PTS,  w, h)
                ear_r = eye_aspect_ratio(face_lms, RIGHT_EAR_PTS, w, h)
                blink_left  = ear_l < 0.20
                blink_right = ear_r < 0.20
            except Exception:
                pass

        # ── Draw HANDS ─────────────────────────────────────────
        now = time.perf_counter()
        active_sides = set()
        hand_data = []
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            side    = hand_result.handedness[i][0].category_name  # "Left"/"Right"
            up      = fingers_up(hand_lms, w, h)
            pinch   = is_ok_pinch(hand_lms, w, h, cfg.OK_PINCH_PX)
            gesture = classify_gesture(up, pinch)
            active_sides.add(side)

            # ── Gesture hold tracking (per hand) ────────────────
            prev = gesture_hold.get(side)
            if prev and prev["gesture"] == gesture:
                held = now - prev["start"]
            else:
                gesture_hold[side] = {"gesture": gesture, "start": now}
                held = 0.0
            hold_ratio = min(1.0, held / cfg.GESTURE_HOLD_SECONDS)

            # ── Hold-to-trigger: OK Sign → auto screenshot ──────
            if ("OK Sign" in gesture and held >= cfg.GESTURE_HOLD_SECONDS
                    and (now - last_action_ts) > cfg.GESTURE_COOLDOWN):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(cfg.SCREENSHOT_DIR, f"okgesture_{ts}.png")
                cv2.imwrite(path, frame)
                last_action_ts = now
                print(f"[👌] OK Sign held {cfg.GESTURE_HOLD_SECONDS}s — auto screenshot: {path}")

            hand_data.append((side, gesture, hold_ratio))

            # ── Pinch-to-zoom: thumb+index apart, other 3 fingers curled ──
            if not middle_ring_pinky_up(up):
                zoom_target = pinch_zoom_factor(hand_lms, w, h, cfg)
                zoom_current += (zoom_target - zoom_current) * cfg.ZOOM_SMOOTHING

            # ── Swipe: open hand moving fast left/right ────────────────
            if gesture.strip().endswith("Open Hand"):
                wrist_x_norm = hand_lms[0].x  # already mirrored if mirror=True
                swipe_tracker.update(side, wrist_x_norm, now)

            dot_c  = cfg.HAND_R_DOT  if side == "Right" else cfg.HAND_L_DOT
            line_c = cfg.HAND_R_LINE if side == "Right" else cfg.HAND_L_LINE
            draw_hand(canvas, hand_lms, dot_c, line_c, w, h)

        # Drop hold-state for hands that left the frame
        for side in list(gesture_hold.keys()):
            if side not in active_sides:
                del gesture_hold[side]
        swipe_tracker.drop_missing(active_sides)

        # ── Resolve swipe trigger ────────────────────────────────────
        detected_swipe = swipe_tracker.detect(now)
        if detected_swipe:
            swipe_label = detected_swipe
            swipe_label_ts = now
            print(f"[↔] Swipe {detected_swipe} detected — next/prev slide action")
        if swipe_label and (now - swipe_label_ts) > cfg.SWIPE_DISPLAY_SEC:
            swipe_label = ""

        # ── Blend overlay onto frame ────────────────────────────
        frame = blend_overlay(canvas, frame, overlay_alpha)

        # ── Apply pinch-controlled digital zoom ─────────────────
        frame = apply_digital_zoom(frame, zoom_current)

        # ── FPS calculation ────────────────────────────────────
        fps_times.append(time.perf_counter())
        if len(fps_times) >= 2:
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
        else:
            fps = 0.0

        # ── HUD ────────────────────────────────────────────────
        draw_hud(frame, fps, len(face_result.face_landmarks),
                 hand_data, blink_left, blink_right,
                 overlay_alpha, mirror, cfg)
        draw_keybinds(frame, cfg)
        draw_zoom_swipe_hud(frame, zoom_current, swipe_label, w, cfg)

        # ── Display ────────────────────────────────────────────
        cv2.imshow("Vision Tracker — MediaPipe", frame)

        # ── Key handling ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[→] Quit.")
            break

        elif key == ord("s"):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(cfg.SCREENSHOT_DIR, f"capture_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[📷] Screenshot saved: {path}")

        elif key == ord("m"):
            mirror = not mirror
            print(f"[↔] Mirror: {'ON' if mirror else 'OFF'}")

        elif key == ord("+") or key == ord("="):
            overlay_alpha = min(1.0, overlay_alpha + 0.05)

        elif key == ord("-"):
            overlay_alpha = max(0.1, overlay_alpha - 0.05)

    # ── Cleanup ────────────────────────────────────────────────
    cap.release()
    face_det.close()
    hand_det.close()
    cv2.destroyAllWindows()
    print("Resources released. Goodbye.")


if __name__ == "__main__":
    main()
