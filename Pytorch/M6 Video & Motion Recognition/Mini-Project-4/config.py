import cv2
import numpy as np

# ── screen size detection ──────────────────────────────────────────────────────

def get_screen_size():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return 1920, 1080

    cv2.namedWindow('GestureCanvas', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('GestureCanvas', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('GestureCanvas', frame)
    cv2.waitKey(1)

    w = cv2.getWindowImageRect('GestureCanvas')[2]
    h = cv2.getWindowImageRect('GestureCanvas')[3]
    cap.release()

    if w <= 0 or h <= 0:
        return 1920, 1080
    return w, h

SCREEN_W, SCREEN_H = get_screen_size()

# ── gesture thresholds (normalized 0.0–1.0) ────────────────────────────────────

PINCH_THRESHOLD         = 0.07   # thumb + index tip distance (increased for easier pinch)
SPREAD_THRESHOLD        = 0.08   # avg fingertip gap → scale mode
TOGETHER_THRESHOLD      = 0.08   # avg fingertip gap → erase mode (loosened)
SCALE_DELTA_THRESHOLD   = 0.01   # gap change per frame → is_scaling
ROTATE_DELTA_THRESHOLD  = 0.05   # angle change per frame (~3 deg) → is_rotating

# ── smoothing ──────────────────────────────────────────────────────────────────

# exponential blend applied to raw landmarks each frame
# 0.0 = fully laggy (never updates), 1.0 = raw unsmoothed
LANDMARK_SMOOTH_ALPHA   = 0.5

# rolling average window for pinch distance (frames)
PINCH_SMOOTH_WINDOW     = 5

# ── gesture confirmation ────────────────────────────────────────────────────────

# a gesture must appear this many consecutive frames before it is acted on
# applies to ALL gestures including idle — prevents flicker mid-stroke
GESTURE_CONFIRM_FRAMES  = 3

# ── shape proximity (normalized) ───────────────────────────────────────────────

GRAB_PROXIMITY          = 0.15   # how close hand must be to shape to interact
ERASE_PROXIMITY         = 0.15   # how close palm must be to shape to erase
PRECISE_ERASE_RADIUS    = 0.03   # thumb tip eraser radius
GRAB_BBOX_PADDING       = 0.03   # forgiveness when selecting a shape with fingertips
GRAB_PROXIMITY          = 0.03   # how close fingertips must be to shape bbox to trigger grab
CURL_THRESHOLD          = 1.3    # tip_to_wrist / mcp_to_wrist ratio for middle/ring/pinky

# ── canvas protection zone ─────────────────────────────────────────────────────
# normalized x boundary — no drawing allowed left of this (protects palette)
# set to width of 2 palette columns + padding
DRAW_EXCLUSION_X        = 0.15   # ~15% of screen width from left edge

# ── draw mode ──────────────────────────────────────────────────────────────────
# 'pinch' = thumb+index pinch draws | 'point' = index extended, others curled
DRAW_MODE               = 'pinch'

# ── dwell (seconds) ────────────────────────────────────────────────────────────

DWELL_TIME              = 1.0    # seconds to dwell over UI element to select

# ── dustbin (top right, as % of screen) ───────────────────────────────────────

DUSTBIN_X       = int(SCREEN_W * 0.92)
DUSTBIN_Y       = int(SCREEN_H * 0.05)
DUSTBIN_SIZE    = int(min(SCREEN_W, SCREEN_H) * 0.06)

# ── palette (left edge, as % of screen) ───────────────────────────────────────

PALETTE_X           = int(SCREEN_W * 0.01)
PALETTE_ITEM_SIZE   = int(min(SCREEN_W, SCREEN_H) * 0.05)
PALETTE_PADDING     = int(min(SCREEN_W, SCREEN_H) * 0.01)

# ── default colors (BGR) ───────────────────────────────────────────────────────

DEFAULT_COLORS = [
    (255, 255, 255),   # white
    (0,   0,   255),   # red
    (0,   165, 255),   # orange
    (0,   255, 255),   # yellow
    (0,   255, 0  ),   # green
    (255, 0,   0  ),   # blue
    (130, 0,   75 ),   # indigo
    (255, 0,   145),   # violet
    (0,   0,   0  ),   # black
]

# ── pen sizes (as % of screen diagonal) ───────────────────────────────────────

SCREEN_DIAG = (SCREEN_W**2 + SCREEN_H**2) ** 0.5

PEN_SIZES = [
    int(SCREEN_DIAG * 0.002),   # thin
    int(SCREEN_DIAG * 0.004),   # medium
    int(SCREEN_DIAG * 0.007),   # thick
]

# ── stroke styles ──────────────────────────────────────────────────────────────

STROKE_STYLES = ['solid', 'dashed', 'dotted', 'glow', 'chalk', 'tapered']
