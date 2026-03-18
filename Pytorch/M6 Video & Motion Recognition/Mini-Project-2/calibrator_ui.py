# calibrator_ui.py
# All OpenCV drawing helpers for the Calibrator.
# Import this in calibrator.py — keeps UI separate from logic.

import cv2
import time

WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)

FLASH_SPEED = 0.3


def draw_target(frame, pos, active=False):
    """Draws target dot — flashes cyan when waiting, solid green when gaze confirmed."""
    if active:
        color = GREEN
    else:
        t       = time.time()
        visible = int(t / FLASH_SPEED) % 2 == 0
        color   = CYAN if visible else WHITE

    cv2.circle(frame, pos, 20, color, -1)
    cv2.circle(frame, pos, 24, WHITE, 2)
    cv2.circle(frame, pos, 5,  (0, 0, 0), -1)
    return frame


def draw_progress_bar(frame, progress, color=GREEN, y=50):
    bar_x1, bar_y1 = 20, y
    bar_x2, bar_y2 = 320, y + 20
    filled_x2      = int(bar_x1 + (bar_x2 - bar_x1) * progress)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), WHITE, 1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (filled_x2, bar_y2), color, -1)
    return frame


def draw_centered_text(frame, lines, start_y=None):
    h, w   = frame.shape[:2]
    line_h = 45
    y      = start_y if start_y else (h - len(lines) * line_h) // 2
    for i, (text, color, scale) in enumerate(lines):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        x         = (w - text_size[0]) // 2
        cv2.putText(frame, text, (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return frame


def draw_dwell_box(frame, coords, label, progress):
    x1, y1, x2, y2 = coords
    r, g = int(255 * (1 - progress)), int(255 * progress)
    color = (0, g, r)
    if progress > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 + int(progress * 3))
    fs        = 0.7
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
    tx        = x1 + (x2 - x1 - text_size[0]) // 2
    ty        = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)
    return frame