import cv2
import time
import numpy as np
from collections import deque

# Constants
TRAIL_LENGTH    = 60       # maximum stored points
POINT_LIFETIME  = 2.0      # seconds before a point fully fades
TRAIL_COLOR     = (0, 255, 0)
MAX_THICKNESS   = 10
MIN_THICKNESS   = 2

# Module state — each entry is (x, y, timestamp)
trail = deque(maxlen=TRAIL_LENGTH)


def reset():
    global trail
    trail = deque(maxlen=TRAIL_LENGTH)


def interpolate_points(p1, p2, num_points=8):
    """Inserts extra points between p1 and p2 for a smoother line."""
    return [
        (
            int(p1[0] + (p2[0] - p1[0]) * t / num_points),
            int(p1[1] + (p2[1] - p1[1]) * t / num_points),
            p1[2] + (p2[2] - p1[2]) * t / num_points   # interpolate timestamp too
        )
        for t in range(num_points + 1)
    ]


def run(frame, fingertip=None):
    global trail

    now = time.time()

    # Add new point with timestamp
    if fingertip is not None:
        trail.append((fingertip[0], fingertip[1], now))

    # Drop expired points
    trail = deque(
        [p for p in trail if now - p[2] < POINT_LIFETIME],
        maxlen=TRAIL_LENGTH
    )

    trail_list = list(trail)

    if len(trail_list) < 2:
        return frame

    # Build dense interpolated point list
    dense = []
    for i in range(1, len(trail_list)):
        segment = interpolate_points(trail_list[i - 1], trail_list[i])
        dense.extend(segment)

    total = len(dense)

    for i in range(1, total):
        x, y, ts = dense[i]
        px, py, _ = dense[i - 1]

        # Age-based fade — older = more transparent
        age       = now - ts
        life      = max(0.0, 1.0 - (age / POINT_LIFETIME))

        # Position-based taper — tip is thicker
        pos_alpha = i / total

        color     = tuple(int(c * life) for c in TRAIL_COLOR)
        thickness = max(MIN_THICKNESS, int(MAX_THICKNESS * pos_alpha * life))

        cv2.line(frame, (px, py), (x, y), color, thickness)

    # Blob at current fingertip — fades with life too
    if trail_list:
        x, y, ts = trail_list[-1]
        life = max(0.0, 1.0 - ((now - ts) / POINT_LIFETIME))
        color = tuple(int(c * life) for c in TRAIL_COLOR)
        cv2.circle(frame, (x, y), MAX_THICKNESS // 2, color, -1)

    return frame
