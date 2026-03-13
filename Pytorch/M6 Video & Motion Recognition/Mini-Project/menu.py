import cv2
import time
import numpy as np

# Dwell configuration
DWELL_TIME = 0.5   # seconds to hold finger on zone before triggering

# Menu trigger box — top right corner
TRIGGER_BOX = {
    "label"  : "MENU",
    "coords" : None,   # set dynamically based on frame size
    "action" : "MENU"
}

# Menu function boxes — shown at center when menu is open
MENU_ITEMS = [
    {"label": "Object Detection",    "action": "OBJECT_DETECTION"},
    {"label": "Action Recognition",  "action": "ACTION_RECOGNITION"},
    {"label": "3D Face Mesh",        "action": "FACE_MESH"},
    {"label": "Motion Trail",        "action": "MOTION_TRAIL"},
    {"label": "Exit",                "action": "EXIT"},
]

# Back box — shown in top right when a function is running
BACK_BOX = {
    "label"  : "BACK",
    "action" : "MENU"
}

# Dwell state
active_zone  = None
dwell_start  = None


def get_dwell_progress():
    """Returns dwell progress from 0.0 to 1.0."""
    if dwell_start is None:
        return 0.0
    elapsed  = time.time() - dwell_start
    return min(elapsed / DWELL_TIME, 1.0)


def get_zone_color(progress):
    """
    Interpolates box color from red → yellow → green based on dwell progress.

    progress = 0.0  →  red    (0,   0,   255) in BGR
    progress = 0.5  →  yellow (0,   255, 255) in BGR
    progress = 1.0  →  green  (0,   255, 0  ) in BGR
    """
    r = int(255 * (1 - progress))
    g = int(255 * progress)
    return (0, g, r)


def is_inside(point, box_coords):
    """Returns True if point (x,y) is inside box (x1,y1,x2,y2)."""
    if point is None or box_coords is None:
        return False
    x, y = point
    x1, y1, x2, y2 = box_coords
    return x1 < x < x2 and y1 < y < y2


def draw_zone(frame, coords, label, progress):
    """
    Draws a single interactive zone box with:
    - dynamic color based on dwell progress
    - label centered inside
    - filled background at low opacity when active
    """

    x1, y1, x2, y2 = coords
    color = get_zone_color(progress)

    # Filled background when hovering
    if progress > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    # Border
    thickness = 2 + int(progress * 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label
    font_scale = 0.6
    text_size  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    text_x     = x1 + (x2 - x1 - text_size[0]) // 2
    text_y     = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return frame


def build_menu_coords(frame_w, frame_h):
    """
    Builds centered menu box coordinates based on frame size.
    Returns list of (coords, label, action) tuples.
    """

    num_items  = len(MENU_ITEMS)
    box_w      = 280
    box_h      = 60
    padding    = 20
    total_h    = num_items * box_h + (num_items - 1) * padding
    start_x    = (frame_w - box_w) // 2
    start_y    = (frame_h - total_h) // 2

    zones = []
    for i, item in enumerate(MENU_ITEMS):
        x1 = start_x
        y1 = start_y + i * (box_h + padding)
        x2 = x1 + box_w
        y2 = y1 + box_h
        zones.append(((x1, y1, x2, y2), item["label"], item["action"]))

    return zones


def build_trigger_coords(frame_w, frame_h):
    """Returns top-right corner trigger box coordinates."""
    margin = 15
    box_w  = 100
    box_h  = 45
    x1     = frame_w - box_w - margin
    y1     = margin
    x2     = frame_w - margin
    y2     = y1 + box_h
    return (x1, y1, x2, y2)


def build_back_coords(frame_w, frame_h):
    """Returns top-right corner back box coordinates."""
    return build_trigger_coords(frame_w, frame_h)


def update(frame, fingertip, zones):
    """
    Core menu update — checks fingertip against all zones,
    updates dwell timer, returns triggered action or None.

    zones: list of (coords, label, action) tuples
    """

    global active_zone, dwell_start

    triggered = None

    for coords, label, action in zones:

        in_zone = is_inside(fingertip, coords)

        if in_zone:
            if active_zone != action:
                # Finger just entered this zone — reset timer
                active_zone = action
                dwell_start = time.time()

            progress = get_dwell_progress()
            frame    = draw_zone(frame, coords, label, progress)

            if progress >= 1.0:
                triggered   = action
                active_zone = None
                dwell_start = None

        else:
            # Not in this zone — draw at rest state
            if active_zone == action:
                active_zone = None
                dwell_start = None
            frame = draw_zone(frame, coords, label, 0.0)

    return frame, triggered


def draw_menu(frame, fingertip):
    """
    Draws full menu overlay with all function boxes.
    Returns triggered action or None.
    """

    h, w = frame.shape[:2]
    zones = build_menu_coords(w, h)

    # Dim background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    frame, triggered = update(frame, fingertip, zones)
    return frame, triggered


def draw_trigger(frame, fingertip):
    """
    Draws the always-visible menu trigger box in top-right corner.
    Returns 'MENU' if triggered, else None.
    """

    h, w   = frame.shape[:2]
    coords = build_trigger_coords(w, h)
    zones  = [(coords, "MENU", "MENU")]

    frame, triggered = update(frame, fingertip, zones)
    return frame, triggered


def draw_back(frame, fingertip):
    """
    Draws the always-visible back box when a function is running.
    Returns 'MENU' if triggered, else None.
    """

    h, w   = frame.shape[:2]
    coords = build_back_coords(w, h)
    zones  = [(coords, "BACK", "MENU")]

    frame, triggered = update(frame, fingertip, zones)
    return frame, triggered
