import cv2
import time

HAND_DWELL_TIME = 0.5
GAZE_DWELL_TIME = 1.0

CORNER_ZONES = {
    "TOP_LEFT"     : {"label": "Object Detection",   "action": "OBJECT_DETECTION"},
    "TOP_RIGHT"    : {"label": "Motion Trail",        "action": "MOTION_TRAIL"},
    "BOTTOM_LEFT"  : {"label": "Action Recognition",  "action": "ACTION_RECOGNITION"},
    "BOTTOM_RIGHT" : {"label": "Profile",             "action": "PROFILE"},
}

HAND_MENU_ITEMS = [
    {"label": "Object Detection",   "action": "OBJECT_DETECTION"},
    {"label": "Action Recognition", "action": "ACTION_RECOGNITION"},
    {"label": "Motion Trail",       "action": "MOTION_TRAIL"},
    {"label": "Profile",            "action": "PROFILE"},
    {"label": "Exit",               "action": "EXIT"},
]

hand_active_zone = None
hand_dwell_start = None
gaze_active_zone = None
gaze_dwell_start = None
hand_menu_open   = False


def get_dwell_progress(dwell_start, dwell_time):
    if dwell_start is None:
        return 0.0
    return min((time.time() - dwell_start) / dwell_time, 1.0)


def get_zone_color(progress):
    return (0, int(255 * progress), int(255 * (1 - progress)))


def is_inside(point, coords):
    if point is None or coords is None:
        return False
    x, y = point
    x1, y1, x2, y2 = coords
    return x1 < x < x2 and y1 < y < y2


def draw_zone(frame, coords, label, progress):
    x1, y1, x2, y2 = coords
    color = get_zone_color(progress)

    if progress > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 + int(progress * 3))

    fs        = 0.55
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
    tx        = x1 + (x2 - x1 - text_size[0]) // 2
    ty        = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)
    return frame


def build_corner_coords(frame_w, frame_h, zone_size=0.25):
    zone_w = int(frame_w * zone_size)
    zone_h = int(frame_h * zone_size)
    pad    = 15
    return {
        "TOP_LEFT"     : (pad,              pad,              zone_w,              zone_h),
        "TOP_RIGHT"    : (frame_w - zone_w, pad,              frame_w - pad,       zone_h),
        "BOTTOM_LEFT"  : (pad,              frame_h - zone_h, zone_w,              frame_h - pad),
        "BOTTOM_RIGHT" : (frame_w - zone_w, frame_h - zone_h, frame_w - pad,       frame_h - pad),
    }


def build_trigger_coords(frame_w, frame_h):
    margin = 15
    box_w  = 100
    box_h  = 45
    x1     = frame_w - box_w - margin
    y1     = margin
    return (x1, y1, x1 + box_w, y1 + box_h)


def build_hand_menu_coords(frame_w, frame_h):
    num_items = len(HAND_MENU_ITEMS)
    box_w, box_h, padding = 280, 60, 20
    total_h  = num_items * box_h + (num_items - 1) * padding
    start_x  = (frame_w - box_w) // 2
    start_y  = (frame_h - total_h) // 2
    zones    = []
    for i, item in enumerate(HAND_MENU_ITEMS):
        x1 = start_x
        y1 = start_y + i * (box_h + padding)
        zones.append(((x1, y1, x1 + box_w, y1 + box_h), item["label"], item["action"]))
    return zones


def update_hand_zone(frame, fingertip, zones, dwell_time):
    global hand_active_zone, hand_dwell_start
    triggered = None

    for coords, label, action in zones:
        in_zone = is_inside(fingertip, coords)
        if in_zone:
            if hand_active_zone != action:
                hand_active_zone = action
                hand_dwell_start = time.time()
            progress = get_dwell_progress(hand_dwell_start, dwell_time)
            frame    = draw_zone(frame, coords, label, progress)
            if progress >= 1.0:
                triggered        = action
                hand_active_zone = None
                hand_dwell_start = None
        else:
            if hand_active_zone == action:
                hand_active_zone = None
                hand_dwell_start = None
            frame = draw_zone(frame, coords, label, 0.0)

    return frame, triggered


def update_gaze_zone(frame, gaze_zone, corner_coords):
    global gaze_active_zone, gaze_dwell_start
    triggered = None

    for corner, zone_info in CORNER_ZONES.items():
        coords  = corner_coords[corner]
        label   = zone_info["label"]
        action  = zone_info["action"]
        in_zone = gaze_zone == corner

        if in_zone:
            if gaze_active_zone != corner:
                gaze_active_zone = corner
                gaze_dwell_start = time.time()
            progress = get_dwell_progress(gaze_dwell_start, GAZE_DWELL_TIME)
            frame    = draw_zone(frame, coords, label, progress)
            if progress >= 1.0:
                triggered        = action
                gaze_active_zone = None
                gaze_dwell_start = None
        else:
            if gaze_active_zone == corner:
                gaze_active_zone = None
                gaze_dwell_start = None
            frame = draw_zone(frame, coords, label, 0.0)

    return frame, triggered


def draw_gaze_corners(frame, gaze_zone):
    h, w          = frame.shape[:2]
    corner_coords = build_corner_coords(w, h)
    return update_gaze_zone(frame, gaze_zone, corner_coords)


def draw_trigger(frame, fingertip):
    h, w   = frame.shape[:2]
    coords = build_trigger_coords(w, h)
    return update_hand_zone(frame, fingertip, [(coords, "MENU", "MENU")], HAND_DWELL_TIME)


def draw_hand_menu(frame, fingertip):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    zones = build_hand_menu_coords(w, h)
    return update_hand_zone(frame, fingertip, zones, HAND_DWELL_TIME)
