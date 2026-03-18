# gaze_tracker.py
# Hybrid gaze tracking:
#   Horizontal — head yaw + iris X offset (50/50)
#   Vertical   — head pitch2 only (iris Y too noisy)

import cv2
import numpy as np

WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)

# Head pose landmarks
NOSE_TIP = 4
FOREHEAD = 10
CHIN     = 152
L_CHEEK  = 234
R_CHEEK  = 454

# Iris + eye landmarks
L_IRIS       = 468
L_EYE_LEFT   = 33
L_EYE_RIGHT  = 133
L_EYE_TOP    = 159
L_EYE_BOTTOM = 145
R_IRIS       = 473
R_EYE_LEFT   = 362
R_EYE_RIGHT  = 263
R_EYE_TOP    = 386
R_EYE_BOTTOM = 374

# Blend weights
HEAD_WEIGHT = 0.5
IRIS_WEIGHT = 0.5

# Smoothing
SMOOTH_WINDOW = 8
yaw_history   = []
pitch_history = []


def reset_smooth():
    global yaw_history, pitch_history
    yaw_history   = []
    pitch_history = []


def get_head_pose(landmarks):
    """
    Returns (yaw, pitch2) from head geometry.
    Yaw    — nose horizontal offset from face center / face width
    Pitch2 — nose_to_chin / forehead_to_nose ratio
    """
    if landmarks is None:
        return None, None

    nose    = landmarks[NOSE_TIP]
    fore    = landmarks[FOREHEAD]
    chin    = landmarks[CHIN]
    l_cheek = landmarks[L_CHEEK]
    r_cheek = landmarks[R_CHEEK]

    face_center_x = (l_cheek.x + r_cheek.x) / 2
    face_width    =  r_cheek.x - l_cheek.x

    if face_width < 1e-6:
        return None, None

    yaw = (nose.x - face_center_x) / face_width

    nose_to_chin     = abs(nose.y - chin.y)
    forehead_to_nose = abs(fore.y - nose.y)

    if forehead_to_nose < 1e-6:
        return float(yaw), None

    pitch = nose_to_chin / forehead_to_nose

    return float(yaw), float(pitch)


def get_iris_offset(landmarks):
    """
    Returns average iris X offset across both eyes.
    Normalized to eye width, centered at 0.0.
    Returns None if landmarks unavailable.
    """
    if landmarks is None:
        return None

    def eye_x_offset(iris_idx, left_idx, right_idx):
        iris  = landmarks[iris_idx]
        left  = landmarks[left_idx]
        right = landmarks[right_idx]
        eye_w = abs(right.x - left.x)
        if eye_w < 1e-6:
            return None
        return ((iris.x - left.x) / eye_w) - 0.5

    l_offset = eye_x_offset(L_IRIS, L_EYE_LEFT, L_EYE_RIGHT)
    r_offset = eye_x_offset(R_IRIS, R_EYE_LEFT, R_EYE_RIGHT)

    if l_offset is None and r_offset is None:
        return None
    if l_offset is None:
        return r_offset
    if r_offset is None:
        return l_offset

    return (l_offset + r_offset) / 2.0


def get_gaze(landmarks, w, h, calibration=None):
    """
    Returns smoothed (yaw, pitch).
    Yaw   — head yaw + iris X blended
    Pitch — head pitch2 only
    """
    global yaw_history, pitch_history

    head_yaw, pitch = get_head_pose(landmarks)
    iris_x          = get_iris_offset(landmarks)

    # Blend yaw — head + iris
    if head_yaw is not None and iris_x is not None:
        yaw = head_yaw * HEAD_WEIGHT + iris_x * IRIS_WEIGHT
    elif head_yaw is not None:
        yaw = head_yaw
    else:
        yaw = None

    # Smooth yaw
    if yaw is not None:
        yaw_history.append(yaw)
        if len(yaw_history) > SMOOTH_WINDOW:
            yaw_history.pop(0)
        yaw = sum(yaw_history) / len(yaw_history)

    # Smooth pitch
    if pitch is not None:
        pitch_history.append(pitch)
        if len(pitch_history) > SMOOTH_WINDOW:
            pitch_history.pop(0)
        pitch = sum(pitch_history) / len(pitch_history)

    return yaw, pitch


def get_corner_zone(yaw, pitch, calibration):
    if yaw is None or pitch is None or calibration is None:
        return None

    yaw_left   = calibration.get("yaw_left")
    yaw_right  = calibration.get("yaw_right")
    pitch_up   = calibration.get("pitch_up")
    pitch_down = calibration.get("pitch_down")

    if any(v is None for v in [yaw_left, yaw_right, pitch_up, pitch_down]):
        return None

    is_left  = yaw   < yaw_left
    is_right = yaw   > yaw_right
    is_up    = pitch > pitch_up    # pitch2 higher = looking up
    is_down  = pitch < pitch_down  # pitch2 lower  = looking down

    if is_left  and is_up:   return "TOP_LEFT"
    if is_right and is_up:   return "TOP_RIGHT"
    if is_left  and is_down: return "BOTTOM_LEFT"
    if is_right and is_down: return "BOTTOM_RIGHT"

    return None


# gaze_tracker.py → draw_gaze_debug()
def draw_gaze_debug(frame, yaw, pitch, calibration):
    h, w = frame.shape[:2]

    if yaw is not None:
        cv2.putText(frame, f"Yaw:   {yaw:.3f}", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    if pitch is not None:
        cv2.putText(frame, f"Pitch: {pitch:.3f}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    zone = get_corner_zone(yaw, pitch, calibration)
    if zone is not None:
        cv2.putText(frame, f"Zone: {zone}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)

    # Gaze circle
    cx, cy = None, None
    if yaw is not None and pitch is not None:
        if calibration is not None:
            cx = int(np.interp(yaw,
                               [calibration["yaw_left"],  calibration["yaw_right"]],
                               [0, w]))
            cy = int(np.interp(pitch,
                               [calibration["pitch_down"], calibration["pitch_up"]],
                               [h, 0]))
        else:
            cx = int(w * 0.5 + yaw   * w * 2.0)
            cy = int(h * 0.5 - (pitch - 1.2) * h * 2.0)

        cx = max(20, min(w - 20, cx))
        cy = max(20, min(h - 20, cy))

        color = GREEN if zone is not None else YELLOW
        cv2.circle(frame, (cx, cy), 18, color, 2)
        cv2.circle(frame, (cx, cy), 4,  color, -1)
        cv2.line(frame, (cx - 25, cy), (cx + 25, cy), color, 1)
        cv2.line(frame, (cx, cy - 25), (cx, cy + 25), color, 1)

    return frame, (cx, cy)  # ← return gaze point