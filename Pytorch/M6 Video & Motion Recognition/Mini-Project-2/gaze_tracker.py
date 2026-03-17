import cv2
import time
import os
import mediapipe as mp

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task")),
    running_mode = VisionRunningMode.VIDEO,
    num_faces    = 1
)

landmarker = FaceLandmarker.create_from_options(options)

# Iris landmark indices
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473

# Eye corner indices for head movement removal
R_EYE_INNER = 362
R_EYE_OUTER = 263
L_EYE_INNER = 133
L_EYE_OUTER = 33

# Smoothing — averages last N gaze positions to reduce jitter
SMOOTH_WINDOW = 8
gaze_history  = []


def reset_smooth():
    global gaze_history
    gaze_history = []


def get_iris_offset(landmarks):
    """
    Returns normalized iris offset relative to eye corners.

    Raw iris position moves with both head AND eyes.
    Eye corner position moves with head only.
    Subtracting eye center from iris position removes head movement —
    leaving only pure eyeball rotation.

    offset_x > 0 → looking right
    offset_x < 0 → looking left
    offset_y > 0 → looking down
    offset_y < 0 → looking up
    """

    r_iris  = landmarks[RIGHT_IRIS_CENTER]
    l_iris  = landmarks[LEFT_IRIS_CENTER]

    r_inner = landmarks[R_EYE_INNER]
    r_outer = landmarks[R_EYE_OUTER]
    l_inner = landmarks[L_EYE_INNER]
    l_outer = landmarks[L_EYE_OUTER]

    # Eye center = midpoint of inner and outer corners
    r_eye_cx = (r_inner.x + r_outer.x) / 2
    r_eye_cy = (r_inner.y + r_outer.y) / 2
    l_eye_cx = (l_inner.x + l_outer.x) / 2
    l_eye_cy = (l_inner.y + l_outer.y) / 2

    # Eye width for normalization — removes scale differences
    r_eye_w = abs(r_outer.x - r_inner.x)
    l_eye_w = abs(l_outer.x - l_inner.x)

    if r_eye_w < 1e-6 or l_eye_w < 1e-6:
        return None

    # Offset = iris minus eye center, normalized by eye width
    r_offset_x = (r_iris.x - r_eye_cx) / r_eye_w
    r_offset_y = (r_iris.y - r_eye_cy) / r_eye_w
    l_offset_x = (l_iris.x - l_eye_cx) / l_eye_w
    l_offset_y = (l_iris.y - l_eye_cy) / l_eye_w

    # Average both eyes for stability
    return ((r_offset_x + l_offset_x) / 2,
            (r_offset_y + l_offset_y) / 2)


def get_gaze_point(frame, calibration=None):
    """
    Returns smoothed screen position of gaze.

    Without calibration → returns raw normalized offset as rough screen position.
    With calibration    → maps offset to accurate screen coordinates.
    Returns None if no face detected.
    """

    global gaze_history

    h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if not result.face_landmarks:
        gaze_history = []
        return None

    landmarks = result.face_landmarks[0]
    offset    = get_iris_offset(landmarks)

    if offset is None:
        return None

    if calibration is not None:
        # Map offset to screen coordinates using calibration data
        ox, oy = offset
        sx = int(_interp(ox,
                         calibration["offset_x_min"], calibration["offset_x_max"],
                         0, w))
        sy = int(_interp(oy,
                         calibration["offset_y_min"], calibration["offset_y_max"],
                         0, h))
    else:
        # Fallback — rough mapping without calibration
        sx = int(w * 0.5 + offset[0] * w * 3)
        sy = int(h * 0.5 + offset[1] * h * 3)

    sx = max(0, min(w, sx))
    sy = max(0, min(h, sy))

    # Smooth over history
    gaze_history.append((sx, sy))
    if len(gaze_history) > SMOOTH_WINDOW:
        gaze_history.pop(0)

    smoothed_x = int(sum(p[0] for p in gaze_history) / len(gaze_history))
    smoothed_y = int(sum(p[1] for p in gaze_history) / len(gaze_history))

    return (smoothed_x, smoothed_y)


def _interp(value, in_min, in_max, out_min, out_max):
    """Linear interpolation — maps value from input range to output range."""
    if in_max == in_min:
        return (out_min + out_max) / 2
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


def get_corner_zone(gaze, frame_w, frame_h, zone_size=0.25):
    """
    Maps gaze point to one of four corner zones or None if not in any corner.
    zone_size controls how large each corner zone is as a fraction of frame size.
    Returns one of: 'TOP_LEFT', 'TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', None
    """

    if gaze is None:
        return None

    x, y   = gaze
    zone_w = int(frame_w * zone_size)
    zone_h = int(frame_h * zone_size)

    in_left   = x < zone_w
    in_right  = x > frame_w - zone_w
    in_top    = y < zone_h
    in_bottom = y > frame_h - zone_h

    if in_top    and in_left:  return "TOP_LEFT"
    if in_top    and in_right: return "TOP_RIGHT"
    if in_bottom and in_left:  return "BOTTOM_LEFT"
    if in_bottom and in_right: return "BOTTOM_RIGHT"

    return None


def draw_gaze(frame, gaze):
    """Draws a small dot at the estimated gaze position."""
    if gaze is not None:
        cv2.circle(frame, gaze, 8,  (0, 200, 255), -1)
        cv2.circle(frame, gaze, 10, (255, 255, 255), 1)
    return frame
