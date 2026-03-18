import cv2
import time
import hand_tracker
from display import crop_to_fill
import gaze_tracker
import menu
import face_recognizer
from registrar import Registrar
from calibrator import Calibrator, load_calibration
from profile_manager import ProfileManager

from functions import object_detection
from functions import action_recognition
from functions import motion_trail

# Colors
WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)

# System states
STATE_RECOGNIZING = "RECOGNIZING"
STATE_WELCOME     = "WELCOME"
STATE_REGISTERING = "REGISTERING"
STATE_CALIBRATING = "CALIBRATING"
STATE_RUNNING     = "RUNNING"
STATE_PROFILE     = "PROFILE"

FUNCTION_MAP = {
    "OBJECT_DETECTION"   : object_detection,
    "ACTION_RECOGNITION" : action_recognition,
    "MOTION_TRAIL"       : motion_trail,
}

PROFILE_ACTION = "PROFILE"

WELCOME_DURATION = 2.0


def reset_functions():
    for module in FUNCTION_MAP.values():
        module.reset()
    gaze_tracker.reset_smooth()
    menu.close_menu()


# ── Main ────────────────────────────────────────────────────────────────

cap   = cv2.VideoCapture(0)
ret, frame = cap.read()

# Create fullscreen window first to get screen dimensions
cv2.namedWindow("ARIA", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("ARIA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get screen size from a temporary fullscreen display
cv2.imshow("ARIA", frame)
cv2.waitKey(1)
screen_w = cv2.getWindowImageRect("ARIA")[2]
screen_h = cv2.getWindowImageRect("ARIA")[3]

# Use screen dimensions as w, h throughout — all zones map to screen space
w, h = screen_w, screen_h

# System state
state        = STATE_RECOGNIZING
active_func  = None

# Per-session objects — recreated as needed
registrar    = None
calibrator   = None
profile_mgr  = None

# Current user
current_user_id   = None
current_user_name = None
calibration       = None

# Welcome state
welcome_start = None

# Profiles — loaded once, refreshed after registration/deletion
profiles = face_recognizer.load_profiles()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = crop_to_fill(frame, w, h)

    # Always get inputs
    # Always get inputs
    fingertip = hand_tracker.get_fingertip(frame)
    landmarks, user_id, name, mesh_color, fingerprint = face_recognizer.process(frame, profiles)
    yaw, pitch = gaze_tracker.get_gaze(landmarks, w, h, calibration)
    gaze_zone  = gaze_tracker.get_corner_zone(yaw, pitch, calibration)
    frame, gaze_point = gaze_tracker.draw_gaze_debug(frame, yaw, pitch, calibration)
    frame      = hand_tracker.draw_fingertip(frame, fingertip)

    # Key input
    key = cv2.waitKey(1) & 0xFF
    if key == 17:   # Ctrl+Q
        break


    # ── STATE: RECOGNIZING ───────────────────────────────────────────

    if state == STATE_RECOGNIZING:

        # landmarks, user_id, name, mesh_color, fingerprint already computed above
        frame = face_recognizer.draw_mesh(frame, landmarks, mesh_color)

        if user_id is not None:
            if mesh_color is None:
                current_user_id   = user_id
                current_user_name = name
                calibration       = load_calibration(user_id)
                welcome_start     = time.time()
                state             = STATE_WELCOME

        elif landmarks is not None and fingerprint is not None:
            registrar = Registrar()
            state     = STATE_REGISTERING

        else:
            cv2.putText(frame, "Please face the camera", (w//2 - 200, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)


    # ── STATE: WELCOME ───────────────────────────────────────────────

    elif state == STATE_WELCOME:

        elapsed = time.time() - welcome_start

        cv2.putText(frame, f"Welcome, {current_user_name}!",
                    (w//2 - 250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 2)

        if elapsed >= WELCOME_DURATION:
            if calibration is None:
                calibrator = Calibrator(current_user_id, w, h)
                state      = STATE_CALIBRATING
            else:
                state = STATE_RUNNING


    # ── STATE: REGISTERING ───────────────────────────────────────────

    elif state == STATE_REGISTERING:

        # landmarks already computed above — just draw white mesh
        frame = face_recognizer.draw_mesh(frame, landmarks, (255, 255, 255))
        frame = registrar.update(frame, landmarks, fingertip, key)

        if registrar.done:
            profiles          = face_recognizer.load_profiles()
            current_user_id   = registrar.user_id
            current_user_name = registrar.name.strip()
            calibrator        = Calibrator(current_user_id, w, h)
            face_recognizer.reset_flash()
            state             = STATE_CALIBRATING

        elif registrar.cancelled:
            face_recognizer.reset_flash()
            state = STATE_RECOGNIZING


    # ── STATE: CALIBRATING ───────────────────────────────────────────

    elif state == STATE_CALIBRATING:
        frame = calibrator.update(frame, landmarks, fingertip, yaw, pitch)
        if calibrator.done:
            calibration = load_calibration(current_user_id)
            state       = STATE_RUNNING


    # ── STATE: RUNNING ───────────────────────────────────────────────

    # main.py — replace STATE_RUNNING block entirely
    elif state == STATE_RUNNING:

        frame, triggered = menu.draw_corners(frame, fingertip, gaze_point)

        if menu.hand_menu_open:
            frame, selected = menu.update_hand_menu(frame, fingertip)
            if selected == "EXIT":
                break
            elif selected in FUNCTION_MAP:
                reset_functions()
                active_func         = selected
                menu.hand_menu_open = False
            elif selected == PROFILE_ACTION:
                profile_mgr         = ProfileManager(current_user_id, current_user_name)
                state               = STATE_PROFILE
                menu.hand_menu_open = False

        if triggered == "MENU":
            menu.hand_menu_open = True
        elif triggered == PROFILE_ACTION:
            profile_mgr         = ProfileManager(current_user_id, current_user_name)
            state               = STATE_PROFILE
            menu.hand_menu_open = False
        elif triggered in FUNCTION_MAP:
            reset_functions()
            active_func         = triggered
            menu.hand_menu_open = False

        if active_func is not None:
            if active_func == "MOTION_TRAIL":
                frame = motion_trail.run(frame, fingertip)
            else:
                frame = FUNCTION_MAP[active_func].run(frame)


    # ── STATE: PROFILE ───────────────────────────────────────────────

    elif state == STATE_PROFILE:

        # landmarks, verified_id, mesh_color already computed above
        frame = profile_mgr.update(
            frame, fingertip, key,
            landmarks=landmarks,
            verified_id=user_id,
            mesh_color=mesh_color
        )

        if profile_mgr.deleted:
            profiles          = face_recognizer.load_profiles()
            current_user_id   = None
            current_user_name = None
            calibration       = None
            face_recognizer.reset_flash()
            state             = STATE_RECOGNIZING

        elif profile_mgr.done:
            current_user_name = profile_mgr.name
            state             = STATE_RUNNING


    # ── Display ──────────────────────────────────────────────────────

    cv2.imshow("ARIA", frame)


cap.release()
cv2.destroyAllWindows()
