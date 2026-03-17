import cv2
import time
import os
import numpy as np
import mediapipe as mp
from face_recognizer import extract_fingerprint, save_profile, generate_user_id

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections

TOTAL_LANDMARKS = 468
COLLECT_FRAMES  = 30

WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)

STATE_CONFIRM = "CONFIRM"
STATE_NAME    = "NAME"
STATE_SCAN    = "SCAN"
STATE_DONE    = "DONE"

# Head pose zones — each must reach MIN_ZONE_LANDMARKS confirmed landmarks
# Nose tip = landmark 4, nose bridge = landmark 6
# Offset thresholds are normalized (0-1 range)
POSE_ZONES = {
    "front"  : {"label": "Front",       "confirmed": set()},
    "left"   : {"label": "Turn Left",   "confirmed": set()},
    "right"  : {"label": "Turn Right",  "confirmed": set()},
    "up"     : {"label": "Tilt Up",     "confirmed": set()},
}

# Minimum unique landmarks per zone to consider it complete
MIN_ZONE_LANDMARKS = 200

# Nose offset thresholds for pose detection
SIDE_THRESHOLD = 0.04    # nose x offset from center to count as left/right
UP_THRESHOLD   = 0.03    # nose y offset from center to count as up


class Registrar:

    def __init__(self):
        self.reset()

    def reset(self):
        self.state    = STATE_CONFIRM
        self.name     = ""
        self.user_id  = None

        # Per-zone confirmed landmark sets
        self.zone_landmarks = {
            "front" : set(),
            "left"  : set(),
            "right" : set(),
            "up"    : set(),
        }

        self.fingerprints    = []
        self.done            = False
        self.cancelled       = False
        self.active_zone     = None
        self.dwell_start     = None
        self.DWELL_TIME      = 0.5


    # ── Head pose detection ─────────────────────────────────────────────

    def get_head_pose(self, landmarks):
        """
        Returns current head pose zone based on nose position
        relative to face center.

        Uses midpoint between left and right cheek as face center reference.
        """

        nose    = landmarks[4]
        l_cheek = landmarks[234]
        r_cheek = landmarks[454]

        face_center_x = (l_cheek.x + r_cheek.x) / 2
        face_center_y = (l_cheek.y + r_cheek.y) / 2

        offset_x = nose.x - face_center_x
        offset_y = nose.y - face_center_y

        if offset_x < -SIDE_THRESHOLD:
            return "left"
        elif offset_x > SIDE_THRESHOLD:
            return "right"
        elif offset_y < -UP_THRESHOLD:
            return "up"
        else:
            return "front"


    # ── Dwell helpers ───────────────────────────────────────────────────

    def get_dwell_progress(self):
        if self.dwell_start is None:
            return 0.0
        return min((time.time() - self.dwell_start) / self.DWELL_TIME, 1.0)

    def get_zone_color(self, progress):
        return (0, int(255 * progress), int(255 * (1 - progress)))

    def is_inside(self, point, coords):
        if point is None or coords is None:
            return False
        x, y = point
        x1, y1, x2, y2 = coords
        return x1 < x < x2 and y1 < y < y2

    def draw_dwell_box(self, frame, coords, label, progress):
        x1, y1, x2, y2 = coords
        color = self.get_zone_color(progress)
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


    # ── State: CONFIRM ──────────────────────────────────────────────────

    def draw_confirm(self, frame, fingertip):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "New user detected!", (w//2 - 180, h//2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        box_w, box_h = 160, 60
        pad          = 30
        center_y     = h // 2

        yes_coords = (w//2 - box_w - pad, center_y, w//2 - pad, center_y + box_h)
        no_coords  = (w//2 + pad, center_y, w//2 + box_w + pad, center_y + box_h)

        for coords, label, action in [(yes_coords, "YES", "YES"), (no_coords, "NO", "NO")]:
            in_zone = self.is_inside(fingertip, coords)
            if in_zone:
                if self.active_zone != action:
                    self.active_zone = action
                    self.dwell_start = time.time()
                progress = self.get_dwell_progress()
                frame    = self.draw_dwell_box(frame, coords, label, progress)
                if progress >= 1.0:
                    if action == "YES":
                        self.state = STATE_NAME
                    else:
                        self.cancelled = True
                    self.active_zone = None
                    self.dwell_start = None
            else:
                if self.active_zone == action:
                    self.active_zone = None
                    self.dwell_start = None
                frame = self.draw_dwell_box(frame, coords, label, 0.0)

        return frame


    # ── State: NAME ─────────────────────────────────────────────────────

    def handle_key(self, key):
        if self.state != STATE_NAME:
            return
        if key == 13:
            if len(self.name.strip()) > 0:
                self.user_id = generate_user_id()
                self.state   = STATE_SCAN
        elif key == 8:
            self.name = self.name[:-1]
        elif 32 <= key <= 126:
            self.name += chr(key)

    def draw_name_input(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "Enter your name:", (w//2 - 180, h//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        bx1, by1 = w//2 - 200, h//2 - 10
        bx2, by2 = w//2 + 200, h//2 + 55
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), WHITE, 2)
        cv2.putText(frame, self.name + "|", (bx1 + 15, by2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
        cv2.putText(frame, "Press Enter to confirm", (w//2 - 180, h//2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)
        return frame


    # ── State: SCAN ─────────────────────────────────────────────────────

    def update_scan(self, frame, landmarks):
        h, w = frame.shape[:2]

        if landmarks is not None:
            pose = self.get_head_pose(landmarks)

            # Add visible landmarks to current pose zone
            for i, lm in enumerate(landmarks):
                if 0.0 < lm.x < 1.0 and 0.0 < lm.y < 1.0:
                    self.zone_landmarks[pose].add(i)

            # Collect fingerprint
            fp = extract_fingerprint(landmarks)
            if fp is not None:
                self.fingerprints.append(fp)
                if len(self.fingerprints) > COLLECT_FRAMES * 3:
                    self.fingerprints.pop(0)

            # Draw mesh — green if landmark confirmed in ANY zone, white otherwise
            all_confirmed = set().union(*self.zone_landmarks.values())
            for connection in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                start_idx = connection.start
                end_idx   = connection.end
                start     = landmarks[start_idx]
                end       = landmarks[end_idx]
                both_confirmed = (start_idx in all_confirmed and
                                  end_idx   in all_confirmed)
                color  = GREEN if both_confirmed else WHITE
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w),   int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)

            # Show current pose
            pose_label = POSE_ZONES[pose]["label"]
            cv2.putText(frame, f"Pose: {pose_label}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN if True else WHITE, 1)

        # Zone progress bars
        bar_x = w - 220
        cv2.putText(frame, "Coverage:", (bar_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        zones_done = 0
        for i, (zone_key, zone_info) in enumerate(POSE_ZONES.items()):
            count    = len(self.zone_landmarks[zone_key])
            progress = min(count / MIN_ZONE_LANDMARKS, 1.0)
            done     = progress >= 1.0
            if done:
                zones_done += 1

            bar_y  = 45 + i * 35
            bar_w  = 180
            filled = int(bar_w * progress)
            color  = GREEN if done else YELLOW

            cv2.putText(frame, zone_info["label"], (bar_x, bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(frame, (bar_x, bar_y + 5), (bar_x + bar_w, bar_y + 18), WHITE, 1)
            cv2.rectangle(frame, (bar_x, bar_y + 5), (bar_x + filled, bar_y + 18), color, -1)

        cv2.putText(frame, "Tilt your head in all directions", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)

        # Complete when all zones done
        if zones_done == len(POSE_ZONES):
            self._complete_registration()

        return frame

    def _complete_registration(self):
        if len(self.fingerprints) == 0:
            return
        avg_fingerprint = np.mean(self.fingerprints, axis=0)
        save_profile(self.user_id, self.name.strip(), avg_fingerprint)
        self.state = STATE_DONE
        self.done  = True


    # ── Main update ─────────────────────────────────────────────────────

    def update(self, frame, landmarks, fingertip, key):
        if self.state == STATE_CONFIRM:
            frame = self.draw_confirm(frame, fingertip)
        elif self.state == STATE_NAME:
            self.handle_key(key)
            frame = self.draw_name_input(frame)
        elif self.state == STATE_SCAN:
            frame = self.update_scan(frame, landmarks)
        return frame
