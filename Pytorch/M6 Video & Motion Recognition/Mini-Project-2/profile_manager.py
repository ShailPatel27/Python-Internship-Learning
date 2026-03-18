import cv2
import time
import os
import numpy as np
from face_recognizer import update_name, delete_profile, process, draw_mesh

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)

STATE_MAIN           = "MAIN"
STATE_RENAME         = "RENAME"
STATE_CONFIRM_DELETE = "CONFIRM_DELETE"
STATE_VERIFY_FACE    = "VERIFY_FACE"

DWELL_TIME        = 0.5
DELETE_DWELL_TIME = 2.0


class ProfileManager:

    def __init__(self, user_id, name):
        self.user_id  = user_id
        self.name     = name
        self.state    = STATE_MAIN
        self.done     = False
        self.deleted  = False
        self.new_name = ""

        self.active_zone = None
        self.dwell_start = None

        # Face verify state
        self.verify_message       = ""
        self.verify_message_timer = 0


    # ── Dwell helpers ───────────────────────────────────────────────────

    def get_dwell_progress(self, dwell_time=DWELL_TIME):
        if self.dwell_start is None:
            return 0.0
        return min((time.time() - self.dwell_start) / dwell_time, 1.0)

    def get_zone_color(self, progress):
        return (0, int(255 * progress), int(255 * (1 - progress)))

    def is_inside(self, point, coords):
        if point is None or coords is None:
            return False
        x, y = point
        x1, y1, x2, y2 = coords
        return x1 < x < x2 and y1 < y < y2

    def draw_dwell_box(self, frame, coords, label, progress, dwell_time=DWELL_TIME):
        x1, y1, x2, y2 = coords
        color = self.get_zone_color(progress)

        if progress > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 + int(progress * 3))

        # Progress arc for delete button
        if dwell_time == DELETE_DWELL_TIME and progress > 0:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(frame, (cx, cy), (30, 30), 0,
                        -90, -90 + int(360 * progress), color, 3)

        fs        = 0.7
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
        tx        = x1 + (x2 - x1 - text_size[0]) // 2
        ty        = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)
        return frame

    def update_zone(self, frame, fingertip, coords, label, action, dwell_time=DWELL_TIME):
        in_zone   = self.is_inside(fingertip, coords)
        triggered = False

        if in_zone:
            if self.active_zone != action:
                self.active_zone = action
                self.dwell_start = time.time()
            progress = self.get_dwell_progress(dwell_time)
            frame    = self.draw_dwell_box(frame, coords, label, progress, dwell_time)
            if progress >= 1.0:
                triggered        = True
                self.active_zone = None
                self.dwell_start = None
        else:
            if self.active_zone == action:
                self.active_zone = None
                self.dwell_start = None
            frame = self.draw_dwell_box(frame, coords, label, 0.0, dwell_time)

        return frame, triggered


    # ── State: MAIN ─────────────────────────────────────────────────────

    def draw_main(self, frame, fingertip):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Profile: {self.name}", (w//2 - 200, h//2 - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, f"ID: {self.user_id}", (w//2 - 200, h//2 - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

        box_w, box_h = 200, 60
        pad          = 25
        row_y        = h//2 - 30

        rename_coords = (w//2 - box_w - pad, row_y, w//2 - pad, row_y + box_h)
        delete_coords = (w//2 + pad, row_y, w//2 + box_w + pad, row_y + box_h)
        back_coords   = (w//2 - 100, row_y + box_h + pad,
                         w//2 + 100, row_y + box_h + pad + box_h)

        frame, rename_t = self.update_zone(frame, fingertip, rename_coords, "Change Name", "RENAME")
        frame, delete_t = self.update_zone(frame, fingertip, delete_coords, "Delete Account",
                                           "DELETE", dwell_time=DELETE_DWELL_TIME)
        frame, back_t   = self.update_zone(frame, fingertip, back_coords, "Back", "BACK")

        if rename_t:
            self.state    = STATE_RENAME
            self.new_name = ""
        elif delete_t:
            self.state = STATE_CONFIRM_DELETE
        elif back_t:
            self.done = True

        return frame


    # ── State: RENAME ───────────────────────────────────────────────────

    def handle_key(self, key):
        if self.state != STATE_RENAME:
            return
        if key == 13:
            if len(self.new_name.strip()) > 0:
                update_name(self.user_id, self.new_name.strip())
                self.name  = self.new_name.strip()
                self.state = STATE_MAIN
        elif key == 8:
            self.new_name = self.new_name[:-1]
        elif 32 <= key <= 126:
            self.new_name += chr(key)

    def draw_rename(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "Enter new name:", (w//2 - 180, h//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        bx1, by1 = w//2 - 200, h//2 - 10
        bx2, by2 = w//2 + 200, h//2 + 55
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), WHITE, 2)
        cv2.putText(frame, self.new_name + "|", (bx1 + 15, by2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
        cv2.putText(frame, "Press Enter to confirm", (w//2 - 180, h//2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)
        return frame


    # ── State: CONFIRM DELETE ───────────────────────────────────────────

    def draw_confirm_delete(self, frame, fingertip):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Delete account?", (w//2 - 180, h//2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
        cv2.putText(frame, "Face verification required", (w//2 - 210, h//2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 1)

        box_w, box_h = 160, 60
        pad          = 30
        row_y        = h // 2

        confirm_coords = (w//2 - box_w - pad, row_y, w//2 - pad, row_y + box_h)
        cancel_coords  = (w//2 + pad, row_y, w//2 + box_w + pad, row_y + box_h)

        frame, confirm_t = self.update_zone(frame, fingertip, confirm_coords,
                                            "Confirm", "CONFIRM", dwell_time=DELETE_DWELL_TIME)
        frame, cancel_t  = self.update_zone(frame, fingertip, cancel_coords, "Cancel", "CANCEL")

        if confirm_t:
            self.state = STATE_VERIFY_FACE
        elif cancel_t:
            self.state = STATE_MAIN

        return frame


    # ── State: VERIFY FACE ──────────────────────────────────────────────

    def draw_verify_face(self, frame, landmarks, verified_id, mesh_color):
        """
        Shows face verification screen before deletion.
        Runs face recognition — only deletes if recognized user matches account owner.
        """

        h, w = frame.shape[:2]

        # Draw mesh with recognition color
        frame = draw_mesh(frame, landmarks, mesh_color)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "Please face the camera to verify identity",
                    (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Show feedback message if any
        if self.verify_message and self.verify_message_timer > 0:
            color = GREEN if "confirmed" in self.verify_message.lower() else RED
            cv2.putText(frame, self.verify_message, (w//2 - 200, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            self.verify_message_timer -= 1

        # Check if recognized user matches account owner
        if verified_id == self.user_id and self.verify_message_timer == 0 and not self.deleted:
            self.verify_message       = "Identity confirmed. Deleting..."
            self.verify_message_timer = 45
            delete_profile(self.user_id)
            self.deleted = True

        elif verified_id is not None and verified_id != self.user_id:
            self.verify_message       = "Face not recognized. Try again."
            self.verify_message_timer = 60
            self.state                = STATE_CONFIRM_DELETE

        return frame


    # ── Main update ─────────────────────────────────────────────────────

    def update(self, frame, fingertip, key, landmarks=None, verified_id=None, mesh_color=None):
        if self.state == STATE_MAIN:
            frame = self.draw_main(frame, fingertip)
        elif self.state == STATE_RENAME:
            self.handle_key(key)
            frame = self.draw_rename(frame)
        elif self.state == STATE_CONFIRM_DELETE:
            frame = self.draw_confirm_delete(frame, fingertip)
        elif self.state == STATE_VERIFY_FACE:
            frame = self.draw_verify_face(frame, landmarks, verified_id, mesh_color)
        return frame
