import cv2
import time
import os
import json
import random
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

WHITE  = (255, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 255, 0)

# Calibration constants
COLLECT_DURATION  = 1.0    # seconds of gaze-confirmed data needed per corner
TIMEOUT_DURATION  = 10.0   # seconds before auto-advancing if no gaze detected
TEST_POINTS       = 3
TEST_DURATION     = 1.5
FLASH_SPEED       = 0.3

CORNERS = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]

CORNER_LABELS = {
    "TOP_LEFT"     : "Top Left",
    "TOP_RIGHT"    : "Top Right",
    "BOTTOM_LEFT"  : "Bottom Left",
    "BOTTOM_RIGHT" : "Bottom Right",
}

STATE_WAITING = "WAITING"
STATE_CORNER  = "CORNER"
STATE_TEST    = "TEST"
STATE_RESULT  = "RESULT"
STATE_DONE    = "DONE"


class Calibrator:

    def __init__(self, name, frame_w, frame_h):
        self.name     = name
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        self.reset()

    def reset(self):
        self.state           = STATE_WAITING
        self.corner_idx      = 0
        self.corner_data     = {}
        self.gaze_time       = 0.0      # accumulated gaze-confirmed seconds
        self.corner_start    = None     # when we started this corner (for timeout)
        self.gaze_on_corner  = False    # is gaze currently on the target corner
        self.done            = False
        self.calibration     = None

        self.test_points       = []
        self.test_idx          = 0
        self.test_start        = None
        self.test_gaze_samples = []
        self.test_results      = []
        self.accuracy_score    = 0.0

        self.active_zone  = None
        self.dwell_start  = None
        self.DWELL_TIME   = 0.5
        self.result_start = None        # properly initialized here, not as class attr


    # ── Corner position helpers ─────────────────────────────────────────

    def get_corner_pos(self, corner):
        pad = 80
        positions = {
            "TOP_LEFT"     : (pad, pad),
            "TOP_RIGHT"    : (self.frame_w - pad, pad),
            "BOTTOM_LEFT"  : (pad, self.frame_h - pad),
            "BOTTOM_RIGHT" : (self.frame_w - pad, self.frame_h - pad),
        }
        return positions[corner]

    def is_gaze_at_corner(self, gaze, corner):
        """
        Returns True if gaze point is within the target corner zone.
        Uses same zone_size as menu.py so gaze zones match visually.
        """
        if gaze is None:
            return False

        x, y   = gaze
        zone_w = int(self.frame_w * 0.25)
        zone_h = int(self.frame_h * 0.25)

        checks = {
            "TOP_LEFT"     : x < zone_w       and y < zone_h,
            "TOP_RIGHT"    : x > self.frame_w - zone_w and y < zone_h,
            "BOTTOM_LEFT"  : x < zone_w       and y > self.frame_h - zone_h,
            "BOTTOM_RIGHT" : x > self.frame_w - zone_w and y > self.frame_h - zone_h,
        }
        return checks.get(corner, False)


    # ── Iris offset extraction ──────────────────────────────────────────

    def get_iris_offset(self, landmarks):
        """
        Returns normalized iris offset relative to eye corners.
        Subtracts eye center to remove head movement — leaving only eyeball rotation.
        """
        if landmarks is None:
            return None

        r_iris  = landmarks[468]
        l_iris  = landmarks[473]
        r_inner = landmarks[362]
        r_outer = landmarks[263]
        l_inner = landmarks[133]
        l_outer = landmarks[33]

        r_eye_cx = (r_inner.x + r_outer.x) / 2
        r_eye_cy = (r_inner.y + r_outer.y) / 2
        l_eye_cx = (l_inner.x + l_outer.x) / 2
        l_eye_cy = (l_inner.y + l_outer.y) / 2

        r_eye_w = abs(r_outer.x - r_inner.x)
        l_eye_w = abs(l_outer.x - l_inner.x)

        if r_eye_w < 1e-6 or l_eye_w < 1e-6:
            return None

        r_offset_x = (r_iris.x - r_eye_cx) / r_eye_w
        r_offset_y = (r_iris.y - r_eye_cy) / r_eye_w
        l_offset_x = (l_iris.x - l_eye_cx) / l_eye_w
        l_offset_y = (l_iris.y - l_eye_cy) / l_eye_w

        return ((r_offset_x + l_offset_x) / 2,
                (r_offset_y + l_offset_y) / 2)


    # ── Calibration mapping ─────────────────────────────────────────────

    def build_calibration(self):
        corner_offsets = {}
        for corner in CORNERS:
            data = self.corner_data.get(corner, [])
            if len(data) == 0:
                return None
            corner_offsets[corner] = (
                float(np.mean([d[0] for d in data])),
                float(np.mean([d[1] for d in data]))
            )

        all_x = [v[0] for v in corner_offsets.values()]
        all_y = [v[1] for v in corner_offsets.values()]

        return {
            "offset_x_min" : float(min(all_x)),
            "offset_x_max" : float(max(all_x)),
            "offset_y_min" : float(min(all_y)),
            "offset_y_max" : float(max(all_y)),
            "frame_w"      : self.frame_w,
            "frame_h"      : self.frame_h,
        }

    def offset_to_screen(self, offset, calibration):
        if offset is None or calibration is None:
            return None
        ox, oy = offset
        sx = np.interp(ox,
                       [calibration["offset_x_min"], calibration["offset_x_max"]],
                       [0, calibration["frame_w"]])
        sy = np.interp(oy,
                       [calibration["offset_y_min"], calibration["offset_y_max"]],
                       [0, calibration["frame_h"]])
        return (int(sx), int(sy))

    def save_calibration(self, calibration):
        folder = os.path.join(PROFILES_DIR, self.name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "calibration.json"), "w") as f:
            json.dump(calibration, f, indent=2)


    # ── Drawing helpers ─────────────────────────────────────────────────

    def draw_target(self, frame, pos, active=False):
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

    def draw_progress_bar(self, frame, progress, color=GREEN, y=50):
        bar_x1, bar_y1 = 20, y
        bar_x2, bar_y2 = 320, y + 20
        filled_x2      = int(bar_x1 + (bar_x2 - bar_x1) * progress)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), WHITE, 1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (filled_x2, bar_y2), color, -1)
        return frame

    def draw_centered_text(self, frame, lines, start_y=None):
        h, w   = frame.shape[:2]
        line_h = 45
        y      = start_y if start_y else (h - len(lines) * line_h) // 2
        for i, (text, color, scale) in enumerate(lines):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            x         = (w - text_size[0]) // 2
            cv2.putText(frame, text, (x, y + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        return frame


    # ── State: WAITING ──────────────────────────────────────────────────

    def update_waiting(self, frame, landmarks):
        if landmarks is not None:
            self.state        = STATE_CORNER
            self.corner_start = time.time()
            self.gaze_time    = 0.0
            return frame

        frame = self.draw_centered_text(frame, [
            ("Gaze Calibration", WHITE, 0.9),
            ("Please face the camera", YELLOW, 0.7),
        ])
        return frame


    # ── State: CORNER ───────────────────────────────────────────────────

    def update_corner(self, frame, landmarks, gaze):
        """
        Collects iris offset data for current corner.

        Timer only runs when gaze is confirmed at the target corner.
        Auto-advances after TIMEOUT_DURATION regardless.
        """

        corner = CORNERS[self.corner_idx]
        pos    = self.get_corner_pos(corner)
        label  = CORNER_LABELS[corner]

        now     = time.time()
        timeout = now - self.corner_start > TIMEOUT_DURATION

        # Handle face disappearing
        if landmarks is None:
            cv2.putText(frame, "Please return to frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            frame = self.draw_target(frame, pos, active=False)
            return frame

        # Check if gaze is at this corner
        self.gaze_on_corner = self.is_gaze_at_corner(gaze, corner)

        if self.gaze_on_corner:
            # Collect iris offset only when gaze confirmed at corner
            offset = self.get_iris_offset(landmarks)
            if offset is not None:
                if corner not in self.corner_data:
                    self.corner_data[corner] = []
                self.corner_data[corner].append(offset)
                self.gaze_time += 1 / 30   # approximate per-frame time increment

        # Progress based on gaze-confirmed time
        gaze_progress    = min(self.gaze_time / COLLECT_DURATION, 1.0)
        timeout_progress = min((now - self.corner_start) / TIMEOUT_DURATION, 1.0)

        # Draw target — green when gaze confirmed, flashing otherwise
        frame = self.draw_target(frame, pos, active=self.gaze_on_corner)

        # Gaze progress bar (green)
        frame = self.draw_progress_bar(frame, gaze_progress, GREEN, y=50)

        # Timeout bar (yellow) — shows how much time is left before auto-advance
        frame = self.draw_progress_bar(frame, timeout_progress, YELLOW, y=78)

        cv2.putText(frame, f"Look at: {label}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(frame, f"Corner {self.corner_idx + 1} of {len(CORNERS)}",
                    (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)

        if self.gaze_on_corner:
            cv2.putText(frame, "Gaze detected!", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
        else:
            cv2.putText(frame, "Look at the flashing dot", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)

        # Advance if gaze collected enough OR timeout reached
        if gaze_progress >= 1.0 or timeout:
            self.corner_idx  += 1
            self.gaze_time    = 0.0
            self.corner_start = time.time()

            if self.corner_idx >= len(CORNERS):
                self.calibration = self.build_calibration()
                self._setup_test()
                self.state = STATE_TEST

        return frame


    # ── State: TEST ─────────────────────────────────────────────────────

    def _setup_test(self):
        pad = 100
        self.test_points = [
            (random.randint(pad, self.frame_w - pad),
             random.randint(pad, self.frame_h - pad))
            for _ in range(TEST_POINTS)
        ]
        self.test_idx          = 0
        self.test_start        = time.time()
        self.test_gaze_samples = []
        self.test_results      = []

    def update_test(self, frame, landmarks):
        if self.test_idx >= TEST_POINTS:
            self.accuracy_score = sum(self.test_results) / max(len(self.test_results), 1)
            self.state          = STATE_RESULT
            self.result_start   = time.time()
            return frame

        target_pos = self.test_points[self.test_idx]

        if landmarks is not None:
            offset = self.get_iris_offset(landmarks)
            if offset is not None:
                gaze_screen = self.offset_to_screen(offset, self.calibration)
                if gaze_screen is not None:
                    self.test_gaze_samples.append(gaze_screen)

        elapsed  = time.time() - self.test_start
        progress = min(elapsed / TEST_DURATION, 1.0)

        frame = self.draw_target(frame, target_pos, active=False)
        frame = self.draw_progress_bar(frame, progress, GREEN, y=50)

        cv2.putText(frame, f"Look at the dot  ({self.test_idx + 1}/{TEST_POINTS})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        if progress >= 1.0:
            if len(self.test_gaze_samples) > 0:
                avg_x        = int(np.mean([g[0] for g in self.test_gaze_samples]))
                avg_y        = int(np.mean([g[1] for g in self.test_gaze_samples]))
                cx, cy       = self.frame_w // 2, self.frame_h // 2
                target_quad  = (target_pos[0] > cx, target_pos[1] > cy)
                gaze_quad    = (avg_x > cx, avg_y > cy)
                passed       = target_quad == gaze_quad
            else:
                passed = False

            self.test_results.append(passed)
            self.test_idx          += 1
            self.test_start         = time.time()
            self.test_gaze_samples  = []

        return frame


    # ── State: RESULT ───────────────────────────────────────────────────

    def update_result(self, frame, fingertip):
        pct    = int(self.accuracy_score * 100)
        passed = self.accuracy_score >= 0.66
        h, w   = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        color = GREEN if passed else RED
        frame = self.draw_centered_text(frame, [
            ("Calibration Complete", WHITE, 0.9),
            (f"Accuracy: {pct}%", color, 1.0),
        ], start_y=h//2 - 80)

        if passed:
            if time.time() - self.result_start > 1.5:
                self.save_calibration(self.calibration)
                self.state = STATE_DONE
                self.done  = True
        else:
            cv2.putText(frame, "Accuracy too low", (w//2 - 150, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 1)

            box_w, box_h = 200, 55
            pad          = 30
            center_y     = h//2 + 60

            redo_coords = (w//2 - box_w - pad, center_y,
                           w//2 - pad,          center_y + box_h)
            skip_coords = (w//2 + pad,          center_y,
                           w//2 + box_w + pad,  center_y + box_h)

            for coords, label, action in [
                (redo_coords, "Redo", "REDO"),
                (skip_coords, "Skip", "SKIP")
            ]:
                in_zone = self._is_inside(fingertip, coords)
                if in_zone:
                    if self.active_zone != action:
                        self.active_zone = action
                        self.dwell_start = time.time()
                    progress = self._get_dwell_progress()
                    frame    = self._draw_dwell_box(frame, coords, label, progress)
                    if progress >= 1.0:
                        if action == "REDO":
                            uid = self.name
                            w_  = self.frame_w
                            h_  = self.frame_h
                            self.reset()
                            self.name    = uid
                            self.frame_w = w_
                            self.frame_h = h_
                        else:
                            self.save_calibration(self.calibration)
                            self.state = STATE_DONE
                            self.done  = True
                        self.active_zone = None
                        self.dwell_start = None
                else:
                    if self.active_zone == action:
                        self.active_zone = None
                        self.dwell_start = None
                    frame = self._draw_dwell_box(frame, coords, label, 0.0)

        return frame


    # ── Dwell helpers ───────────────────────────────────────────────────

    def _get_dwell_progress(self):
        if self.dwell_start is None:
            return 0.0
        return min((time.time() - self.dwell_start) / self.DWELL_TIME, 1.0)

    def _is_inside(self, point, coords):
        if point is None or coords is None:
            return False
        x, y = point
        x1, y1, x2, y2 = coords
        return x1 < x < x2 and y1 < y < y2

    def _draw_dwell_box(self, frame, coords, label, progress):
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


    # ── Main update ─────────────────────────────────────────────────────

    def update(self, frame, landmarks, fingertip, gaze=None):
        """Main update — call every frame. Returns annotated frame."""

        if self.state == STATE_WAITING:
            frame = self.update_waiting(frame, landmarks)
        elif self.state == STATE_CORNER:
            frame = self.update_corner(frame, landmarks, gaze)
        elif self.state == STATE_TEST:
            frame = self.update_test(frame, landmarks)
        elif self.state == STATE_RESULT:
            frame = self.update_result(frame, fingertip)

        return frame


def load_calibration(name):
    """Loads saved calibration from profiles/name/calibration.json. Returns dict or None."""
    path = os.path.join(PROFILES_DIR, name, "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
