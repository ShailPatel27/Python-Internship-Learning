# calibrator.py
# Core calibration logic — state machine, data collection, saving.
# UI/drawing is handled by calibrator_ui.py

import cv2
import time
import os
import json
import random
import numpy as np
from calibrator_ui import (
    draw_target, draw_progress_bar,
    draw_centered_text, draw_dwell_box,
    WHITE, GREEN, RED, YELLOW, CYAN
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

# Calibration constants
COLLECT_DURATION  = 1.0
TIMEOUT_DURATION  = 10.0
TEST_POINTS       = 3
TEST_DURATION     = 1.5

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
        self.name    = name
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.reset()

    def reset(self):
        self.state           = STATE_WAITING
        self.corner_idx      = 0
        self.corner_data     = {}
        self.gaze_time       = 0.0
        self.corner_start    = None
        self.gaze_on_corner  = False
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
        self.result_start = None


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

    def is_gaze_at_corner(self, yaw, pitch, corner):
        if yaw is None or pitch is None:
            return False

        is_left  = yaw   < -0.15
        is_right = yaw   >  0.15
        is_up    = pitch >  1.30   # pitch2 high = looking up
        is_down  = pitch <  1.05   # pitch2 low  = looking down

        checks = {
            "TOP_LEFT"     : is_left  and is_up,
            "TOP_RIGHT"    : is_right and is_up,
            "BOTTOM_LEFT"  : is_left  and is_down,
            "BOTTOM_RIGHT" : is_right and is_down,
        }
        return checks.get(corner, False)

    # ── Calibration mapping ─────────────────────────────────────────────

    def build_calibration(self):
        corner_offsets = {}
        for corner in CORNERS:
            data = self.corner_data.get(corner, [])
            if len(data) == 0:
                return None
            avg_yaw   = float(np.mean([d[0] for d in data]))
            avg_pitch = float(np.mean([d[1] for d in data]))
            corner_offsets[corner] = (avg_yaw, avg_pitch)

        return {
            # Yaw — from left/right corners
            "yaw_left"  : float(np.mean([corner_offsets["TOP_LEFT"][0],
                                        corner_offsets["BOTTOM_LEFT"][0]])),
            "yaw_right" : float(np.mean([corner_offsets["TOP_RIGHT"][0],
                                        corner_offsets["BOTTOM_RIGHT"][0]])),
            # Pitch — from top/bottom corners
            "pitch_up"  : float(np.mean([corner_offsets["TOP_LEFT"][1],
                                        corner_offsets["TOP_RIGHT"][1]])),
            "pitch_down": float(np.mean([corner_offsets["BOTTOM_LEFT"][1],
                                        corner_offsets["BOTTOM_RIGHT"][1]])),
            "frame_w"   : self.frame_w,
            "frame_h"   : self.frame_h,
        }

    def save_calibration(self, calibration):
        folder = os.path.join(PROFILES_DIR, self.name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "calibration.json"), "w") as f:
            json.dump(calibration, f, indent=2)


    # ── State: WAITING ──────────────────────────────────────────────────

    def update_waiting(self, frame, landmarks):
        if landmarks is not None:
            self.state        = STATE_CORNER
            self.corner_start = time.time()
            self.gaze_time    = 0.0
            return frame

        frame = draw_centered_text(frame, [
            ("Gaze Calibration", WHITE, 0.9),
            ("Please face the camera", YELLOW, 0.7),
        ])
        return frame


    # ── State: CORNER ───────────────────────────────────────────────────

    def update_corner(self, frame, landmarks, yaw, pitch):
        corner = CORNERS[self.corner_idx]
        pos    = self.get_corner_pos(corner)
        label  = CORNER_LABELS[corner]

        now     = time.time()
        timeout = now - self.corner_start > TIMEOUT_DURATION

        if landmarks is None:
            cv2.putText(frame, "Please return to frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            frame = draw_target(frame, pos, active=False)
            return frame

        self.gaze_on_corner = self.is_gaze_at_corner(yaw, pitch, corner)
        if self.gaze_on_corner:
            if corner not in self.corner_data:
                self.corner_data[corner] = []
            self.corner_data[corner].append((yaw, pitch))
            self.gaze_time += 1 / 30

        gaze_progress    = min(self.gaze_time / COLLECT_DURATION, 1.0)
        timeout_progress = min((now - self.corner_start) / TIMEOUT_DURATION, 1.0)

        frame = draw_target(frame, pos, active=self.gaze_on_corner)
        frame = draw_progress_bar(frame, gaze_progress, GREEN, y=50)
        frame = draw_progress_bar(frame, timeout_progress, YELLOW, y=78)

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

    def update_test(self, frame, landmarks, yaw=None, pitch=None):
        if self.test_idx >= TEST_POINTS:
            self.accuracy_score = sum(self.test_results) / max(len(self.test_results), 1)
            self.state          = STATE_RESULT
            self.result_start   = time.time()
            return frame

        target_pos = self.test_points[self.test_idx]

        if yaw is not None and pitch is not None:
            h, w = frame.shape[:2]
            sx   = int(w * 0.5 + yaw   * w * 1.5)
            sy   = int(h * 0.5 + pitch * h * 1.5)
            self.test_gaze_samples.append((sx, sy))

        elapsed  = time.time() - self.test_start
        progress = min(elapsed / TEST_DURATION, 1.0)

        frame = draw_target(frame, target_pos, active=False)
        frame = draw_progress_bar(frame, progress, GREEN, y=50)

        cv2.putText(frame, f"Look at the dot  ({self.test_idx + 1}/{TEST_POINTS})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        if progress >= 1.0:
            if len(self.test_gaze_samples) > 0:
                avg_x       = int(np.mean([g[0] for g in self.test_gaze_samples]))
                avg_y       = int(np.mean([g[1] for g in self.test_gaze_samples]))
                cx, cy      = self.frame_w // 2, self.frame_h // 2
                target_quad = (target_pos[0] > cx, target_pos[1] > cy)
                gaze_quad   = (avg_x > cx, avg_y > cy)
                passed      = target_quad == gaze_quad
            else:
                passed = False

            self.test_results.append(passed)
            self.test_idx         += 1
            self.test_start        = time.time()
            self.test_gaze_samples = []

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
        frame = draw_centered_text(frame, [
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
                    frame    = draw_dwell_box(frame, coords, label, progress)
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
                    frame = draw_dwell_box(frame, coords, label, 0.0)

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


    # ── Main update ─────────────────────────────────────────────────────

    def update(self, frame, landmarks, fingertip, yaw=None, pitch=None):
        if self.state == STATE_WAITING:
            frame = self.update_waiting(frame, landmarks)
        elif self.state == STATE_CORNER:
            frame = self.update_corner(frame, landmarks, yaw, pitch)
        elif self.state == STATE_TEST:
            frame = self.update_test(frame, landmarks, yaw, pitch)
        elif self.state == STATE_RESULT:
            frame = self.update_result(frame, fingertip)
        return frame


def load_calibration(name):
    path = os.path.join(PROFILES_DIR, name, "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)