import numpy as np
import cv2
import math

class Shape:
    def __init__(self, color, pen_size, stroke_style):
        self.raw_points   = []        # normalized coords while drawing (pre-finalize)
        self.local_points = []        # local space (post-finalize, centered 0,0)
        self.position     = (0, 0)    # world center normalized
        self.rotation     = 0.0
        self.scale        = 1.0
        self.color        = color
        self.pen_size     = pen_size
        self.stroke_style = stroke_style
        self.finalized    = False

    # ── building ───────────────────────────────────────────────────────────────

    def add_point(self, nx, ny):
        self.raw_points.append((nx, ny))

    def finalize(self):
        if len(self.raw_points) < 2:
            return False
        cx = sum(p[0] for p in self.raw_points) / len(self.raw_points)
        cy = sum(p[1] for p in self.raw_points) / len(self.raw_points)
        self.position     = (cx, cy)
        self.local_points = [(p[0] - cx, p[1] - cy) for p in self.raw_points]
        self.raw_points   = []
        self.finalized    = True
        return True

    # ── transform ──────────────────────────────────────────────────────────────

    def get_world_points(self, screen_w, screen_h):
        cos_r  = math.cos(self.rotation)
        sin_r  = math.sin(self.rotation)
        px, py = self.position
        pts    = []
        for lx, ly in self.local_points:
            sx = lx * self.scale
            sy = ly * self.scale
            rx = sx * cos_r - sy * sin_r
            ry = sx * sin_r + sy * cos_r
            pts.append((int((px + rx) * screen_w), int((py + ry) * screen_h)))
        return pts

    # ── proximity ──────────────────────────────────────────────────────────────

    def get_bounding_box(self, screen_w, screen_h):
        # returns (min_x, min_y, max_x, max_y) in normalized coords
        pts = self.get_world_points(screen_w, screen_h)
        if not pts:
            px, py = self.position
            return px, py, px, py
        xs = [p[0] / screen_w for p in pts]
        ys = [p[1] / screen_h for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def is_hand_overlapping(self, slm, screen_w, screen_h, padding):
        # grab is true if ANY fingertip is within the shape's bounding box
        min_x, min_y, max_x, max_y = self.get_bounding_box(screen_w, screen_h)
        tip_ids = [8, 12, 16, 20]
        for i in tip_ids:
            tx = slm[i][0]
            ty = slm[i][1]
            if (min_x - padding <= tx <= max_x + padding and
                    min_y - padding <= ty <= max_y + padding):
                return True
        return False

    def distance_to(self, nx, ny):
        px, py = self.position
        return ((nx - px)**2 + (ny - py)**2) ** 0.5

    def is_near(self, nx, ny, threshold):
        return self.distance_to(nx, ny) < threshold

    def erase_near_point(self, nx, ny, screen_w, screen_h, radius_norm):
        world_pts = self.get_world_points(screen_w, screen_h)
        tx = nx * screen_w
        ty = ny * screen_h
        r  = radius_norm * screen_w
        self.local_points = [
            lp for lp, (wx, wy) in zip(self.local_points, world_pts)
            if ((tx - wx)**2 + (ty - wy)**2) ** 0.5 >= r
        ]

    # ── render ─────────────────────────────────────────────────────────────────

    def draw_active(self, canvas, screen_w, screen_h):
        # renders raw_points directly — no finalize, no transform
        if len(self.raw_points) < 2:
            return
        pts = [(int(p[0] * screen_w), int(p[1] * screen_h))
               for p in self.raw_points]
        draw_pts(canvas, pts, self.color, self.pen_size, self.stroke_style)

    def draw(self, canvas, screen_w, screen_h):
        pts = self.get_world_points(screen_w, screen_h)
        if len(pts) < 2:
            return
        thickness = max(1, int(self.pen_size * self.scale))
        draw_pts(canvas, pts, self.color, thickness, self.stroke_style)


# ── shared drawing pipeline ────────────────────────────────────────────────────

def draw_pts(canvas, pts, color, pen_size, stroke_style):
    thickness = max(1, int(pen_size))
    n         = len(pts)

    if stroke_style == 'solid':
        for i in range(n - 1):
            cv2.line(canvas, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

    elif stroke_style == 'dashed':
        for i in range(n - 1):
            if i % 2 == 0:
                cv2.line(canvas, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

    elif stroke_style == 'dotted':
        step = max(1, thickness * 2)
        for i in range(0, n, step):
            cv2.circle(canvas, pts[i], thickness, color, -1, cv2.LINE_AA)

    elif stroke_style == 'glow':
        for t, alpha in [(thickness * 5, 0.08), (thickness * 3, 0.2), (thickness, 1.0)]:
            col = tuple(int(c * alpha) for c in color)
            for i in range(n - 1):
                cv2.line(canvas, pts[i], pts[i+1], col, max(1, t), cv2.LINE_AA)

    elif stroke_style == 'chalk':
        rng = np.random.default_rng(0)
        for i in range(n - 1):
            p1  = (pts[i][0]   + int(rng.integers(-2, 3)),
                   pts[i][1]   + int(rng.integers(-2, 3)))
            p2  = (pts[i+1][0] + int(rng.integers(-2, 3)),
                   pts[i+1][1] + int(rng.integers(-2, 3)))
            col = tuple(int(c * rng.uniform(0.55, 1.0)) for c in color)
            cv2.line(canvas, p1, p2, col, thickness, cv2.LINE_AA)

    elif stroke_style == 'tapered':
        for i in range(n - 1):
            t = max(1, int(thickness * (1.0 - i / n) + 1))
            cv2.line(canvas, pts[i], pts[i+1], color, t, cv2.LINE_AA)
