import numpy as np
import cv2
from shapes import Shape

class Canvas:
    def __init__(self, screen_w, screen_h):
        self.screen_w     = screen_w
        self.screen_h     = screen_h
        self.shapes       = []
        self.active_shape = None   # shape currently being drawn
        self.selected     = None   # shape being transformed

        # tool state
        self.color        = (255, 255, 255)
        self.pen_size     = 3
        self.stroke_style = 'solid'

    # ── drawing ────────────────────────────────────────────────────────────────

    def start_stroke(self, nx, ny):
        self.active_shape = Shape(self.color, self.pen_size, self.stroke_style)
        self.active_shape.add_point(nx, ny)

    def continue_stroke(self, nx, ny):
        if self.active_shape:
            self.active_shape.add_point(nx, ny)

    def end_stroke(self):
        if self.active_shape:
            if self.active_shape.finalize():
                self.shapes.append(self.active_shape)
            self.active_shape = None

    # ── selection ──────────────────────────────────────────────────────────────

    def select_closest(self, nx, ny, proximity):
        best_dist = float('inf')
        best      = None
        for shape in self.shapes:
            d = shape.distance_to(nx, ny)
            if d < proximity and d < best_dist:
                best_dist = d
                best      = shape
        self.selected = best
        return best

    # ── transform ──────────────────────────────────────────────────────────────

    def move_selected(self, dx, dy):
        if self.selected:
            px, py = self.selected.position
            self.selected.position = (px + dx, py + dy)

    def rotate_selected(self, delta_angle):
        if self.selected:
            self.selected.rotation += delta_angle

    def scale_selected(self, delta_gap):
        if self.selected:
            factor = 1.0 + (delta_gap * 10.0)
            self.selected.scale = max(0.1, self.selected.scale * factor)

    # ── erase ──────────────────────────────────────────────────────────────────

    def erase_near(self, slm, padding):
        # erase every shape any fingertip is touching
        tip_ids  = [8, 12, 16, 20]
        tip_coords = [(slm[i][0], slm[i][1]) for i in tip_ids]
        def touched(shape):
            min_x, min_y, max_x, max_y = shape.get_bounding_box(self.screen_w, self.screen_h)
            return any(
                min_x - padding <= tx <= max_x + padding and
                min_y - padding <= ty <= max_y + padding
                for tx, ty in tip_coords
            )
        self.shapes = [s for s in self.shapes if not touched(s)]
        if self.selected and self.selected not in self.shapes:
            self.selected = None

    def precise_erase(self, nx, ny, radius):
        # erase entire shape if thumb tip is within its bounding box
        def thumb_touches(shape):
            min_x, min_y, max_x, max_y = shape.get_bounding_box(self.screen_w, self.screen_h)
            return (min_x - radius <= nx <= max_x + radius and
                    min_y - radius <= ny <= max_y + radius)
        self.shapes = [s for s in self.shapes if not thumb_touches(s)]
        if self.selected and self.selected not in self.shapes:
            self.selected = None

    def erase_if_in_dustbin(self, shape, dustbin_x, dustbin_y, dustbin_size):
        px = shape.position[0] * self.screen_w
        py = shape.position[1] * self.screen_h
        if abs(px - dustbin_x) < dustbin_size and abs(py - dustbin_y) < dustbin_size:
            if shape in self.shapes:
                self.shapes.remove(shape)
            if self.selected == shape:
                self.selected = None
            return True
        return False

    # ── render ─────────────────────────────────────────────────────────────────

    def render(self):
        # pure black canvas
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

        for shape in self.shapes:
            shape.draw(canvas, self.screen_w, self.screen_h)

        # highlight selected shape
        if self.selected:
            pts = self.selected.get_world_points(self.screen_w, self.screen_h)
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.line(canvas, pts[i], pts[i+1], (100, 200, 255), 
                             max(1, int(self.selected.pen_size * self.selected.scale) + 3),
                             cv2.LINE_AA)
                self.selected.draw(canvas, self.screen_w, self.screen_h)

        # active stroke drawn directly from raw points
        if self.active_shape:
            self.active_shape.draw_active(canvas, self.screen_w, self.screen_h)

        return canvas
