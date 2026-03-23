import cv2
import numpy as np
import math
import time
import config as cfg_mod


class DwellTimer:
    def __init__(self, dwell_time):
        self.dwell_time = dwell_time
        self.target     = None
        self.start_time = None

    def update(self, target_id):
        if target_id != self.target:
            self.target     = target_id
            self.start_time = time.time()
            return False
        if self.start_time and (time.time() - self.start_time) >= self.dwell_time:
            self.start_time = time.time()   # reset so it doesn't retrigger instantly
            return True
        return False

    def get_progress(self, target_id):
        if target_id != self.target or self.start_time is None:
            return 0.0
        return min(1.0, (time.time() - self.start_time) / self.dwell_time)

    def reset(self):
        self.target     = None
        self.start_time = None


class UI:
    def __init__(self, screen_w, screen_h, cfg):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cfg      = cfg
        self.dwell    = DwellTimer(cfg.DWELL_TIME)

        s = cfg.PALETTE_ITEM_SIZE
        p = cfg.PALETTE_PADDING

        raw_items = []

        for i, color in enumerate(cfg.DEFAULT_COLORS):
            raw_items.append({'id': f'color_{i}', 'type': 'color', 'value': color})

        raw_items.append({'id': 'color_wheel', 'type': 'wheel'})

        for i, size in enumerate(cfg.PEN_SIZES):
            raw_items.append({'id': f'size_{i}', 'type': 'size', 'value': size})

        for style in cfg.STROKE_STYLES:
            raw_items.append({'id': f'style_{style}', 'type': 'stroke', 'value': style})

        # draw mode toggle
        raw_items.append({'id': 'draw_pinch', 'type': 'draw_mode', 'value': 'pinch'})
        raw_items.append({'id': 'draw_point', 'type': 'draw_mode', 'value': 'point'})

        # layout — wrap to second column if overflow
        self.items     = []
        col            = 0
        y              = p
        max_col_h      = screen_h - p * 2

        for item in raw_items:
            if y + s > max_col_h:
                col += 1
                y    = p
            x = cfg.PALETTE_X + col * (s + p * 2)
            item['rect'] = (x, y, s, s)
            self.items.append(item)
            y += s + p

        # color wheel geometry
        wheel         = next(it for it in self.items if it['id'] == 'color_wheel')
        wx, wy, ww, _ = wheel['rect']
        self.wheel_cx = wx + ww // 2
        self.wheel_cy = wy + ww // 2
        self.wheel_r  = ww // 2

        # dustbin top right
        ds = cfg.DUSTBIN_SIZE
        self.dustbin_rect = (screen_w - ds - p, p, ds, ds)

    # ── hit testing ────────────────────────────────────────────────────────────

    def get_hovered_item(self, nx, ny):
        px = nx * self.screen_w
        py = ny * self.screen_h
        for item in self.items:
            x, y, w, h = item['rect']
            if x <= px <= x + w and y <= py <= y + h:
                return item
        return None

    def get_color_from_wheel(self, nx, ny):
        px  = nx * self.screen_w
        py  = ny * self.screen_h
        dx  = px - self.wheel_cx
        dy  = py - self.wheel_cy
        dst = (dx**2 + dy**2) ** 0.5
        if dst > self.wheel_r:
            return None
        hue = int((math.atan2(dy, dx) + math.pi) / (2 * math.pi) * 180)
        sat = int(min(dst / self.wheel_r, 1.0) * 255)
        hsv = np.uint8([[[hue, sat, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr[0][0])

    # ── update ─────────────────────────────────────────────────────────────────

    def update(self, index_tip_norm, canvas_obj):
        # color wheel uses index tip — instant selection
        wheel_item = next((it for it in self.items if it['id'] == 'color_wheel'), None)
        if wheel_item:
            ix, iy = index_tip_norm
            ipx = ix * self.screen_w
            ipy = iy * self.screen_h
            wx, wy, ww, wh = wheel_item['rect']
            if wx <= ipx <= wx + ww and wy <= ipy <= wy + wh:
                color = self.get_color_from_wheel(*index_tip_norm)
                if color:
                    canvas_obj.color = color
                    return

        # everything else uses index tip for dwell too
        item = self.get_hovered_item(*index_tip_norm)
        if item and item['id'] != 'color_wheel':
            if self.dwell.update(item['id']):
                self._apply(item, canvas_obj)
        else:
            self.dwell.reset()

    def _apply(self, item, canvas_obj):
        t = item['type']
        if t == 'color':
            canvas_obj.color = item['value']
        elif t == 'size':
            canvas_obj.pen_size = item['value']
        elif t == 'stroke':
            canvas_obj.stroke_style = item['value']
        elif t == 'draw_mode':
            # write back to config so classifier picks it up
            import config as cfg_mod
            cfg_mod.DRAW_MODE = item['value']

    # ── render ─────────────────────────────────────────────────────────────────

    def draw(self, frame, canvas_obj, index_tip_norm):
        import config as cfg_mod
        hovered = self.get_hovered_item(*index_tip_norm)

        for item in self.items:
            x, y, w, h = item['rect']
            is_sel      = False
            is_hov      = hovered and hovered['id'] == item['id']

            if item['type'] == 'color':
                cv2.rectangle(frame, (x, y), (x+w, y+h), item['value'], -1)
                is_sel = item['value'] == canvas_obj.color

            elif item['type'] == 'wheel':
                self._draw_wheel(frame, x, y, w, h)

            elif item['type'] == 'size':
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), -1)
                r = max(2, item['value'])
                cv2.circle(frame, (x + w//2, y + h//2), r, (255, 255, 255), -1)
                is_sel = item['value'] == canvas_obj.pen_size

            elif item['type'] == 'stroke':
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), -1)
                cv2.putText(frame, item['value'][:3].upper(),
                            (x + 4, y + h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                is_sel = item['value'] == canvas_obj.stroke_style

            elif item['type'] == 'draw_mode':
                cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 60), -1)
                label = 'PCH' if item['value'] == 'pinch' else 'PNT'
                cv2.putText(frame, label, (x + 4, y + h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1)
                is_sel = item['value'] == cfg_mod.DRAW_MODE

            if is_sel:
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (255, 255, 255), 2)

            if is_hov and item['id'] != 'color_wheel':
                cv2.rectangle(frame, (x-1, y-1), (x+w+1, y+h+1), (180, 180, 180), 1)
                prog = self.dwell.get_progress(item['id'])
                if prog > 0:
                    cv2.ellipse(frame, (x+w//2, y+h//2), (w//2, h//2),
                                -90, 0, int(360 * prog), (255, 255, 0), 2)

        # dustbin
        dx, dy, dw, dh = self.dustbin_rect
        cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), (60, 30, 30), -1)
        cv2.rectangle(frame, (dx, dy), (dx+dw, dy+dh), (180, 60, 60), 1)
        cv2.putText(frame, 'DEL', (dx + 4, dy + dh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 80, 80), 1)

    def _draw_wheel(self, frame, x, y, w, h):
        cx, cy = x + w//2, y + h//2
        r      = w // 2
        for deg in range(0, 360, 4):
            rad   = math.radians(deg)
            hue   = deg // 2
            hsv   = np.uint8([[[hue, 255, 255]]])
            bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            color = tuple(int(c) for c in bgr[0][0])
            ex    = int(cx + r * math.cos(rad))
            ey    = int(cy + r * math.sin(rad))
            cv2.line(frame, (cx, cy), (ex, ey), color, 2)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
