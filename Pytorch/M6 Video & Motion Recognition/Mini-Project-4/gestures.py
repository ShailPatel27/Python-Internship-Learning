import math
from collections import deque
import config as cfg

# ── smoothers / confirmer ──────────────────────────────────────────────────────

class LandmarkSmoother:
    def __init__(self, alpha):
        self.alpha    = alpha
        self.smoothed = None

    def update(self, lm):
        raw = [(lm[i].x, lm[i].y) for i in range(21)]
        if self.smoothed is None:
            self.smoothed = raw
        else:
            self.smoothed = [
                (self.alpha * rx + (1 - self.alpha) * sx,
                 self.alpha * ry + (1 - self.alpha) * sy)
                for (rx, ry), (sx, sy) in zip(raw, self.smoothed)
            ]
        return self.smoothed

    def reset(self):
        self.smoothed = None


class PinchSmoother:
    def __init__(self, window):
        self.window  = window
        self.history = deque(maxlen=window)

    def update(self, dist):
        self.history.append(dist)
        return sum(self.history) / len(self.history)

    def reset(self):
        self.history.clear()


class GestureConfirmer:
    def __init__(self, n_frames):
        self.n_frames  = n_frames
        self.candidate = None
        self.count     = 0
        self.confirmed = "idle"

    def update(self, raw):
        if raw == self.candidate:
            self.count += 1
        else:
            self.candidate = raw
            self.count     = 1
        if self.count >= self.n_frames:
            self.confirmed = self.candidate
            self.count     = self.n_frames
        return self.confirmed

    def reset(self):
        self.candidate = None
        self.count     = 0
        self.confirmed = "idle"


# ── landmark accessors ─────────────────────────────────────────────────────────

def _lx(slm, i): return slm[i][0]
def _ly(slm, i): return slm[i][1]

def _dist(slm, a, b):
    dx = _lx(slm, a) - _lx(slm, b)
    dy = _ly(slm, a) - _ly(slm, b)
    return (dx**2 + dy**2) ** 0.5

def get_pinch_distance_smooth(slm):
    return _dist(slm, 4, 8)

def get_pinch_midpoint(slm):
    return (_lx(slm,4)+_lx(slm,8))/2, (_ly(slm,4)+_ly(slm,8))/2

def get_index_tip(slm):  return _lx(slm, 8), _ly(slm, 8)
def get_thumb_tip(slm):  return _lx(slm, 4), _ly(slm, 4)

def get_palm_center(slm):
    ids = [0, 5, 9, 13, 17]
    return (sum(_lx(slm,i) for i in ids)/5,
            sum(_ly(slm,i) for i in ids)/5)

def get_wrist_to_middle_angle(slm):
    return math.atan2(_ly(slm,12)-_ly(slm,0), _lx(slm,12)-_lx(slm,0))

def get_avg_fingertip_gap(slm):
    tips = [8, 12, 16, 20]
    gaps = []
    for i in range(len(tips)-1):
        gaps.append(_dist(slm, tips[i], tips[i+1]))
    return sum(gaps) / len(gaps)


# ── finger state helpers ───────────────────────────────────────────────────────

def _is_curled(slm, tip_id, pip_id, mcp_id, threshold):
    tip_below_base = _ly(slm, tip_id) > _ly(slm, pip_id)
    tip_near_wrist = _dist(slm, tip_id, 0) < _dist(slm, mcp_id, 0) * threshold
    return tip_below_base and tip_near_wrist

def _is_extended(slm, tip_id, pip_id, mcp_id, threshold):
    return not _is_curled(slm, tip_id, pip_id, mcp_id, threshold)

_IDX_THRESH  = 1.5
_STD_THRESH  = cfg.CURL_THRESHOLD

def _index_curled(slm):  return _is_curled(slm,  8,  6, 5,  _IDX_THRESH)
def _middle_curled(slm): return _is_curled(slm, 12, 10, 9,  _STD_THRESH)
def _ring_curled(slm):   return _is_curled(slm, 16, 14, 13, _STD_THRESH)
def _pinky_curled(slm):  return _is_curled(slm, 20, 18, 17, _STD_THRESH)
def _thumb_extended(slm):
    return _dist(slm, 4, 9) > 0.1


# ── gesture checks ─────────────────────────────────────────────────────────────

def is_draw_pinch(slm, smooth_dist):
    thumb_out     = _thumb_extended(slm)
    index_ext     = _is_extended(slm, 8, 6, 5, _IDX_THRESH)
    others_curled = _middle_curled(slm) and _ring_curled(slm) and _pinky_curled(slm)
    pinching      = smooth_dist < cfg.PINCH_THRESHOLD
    return thumb_out and index_ext and others_curled and pinching

def is_draw_point(slm):
    index_ext     = _is_extended(slm, 8, 6, 5, _IDX_THRESH)
    others_curled = (not _thumb_extended(slm) and
                     _middle_curled(slm) and
                     _ring_curled(slm) and
                     _pinky_curled(slm))
    return index_ext and others_curled

def is_erase(slm):
    # thumb extended, all 4 fingers curled
    return (_thumb_extended(slm) and
            _index_curled(slm) and
            _middle_curled(slm) and
            _ring_curled(slm) and
            _pinky_curled(slm))

def is_grabbing(slm):
    # all 4 fingers curled (thumb state ignored — thumb handles erase separately)
    return (_index_curled(slm) and
            _middle_curled(slm) and
            _ring_curled(slm) and
            _pinky_curled(slm))


# ── proximity ──────────────────────────────────────────────────────────────────

def is_hand_near_any_shape(slm, shapes, screen_w, screen_h, padding):
    if not shapes:
        return False
    return any(s.is_hand_overlapping(slm, screen_w, screen_h, padding) for s in shapes)

def get_closest_shape_to_fingertips(slm, shapes, screen_w, screen_h, padding):
    palm_x, palm_y = get_palm_center(slm)
    best, best_dist = None, float('inf')
    for s in shapes:
        if s.is_hand_overlapping(slm, screen_w, screen_h, padding):
            d = s.distance_to(palm_x, palm_y)
            if d < best_dist:
                best_dist = d
                best      = s
    return best


# ── classifier ─────────────────────────────────────────────────────────────────

def classify_gesture(slm, smooth_dist, shapes, screen_w, screen_h):
    # erase wins first — thumb out + all fingers curled
    if is_erase(slm):
        return "erase"

    # grab/move — requires fingertips on shape
    near = is_hand_near_any_shape(slm, shapes, screen_w, screen_h, cfg.GRAB_BBOX_PADDING)
    if is_grabbing(slm) and near:
        return "move"   # scale + rotate applied simultaneously in main loop

    # draw — outside exclusion zone
    draw_x = (_lx(slm,4)+_lx(slm,8))/2 if cfg.DRAW_MODE == 'pinch' else _lx(slm,8)
    if draw_x > cfg.DRAW_EXCLUSION_X:
        if cfg.DRAW_MODE == 'pinch' and is_draw_pinch(slm, smooth_dist):
            return "draw"
        if cfg.DRAW_MODE == 'point' and is_draw_point(slm):
            return "draw"

    return "idle"
