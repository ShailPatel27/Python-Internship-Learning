import cv2
import numpy as np


def crop_to_fill(frame, target_w, target_h):
    """
    Crops frame to fill target dimensions while preserving aspect ratio.
    Centers the crop — no stretching, no black bars.
    """

    src_h, src_w = frame.shape[:2]

    # Scale factor to fill target — use the larger ratio
    scale = max(target_w / src_w, target_h / src_h)

    # Scaled dimensions
    scaled_w = int(src_w * scale)
    scaled_h = int(src_h * scale)

    # Resize
    resized = cv2.resize(frame, (scaled_w, scaled_h))

    # Center crop
    x1 = (scaled_w - target_w) // 2
    y1 = (scaled_h - target_h) // 2

    return resized[y1:y1 + target_h, x1:x1 + target_w]
