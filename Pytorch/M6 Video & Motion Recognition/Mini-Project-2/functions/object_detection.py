import cv2
import random
from ultralytics import YOLO

# Model
model = YOLO('yolov8n.pt')

# Colors
random.seed(42)
COLORS = {
    class_id: tuple(random.randint(50, 255) for _ in range(3))
    for class_id in range(80)
}

# Constants
CONF_THRESHOLD = 0.5
DETECT_EVERY   = 5

# Module state
frame_count       = 0
stored_detections = []


def reset():
    global frame_count, stored_detections
    frame_count       = 0
    stored_detections = []


def detect(frame):
    """Runs YOLO on frame and returns list of (label, class_id, confidence, x1, y1, x2, y2)."""

    results     = model(frame, conf=CONF_THRESHOLD, verbose=False)
    detections  = []

    for box in results[0].boxes:
        class_id   = int(box.cls.item())
        label      = model.names[class_id]
        confidence = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append((label, class_id, confidence, x1, y1, x2, y2))

    return detections


def draw(frame, detections):
    """Draws bounding boxes and labels for all detections."""

    for label, class_id, confidence, x1, y1, x2, y2 in detections:
        color      = COLORS[class_id]
        text       = f"{label} {confidence:.2f}"
        text_size  = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 8), (x1 + text_size[0], y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def run(frame):
    global frame_count, stored_detections

    frame_count += 1

    if frame_count % DETECT_EVERY == 0:
        stored_detections = detect(frame)

    frame = draw(frame, stored_detections)

    return frame
