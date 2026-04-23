import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
CLIP_LENGTH    = 16
STEP           = 8
TOP_K          = 3
CONF_THRESHOLD = 0.5
DETECT_EVERY   = 5
MAX_MISS       = 20

# YOLO
yolo = YOLO('yolov8n.pt')

# R3D
weights = R3D_18_Weights.DEFAULT
r3d     = r3d_18(weights=weights).eval().to(device)
labels  = weights.meta["categories"]

# Preprocessing
preprocess = Compose([
    Resize((128, 171)),
    CenterCrop((112, 112)),
    Normalize(mean=[0.43216, 0.394666, 0.37645],
              std=[0.22803, 0.22145, 0.216989])
])

# Module state
frame_count   = 0
crop_buffer   = []
current_preds = []
current_box   = None
miss_count    = 0


def reset():
    global frame_count, crop_buffer, current_preds, current_box, miss_count
    frame_count   = 0
    crop_buffer   = []
    current_preds = []
    current_box   = None
    miss_count    = 0


def get_person_crop(frame):
    """Returns largest person crop and its bounding box."""

    results   = yolo(frame, conf=CONF_THRESHOLD, verbose=False)
    best_box  = None
    best_area = 0

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        if yolo.names[class_id] != "person":
            continue
        if box.conf.item() < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            best_box  = (x1, y1, x2, y2)

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = best_box
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, best_box


def prepare_clip(crops):
    """Converts list of BGR crops into R3D input tensor."""

    processed = []

    for crop in crops:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor   = torch.tensor(crop_rgb).permute(2, 0, 1).float() / 255.0
        tensor   = preprocess(tensor)
        processed.append(tensor)

    clip = torch.stack(processed, dim=0).permute(1, 0, 2, 3)
    return clip.unsqueeze(0).to(device)


def predict_action(crops):
    """Runs R3D on clip and returns top K predictions."""

    clip = prepare_clip(crops)

    with torch.no_grad():
        output = r3d(clip)

    probs                  = F.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probs, TOP_K)

    return [(labels[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]


def run(frame):
    global frame_count, crop_buffer, current_preds, current_box, miss_count

    frame_count += 1

    # Detection
    if frame_count % DETECT_EVERY == 0:
        crop, box = get_person_crop(frame)

        if crop is not None:
            miss_count  = 0
            current_box = box
            crop_buffer.append(crop)

            if len(crop_buffer) > CLIP_LENGTH:
                crop_buffer.pop(0)

        else:
            miss_count += 1
            if miss_count > MAX_MISS:
                crop_buffer   = []
                current_preds = []
                current_box   = None
                miss_count    = 0

    # Action recognition
    if len(crop_buffer) == CLIP_LENGTH and frame_count % STEP == 0:
        current_preds = predict_action(crop_buffer)

    # Draw person box
    if current_box is not None:
        x1, y1, x2, y2 = current_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw predictions
    for i, (label, prob) in enumerate(current_preds):
        text = f"{label}: {prob:.2f}"
        cv2.putText(frame, text, (20, 40 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Buffer progress
    cv2.putText(frame, f"Buffer: {len(crop_buffer)}/{CLIP_LENGTH}", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame
