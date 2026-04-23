import cv2
import time
import os
import mediapipe as mp

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options      = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task")),
    running_mode      = VisionRunningMode.VIDEO,
    num_hands         = 1,
    min_hand_detection_confidence = 0.7,
    min_hand_presence_confidence  = 0.7,
    min_tracking_confidence       = 0.7
)

landmarker = HandLandmarker.create_from_options(options)

# Landmark indices
INDEX_TIP   = 8
INDEX_BASE  = 5
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

# def is_pointing(landmarks):

#     index_up  = landmarks[INDEX_TIP].y  < landmarks[INDEX_BASE].y
#     middle_up = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_BASE].y
#     ring_up   = landmarks[RING_TIP].y   < landmarks[RING_BASE].y
#     pinky_up  = landmarks[PINKY_TIP].y  < landmarks[PINKY_BASE].y

#     return index_up and not middle_up and not ring_up and not pinky_up

def is_pointing(landmarks):
    """
    Returns True if index finger is the highest extended finger.
    More lenient than strict closed-finger check — works even when
    pointing directly at camera where finger shape is hard to detect.
    """

    index_up           = landmarks[INDEX_TIP].y < landmarks[INDEX_BASE].y
    higher_than_middle = landmarks[INDEX_TIP].y < landmarks[MIDDLE_TIP].y
    higher_than_ring   = landmarks[INDEX_TIP].y < landmarks[RING_TIP].y
    higher_than_pinky  = landmarks[INDEX_TIP].y < landmarks[PINKY_TIP].y

    return index_up and higher_than_middle and higher_than_ring and higher_than_pinky


def get_fingertip(frame):
    """
    Processes a frame and returns:
    - (x, y) pixel position of index fingertip if pointing gesture detected
    - None if no hand detected or not pointing
    """

    h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    if not is_pointing(landmarks):
        return None

    tip_x = int(landmarks[INDEX_TIP].x * w)
    tip_y = int(landmarks[INDEX_TIP].y * h)

    return (tip_x, tip_y)


def draw_fingertip(frame, tip):
    """Draws a small circle at the fingertip position."""

    if tip is not None:
        cv2.circle(frame, tip, 10, (0, 255, 255), -1)
        cv2.circle(frame, tip, 12, (255, 255, 255), 2)

    return frame
