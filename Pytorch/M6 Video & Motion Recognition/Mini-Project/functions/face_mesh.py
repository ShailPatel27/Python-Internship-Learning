import cv2
import time
import os
import mediapipe as mp

# Tasks API setup
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "..", "face_landmarker.task")),
    running_mode = VisionRunningMode.VIDEO,
    num_faces    = 2
)

landmarker = FaceLandmarker.create_from_options(options)

# Colors
TESSELATION_COLOR = (0, 255, 0)
CONTOUR_COLOR     = (255, 255, 255)


def draw_connections(frame, landmarks, connections, color, thickness, h, w):
    """Draws lines between connected landmark pairs."""

    for connection in connections:
        start_idx = connection.start
        end_idx   = connection.end

        start = landmarks[start_idx]
        end   = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)
        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def reset():
    pass


def run(frame):

    h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if not result.face_landmarks:
        return frame

    for face_landmarks in result.face_landmarks:

        draw_connections(
            frame, face_landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            TESSELATION_COLOR, 1, h, w
        )

        draw_connections(
            frame, face_landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            CONTOUR_COLOR, 1, h, w
        )

    return frame
