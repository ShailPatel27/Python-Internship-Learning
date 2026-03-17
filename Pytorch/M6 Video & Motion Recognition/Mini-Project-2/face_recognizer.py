import cv2
import time
import os
import json
import numpy as np
import mediapipe as mp

BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR          = os.path.join(BASE_DIR, "profiles")

BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=os.path.join(BASE_DIR, "face_landmarker.task")),
    running_mode = VisionRunningMode.VIDEO,
    num_faces    = 1
)

landmarker = FaceLandmarker.create_from_options(options)

# Landmark indices
L_EYE_INNER  = 133
L_EYE_OUTER  = 33
L_EYE_TOP    = 159
L_EYE_BOTTOM = 145
R_EYE_INNER  = 362
R_EYE_OUTER  = 263
R_EYE_TOP    = 386
R_EYE_BOTTOM = 374
FOREHEAD     = 10
CHIN         = 152
L_CHEEK      = 234
R_CHEEK      = 454
L_JAW        = 172
R_JAW        = 397
NOSE_TIP     = 4
NOSE_BRIDGE  = 6
NOSE_L       = 64
NOSE_R       = 294
MOUTH_L      = 61
MOUTH_R      = 291
UPPER_LIP    = 13
LOWER_LIP    = 14

SIMILARITY_THRESHOLD = 0.85
GREEN_FLASH_DURATION = 1.0

# State
flash_start = None


def dist(a, b):
    """Euclidean distance between two landmarks."""
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def extract_fingerprint(landmarks):
    """
    Extracts normalized geometric measurements from face landmarks.
    All distances divided by face width to remove scale variation
    from camera distance.
    Returns a 1D numpy array — the geometric fingerprint.
    """

    face_width = dist(landmarks[L_CHEEK], landmarks[R_CHEEK])
    if face_width < 1e-6:
        return None

    features = [
        dist(landmarks[L_EYE_INNER],  landmarks[L_EYE_OUTER]),
        dist(landmarks[R_EYE_INNER],  landmarks[R_EYE_OUTER]),
        dist(landmarks[L_EYE_TOP],    landmarks[L_EYE_BOTTOM]),
        dist(landmarks[R_EYE_TOP],    landmarks[R_EYE_BOTTOM]),
        dist(landmarks[L_EYE_OUTER],  landmarks[R_EYE_OUTER]),
        dist(landmarks[FOREHEAD],     landmarks[CHIN]),
        dist(landmarks[L_JAW],        landmarks[R_JAW]),
        dist(landmarks[L_CHEEK],      landmarks[R_CHEEK]),
        dist(landmarks[NOSE_L],       landmarks[NOSE_R]),
        dist(landmarks[NOSE_BRIDGE],  landmarks[NOSE_TIP]),
        dist(landmarks[MOUTH_L],      landmarks[MOUTH_R]),
        dist(landmarks[UPPER_LIP],    landmarks[LOWER_LIP]),
        dist(landmarks[NOSE_TIP],     landmarks[MOUTH_L]),
        dist(landmarks[NOSE_TIP],     landmarks[MOUTH_R]),
        dist(landmarks[MOUTH_L],      landmarks[CHIN]),
    ]

    return np.array(features) / face_width


def cosine_similarity(a, b):
    """
    Cosine similarity between two vectors.
    1.0 = identical, 0.0 = completely different.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── Profile I/O ─────────────────────────────────────────────────────────

def generate_user_id():
    """Generates unique user ID from current timestamp."""
    return f"usr_{int(time.time())}"


def save_profile(user_id, name, fingerprint):
    """
    Saves profile to profiles/user_id/.
    Creates embedding.npy and meta.json.
    """
    folder = os.path.join(PROFILES_DIR, user_id)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "embedding.npy"), fingerprint)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({
            "id"      : user_id,
            "name"    : name,
            "created" : int(time.time())
        }, f, indent=2)


def load_profiles():
    """
    Loads all saved profiles.
    Returns dict of {user_id: {"name": ..., "embedding": np.array}}.
    """
    profiles = {}

    if not os.path.exists(PROFILES_DIR):
        return profiles

    for user_id in os.listdir(PROFILES_DIR):
        emb_path  = os.path.join(PROFILES_DIR, user_id, "embedding.npy")
        meta_path = os.path.join(PROFILES_DIR, user_id, "meta.json")

        if os.path.exists(emb_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            profiles[user_id] = {
                "name"      : meta["name"],
                "embedding" : np.load(emb_path)
            }

    return profiles


def update_name(user_id, new_name):
    """Updates display name in meta.json without touching folder or embedding."""
    meta_path = os.path.join(PROFILES_DIR, user_id, "meta.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        meta = json.load(f)
    meta["name"] = new_name
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def delete_profile(user_id):
    """Deletes entire profile folder for user_id."""
    import shutil
    path = os.path.join(PROFILES_DIR, user_id)
    if os.path.exists(path):
        shutil.rmtree(path)


# ── Recognition ─────────────────────────────────────────────────────────

def identify(fingerprint, profiles):
    """
    Compares fingerprint against all saved profiles.
    Returns (user_id, name, similarity) of best match, or (None, None, 0) if no match.
    """
    best_id    = None
    best_name  = None
    best_score = 0.0

    for user_id, data in profiles.items():
        score = cosine_similarity(fingerprint, data["embedding"])
        if score > best_score:
            best_score = score
            best_id    = user_id
            best_name  = data["name"]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_id, best_name, best_score

    return None, None, best_score


def get_mesh_color(recognized):
    """
    Returns mesh draw color based on recognition state.
    Red = scanning, Green = recognized (during flash), None = flash done.
    """
    global flash_start

    if recognized:
        if flash_start is None:
            flash_start = time.time()
        elapsed = time.time() - flash_start
        if elapsed < GREEN_FLASH_DURATION:
            return (0, 255, 0)
        return None   # flash done — caller transitions to welcome

    flash_start = None
    return (0, 0, 255)   # red — scanning


def reset_flash():
    global flash_start
    flash_start = None


# ── Frame processing ─────────────────────────────────────────────────────

def process(frame, profiles):
    """
    Runs face detection and recognition on a single frame.

    Returns:
    - landmarks   : raw landmark list or None
    - user_id     : matched user ID or None
    - name        : matched display name or None
    - mesh_color  : color to draw mesh with
    - fingerprint : extracted fingerprint or None
    """

    h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if not result.face_landmarks:
        reset_flash()
        return None, None, None, (0, 0, 255), None

    landmarks   = result.face_landmarks[0]
    fingerprint = extract_fingerprint(landmarks)

    if fingerprint is None:
        return landmarks, None, None, (0, 0, 255), None

    user_id, name, score = identify(fingerprint, profiles)
    color                = get_mesh_color(user_id is not None)

    return landmarks, user_id, name, color, fingerprint


# ── Drawing ──────────────────────────────────────────────────────────────

def draw_mesh(frame, landmarks, color):
    """Draws face mesh tessellation and contours in the given color."""

    if landmarks is None or color is None:
        return frame

    h, w = frame.shape[:2]
    FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections

    for connection in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        start = landmarks[connection.start]
        end   = landmarks[connection.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w),   int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)

    for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        start = landmarks[connection.start]
        end   = landmarks[connection.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w),   int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame
