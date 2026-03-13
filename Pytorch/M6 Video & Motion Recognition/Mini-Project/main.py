import cv2
import hand_tracker
import menu

from functions import object_detection
from functions import action_recognition
from functions import face_mesh
from functions import motion_trail

FUNCTION_MAP = {
    "OBJECT_DETECTION"   : object_detection,
    "ACTION_RECOGNITION" : action_recognition,
    "FACE_MESH"          : face_mesh,
    "MOTION_TRAIL"       : motion_trail,
}


def reset_all():
    for module in FUNCTION_MAP.values():
        module.reset()


cap   = cv2.VideoCapture(0)
state = "TRIGGER"

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Mirror frame so movements feel natural
    frame = cv2.flip(frame, 1)

    fingertip = hand_tracker.get_fingertip(frame)
    frame     = hand_tracker.draw_fingertip(frame, fingertip)

    if state == "TRIGGER":
        frame, triggered = menu.draw_trigger(frame, fingertip)
        if triggered == "MENU":
            state = "MENU"

    elif state == "MENU":
        frame, triggered = menu.draw_menu(frame, fingertip)
        if triggered == "EXIT":
            break
        elif triggered in FUNCTION_MAP:
            reset_all()
            state = triggered

    else:
        # Motion trail gets fingertip passed in — others just get frame
        if state == "MOTION_TRAIL":
            frame = motion_trail.run(frame, fingertip)
        else:
            module = FUNCTION_MAP[state]
            frame  = module.run(frame)

        frame, triggered = menu.draw_trigger(frame, fingertip)
        if triggered == "MENU":
            reset_all()
            state = "MENU"

    cv2.imshow("All In One System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
