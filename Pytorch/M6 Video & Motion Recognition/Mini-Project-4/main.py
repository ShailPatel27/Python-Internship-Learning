import cv2
import mediapipe as mp
import time
import os

import config as cfg
from gestures import (
    LandmarkSmoother, PinchSmoother, GestureConfirmer,
    classify_gesture,
    get_pinch_midpoint, get_palm_center,
    get_wrist_to_middle_angle, get_avg_fingertip_gap,
    get_pinch_distance_smooth, get_thumb_tip, get_index_tip,
    get_closest_shape_to_fingertips, is_hand_near_any_shape,
    is_erase, is_grabbing
)
from canvas        import Canvas
from ui            import UI
from hand_skeleton import draw_skeleton

BASE_DIR = os.getcwd()

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options = BaseOptions(
        model_asset_path = os.path.join(BASE_DIR, '../hand_landmarker.task')
    ),
    running_mode                  = VisionRunningMode.VIDEO,
    num_hands                     = 1,
    min_hand_detection_confidence = 0.7,
    min_hand_presence_confidence  = 0.7,
    min_tracking_confidence       = 0.7
)

def crop_to_fill(frame, target_w, target_h):
    src_h, src_w = frame.shape[:2]
    scale        = max(target_w / src_w, target_h / src_h)
    scaled_w     = int(src_w * scale)
    scaled_h     = int(src_h * scale)
    resized      = cv2.resize(frame, (scaled_w, scaled_h))
    x1 = (scaled_w - target_w) // 2
    y1 = (scaled_h - target_h) // 2
    return resized[y1:y1 + target_h, x1:x1 + target_w]

def main():
    W, H   = cfg.SCREEN_W, cfg.SCREEN_H
    canvas = Canvas(W, H)
    ui     = UI(W, H, cfg)

    lm_smoother    = LandmarkSmoother(cfg.LANDMARK_SMOOTH_ALPHA)
    pinch_smoother = PinchSmoother(cfg.PINCH_SMOOTH_WINDOW)
    confirmer      = GestureConfirmer(cfg.GESTURE_CONFIRM_FRAMES)

    cap = cv2.VideoCapture(0)

    prev_slm     = None
    prev_gesture = "idle"
    prev_palm    = None
    prev_angle   = None
    prev_gap     = None
    pinch_pt     = (0.5, 0.5)
    index_tip    = (0.5, 0.5)

    with HandLandmarker.create_from_options(hand_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            frame  = crop_to_fill(frame, W, H)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts     = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_img, ts)

            if result.hand_landmarks:
                raw_lm = result.hand_landmarks[0]
                slm    = lm_smoother.update(raw_lm)

                smooth_dist = pinch_smoother.update(get_pinch_distance_smooth(slm))

                raw_gesture = classify_gesture(slm, smooth_dist, canvas.shapes, W, H)
                gesture     = confirmer.update(raw_gesture)

                pinch_pt  = get_pinch_midpoint(slm)
                index_tip = get_index_tip(slm)
                palm_pt   = get_palm_center(slm)
                angle     = get_wrist_to_middle_angle(slm)
                gap       = get_avg_fingertip_gap(slm)

                draw_tip = index_tip if cfg.DRAW_MODE == 'point' else pinch_pt

                # ── gesture actions ────────────────────────────────────────────

                if gesture == "draw":
                    if prev_gesture != "draw":
                        canvas.start_stroke(*draw_tip)
                    else:
                        canvas.continue_stroke(*draw_tip)

                elif gesture == "move":
                    # select shape once on entry
                    if canvas.selected is None:
                        canvas.selected = get_closest_shape_to_fingertips(
                            slm, canvas.shapes, W, H, cfg.GRAB_BBOX_PADDING)

                    if canvas.selected:
                        # move — palm delta
                        if prev_palm:
                            dx = palm_pt[0] - prev_palm[0]
                            dy = palm_pt[1] - prev_palm[1]
                            canvas.move_selected(dx, dy)

                        # scale — fingertip gap delta, deadzone applied
                        if prev_gap is not None:
                            gap_delta = gap - prev_gap
                            if abs(gap_delta) > cfg.SCALE_DELTA_THRESHOLD:
                                canvas.scale_selected(gap_delta)

                        # rotate — wrist angle delta, deadzone applied
                        if prev_angle is not None:
                            angle_delta = angle - prev_angle
                            if abs(angle_delta) > cfg.ROTATE_DELTA_THRESHOLD:
                                canvas.rotate_selected(angle_delta)

                        # dustbin check
                        canvas.erase_if_in_dustbin(
                            canvas.selected,
                            cfg.DUSTBIN_X, cfg.DUSTBIN_Y, cfg.DUSTBIN_SIZE
                        )

                elif gesture == "erase":
                    canvas.precise_erase(*get_thumb_tip(slm), cfg.PRECISE_ERASE_RADIUS)

                elif gesture == "idle":
                    canvas.selected = None

                # end stroke on leaving draw
                if prev_gesture == "draw" and gesture != "draw":
                    canvas.end_stroke()

                ui.update(index_tip, canvas)

                output = canvas.render()
                draw_skeleton(output, slm, W, H, gesture)

                # debug overlay
                debug_lines = [
                    f"raw: {raw_gesture}  confirmed: {gesture}",
                    f"grabbing: {is_grabbing(slm)}  erase: {is_erase(slm)}",
                    f"near: {is_hand_near_any_shape(slm, canvas.shapes, W, H, cfg.GRAB_BBOX_PADDING)}",
                ]
                for i, line in enumerate(debug_lines):
                    cv2.putText(output, line, (10, H - 60 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                prev_slm     = slm
                prev_gesture = gesture
                prev_palm    = palm_pt
                prev_angle   = angle
                prev_gap     = gap

            else:
                if prev_gesture == "draw":
                    canvas.end_stroke()

                lm_smoother.reset()
                pinch_smoother.reset()
                confirmer.reset()

                prev_slm     = None
                prev_gesture = "idle"
                prev_palm    = None
                prev_angle   = None
                prev_gap     = None
                canvas.selected = None

                output = canvas.render()

            ui.draw(output, canvas, index_tip)

            cv2.imshow('GestureCanvas', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
