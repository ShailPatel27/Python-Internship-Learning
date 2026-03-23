import cv2

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

FINGERTIPS = {4, 8, 12, 16, 20}
COLOR      = (255, 255, 255)
DIM        = (160, 160, 160)

def draw_skeleton(frame, slm, screen_w, screen_h, gesture):
    pts = [
        (int(slm[i][0] * screen_w), int(slm[i][1] * screen_h))
        for i in range(21)
    ]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], DIM, 1, cv2.LINE_AA)

    for i, (px, py) in enumerate(pts):
        r = 5 if i in FINGERTIPS else 3
        cv2.circle(frame, (px, py), r, COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), r, (60, 60, 60), 1, cv2.LINE_AA)

    wx, wy = pts[0]
    cv2.putText(frame, gesture, (wx - 30, wy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
