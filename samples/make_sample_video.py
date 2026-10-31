"""Creates video_demo.avi: a coloured shape that changes (circle, square, triangle, star, heart) while drifting."""

import os
import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
W = 320
w = cv2.VideoWriter(
    os.path.join(here, "video_demo.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, W)
)
cols = [
    (30, 140, 255),
    (60, 200, 90),
    (230, 90, 60),
    (40, 200, 230),
    (90, 60, 230),
]  # BGR
for i in range(40):
    f = np.full((W, W, 3), 205, np.uint8)
    k = i // 8
    cx = int(120 + 3 * i)
    cy = 160
    col = cols[k]
    if k == 0:
        cv2.circle(f, (cx, cy), 60, col, -1, cv2.LINE_AA)
    elif k == 1:
        cv2.rectangle(f, (cx - 55, cy - 55), (cx + 55, cy + 55), col, -1)
    elif k == 2:
        cv2.fillPoly(
            f,
            [np.array([(cx, cy - 65), (cx - 60, cy + 50), (cx + 60, cy + 50)])],
            col,
            cv2.LINE_AA,
        )
    elif k == 3:
        pts = [
            (
                cx + (70 if j % 2 == 0 else 28) * np.cos(-np.pi / 2 + j * np.pi / 5),
                cy + (70 if j % 2 == 0 else 28) * np.sin(-np.pi / 2 + j * np.pi / 5),
            )
            for j in range(10)
        ]
        cv2.fillPoly(f, [np.array(pts, np.int32)], col, cv2.LINE_AA)
    else:
        t = np.linspace(0, 2 * np.pi, 200)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        cv2.fillPoly(
            f,
            [np.column_stack([cx + 4.2 * x, cy - 4.2 * (y + 2.5)]).astype(np.int32)],
            col,
            cv2.LINE_AA,
        )
    w.write(f)
w.release()
print("video_demo.avi written")
