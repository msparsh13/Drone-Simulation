"""Generate repeatable test shapes: star, diamond, triangle, transparent star."""
import os
import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def star(size=400, r_out=180, r_in=75):
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / 5
        pts.append([size / 2 + r * np.cos(a), size / 2 + r * np.sin(a)])
    return np.array(pts, np.int32)


def canvas(pts, size=400):
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    return img


s = star()
cv2.imwrite(f"{here}/star.png", canvas(s))
cv2.imwrite(f"{here}/diamond.png", canvas(np.array([[200, 30], [370, 200], [200, 370], [30, 200]], np.int32)))
cv2.imwrite(f"{here}/triangle.png", canvas(np.array([[200, 40], [370, 350], [30, 350]], np.int32)))
rgba = np.zeros((400, 400, 4), np.uint8)   # two-tone star, transparent background
cv2.fillPoly(rgba, [s], (0, 140, 255, 255))
rgba[:200][rgba[:200, :, 3] > 0] = (200, 80, 40, 255)
cv2.imwrite(f"{here}/star_transparent.png", rgba)
