"""Generate show-style formation images (transparent PNGs, thick shapes that survive ~300-1000 drones).
Run:  py samples/make_show_samples.py      (adds files next to this script)"""
import os, sys
import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from swarm.lights import hsv

AA = cv2.LINE_AA
def C(r, g, b): return (b, g, r, 255)
def canvas(w=500, h=500): return np.zeros((h, w, 4), np.uint8)
def poly(img, pts, col): cv2.fillPoly(img, [np.array(pts, np.int32)], col, AA)
def disc(img, c, r, col): cv2.circle(img, tuple(map(int, c)), int(r), col, -1, AA)
def save(name, img): cv2.imwrite(os.path.join(here, name), img)

def star_pts(cx, cy, ro, ri, rot=-np.pi / 2):
    return [(cx + (ro if i % 2 == 0 else ri) * np.cos(rot + i * np.pi / 5),
             cy + (ro if i % 2 == 0 else ri) * np.sin(rot + i * np.pi / 5)) for i in range(10)]

# heart
im = canvas(); t = np.linspace(0, 2 * np.pi, 400)
x = 16 * np.sin(t) ** 3; y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
poly(im, np.column_stack([250 + 13 * x, 250 - 13 * (y + 2.5)]), C(230, 30, 60)); save("heart.png", im)

# three overlapping rings
im = canvas()
for c, col in (((250, 185), C(240, 50, 60)), ((185, 300), C(60, 200, 90)), ((315, 300), C(60, 110, 240))):
    cv2.circle(im, c, 100, col, 36, AA)
save("venn_rings.png", im)

# smiley (eyes and mouth are real holes)
im = canvas(); disc(im, (250, 250), 225, C(255, 210, 0))
for c in ((180, 190), (320, 190)): cv2.ellipse(im, c, (28, 48), 0, 0, 360, (0, 0, 0, 0), -1)
cv2.ellipse(im, (250, 270), (125, 105), 0, 15, 165, (0, 0, 0, 0), 34); save("smiley.png", im)

# gold star
im = canvas(); poly(im, star_pts(250, 265, 235, 98), C(255, 200, 40)); save("star_gold.png", im)

# christmas tree
im = canvas()
for tri in ([(250, 40), (130, 190), (370, 190)], [(250, 110), (100, 310), (400, 310)], [(250, 190), (60, 420), (440, 420)]):
    poly(im, tri, C(30, 150, 60))
cv2.rectangle(im, (222, 415), (278, 485), C(130, 85, 40), -1)
poly(im, star_pts(250, 55, 48, 20), C(255, 220, 40))
for c, col in (((222, 190), C(230, 40, 50)), ((285, 255), C(70, 120, 240)), ((200, 330), C(255, 220, 60)),
               ((310, 375), C(230, 40, 50)), ((160, 390), C(70, 120, 240)), ((380, 395), C(255, 220, 60))):
    disc(im, c, 17, col)
save("tree.png", im)

# rocket
im = canvas()
poly(im, [(250, 20), (320, 120), (320, 335), (180, 335), (180, 120)], C(235, 235, 245))
poly(im, [(250, 20), (320, 125), (180, 125)], C(230, 50, 50))
poly(im, [(180, 250), (105, 375), (180, 335)], C(230, 50, 50)); poly(im, [(320, 250), (395, 375), (320, 335)], C(230, 50, 50))
disc(im, (250, 205), 40, C(50, 120, 230))
poly(im, [(200, 340), (300, 340), (250, 480)], C(255, 140, 0)); poly(im, [(223, 340), (277, 340), (250, 425)], C(255, 225, 60))
save("rocket.png", im)

# flower
im = canvas()
for k in range(8):
    a = k * np.pi / 4
    cv2.ellipse(im, (int(250 + 105 * np.cos(a)), int(250 + 105 * np.sin(a))), (82, 42), np.degrees(a), 0, 360, C(255, 105, 180), -1, AA)
disc(im, (250, 250), 62, C(255, 215, 0)); save("flower.png", im)

# butterfly
im = canvas()
cv2.ellipse(im, (150, 170), (115, 80), -30, 0, 360, C(255, 140, 0), -1, AA); cv2.ellipse(im, (350, 170), (115, 80), 30, 0, 360, C(255, 140, 0), -1, AA)
cv2.ellipse(im, (172, 330), (85, 62), 25, 0, 360, C(160, 70, 210), -1, AA); cv2.ellipse(im, (328, 330), (85, 62), -25, 0, 360, C(160, 70, 210), -1, AA)
cv2.ellipse(im, (250, 255), (18, 135), 0, 0, 360, C(245, 235, 255), -1, AA); save("butterfly.png", im)

# globe
im = canvas(); disc(im, (250, 250), 225, C(30, 110, 230))
land = np.zeros((500, 500), np.uint8)
for p in ([(120, 120), (200, 90), (260, 130), (230, 200), (250, 260), (210, 330), (170, 300), (150, 220), (100, 190)],
          [(300, 150), (390, 140), (430, 220), (380, 265), (330, 230)], [(290, 300), (365, 312), (385, 385), (320, 425), (280, 360)]):
    cv2.fillPoly(land, [np.array(p, np.int32)], 255)
circ = np.zeros((500, 500), np.uint8); cv2.circle(circ, (250, 250), 225, 255, -1)
im[(land > 0) & (circ > 0)] = C(70, 185, 90); save("globe.png", im)

# spiral galaxy
im = canvas()
for arm in (0, np.pi):
    for r in np.arange(15, 232, 1.0):
        th = r / 38 + arm; h = 0.5 + 0.35 * r / 232
        col = tuple(int(v) for v in hsv(np.array([h]))[0][::-1]) + (255,)
        disc(im, (250 + r * np.cos(th), 250 + r * np.sin(th)), 20, col)
disc(im, (250, 250), 42, C(255, 255, 255)); save("galaxy.png", im)

# text
def text(name, s, w, h, col, margin=40):
    im = canvas(w, h); (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    sc = (w - 2 * margin) / tw; thick = max(int(sc * 4.2), 2)
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, sc, thick)
    cv2.putText(im, s, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX, sc, col, thick, AA); save(name, im)
text("text_hello.png", "HELLO", 900, 300, C(120, 230, 255)); text("text_2027.png", "2027", 700, 300, C(255, 170, 40))

# fireworks burst
im = canvas(); cols = hsv(np.arange(14) / 14)
for k in range(14):
    a = k * 2 * np.pi / 14; col = tuple(int(v) for v in cols[k][::-1]) + (255,)
    cv2.line(im, (int(250 + 45 * np.cos(a)), int(250 + 45 * np.sin(a))), (int(250 + 190 * np.cos(a)), int(250 + 190 * np.sin(a))), col, 22, AA)
    disc(im, (250 + 215 * np.cos(a), 250 + 215 * np.sin(a)), 26, col)
disc(im, (250, 250), 52, C(255, 255, 255)); save("fireworks.png", im)

# crown
im = canvas(); gold = C(255, 195, 30)
poly(im, [(80, 335), (60, 130), (160, 235), (250, 100), (340, 235), (440, 130), (420, 335)], gold)
cv2.rectangle(im, (75, 335), (425, 410), gold, -1)
for c, col in (((60, 130), C(230, 40, 50)), ((250, 100), C(70, 120, 240)), ((440, 130), C(230, 40, 50))): disc(im, c, 28, col)
for c, col in (((160, 372), C(70, 120, 240)), ((250, 372), C(230, 40, 50)), ((340, 372), C(70, 120, 240))): disc(im, c, 19, col)
save("crown.png", im)
print("done")
