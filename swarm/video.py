"""Video -> keyframes. Each keyframe becomes one formation (stage) in the show.

Real drones cannot play video frame rates (they are limited to a few m/s), so a video becomes a
handful of keyframes spaced by flight time, not a 24 fps animation.
"""
from __future__ import annotations
import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _resize(fr, max_side):
    h, w = fr.shape[:2]
    if max(h, w) <= max_side:
        return fr
    f = max_side / max(h, w)
    return cv2.resize(fr, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


def _read_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, fr = cap.read()
    return fr if ok else None


def extract_keyframes(path, count=8, max_side=400, mode="motion", diff_thresh=30):
    """Return (frames, meta). frames = list of dict(bgr, alpha|None, t).

    mode 'motion'   : foreground = pixels that differ from the median background of the video
                      (needs a moving subject or an empty static background); colours kept.
    mode 'threshold': keep the whole frame; the usual threshold/invert mask is applied later.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video. Use MP4/AVI/MOV/MKV/WEBM with a common codec.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    if total <= 0:
        cap.release(); raise ValueError("Could not determine the video length.")
    count = int(np.clip(count, 2, 16))
    idx = np.unique(np.linspace(0, total - 1, count).round().astype(int))
    raw = {}
    for i in idx:
        fr = _read_at(cap, i)
        if fr is not None:
            raw[int(i)] = _resize(fr, max_side)
    if not raw:
        cap.release(); raise ValueError("No frames could be decoded from the video.")
    bg = None
    if mode == "motion":
        samples = [_read_at(cap, i) for i in np.unique(np.linspace(0, total - 1, 15).round().astype(int))]
        samples = [_resize(s, max_side) for s in samples if s is not None]
        shape = next(iter(raw.values())).shape
        bg = np.median(np.stack([s for s in samples if s.shape == shape]), axis=0).astype(np.int16)
    cap.release()
    frames = []
    for i in sorted(raw):
        fr = raw[i]; alpha = None
        if bg is not None:
            d = np.abs(fr.astype(np.int16) - bg).max(axis=2)
            m = ((d > diff_thresh) * 255).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            alpha = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        frames.append(dict(bgr=fr, alpha=alpha, t=i / fps))
    return frames, dict(fps=fps, duration=total / fps, total_frames=total)


def encode_png(bgr, alpha=None) -> bytes:
    img = bgr if alpha is None else cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
