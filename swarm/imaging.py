"""Image loading and foreground-mask extraction (OpenCV)."""
from __future__ import annotations
import cv2
import numpy as np

SUPPORTED = (".png", ".jpg", ".jpeg")


def _split(img):
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / np.iinfo(img.dtype).max)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None
    if img.shape[2] == 4:
        return np.ascontiguousarray(img[:, :, :3]), np.ascontiguousarray(img[:, :, 3])
    return img, None


def load_image(path: str):
    """Return (bgr uint8 HxWx3, alpha uint8 HxW or None)."""
    if not path.lower().endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type (use {', '.join(SUPPORTED)}): {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return _split(img)


def decode_image(data: bytes):
    """Same as load_image but from raw PNG/JPG bytes (used by the web upload)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Uploaded data is not a valid PNG/JPG image.")
    return _split(img)


def resize_keep_aspect(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    f = max_side / max(h, w)
    size = (max(1, round(w * f)), max(1, round(h * f)))
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)


def extract_mask(bgr, alpha=None, threshold: int = 127, invert: bool = False,
                 denoise: bool = True) -> np.ndarray:
    """Boolean foreground mask.

    - If the image has real transparency, foreground = alpha > 127.
    - Otherwise foreground = dark pixels (gray < threshold), i.e. a dark
      logo on a light background. Use invert=True for light-on-dark.
    - threshold=-1 selects Otsu's threshold automatically.
    """
    if alpha is not None and alpha.min() < 255:
        mask = alpha > 127
        if invert:
            mask = ~mask
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if threshold < 0:
            t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            t = threshold
        mask = gray > t if invert else gray < t
    mask = mask.astype(np.uint8)
    if denoise:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask.astype(bool)


def crop_to_content(mask, bgr):
    """Crop to the foreground bounding box so scale is relative to the shape."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty: adjust threshold/invert.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return mask[y0:y1, x0:x1], bgr[y0:y1, x0:x1]
