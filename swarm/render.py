"""Render a formation as small triangles to an image (OpenCV)."""

from __future__ import annotations
import cv2
import numpy as np
from .formation import Formation
from .sampling import nn_distances


def render_formation(
    f: Formation,
    size: int = 900,
    view_extent: float | None = None,
    tri_size: float | None = None,
    bg=(255, 255, 255),
    use_colors: bool = True,
) -> np.ndarray:
    """Draw each particle as an upward-pointing triangle. Returns BGR image.

    view_extent: world width/height shown (default: formation extent * 1.15).
    tri_size: triangle circumradius in world units (default 0.45 * median spacing).
    """
    view = view_extent or f.meta.get("extent", 100.0) * 1.15
    px_per_unit = size / view
    if tri_size is None:
        tri_size = (
            0.45 * float(np.median(nn_distances(f.positions))) if len(f) > 1 else 2.0
        )
    r = max(tri_size * px_per_unit, 2.0)
    canvas = np.full((size, size, 3), bg, np.uint8)
    local = np.array([[0, -r], [-0.866 * r, 0.5 * r], [0.866 * r, 0.5 * r]])
    for p, c in zip(f.positions, f.colors):
        cx, cy = size / 2 + p[0] * px_per_unit, size / 2 - p[1] * px_per_unit
        pts = np.round(local + [cx, cy]).astype(np.int32)
        col = tuple(int(v) for v in c[::-1]) if use_colors else (40, 40, 40)
        cv2.fillConvexPoly(canvas, pts, col, cv2.LINE_AA)
    return canvas
