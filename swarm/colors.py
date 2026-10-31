"""LED colour helpers."""

from __future__ import annotations
import numpy as np

WARM_WHITE = (255, 244, 214)


def lift_dark(colors, min_lum: float = 70.0):
    """LEDs cannot show black. Brighten colours darker than min_lum (0-255 luminance),
    keeping hue; near-black becomes warm white. Returns (uint8 colours, number changed)."""
    c = np.asarray(colors, float)
    if min_lum <= 0 or len(c) == 0:
        return c.astype(np.uint8), 0
    lum = c @ np.array([0.2126, 0.7152, 0.0722])
    low = lum < min_lum
    out = c.copy()
    black = low & (c.max(axis=1) < 8)
    out[black] = WARM_WHITE
    mid = low & ~black
    out[mid] = np.clip(c[mid] * (min_lum / np.maximum(lum[mid], 1e-6))[:, None], 0, 255)
    out = np.round(out).astype(np.uint8)
    return out, int((out != np.round(c).astype(np.uint8)).any(axis=1).sum())
