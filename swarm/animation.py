"""Linear-path animation math (mirrored in static/index.html)."""
from __future__ import annotations
import numpy as np

EASINGS = {
    "linear": lambda x: x,
    "cubic": lambda x: np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2),
    "out": lambda x: 1 - (1 - x) ** 3,
    "smooth": lambda x: x ** 3 * (10 - 15 * x + 6 * x ** 2),  # min-jerk: zero v and a at both ends
}


def positions_at(starts, targets, t: float, duration: float, easing: str = "cubic"):
    """Particle positions at time t. Exactly equals targets for t >= duration."""
    p = min(max(t / duration, 0.0), 1.0)
    e = float(EASINGS[easing](np.float64(p)))
    if p >= 1.0:
        return np.array(targets, float)
    return starts + (targets - starts) * e
