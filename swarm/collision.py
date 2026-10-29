"""Time-sampled 3D collision detection along eased straight paths with optional altitude lift."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .animation import EASINGS

# Peak velocity / acceleration of each easing as multiples of d/T and d/T^2.
SPEED_FACTOR = {"linear": 1.0, "cubic": 3.0, "out": 3.0, "smooth": 1.875}
ACCEL_FACTOR = {"linear": None, "cubic": 12.0, "out": None, "smooth": 5.7735}  # None = impulsive


def positions(starts, targets, delays, T, tau, easing="smooth", lift=None):
    """(n,3) positions at time tau. xy moves along the eased straight line; the third
    axis is a lift bump lift*sin^2(pi*p) (0 at both ends) used for layer separation."""
    p = np.clip((tau - delays) / T, 0.0, 1.0)
    e = np.asarray(EASINGS[easing](p), float)
    xy = starts + (targets - starts) * e[:, None]
    z = np.zeros(len(p)) if lift is None else lift * np.sin(np.pi * p) ** 2
    return np.column_stack([xy, z])


def choose_steps(dmax, T, total, min_sep, easing="smooth"):
    """Step count so two particles move < min_sep/2 relative to each other per step."""
    vpk = SPEED_FACTOR[easing] * dmax / T
    dt = min_sep / (4 * max(vpk, 1e-9))
    return int(np.clip(np.ceil(total / dt), 60, 500))


def scan(starts, targets, delays, T, min_sep, easing="smooth", steps=120, lift=None) -> dict:
    """{'pairs': set of (i,j) that ever came closer than min_sep (3D), 'min_sep': closest approach}.
    Sampling-based: a very short transient between samples could be missed."""
    total = T + float(delays.max())
    pairs, min_d = set(), np.inf
    for tau in np.linspace(0.0, total, steps + 1):
        P = positions(starts, targets, delays, T, tau, easing, lift)
        tree = cKDTree(P)
        pairs |= tree.query_pairs(min_sep - 1e-9)
        if len(P) > 1:
            d, _ = tree.query(P, k=2)
            min_d = min(min_d, float(d[:, 1].min()))
    return dict(pairs=pairs, min_sep=min_d)
