"""Particle -> target assignment. perm[i] = index of the target given to particle i."""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def greedy(starts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Baseline: particles in ID order each grab the nearest still-free target.
    Fast but order-dependent; late particles get leftovers (long, crossing paths)."""
    cost = cdist(starts, targets)
    free = np.ones(len(targets), bool)
    perm = np.empty(len(starts), int)
    for i in range(len(starts)):
        row = np.where(free, cost[i], np.inf)
        j = int(row.argmin())
        perm[i] = j
        free[j] = False
    return perm


def hungarian(starts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Optimal: minimum total Euclidean travel distance (linear_sum_assignment)."""
    rows, cols = linear_sum_assignment(cdist(starts, targets))
    perm = np.empty(len(starts), int)
    perm[rows] = cols
    return perm


def hungarian_sq(starts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Optimal for total SQUARED distance. Collision-safe property: for synchronized
    straight-line motion, any two particles stay at least
    min(start gap, end gap) / sqrt(2) apart, because a swap cannot lower the cost
    (so (s_i - s_j).(t_i - t_j) >= 0 for every pair)."""
    rows, cols = linear_sum_assignment(cdist(starts, targets, "sqeuclidean"))
    perm = np.empty(len(starts), int)
    perm[rows] = cols
    return perm


METHODS = {"greedy": greedy, "hungarian": hungarian, "hungarian_sq": hungarian_sq}


def assign(starts, targets, method: str = "hungarian") -> np.ndarray:
    if len(starts) != len(targets):
        raise ValueError("starts and targets must have the same length")
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}' (use {', '.join(METHODS)})")
    return METHODS[method](np.asarray(starts, float), np.asarray(targets, float))


def travel_stats(starts, targets_assigned) -> dict:
    d = np.linalg.norm(np.asarray(targets_assigned) - np.asarray(starts), axis=1)
    return dict(total=float(d.sum()), mean=float(d.mean()), max=float(d.max()))
