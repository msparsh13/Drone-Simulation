"""Swarm initial positions. Each row index is a particle's stable ID."""

from __future__ import annotations
import numpy as np


def random_scatter(
    n: int, extent: float = 100.0, seed: int = 0, min_sep: float = 0.0
) -> np.ndarray:
    """Random positions inside the square arena [-extent/2, extent/2]^2.

    With min_sep > 0, uses dart throwing so no two particles start closer than
    min_sep. If the arena is too crowded the remainder is placed unconstrained
    (the collision scan will then report it).
    """
    rng = np.random.default_rng(seed + 1)
    if min_sep <= 0:
        return rng.uniform(-extent / 2, extent / 2, size=(n, 2))
    cand = rng.uniform(-extent / 2, extent / 2, size=(n * 60, 2))
    cells, pts = {}, []
    for p in cand:
        cx, cy = int(p[0] // min_sep), int(p[1] // min_sep)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in cells.get((cx + dx, cy + dy), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_sep**2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            cells.setdefault((cx, cy), []).append(p)
            pts.append(p)
            if len(pts) == n:
                break
    pts = np.array(pts).reshape(-1, 2)
    if (
        len(pts) < n
    ):  # dart throwing saturated: jittered grid keeps the spacing guarantee if it fits
        cols = int(np.ceil(np.sqrt(n)))
        cell = extent / cols
        j = max(0.0, (cell - min_sep) / 2)
        cx = (np.arange(cols) + 0.5) * cell - extent / 2
        g = np.array([(x, y) for y in cx for x in cx])
        g = g[rng.permutation(len(g))[:n]] + rng.uniform(-j, j, size=(n, 2))
        pts = g
    return pts


def grid(n: int, extent: float = 100.0) -> np.ndarray:
    """Regular grid (row-major) filling 90% of the arena; first n cells used."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    half = 0.45 * extent
    xs = np.linspace(-half, half, cols) if cols > 1 else np.zeros(1)
    ys = np.linspace(half, -half, rows) if rows > 1 else np.zeros(1)
    pts = np.array([(x, y) for y in ys for x in xs])
    return pts[:n]


def make_start(
    mode: str, n: int, extent: float = 100.0, seed: int = 0, min_sep: float = 0.0
) -> np.ndarray:
    if mode == "random":
        return random_scatter(n, extent, seed, min_sep)
    if mode == "grid":
        return grid(n, extent)
    raise ValueError(f"Unknown init mode '{mode}' (use 'random' or 'grid')")


def random_scatter3d(n, extent=100.0, base_alt=20.0, seed=0, min_sep=0.0):
    """Random airborne positions in the box [-E/2,E/2]^2 x [base_alt, base_alt+E/2], min_sep apart if possible
    (dart throwing; falls back to a random subset of a regular lattice, which guarantees the spacing)."""
    rng = np.random.default_rng(seed + 1)
    lo, hi = (
        np.array([-extent / 2, -extent / 2, base_alt]),
        np.array([extent / 2, extent / 2, base_alt + extent / 2]),
    )
    if min_sep <= 0:
        return rng.uniform(lo, hi, size=(n, 3))
    cells, pts = {}, []
    for p in rng.uniform(lo, hi, size=(n * 40, 3)):
        c = tuple((p // min_sep).astype(int))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in cells.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if ((p - q) ** 2).sum() < min_sep**2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            cells.setdefault(c, []).append(p)
            pts.append(p)
            if len(pts) == n:
                break
    if len(pts) == n:
        return np.array(pts)
    nx = int(extent / min_sep) + 1
    nz = int(extent / 2 / min_sep) + 1
    gx = lo[0] + np.arange(nx) * min_sep
    gz = lo[2] + np.arange(nz) * min_sep
    lat = np.array([(x, y, z) for x in gx for y in gx for z in gz])
    if len(lat) >= n:
        return lat[rng.choice(len(lat), n, replace=False)]
    return rng.uniform(lo, hi, size=(n, 3))


def grid3d(n, extent=100.0, base_alt=20.0):
    """Flat grid at base_alt (the hover/launch grid)."""
    g = grid(n, extent)
    return np.column_stack([g, np.full(len(g), float(base_alt))])


def make_start3d(mode, n, extent=100.0, base_alt=20.0, seed=0, min_sep=0.0):
    if mode == "random":
        return random_scatter3d(n, extent, base_alt, seed, min_sep)
    if mode == "grid":
        return grid3d(n, extent, base_alt)
    raise ValueError(f"Unknown init mode '{mode}' (use 'random' or 'grid')")
