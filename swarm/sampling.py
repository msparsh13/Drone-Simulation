"""Mask -> evenly distributed target positions via Lloyd relaxation."""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .formation import Formation


def sample_targets(
    mask: np.ndarray,
    n: int,
    bgr: np.ndarray | None = None,
    *,
    extent: float = 100.0,
    scale: float = 1.0,
    iterations: int = 30,
    seed: int = 0,
) -> Formation:
    """Place n targets inside the mask with near-uniform spacing.

    Seeds start at random foreground pixels, then Lloyd iterations move each
    seed to the centroid of the foreground pixels closest to it (a centroidal
    Voronoi tessellation). This avoids clumps and sparse holes. Each target's
    color is the mean image color of its cell.

    The longest image side maps to `extent * scale` world units; aspect ratio
    is preserved.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    ys, xs = np.nonzero(mask)
    if len(xs) < n:
        raise ValueError(
            f"Requested {n} particles but mask has only {len(xs)} foreground pixels."
        )
    pix = np.column_stack([xs, ys]).astype(float) + 0.5
    rng = np.random.default_rng(seed)
    seeds = pix[rng.choice(len(pix), n, replace=False)]

    for _ in range(iterations):
        _, lab = cKDTree(seeds).query(pix)
        cnt = np.bincount(lab, minlength=n)
        ok = cnt > 0
        sx = np.bincount(lab, pix[:, 0], n)
        sy = np.bincount(lab, pix[:, 1], n)
        seeds[ok, 0] = sx[ok] / cnt[ok]
        seeds[ok, 1] = sy[ok] / cnt[ok]

    colors = np.full((n, 3), 255, np.uint8)
    if bgr is not None:
        _, lab = cKDTree(seeds).query(pix)
        cnt = np.maximum(np.bincount(lab, minlength=n), 1)
        px_rgb = bgr[ys, xs][:, ::-1].astype(float)
        for c in range(3):
            colors[:, c] = np.clip(np.bincount(lab, px_rgb[:, c], n) / cnt, 0, 255)

    h, w = mask.shape
    s = extent * scale / max(h, w)
    pos = np.column_stack([(seeds[:, 0] - w / 2) * s, (h / 2 - seeds[:, 1]) * s])
    return Formation(
        pos,
        colors,
        np.arange(n),
        meta=dict(world_scale=s, width=w, height=h, extent=extent, scale=scale),
    )


def to_world(formation: Formation, mask: np.ndarray) -> np.ndarray:
    """World coordinates of every foreground pixel (for evaluation)."""
    s = formation.meta["world_scale"]
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    return np.column_stack([(xs + 0.5 - w / 2) * s, (h / 2 - ys - 0.5) * s])


def nn_distances(points: np.ndarray) -> np.ndarray:
    d, _ = cKDTree(points).query(points, k=2)
    return d[:, 1]


def evaluate_formation(formation: Formation, mask: np.ndarray) -> dict:
    """Reconstruction/uniformity metrics (world units).

    coverage_mean/max: distance from each foreground pixel to its nearest
        particle (low = shape is well represented, no holes).
    min/mean_spacing: nearest-neighbor distance between particles
        (min close to mean = even spread).
    """
    fg = to_world(formation, mask)
    d, _ = cKDTree(formation.positions).query(fg)
    nn = nn_distances(formation.positions) if len(formation) > 1 else np.array([np.inf])
    return dict(
        n=len(formation),
        coverage_mean=float(d.mean()),
        coverage_max=float(d.max()),
        min_spacing=float(nn.min()),
        mean_spacing=float(nn.mean()),
    )
