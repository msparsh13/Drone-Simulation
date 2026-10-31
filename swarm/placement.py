"""Place 2D formations (from images) and built-in 3D shapes into the 3D world.
World frame (metres): x = right, y = depth (away from the audience), z = up."""

from __future__ import annotations
import numpy as np
from .lights import hsv

PLACEMENTS = ("vertical", "horizontal", "cylinder", "sphere", "extrude")
SHAPES = ("sphere", "cube", "helix", "torus")


def place_2d(
    pos2,
    placement="vertical",
    base_alt=20.0,
    *,
    wrap_deg=180.0,
    layers=3,
    gap=3.0,
    seed=0,
):
    """2D image positions (n,2) -> world (n,3). The lowest point sits at base_alt (horizontal: whole plane)."""
    p = np.asarray(pos2, float)
    n = len(p)
    xr = p[:, 0] - (p[:, 0].max() + p[:, 0].min()) / 2
    yr = p[:, 1] - p[:, 1].min()
    if placement == "horizontal":
        return np.column_stack([p[:, 0], p[:, 1], np.full(n, float(base_alt))])
    if placement == "vertical":
        return np.column_stack([xr, np.zeros(n), base_alt + yr])
    if placement == "extrude":
        layer = np.random.default_rng(seed + 7).permutation(n) % max(1, int(layers))
        return np.column_stack(
            [xr, (layer - (max(1, int(layers)) - 1) / 2) * gap, base_alt + yr]
        )
    if placement == "cylinder":  # wraps around a vertical axis; arc length is preserved
        R = max(np.ptp(xr), 1e-6) / np.radians(wrap_deg)
        th = xr / R
        y = -R * np.cos(th)
        return np.column_stack([R * np.sin(th), y - y.mean(), base_alt + yr])
    if (
        placement == "sphere"
    ):  # lon/lat map onto a sphere (spacing compresses near the poles)
        R = max(np.ptp(xr), np.ptp(yr), 1e-6) / np.pi
        lon = np.clip(xr / R, -np.pi / 2, np.pi / 2)
        lat = np.clip((yr - yr.max() / 2) / R, -np.pi / 2, np.pi / 2)
        y = -R * np.cos(lat) * np.cos(lon)
        return np.column_stack(
            [
                R * np.cos(lat) * np.sin(lon),
                y - y.mean(),
                base_alt + R * (1 + np.sin(lat)),
            ]
        )
    raise ValueError(f"Unknown placement '{placement}' (use {', '.join(PLACEMENTS)})")


def shape_points(name, n, radius, base_alt=20.0, seed=0):
    """Built-in 3D shape with n roughly evenly spaced points. Returns (positions (n,3), colours (n,3) uint8)."""
    R, rng = float(radius), np.random.default_rng(seed + 11)
    if name == "sphere":  # Fibonacci lattice: near-uniform spacing
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = np.pi * (1 + 5**0.5) * i
        p = R * np.column_stack(
            [np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)]
        )
    elif name == "cube":  # random subset of a regular lattice: spacing >= lattice step
        m = int(np.ceil(n ** (1 / 3)))
        g = np.linspace(-R, R, m) if m > 1 else np.zeros(1)
        lat = np.array([(x, y, z) for x in g for y in g for z in g])
        p = lat[rng.choice(len(lat), n, replace=False)]
    elif name == "helix":  # double helix, even arc-length spacing
        k = n // 2
        t = np.linspace(0, 6 * np.pi, k)
        r = 0.45 * R
        h = np.linspace(-R, R, k)
        a = np.column_stack([r * np.cos(t), r * np.sin(t), h])
        b = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi), h])
        p = np.vstack(
            [
                a,
                b,
                np.column_stack(
                    [
                        r * np.cos(t[: n - 2 * k] + np.pi / 2),
                        r * np.sin(t[: n - 2 * k] + np.pi / 2),
                        h[: n - 2 * k],
                    ]
                ),
            ]
        )
    elif name == "torus":
        Rm, rr = 0.7 * R, 0.3 * R
        nv = max(3, int(np.ceil(np.sqrt(n * rr / Rm))))
        nu = int(np.ceil(n / nv))
        u, v = np.meshgrid(
            np.linspace(0, 2 * np.pi, nu, endpoint=False),
            np.linspace(0, 2 * np.pi, nv, endpoint=False),
        )
        u, v = u.ravel(), v.ravel()
        sel = rng.choice(len(u), n, replace=False)
        u, v = u[sel], v[sel]
        p = np.column_stack(
            [
                (Rm + rr * np.cos(v)) * np.cos(u),
                (Rm + rr * np.cos(v)) * np.sin(u),
                rr * np.sin(v),
            ]
        )
    else:
        raise ValueError(f"Unknown shape '{name}' (use {', '.join(SHAPES)})")
    p = p + [0.0, 0.0, base_alt + R]
    zn = (p[:, 2] - p[:, 2].min()) / max(np.ptp(p[:, 2]), 1e-9)
    return p, np.clip(hsv(0.55 + 0.4 * zn), 0, 255).astype(np.uint8)
