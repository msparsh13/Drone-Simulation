"""Milestone 1 CLI:  python -m swarm.cli IMAGE -n 500 -o out.png"""

from __future__ import annotations
import argparse
import csv
import cv2
from .imaging import load_image
from .pipeline import formation_from_arrays
from .sampling import evaluate_formation
from .render import render_formation


def build_formation(
    path,
    n=500,
    threshold=127,
    invert=False,
    scale=1.0,
    extent=100.0,
    max_side=400,
    iterations=30,
    seed=0,
):
    bgr, alpha = load_image(path)
    return formation_from_arrays(
        bgr, alpha, n, threshold, invert, scale, extent, max_side, iterations, seed
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert an image into a triangle-swarm formation."
    )
    p.add_argument("image")
    p.add_argument("-n", "--particles", type=int, default=500)
    p.add_argument(
        "-t", "--threshold", type=int, default=127, help="0-255, or -1 for Otsu"
    )
    p.add_argument(
        "--invert", action="store_true", help="light shape on dark background"
    )
    p.add_argument(
        "--scale", type=float, default=1.0, help="formation size relative to arena"
    )
    p.add_argument(
        "--extent", type=float, default=100.0, help="arena size in world units"
    )
    p.add_argument("--max-side", type=int, default=400, help="processing resolution")
    p.add_argument("--iterations", type=int, default=30, help="Lloyd relaxation steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("-o", "--out", default="formation.png")
    p.add_argument("--csv", help="also write targets (id,x,y,r,g,b)")
    a = p.parse_args(argv)

    mask, f = build_formation(
        a.image,
        a.particles,
        a.threshold,
        a.invert,
        a.scale,
        a.extent,
        a.max_side,
        a.iterations,
        a.seed,
    )
    cv2.imwrite(a.out, render_formation(f, use_colors=not a.no_color))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "x", "y", "r", "g", "b"])
            for i, (x, y), c in zip(f.ids, f.positions, f.colors):
                w.writerow([i, f"{x:.4f}", f"{y:.4f}", *c])
    m = evaluate_formation(f, mask)
    print(
        f"particles={m['n']}  coverage mean/max={m['coverage_mean']:.3f}/{m['coverage_max']:.3f}  "
        f"spacing min/mean={m['min_spacing']:.3f}/{m['mean_spacing']:.3f}"
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
