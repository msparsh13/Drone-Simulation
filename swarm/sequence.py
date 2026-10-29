"""Show sequence: [takeoff from ground grid] -> formation 1 -> ... -> formation N -> [landing grid -> descent]."""
from __future__ import annotations
import time
import numpy as np
from .pipeline import formation_from_arrays
from .init import make_start
from .assignment import assign, travel_stats
from .planner import plan_segment
from .collision import SPEED_FACTOR, ACCEL_FACTOR
from .colors import lift_dark
from .sampling import evaluate_formation, nn_distances

START_COLOR = (150, 158, 172)
HOLD = 5.0  # default seconds each formation is held (time for light effects)


def _vertical_phase(n, height, climb_speed, amax, waves, gap):
    """Vertical climb/descent in waves. Drones keep their horizontal cell, so horizontal separation
    (checked on the grid layout) already protects them; waves reduce simultaneous prop wash."""
    if height <= 0:
        raise ValueError("Base altitude must be positive for takeoff/landing.")
    tc = SPEED_FACTOR["smooth"] * height / max(climb_speed, 1e-6)
    if amax: tc = max(tc, (ACCEL_FACTOR["smooth"] * height / amax) ** 0.5)
    tc = float(np.ceil(tc * 10) / 10); waves = max(1, int(waves))
    return dict(climb_time=tc, duration=tc + (waves - 1) * gap, height=float(height), waves=waves,
                delay=np.round((np.arange(n) % waves) * gap, 3).tolist(),
                peak_climb_speed=SPEED_FACTOR["smooth"] * height / tc,
                peak_accel=ACCEL_FACTOR["smooth"] * height / tc ** 2)


def build_sequence(images, *, n=500, threshold=127, invert=False, scale=1.0, init="random",
                   method="hungarian_sq", seed=0, extent=100.0, min_sep=2.5, avoid=True,
                   flight_time=10.0, easing="smooth", vmax=6.0, amax=2.0, auto_time=True,
                   pos_error=0.3, max_layers=4, solid_color=None, hold=HOLD, min_brightness=70.0,
                   takeoff=False, landing=False, base_alt=30.0, climb_speed=2.5, waves=4,
                   wave_gap=3.0) -> dict:
    """images: list of (bgr, alpha). Particle i keeps its identity through every segment."""
    if not images:
        raise ValueError("At least one formation is required.")
    forms = [formation_from_arrays(b, a, n, threshold, invert, scale, extent, seed=seed) for b, a in images]
    n = len(forms[0][1])
    notes = []
    if takeoff:
        init = "grid"; notes.append("Takeoff: drones start on a ground grid and climb vertically in waves.")
    cur = make_start(init, n, extent, seed, min_sep * 1.42)  # >= sqrt(2) * min_sep
    r3 = lambda a: np.round(a, 3).tolist()
    segs, lifted = [], 0
    gap_of = lambda P: float(nn_distances(P).min()) if len(P) > 1 else float("inf")

    def add(tpos, tcols, cov, kept=None, kind="formation"):
        nonlocal cur, lifted
        t0 = time.perf_counter(); perm = assign(cur, tpos, method); ms = (time.perf_counter() - t0) * 1000
        tgt = tpos[perm]
        if kept is not None: col = np.asarray(kept, np.uint8)
        elif solid_color is not None: col = np.tile(np.array(solid_color, np.uint8), (n, 1))
        else:
            col, k = lift_dark(tcols[perm], min_brightness); lifted += k
        plan = plan_segment(cur, tgt, flight_time, min_sep, easing, vmax, avoid, half=extent / 2 + 1e-6,
                            amax=amax, auto_time=auto_time, max_layers=max_layers, pos_error=pos_error)
        bound = min(gap_of(cur), gap_of(tgt)) / 2 ** 0.5 if method == "hungarian_sq" else None
        m = dict(particles=n, method=method, assign_ms=ms, coverage_mean=cov,
                 **{f"travel_{k}": v for k, v in travel_stats(cur, tgt).items()},
                 **{k: plan[k] for k in ("collisions_naive", "collisions_planned", "min_sep_naive", "min_sep_planned",
                                         "peak_speed", "peak_accel", "flight_time", "total_time", "layers_used",
                                         "peak_vertical_speed")},
                 clearance=plan.get("clearance"), separation_bound=bound)
        segs.append(dict(kind=kind, start=r3(cur), target=r3(tgt), color=col.tolist(), delay=r3(plan["delays"]),
                         lift=r3(plan["lift"]), flight=plan["flight_time"], total=plan["total_time"],
                         metrics=m, warnings=plan["warnings"]))
        cur = tgt

    for mask, f in forms:
        add(f.positions, f.colors, evaluate_formation(f, mask)["coverage_mean"])
    take = _vertical_phase(n, base_alt, climb_speed, amax, waves, wave_gap) if takeoff else None
    land = None
    if landing:
        add(make_start("grid", n, extent), None, None, kept=segs[-1]["color"], kind="landing_grid")
        land = _vertical_phase(n, base_alt, climb_speed, amax, waves, wave_gap)
        notes.append("Landing: drones return to the ground grid, then descend vertically in waves.")
    if lifted:
        notes.append(f"{lifted} drone colour(s) were brightened because LEDs cannot show black "
                     f"(minimum brightness {min_brightness:g}).")
    tri = min(0.45 * float(np.median(nn_distances(f.positions))) for _, f in forms) if n > 1 else 2.0
    return dict(extent=extent, tri=tri, n=n, start_color=list(START_COLOR), segments=segs, min_sep=min_sep,
                hold=float(hold), takeoff=take, landing=land, notes=notes,
                params=dict(extent=extent, scale=scale, n=n, min_sep=min_sep, vmax=vmax, amax=amax,
                            pos_error=pos_error, easing=easing, method=method, init=init, base_alt=base_alt,
                            takeoff=bool(takeoff), landing=bool(landing)))
