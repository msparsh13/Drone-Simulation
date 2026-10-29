"""Constraint-aware planner for one transition: speed/accel-fit timing, layers, stagger."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .collision import scan, choose_steps, SPEED_FACTOR, ACCEL_FACTOR


def _layers(pairs, n, min_sep, max_layers):
    """Greedy graph colouring of the conflict graph -> per-particle lift (metres)."""
    adj = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    color = {}
    for v in sorted(adj, key=lambda v: -len(adj[v])):
        used = {color[u] for u in adj[v] if u in color}
        c = 0
        while c in used: c += 1
        color[v] = c % max_layers
    lift = np.zeros(n)
    for v, c in color.items(): lift[v] = c * 1.25 * min_sep
    return lift


def plan_segment(starts, targets, flight_time, min_sep, easing="smooth", vmax=None, avoid=True,
                 half=None, max_rounds=6, amax=None, auto_time=False, max_layers=4,
                 pos_error=0.0) -> dict:
    """Check straight paths and, if `avoid`, resolve conflicts.

    1. auto_time: raise flight time until peak speed <= vmax and peak accel <= amax.
    2. Collision scan (3D, sampled in time).
    3. If pairs conflict: try altitude/depth layers (graph colouring), keep if it helps.
    4. Then try staggered departures, keep only if it helps.
    Anything unresolved is listed in `warnings`; a plan is never silently declared valid.
    """
    n, T = len(starts), float(flight_time)
    d = np.linalg.norm(targets - starts, axis=1)
    dmax = float(d.max()) if n else 0.0
    SF, AF = SPEED_FACTOR[easing], ACCEL_FACTOR[easing]
    t_min = 0.0
    if vmax: t_min = max(t_min, SF * dmax / vmax)
    if amax and AF is not None: t_min = max(t_min, (AF * dmax / amax) ** 0.5)
    if auto_time and T < t_min: T = float(np.ceil(t_min * 10) / 10)
    vpk = SF * dmax / T
    apk = None if AF is None else AF * dmax / T ** 2
    warnings, delays, lift = [], np.zeros(n), np.zeros(n)
    if vmax and vpk > vmax + 1e-9:
        warnings.append(f"Peak speed {vpk:.1f} exceeds limit {vmax:.1f}; needs flight time >= {SF * dmax / vmax:.1f} s.")
    if amax and apk is not None and apk > amax + 1e-9:
        warnings.append(f"Peak acceleration {apk:.2f} exceeds limit {amax:.2f}; needs flight time >= {(AF * dmax / amax) ** .5:.1f} s.")
    if amax and AF is None:
        warnings.append("This easing starts/stops impulsively, so the acceleration limit cannot be met; use smooth easing.")
    if half is not None and max(np.abs(starts).max(), np.abs(targets).max()) > half + 1e-6:
        warnings.append("Some positions are outside the operating area.")
    out = dict(collisions_naive=0, collisions_planned=0, min_sep_naive=None, min_sep_planned=None)

    if min_sep > 0 and n > 1:
        eps = min_sep - 1e-9
        static = cKDTree(starts).query_pairs(eps) | cKDTree(targets).query_pairs(eps)
        if static:
            warnings.append(f"{len(static)} pair(s) start or end closer than the minimum separation ({min_sep}); "
                            f"planning cannot fix this. Use fewer particles, a larger arena/scale, or a smaller separation.")
        res = naive = scan(starts, targets, delays, T, min_sep, easing,
                           choose_steps(dmax, T, T, min_sep, easing), lift)
        if avoid and res["pairs"] - static:
            if max_layers > 1:
                trial_lift = _layers(res["pairs"] - static, n, min_sep, max_layers)
                trial = scan(starts, targets, delays, T, min_sep, easing,
                             choose_steps(dmax, T, T, min_sep, easing), trial_lift)
                if len(trial["pairs"]) < len(res["pairs"]): lift, res = trial_lift, trial
            gap, cap = 0.5 * T, 3.0 * T
            for _ in range(max_rounds):
                fix = res["pairs"] - static
                if not fix: break
                new = delays.copy()
                js = np.unique([max(q) for q in fix])
                new[js] = np.minimum(new[js] + gap, cap)
                if np.array_equal(new, delays): break
                trial = scan(starts, targets, new, T, min_sep, easing,
                             choose_steps(dmax, T, T + new.max(), min_sep, easing), lift)
                if len(trial["pairs"]) >= len(res["pairs"]): break
                delays, res = new, trial
        if res["pairs"]:
            warnings.append(f"{len(res['pairs'])} collision pair(s) remain (closest approach {res['min_sep']:.2f} < "
                            f"{min_sep}). Constraints NOT satisfied: this plan is not collision-free.")
        clearance = res["min_sep"] - 2 * pos_error
        if pos_error > 0 and clearance <= 0 and not res["pairs"]:
            warnings.append(f"Closest approach {res['min_sep']:.2f} minus 2x positioning error ({pos_error}) leaves no "
                            f"worst-case clearance. Increase the minimum separation or use better positioning.")
        out.update(collisions_naive=len(naive["pairs"]), collisions_planned=len(res["pairs"]),
                   min_sep_naive=naive["min_sep"], min_sep_planned=res["min_sep"], clearance=clearance)

    out.update(delays=delays, lift=lift, flight_time=T, total_time=T + (float(delays.max()) if n else 0.0),
               peak_speed=vpk, peak_accel=apk, peak_vertical_speed=float(lift.max() * np.pi / T) if n else 0.0,
               layers_used=int(len(np.unique(lift[lift > 0]))), warnings=warnings)
    return out
