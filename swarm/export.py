"""Export a planned show as a trajectory package: CSV, lights CSV, JSON, validation report (zip)."""
from __future__ import annotations
import csv
import io
import json
import zipfile
import numpy as np
from .animation import EASINGS
from .collision import positions
from .lights import resolve_cues, apply_cues, check_cues

DISCLAIMER = """\
SIMULATION-LEVEL OUTPUT - NOT A CERTIFIED FLIGHT PLAN.
These trajectories were checked against separation, speed, acceleration, altitude and
positioning-error limits *in simulation only*. Takeoff/landing are modelled as simple vertical climbs
and descents from a ground grid. Not modelled: wind, battery curves, GPS/RTK dropout, communication
loss, vehicle dynamics/tracking error, prop wash, terrain, obstacles. Without takeoff/landing enabled,
the start layout is treated as already airborne.

Before ANY real flight: get the required aviation/regulatory approval for your location, import
the trajectories into your drone vendor's show software and re-validate there, run software-in-the-loop
and low-altitude tests, use RTK/GNSS-grade positioning, configure geofence + failsafes (return-to-home,
lost-link, low battery), define wind limits and an exclusion zone, and keep a pilot able to abort.
"""
SMOOTH = EASINGS["smooth"]


def to_world(P, plane, base_alt, extent):
    """Internal (x, y, lift) -> world (x, y, z). horizontal: formation lies flat at base_alt.
    vertical: formation stands in the x-z plane (audience view), lift becomes depth (y)."""
    if plane == "vertical":
        return np.column_stack([P[:, 0], P[:, 2], base_alt + P[:, 1] + extent / 2])
    return np.column_stack([P[:, 0], P[:, 1], base_alt + P[:, 2]])


def _taus(a, b, dt):
    t = np.arange(a, b, dt)
    return t if len(t) and abs(t[-1] - b) < 1e-9 else np.append(t, b)


def _samples(seq, easing, dt):
    """Yield (show time, internal positions (n,3), base colours (n,3), phase) every dt seconds."""
    n, hold, segs = seq["n"], seq["hold"], seq["segments"]
    tk, ld = seq.get("takeoff"), seq.get("landing")
    prev, t0 = np.tile(np.array(seq["start_color"], float), (n, 1)), 0.0
    if tk:
        xy, d = np.array(segs[0]["start"], float), np.array(tk["delay"], float)
        for tau in _taus(0.0, tk["duration"], dt):
            p = np.clip((tau - d) / tk["climb_time"], 0, 1)
            yield tau, np.column_stack([xy, -tk["height"] * (1 - SMOOTH(p))]), prev, "takeoff"
        t0 = tk["duration"]
    for k, sg in enumerate(segs):
        s, tg, dl, lf = (np.array(sg[x], float) for x in ("start", "target", "delay", "lift"))
        col, T, total = np.array(sg["color"], float), sg["flight"], sg["total"]
        first = 0.0 if (k == 0 and not tk) else dt
        for tau in _taus(first, total + hold, dt):
            yield (t0 + tau, positions(s, tg, dl, T, min(tau, total), easing, lf),
                   prev + (col - prev) * np.clip((tau - dl) / T, 0, 1)[:, None], "show")
        prev, t0 = col, t0 + total + hold
    if ld:
        xy, d = np.array(segs[-1]["target"], float), np.array(ld["delay"], float)
        for tau in _taus(dt, ld["duration"], dt):
            p = np.clip((tau - d) / ld["climb_time"], 0, 1)
            yield t0 + tau, np.column_stack([xy, -ld["height"] * SMOOTH(p)]), prev, "landing"


def build_export(seq: dict, *, easing="smooth", rate_hz=5.0, plane="horizontal", base_alt=None,
                 max_alt=120.0, min_alt=2.0, endurance_min=15.0, cues=None, light_rate_hz=20.0) -> bytes:
    n, hold, ext, segs = seq["n"], seq["hold"], seq["extent"], seq["segments"]
    tk, ld = seq.get("takeoff"), seq.get("landing")
    if (tk or ld) and plane == "vertical":
        raise ValueError("Takeoff/landing is only supported for the horizontal formation plane. "
                         "Switch the plane to horizontal or turn takeoff/landing off.")
    base_alt = float(seq["params"].get("base_alt", 30.0) if base_alt is None else base_alt)
    dt = 1.0 / float(np.clip(rate_hz, 0.5, 10.0))
    t_tk, t_ld = (tk["duration"] if tk else 0.0), (ld["duration"] if ld else 0.0)
    rc = resolve_cues(cues, segs, hold, t_tk, t_ld); cue_warn = check_cues(cues)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["drone_id", "t_s", "x_m", "y_m", "z_m", "r", "g", "b"])
    zmin, zmax = np.inf, -np.inf
    for t, P, C, phase in _samples(seq, easing, dt):
        W = np.round(to_world(P, plane, base_alt, ext), 3)
        if phase == "show":
            zmin, zmax = min(zmin, W[:, 2].min()), max(zmax, W[:, 2].max())
        for i in range(n):
            w.writerow([i, round(t, 3), *W[i], *C[i].astype(int).tolist()])
    total = t_tk + sum(sg["total"] + hold for sg in segs) + t_ld
    lights_csv = None
    if rc:
        lrate = float(min(np.clip(light_rate_hz, 2.0, 30.0), max(2.0, 1.5e6 / (n * total))))
        lb = io.StringIO(); lw = csv.writer(lb); lw.writerow(["drone_id", "t_s", "r", "g", "b"])
        for t, P, C, _ in _samples(seq, easing, 1.0 / lrate):
            L = np.clip(apply_cues(C, t, P[:, :2], rc, ext), 0, 255).astype(int)
            for i in range(n):
                lw.writerow([i, round(t, 3), *L[i].tolist()])
        lights_csv = lb.getvalue()
    all_ok = all(not sg["warnings"] for sg in segs)
    alt_ok = zmin >= min_alt - 1e-9 and zmax <= max_alt + 1e-9
    dur_ok = total / 60 <= endurance_min
    verdict = "PASS (simulation-level)" if all_ok and alt_ok and dur_ok else "FAIL - see items below"

    L = ["# Trajectory validation report", "", f"**Result: {verdict}**", "",
         f"- Drones: {n}   Formations: {len(segs)}   Show duration: {total:.1f} s ({total / 60:.1f} min; endurance limit {endurance_min} min -> {'ok' if dur_ok else 'EXCEEDED'})",
         f"- Plane: {plane}   Base altitude: {base_alt} m   Show-phase altitude range: {zmin:.1f} .. {zmax:.1f} m (allowed {min_alt}..{max_alt} m -> {'ok' if alt_ok else 'VIOLATION'})",
         f"- Arena: {ext} m   Min separation: {seq['min_sep']} m   Easing: {easing}   Sample rate: {1 / dt:g} Hz", ""]
    for name, ph in (("Takeoff", tk), ("Landing", ld)):
        if ph:
            L += [f"## {name}", f"- {ph['waves']} wave(s), vertical {ph['height']:.0f} m in {ph['climb_time']:.1f} s each, "
                  f"phase duration {ph['duration']:.1f} s", f"- Peak vertical speed {ph['peak_climb_speed']:.2f} m/s, "
                  f"peak accel {ph['peak_accel']:.2f} m/s^2", "- Horizontal cells are the ground grid; separation checked on the grid layout.", ""]
    for k, sg in enumerate(segs):
        m = sg["metrics"]; f2 = lambda x, d=2: "n/a" if x is None else format(x, f".{d}f")
        L += [f"## Transition {k + 1}" + (" (return to landing grid)" if sg.get("kind") == "landing_grid" else ""),
              f"- Assignment: {m['method']} ({m['assign_ms']:.1f} ms), total travel {m['travel_total']:.0f} m, max {m['travel_max']:.1f} m",
              f"- Flight time {m['flight_time']:.1f} s (+stagger -> {m['total_time']:.1f} s), layers used {m['layers_used']}",
              f"- Collision pairs naive -> planned: {m['collisions_naive']} -> {m['collisions_planned']}; closest approach {f2(m['min_sep_planned'])} m",
              f"- Peak speed {m['peak_speed']:.2f} m/s, peak accel {f2(m['peak_accel'])} m/s^2",
              f"- Clearance after 2x positioning error: {f2(m['clearance'])} m"]
        L += [f"- WARNING: {x}" for x in sg["warnings"]] + [""]
    if seq.get("notes"):
        L += ["## Notes"] + [f"- {x}" for x in seq["notes"]] + [""]
    if rc:
        L += ["## Lighting", f"- {len(rc)} cue(s); high-rate colours (with effects) are in lights.csv; the RGB columns in trajectories.csv are the base colours only.",
              f"- Light sample rate: {lrate:g} Hz (reduced automatically for very large shows)"]
        L += [f"- {c['effect']} {c.get('anchor', {'type': 'show'})} scope={c.get('scope', 'all')} t={c['t0']:.1f}..{c['t1']:.1f}s" for c in rc]
        L += [f"- WARNING: {x}" for x in cue_warn] + [""]
    L += ["## Read this first", "", DISCLAIMER]
    meta = dict(schema="swarm-sim/2", units=dict(length="m", time="s"), plane=plane, base_alt_m=base_alt,
                params=seq["params"], hold_s=hold, takeoff=tk, landing=ld,
                validation=dict(verdict=verdict, show_altitude_range=[float(zmin), float(zmax)], show_duration_s=total),
                transitions=segs, light_cues=cues or [], notes=DISCLAIMER)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("trajectories.csv", buf.getvalue())
        if lights_csv: z.writestr("lights.csv", lights_csv)
        z.writestr("plan.json", json.dumps(meta))
        z.writestr("REPORT.md", "\n".join(L))
    return out.getvalue()
