"""Export a planned show as a trajectory package: CSVs, JSON, validation report (zip)."""

from __future__ import annotations
import csv
import io
import json
import zipfile
import numpy as np
from .timeline import Timeline
from .lights import resolve_cues, apply_cues, check_cues

DISCLAIMER = """\
SIMULATION-LEVEL OUTPUT - NOT A CERTIFIED FLIGHT PLAN.
These trajectories were checked against separation, speed, acceleration, altitude and
positioning-error limits *in simulation only*. Takeoff/landing are modelled as simple vertical climbs
and descents from a ground grid. Failure simulations use simplified physics (point-mass drones, quadratic
drag, simple battery model) and are NOT validated against real vehicles. Not modelled: wind, GPS/RTK
dropout dynamics, communication loss behaviour of real autopilots, vehicle tracking error, prop wash,
terrain, obstacles. Without takeoff/landing enabled, the start layout is treated as already airborne.

Before ANY real flight: get the required aviation/regulatory approval for your location, import
the trajectories into your drone vendor's show software and re-validate there, run software-in-the-loop
and low-altitude tests, use RTK/GNSS-grade positioning, configure geofence + failsafes (return-to-home,
lost-link, low battery), define wind limits and an exclusion zone, and keep a pilot able to abort.
"""


def _times(total, dt):
    t = np.arange(0.0, total, dt)
    return np.append(t, total)


def build_export(
    seq: dict,
    *,
    easing="smooth",
    rate_hz=5.0,
    plane=None,
    base_alt=None,
    max_alt=120.0,
    min_alt=2.0,
    endurance_min=15.0,
    cues=None,
    light_rate_hz=20.0,
    sim=None,
) -> bytes:
    """`plane`/`base_alt` are accepted for backward compatibility (placement is now part of the plan)."""
    n, hold, ext, segs = seq["n"], seq["hold"], seq["extent"], seq["segments"]
    tk, ld = seq.get("takeoff"), seq.get("landing")
    tl = Timeline(seq, easing)
    total = tl.total
    dt = 1.0 / float(np.clip(rate_hz, 0.5, 10.0))
    rc = resolve_cues(cues, segs, hold, tl.t_tk, tl.t_ld)
    cue_warn = check_cues(cues)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["drone_id", "t_s", "x_m", "y_m", "z_m", "r", "g", "b"])
    zmin, zmax = np.inf, -np.inf
    for t in _times(total, dt):
        P, C, phase, _ = tl.at(t)
        W = np.round(P, 3)
        if phase == "show":
            zmin, zmax = min(zmin, W[:, 2].min()), max(zmax, W[:, 2].max())
        for i in range(n):
            w.writerow([i, round(t, 3), *W[i], *np.round(C[i]).astype(int).tolist()])
    lights_csv = None
    if rc:
        lrate = float(
            min(np.clip(light_rate_hz, 2.0, 30.0), max(2.0, 1.5e6 / (n * total)))
        )
        lb = io.StringIO()
        lw = csv.writer(lb)
        lw.writerow(["drone_id", "t_s", "r", "g", "b"])
        for t in _times(total, 1.0 / lrate):
            P, C, _, _ = tl.at(t)
            L = np.clip(apply_cues(C, t, P, rc, ext), 0, 255).astype(int)
            for i in range(n):
                lw.writerow([i, round(t, 3), *L[i].tolist()])
        lights_csv = lb.getvalue()
    all_ok = all(not sg["warnings"] for sg in segs)
    alt_ok = zmin >= min_alt - 1e-9 and zmax <= max_alt + 1e-9
    dur_ok = total / 60 <= endurance_min
    verdict = (
        "PASS (simulation-level)"
        if all_ok and alt_ok and dur_ok
        else "FAIL - see items below"
    )

    L = [
        "# Trajectory validation report",
        "",
        f"**Result: {verdict}**",
        "",
        f"- Drones: {n}   Formations: {len(segs)}   Show duration: {total:.1f} s ({total / 60:.1f} min; endurance limit {endurance_min} min -> {'ok' if dur_ok else 'EXCEEDED'})",
        f"- Placement: {seq['params'].get('placement', 'n/a')}   Show-phase altitude range: {zmin:.1f} .. {zmax:.1f} m (allowed {min_alt}..{max_alt} m -> {'ok' if alt_ok else 'VIOLATION'})",
        f"- Arena: {ext} m   Min separation: {seq['min_sep']} m   Easing: {easing}   Sample rate: {1 / dt:g} Hz",
        "",
    ]
    for name, ph in (("Takeoff", tk), ("Landing", ld)):
        if ph:
            L += [
                f"## {name}",
                f"- {ph['waves']} wave(s), vertical {ph['height']:.0f} m in {ph['climb_time']:.1f} s each, "
                f"phase duration {ph['duration']:.1f} s",
                f"- Peak vertical speed {ph['peak_climb_speed']:.2f} m/s, "
                f"peak accel {ph['peak_accel']:.2f} m/s^2",
                "- Horizontal cells are the ground grid; separation checked on the grid layout.",
                "",
            ]
    f2 = lambda x, d=2: "n/a" if x is None else format(x, f".{d}f")
    for k, sg in enumerate(segs):
        m = sg["metrics"]
        L += [
            f"## Transition {k + 1}"
            + (" (return to landing grid)" if sg.get("kind") == "landing_grid" else ""),
            f"- Assignment: {m['method']} ({m['assign_ms']:.1f} ms), total travel {m['travel_total']:.0f} m, max {m['travel_max']:.1f} m",
            f"- Flight time {m['flight_time']:.1f} s (+stagger -> {m['total_time']:.1f} s), layers used {m['layers_used']}",
            f"- Collision pairs naive -> planned: {m['collisions_naive']} -> {m['collisions_planned']}; closest approach {f2(m['min_sep_planned'])} m",
            f"- Peak speed {m['peak_speed']:.2f} m/s, peak accel {f2(m['peak_accel'])} m/s^2",
            f"- Clearance after 2x positioning error: {f2(m['clearance'])} m",
        ]
        L += [f"- WARNING: {x}" for x in sg["warnings"]] + [""]
    if seq.get("notes"):
        L += ["## Notes"] + [f"- {x}" for x in seq["notes"]] + [""]
    if rc:
        L += [
            "## Lighting",
            f"- {len(rc)} cue(s); high-rate colours (with effects) are in lights.csv; the RGB columns in trajectories.csv are the base colours only.",
            f"- Light sample rate: {lrate:g} Hz (reduced automatically for very large shows)",
        ]
        L += [
            f"- {c['effect']} {c.get('anchor', {'type': 'show'})} scope={c.get('scope', 'all')} t={c['t0']:.1f}..{c['t1']:.1f}s"
            for c in rc
        ]
        L += [f"- WARNING: {x}" for x in cue_warn] + [""]
    if sim and sim.get("failures"):
        m = sim["metrics"]
        L += [
            "## Failure simulation",
            f"- Verdict: **{m['verdict']}**",
            f"- Contacts (closer than {m['contact_dist']} m) with swarm reaction: {m['contacts_with_reaction']}; without: {m['contacts_without_reaction']}",
            f"- Near-misses (closer than min separation): {m['near_misses']}; drones that evaded: {m['evaders']}",
        ]
        L += [
            f"- t={e['t']:.1f}s drone {e['drone']}: {e['text']}" for e in sim["events"]
        ]
        L += [f"- WARNING: {x}" for x in sim["warnings"]] + [""]
    L += ["## Read this first", "", DISCLAIMER]
    meta = dict(
        schema="swarm-sim/3",
        units=dict(length="m", time="s"),
        frame="x right, y depth, z up",
        params=seq["params"],
        hold_s=hold,
        takeoff=tk,
        landing=ld,
        validation=dict(
            verdict=verdict,
            show_altitude_range=[float(zmin), float(zmax)],
            show_duration_s=total,
        ),
        transitions=segs,
        light_cues=cues or [],
        failures=(sim or {}).get("failures", []),
        notes=DISCLAIMER,
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("trajectories.csv", buf.getvalue())
        if lights_csv:
            z.writestr("lights.csv", lights_csv)
        if sim and sim.get("telemetry_csv"):
            z.writestr("telemetry.csv", sim["telemetry_csv"])
        z.writestr("plan.json", json.dumps(meta))
        z.writestr("REPORT.md", "\n".join(L))
    return out.getvalue()
