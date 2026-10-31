"""Failure injection, onboard failsafes, swarm reaction and per-drone telemetry.

SIMPLIFIED PHYSICS (point-mass drones, quadratic drag, linear battery model). It shows how a failure plays out
inside this simulator; it is NOT validated against real vehicles.
"""

from __future__ import annotations
import csv
import io
import numpy as np
from scipy.spatial import cKDTree
from .timeline import Timeline

G, DRAG_K = 9.81, 0.03  # gravity; quadratic drag (terminal speed ~ sqrt(G/K) = 18 m/s)
FAIL_TYPES = ("motor", "battery", "gps", "link", "led")
POLICIES = ("land", "hover_land", "rtl")
STATUS = {
    0: "OK",
    1: "FAILSAFE",
    2: "FALLING",
    3: "LANDED",
    4: "CRASHED",
    5: "LED FAULT",
    6: "GPS LOST",
    7: "LINK LOST",
    8: "BATTERY CRITICAL",
    9: "EVADING",
}
DESC = {
    "motor": "motor failure: thrust lost, free fall",
    "battery": "battery critical",
    "gps": "GPS lost: drifting",
    "link": "link lost: flying the stored plan",
    "led": "LED failure: light out",
}
DEFAULT_FS = dict(
    policy="land",
    detect_delay=0.5,
    descent_speed=1.5,
    hover_s=3.0,
    link_timeout=3.0,
    gps_timeout=2.0,
    gps_drift=1.5,
    reaction=True,
    avoid_radius=5.0,
    react_speed=3.0,
    horizon=4.0,
    contact_dist=1.0,
    battery_critical=8.0,
    rtl_speed=3.0,
    accel=4.0,
    endurance_min=15.0,
    dt=0.1,
)


def drain_rate(speed, vz, endurance_min):
    """Battery %/s: hover drains 100% over endurance_min; faster flight and climbing cost more (simple model)."""
    return (
        100.0
        / (endurance_min * 60.0)
        * (1.0 + 0.01 * speed**2 + 0.12 * np.maximum(vz, 0.0))
    )


def _normalize(failures, n, total, fs):
    out = []
    for f in failures or []:
        i, typ = int(f["drone"]), f.get("type", "motor")
        if not 0 <= i < n:
            raise ValueError(f"Drone {i} does not exist (0..{n - 1}).")
        if typ not in FAIL_TYPES:
            raise ValueError(
                f"Unknown failure type '{typ}' (use {', '.join(FAIL_TYPES)})."
            )
        pol = f.get("policy", fs["policy"])
        if pol not in POLICIES:
            raise ValueError(
                f"Unknown failsafe policy '{pol}' (use {', '.join(POLICIES)})."
            )
        t = float(f.get("t", 0.0))
        if not 0 <= t <= total:
            raise ValueError(
                f"Failure time {t:.1f}s is outside the show (0..{total:.1f}s)."
            )
        out.append(dict(drone=i, type=typ, t=t, policy=pol))
    return sorted(out, key=lambda e: e["t"])


def _fly_failed(tl, ev, fs, home_xy, seed):
    """Open-loop trajectory of one failed drone (deterministic, independent of the other drones)."""
    dt, i, typ, t_f, pol = fs["dt"], ev["drone"], ev["type"], ev["t"], ev["policy"]
    nom = lambda t: tl.pos(min(t, tl.total))[i]
    p = nom(t_f)
    v = (p - nom(max(t_f - dt, 0.0))) / dt if t_f >= dt else np.zeros(3)
    rng = np.random.default_rng(seed * 1009 + i)
    a = rng.uniform(0, 2 * np.pi)
    drift = (
        np.array([np.cos(a), np.sin(a), 0.0]) * fs["gps_drift"]
        if typ == "gps"
        else np.zeros(3)
    )
    t_s = t_f + {
        "battery": fs["detect_delay"],
        "link": fs["link_timeout"],
        "gps": fs["gps_timeout"],
    }.get(typ, 0.0)
    fault = {"gps": 6, "link": 7, "battery": 8, "motor": 2}[typ]
    mode = "ballistic" if typ == "motor" else "follow"
    batt = fs["battery_critical"] if typ == "battery" else None
    ts, P, S = [t_f], [p.copy()], [fault if p[2] > 0.05 else 3]
    t, hover_until, impact, final = t_f, None, 0.0, None
    if p[2] <= 0.05:
        return dict(
            t0=t_f,
            P=np.array(P),
            S=np.array(S),
            t_settle=t_f,
            impact=0.0,
            end=p.copy(),
            status=3,
        )
    for _ in range(int(2000 / dt)):
        t += dt
        if mode == "follow":
            pn = nom(t) + drift * (t - t_f)
            v = (pn - p) / dt
            p = pn
            st = fault
            if batt is not None:
                batt -= drain_rate(np.linalg.norm(v), v[2], fs["endurance_min"]) * dt
                if batt <= 0:
                    mode = "ballistic"
            if p[2] <= 0.01 and t >= tl.t_end_show:
                final = 3
            elif t >= t_s:
                mode = (
                    "hover"
                    if pol == "hover_land"
                    else ("rtl" if pol == "rtl" else "descend")
                )
                hover_until = t + fs["hover_s"]
        elif mode == "ballistic":
            st = 2
            for _k in range(2):
                acc = np.array([0, 0, -G]) - DRAG_K * np.linalg.norm(v) * v
                v = v + acc * dt / 2
                p = p + v * dt / 2
        else:
            st = 1
            if mode == "hover":
                vdes = np.zeros(3)
                if t >= hover_until:
                    mode = "descend"
            if mode == "rtl":
                d = home_xy[i] - p[:2]
                dist = np.linalg.norm(d)
                if dist < 1.0:
                    mode = "descend"
                else:
                    vdes = np.append(d / dist * min(fs["rtl_speed"], dist / dt), 0.0)
            if mode == "descend":
                vdes = np.array(
                    [0, 0, -max(0.4, min(fs["descent_speed"], 0.35 * p[2] + 0.4))]
                )
            v = v + np.clip(vdes - v, -fs["accel"] * dt, fs["accel"] * dt)
            p = p + v * dt
            if batt is not None:
                batt -= drain_rate(np.linalg.norm(v), v[2], fs["endurance_min"]) * dt
                if batt <= 0:
                    mode = "ballistic"
        if (
            mode == "follow" and t >= tl.total and final is None
        ):  # show over, timeout never reached: still flying the plan
            ts.append(t)
            P.append(p.copy())
            S.append(st)
            break
        if p[2] <= 0.0 and final is None:
            p[2] = 0.0
            impact = float(np.linalg.norm(v))
            final = 3 if impact <= 2.5 else 4
        ts.append(t)
        P.append(p.copy())
        S.append(final if final is not None else st)
        if final is not None:
            break
    return dict(
        t0=t_f,
        P=np.array(P),
        S=np.array(S),
        t_settle=ts[-1],
        impact=impact,
        end=p.copy(),
        status=S[-1],
    )


def _hazard_at(h, t, dt):
    k = int(round((t - h["t0"]) / dt))
    return h["P"][min(max(k, 0), len(h["P"]) - 1)]


_ANG = np.arange(8) * np.pi / 4
CAND = np.vstack(
    [np.column_stack([np.cos(_ANG), np.sin(_ANG), np.zeros(8)]), [[0.0, 0.0, 1.0]]]
)  # 8 horizontal + up


def _react(tl, t, offsets, hazards, failed, fs, cur):
    """Neighbours sidestep the predicted path of failing drones. Threat = predicted separation < avoid_radius.
    Each threatened drone picks the escape direction with the most free space (so in a dense flat formation it
    leaves the plane instead of shoving its neighbours), then moves along it with a speed that grows with the threat."""
    n, R, dt = len(offsets), fs["avoid_radius"], fs["dt"]
    taus = t + np.arange(int(fs["horizon"] / 0.25) + 1) * 0.25
    fut = np.stack([tl.pos(tt) for tt in taus]) + offsets[None]
    mag, away_acc, ar = np.zeros(n), np.zeros((n, 3)), np.arange(n)
    for h in hazards:
        hp = np.array([_hazard_at(h, tt, dt) for tt in taus])
        d = fut - hp[:, None, :]
        dist = np.linalg.norm(d, axis=2)
        km = dist.argmin(0)
        dmin = dist[km, ar]
        m = (dmin < R) & ~failed
        away = d[km, ar].copy()
        away[:, 2] = 0
        nr = np.maximum(np.linalg.norm(away, axis=1), 1e-9)
        mag = np.where(m, np.maximum(mag, fs["react_speed"] * (R - dmin) / R), mag)
        away_acc[m] += (away / nr[:, None])[m]
    M = np.where(mag > 0)[0]
    push = np.zeros((n, 3))
    if len(M):
        L = 0.6 * R
        C = len(CAND)
        tree = cKDTree(cur)
        d, idx = tree.query((cur[M][:, None, :] + CAND[None] * L).reshape(-1, 3), k=3)
        d = np.where(idx == np.repeat(M, C * 3).reshape(-1, 3), np.inf, d).reshape(
            len(M), C, 3
        )
        clr = np.minimum(d.min(axis=2), L)
        unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        score = (
            clr
            + 0.4 * L * (unit(away_acc[M]) @ CAND.T)
            + 0.3 * L * (unit(offsets[M]) @ CAND.T)
        )
        push[M] = CAND[score.argmax(1)] * mag[M][:, None]
    offsets += push * dt
    offsets[mag == 0] *= np.exp(-dt / 2.0)
    nrm = np.linalg.norm(offsets, axis=1)
    big = nrm > 2 * R
    offsets[big] *= (2 * R / nrm[big])[:, None]


def _run(tl, fails, paths, fs, reaction, telemetry, min_sep, dirs):
    n, dt = tl.n, fs["dt"]
    t_end = max([tl.total] + [pa["t_settle"] + 2 for pa in paths])
    times = np.arange(0.0, t_end + dt / 2, dt)
    failed_ids = np.array(sorted({f["drone"] for f in fails}), int)
    failed = np.zeros(n, bool)
    failed[failed_ids] = True
    t_first = min([f["t"] for f in fails], default=np.inf)
    hazards = [
        dict(paths[k], t_det=f["t"] + fs["detect_delay"], drone=f["drone"])
        for k, f in enumerate(fails)
        if f["type"] != "led"
    ]
    offsets, battery = np.zeros((n, 3)), np.full(n, 100.0)
    tel_stride = max(1, int(round(fs["tel_dt"] / dt))) if telemetry else 0
    tel = dict(t=[], speed=[], battery=[], status=[], pos=[])
    rec = []
    rec_t = []
    contacts, near, base_near, min_dist, min_to_failed = (
        set(),
        set(),
        set(),
        np.inf,
        np.inf,
    )
    contact_log, evader_max = [], np.zeros(n)
    prev = None
    led_off = {}
    for f in fails:
        if f["type"] == "led":
            led_off[f["drone"]] = f["t"]
    for step, t in enumerate(times):
        nom = tl.pos(min(t, tl.total))
        act = nom.copy()
        status = np.zeros(n, np.int8)
        for k, f in enumerate(fails):
            if f["type"] == "led":
                if t >= f["t"]:
                    status[f["drone"]] = 5
                continue
            pa = paths[k]
            if t >= pa["t0"]:
                j = min(int(round((t - pa["t0"]) / dt)), len(pa["P"]) - 1)
                act[f["drone"]] = pa["P"][j]
                status[f["drone"]] = pa["S"][j]
            if f["type"] == "battery" and abs(t - f["t"]) < dt / 2:
                battery[f["drone"]] = fs["battery_critical"]
        act_h = [h for h in hazards if h["t_det"] <= t <= h["t_settle"] + 1.5]
        if reaction and act_h:
            _react(tl, t, offsets, act_h, failed, fs, act + offsets)
        elif offsets.any():
            offsets *= np.exp(-dt / 2.0)
        offsets[failed] = 0
        evader_max = np.maximum(evader_max, np.linalg.norm(offsets, axis=1))
        act = act + offsets
        status[(np.linalg.norm(offsets, axis=1) > 0.3) & (status == 0)] = 9
        vel = (act - prev) / dt if prev is not None else np.zeros((n, 3))
        prev = act.copy()
        speed = np.linalg.norm(vel, axis=1)
        idle = (act[:, 2] <= 0.05) & (speed < 0.05)
        rate = np.where(idle, 0.05, 1.0) * drain_rate(
            speed, vel[:, 2], fs["endurance_min"]
        )
        rate[(status == 2) | (status == 3) | (status == 4)] = 0.0
        battery = np.maximum(battery - rate * dt, 0.0)
        if (
            np.isfinite(t_first)
            and t >= t_first - dt
            and t <= max(h["t_settle"] for h in hazards + [dict(t_settle=t_first)]) + 3
        ):
            tree = cKDTree(act)
            d2, _ = tree.query(act, k=2)
            min_dist = min(min_dist, float(d2[:, 1].min()))
            for a, b in tree.query_pairs(fs["contact_dist"]):
                if (a, b) not in contacts:
                    contacts.add((a, b))
                    contact_log.append(
                        dict(
                            t=float(t),
                            a=int(a),
                            b=int(b),
                            dist=float(np.linalg.norm(act[a] - act[b])),
                        )
                    )
            near |= tree.query_pairs(min_sep - 1e-9)
            base_near |= cKDTree(nom).query_pairs(min_sep - 1e-9)
            if failed.any():
                dd, _ = (
                    cKDTree(act[~failed]).query(act[failed_ids])
                    if (~failed).any()
                    else (np.array([np.inf]), None)
                )
                min_to_failed = min(min_to_failed, float(np.min(dd)))
        if telemetry:
            if step % tel_stride == 0:
                tel["t"].append(float(t))
                tel["speed"].append(speed.copy())
                tel["battery"].append(battery.copy())
                tel["status"].append(status.copy())
                tel["pos"].append(act.copy())
            if t >= t_first - dt:
                rec.append(act.astype(np.float32))
                rec_t.append(t)
    return dict(
        contacts=contacts,
        contact_log=contact_log,
        near=near - base_near,
        min_dist=min_dist,
        min_to_failed=min_to_failed,
        evader_max=evader_max,
        tel=tel,
        rec=rec,
        rec_t=rec_t,
        t_end=t_end,
    )


def simulate(
    seq,
    easing="smooth",
    failures=(),
    settings=None,
    telemetry=True,
    want_csv=False,
    seed=0,
) -> dict:
    """Run the show with injected failures. Returns telemetry for the dashboard plus a failure report."""
    fs = {**DEFAULT_FS, **(settings or {})}
    tl = Timeline(seq, easing)
    n = tl.n
    fs["tel_dt"] = max(1.0, float(np.ceil(n * tl.total / 2.0e5)))
    fails = _normalize(failures, n, tl.total, fs)
    home = tl.segs[0][0][:, :2]
    paths = [
        _fly_failed(tl, f, fs, home, seed)
        if f["type"] != "led"
        else dict(
            t0=f["t"],
            P=np.array([[0, 0, 0]]),
            S=np.array([5]),
            t_settle=f["t"],
            impact=0.0,
            end=np.zeros(3),
            status=5,
        )
        for f in fails
    ]
    ang = np.random.default_rng(seed + 5).uniform(0, 2 * np.pi, n)
    dirs = np.column_stack(
        [np.zeros(n), np.where(np.arange(n) % 2 == 0, 1.0, -1.0), np.zeros(n)]
    )  # out of a vertical plane
    r = _run(
        tl,
        fails,
        paths,
        fs,
        fs["reaction"] and bool(fails),
        telemetry,
        seq["min_sep"],
        dirs,
    )
    base = (
        _run(tl, fails, paths, fs, False, False, seq["min_sep"], dirs)
        if fails and fs["reaction"]
        else r
    )
    half = seq["extent"] / 2
    events, warnings, results = [], [], []
    for f, pa in zip(fails, paths):
        i = f["drone"]
        typ = f["type"]
        events.append(dict(t=f["t"], drone=i, text=DESC[typ]))
        if typ != "led" and fs["reaction"]:
            events.append(
                dict(
                    t=f["t"] + fs["detect_delay"],
                    drone=i,
                    text="detected by ground control; swarm reaction active",
                )
            )
        if (
            typ in ("battery", "gps", "link")
            and pa["status"] != 3
            or (typ in ("battery", "gps", "link") and pa["t_settle"] > f["t"] + 0.5)
        ):
            ts_ = (
                f["t"]
                + {
                    "battery": fs["detect_delay"],
                    "link": fs["link_timeout"],
                    "gps": fs["gps_timeout"],
                }[typ]
            )
            if ts_ <= pa["t_settle"]:
                events.append(
                    dict(
                        t=ts_, drone=i, text=f"onboard failsafe '{f['policy']}' started"
                    )
                )
        if typ != "led":
            outside = bool(np.any(np.abs(pa["end"][:2]) > half))
            state = STATUS[int(pa["status"])]
            events.append(
                dict(
                    t=pa["t_settle"],
                    drone=i,
                    text=f"{state.lower()} at ({pa['end'][0]:.1f}, {pa['end'][1]:.1f}), impact {pa['impact']:.1f} m/s"
                    + (" - OUTSIDE the arena" if outside else ""),
                )
            )
            if outside:
                warnings.append(f"Drone {i} ended outside the arena (geofence breach).")
            if pa["status"] == 4:
                warnings.append(
                    f"Drone {i} crashed (impact {pa['impact']:.1f} m/s)"
                    + (
                        ": thrust was lost, onboard recovery is impossible."
                        if typ == "motor"
                        else "."
                    )
                )
            results.append(
                dict(
                    **f,
                    t_settle=float(pa["t_settle"]),
                    impact=float(pa["impact"]),
                    end=[float(x) for x in pa["end"]],
                    state=state,
                    outside=outside,
                )
            )
        else:
            results.append(
                dict(
                    **f,
                    t_settle=float(f["t"]),
                    impact=0.0,
                    end=None,
                    state="LED FAULT",
                    outside=False,
                )
            )
    events.sort(key=lambda e: e["t"])
    c_with, c_without = len(r["contacts"]), len(base["contacts"])
    evaders = [int(i) for i in np.where(r["evader_max"] > 0.3)[0]]
    if c_with:
        warnings.append(
            f"{c_with} contact(s) between drones (closer than {fs['contact_dist']} m) even with swarm reaction."
        )
    verdict = (
        "no failures injected"
        if not fails
        else "SAFE: no drone contacts"
        if c_with == 0
        else f"HAZARD: {c_with} contact(s)"
    )
    metrics = dict(
        verdict=verdict,
        contact_dist=fs["contact_dist"],
        contacts_with_reaction=c_with,
        contacts_without_reaction=c_without,
        near_misses=len(r["near"]),
        evaders=len(evaders),
        max_evasion_m=float(r["evader_max"].max()),
        min_distance=None if not np.isfinite(r["min_dist"]) else r["min_dist"],
        min_distance_to_failed=None
        if not np.isfinite(r["min_to_failed"])
        else r["min_to_failed"],
        reaction_enabled=bool(fs["reaction"]),
    )
    out = dict(
        failures=results,
        events=events,
        warnings=warnings,
        metrics=metrics,
        contacts=r["contact_log"][:20],
        settings={
            k: fs[k]
            for k in (
                "policy",
                "detect_delay",
                "descent_speed",
                "reaction",
                "avoid_radius",
                "endurance_min",
            )
        },
        t_end=r["t_end"],
    )
    if telemetry:
        tel = r["tel"]
        T = len(tel["t"])
        out.update(
            tel_dt=fs["tel_dt"],
            speed=np.round(np.array(tel["speed"]).T * 10).astype(int).tolist(),
            battery=np.round(np.array(tel["battery"]).T * 10).astype(int).tolist(),
            status=np.array(tel["status"]).T.astype(int).tolist(),
            status_names=STATUS,
        )
        ov = {}
        if r["rec"]:
            rec = np.array(r["rec"])
            t0 = r["rec_t"][0]
            ids = set(evaders) | {f["drone"] for f in fails if f["type"] != "led"}
            for i in sorted(ids):
                k = len(rec)
                if i in {f["drone"] for f in fails}:
                    pa = paths[[f["drone"] for f in fails].index(i)]
                    k = min(len(rec), int((pa["t_settle"] + 1.0 - t0) / fs["dt"]) + 2)
                ov[str(i)] = dict(
                    t0=float(t0), dt=fs["dt"], pos=np.round(rec[:k, i], 2).tolist()
                )
        out["override"] = ov
        if want_csv:
            b = io.StringIO()
            w = csv.writer(b)
            w.writerow(
                [
                    "drone_id",
                    "t_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "speed_mps",
                    "battery_pct",
                    "status",
                ]
            )
            for k in range(T):
                for i in range(n):
                    w.writerow(
                        [
                            i,
                            round(tel["t"][k], 2),
                            *np.round(tel["pos"][k][i], 2).tolist(),
                            round(float(tel["speed"][k][i]), 2),
                            round(float(tel["battery"][k][i]), 1),
                            STATUS[int(tel["status"][k][i])],
                        ]
                    )
            out["telemetry_csv"] = b.getvalue()
    return out
