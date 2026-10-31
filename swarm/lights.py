"""Light-show cues: LED effects layered on top of each drone's base colour (3D aware).

A cue = {effect, anchor, scope, axis, period, duty, color?}
  effect : on | off | color | blink | pulse | rainbow | wave | fade_in | fade_out
  anchor : {"type":"show"} | {"type":"takeoff"} | {"type":"landing"} | {"type":"hold","stage":k}
           | {"type":"flight","stage":k} | {"type":"abs","t0":s,"t1":s}        (k is 0-based)
  scope  : all | left | right | front | back | top | bottom | even | odd   (by current 3D position / id)
  axis   : x | y | z | id | radial       (rainbow / wave sweep direction; radial = horizontal distance)
  period : seconds per cycle;  duty: fraction "on" for blink;  color: [r,g,b] override
Later cues override earlier ones where they overlap. Mirrored in static/lights.js.
"""

from __future__ import annotations
import numpy as np

EFFECTS = (
    "on",
    "off",
    "color",
    "blink",
    "pulse",
    "rainbow",
    "wave",
    "fade_in",
    "fade_out",
)
MAX_SAFE_FLASH_HZ = 3.0  # common photosensitivity guidance: <= 3 flashes per second


def resolve_cues(cues, segments, hold, takeoff=0.0, landing=0.0):
    """Turn anchors into absolute [t0, t1) show-time intervals.
    Timeline: [takeoff] [transition+hold]* [landing]; anchors: show | takeoff | landing | hold k | flight k | abs."""
    offs, a = [], float(takeoff)
    for sg in segments:
        offs.append(a)
        a += sg["total"] + hold
    out = []
    for c in cues or []:
        if c.get("effect") not in EFFECTS:
            raise ValueError(
                f"Unknown light effect '{c.get('effect')}' (use {', '.join(EFFECTS)})"
            )
        anc = c.get("anchor") or {"type": "show"}
        typ = anc.get("type", "show")
        if typ == "show":
            t0, t1 = 0.0, a + landing
        elif typ == "takeoff":
            t0, t1 = 0.0, float(takeoff)
        elif typ == "landing":
            t0, t1 = a, a + landing
        elif typ in ("hold", "flight"):
            k = int(anc.get("stage", 0))
            if not 0 <= k < len(segments):
                raise ValueError(
                    f"Cue refers to formation {k + 1}, but there are only {len(segments)}."
                )
            t0 = offs[k] + (segments[k]["total"] if typ == "hold" else 0.0)
            t1 = offs[k] + segments[k]["total"] + (hold if typ == "hold" else 0.0)
        elif typ == "abs":
            t0, t1 = float(anc["t0"]), float(anc["t1"])
        else:
            raise ValueError(f"Unknown anchor type '{typ}'")
        out.append(dict(c, t0=t0, t1=t1))
    return out


def check_cues(cues) -> list:
    """Photosensitivity warnings for fast flashing."""
    w = []
    for i, c in enumerate(cues or []):
        if c.get("effect") in ("blink", "pulse"):
            hz = 1.0 / max(float(c.get("period", 1.0)), 1e-3)
            if hz > MAX_SAFE_FLASH_HZ:
                w.append(
                    f"Cue {i + 1} ({c['effect']}) flashes {hz:.1f} times per second; more than "
                    f"{MAX_SAFE_FLASH_HZ:g}/s can trigger photosensitive seizures. Lengthen the period."
                )
    return w


def hsv(h):
    h = np.asarray(h, float)
    h6 = (h % 1.0) * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    q = 1 - f
    z, o = np.zeros_like(h), np.ones_like(h)
    r = np.choose(i, [o, q, z, z, f, o])
    g = np.choose(i, [f, o, o, q, z, z])
    b = np.choose(i, [z, z, f, o, o, q])
    return np.column_stack([r, g, b]) * 255


def _axis(axis, xyz, ids, n, extent, zr):
    if axis == "y":
        return np.clip(xyz[:, 1] / extent + 0.5, 0, 1)
    if axis == "z":
        return np.clip((xyz[:, 2] - zr[0]) / max(zr[1] - zr[0], 1e-9), 0, 1)
    if axis == "id":
        return ids / max(n - 1, 1)
    if axis == "radial":
        return np.clip(np.hypot(xyz[:, 0], xyz[:, 1]) / (extent / 2), 0, 1)
    return np.clip(xyz[:, 0] / extent + 0.5, 0, 1)


def _scope(scope, xyz, ids, zr):
    zmid = (zr[0] + zr[1]) / 2
    return {
        "left": xyz[:, 0] < 0,
        "right": xyz[:, 0] >= 0,
        "front": xyz[:, 1] < 0,
        "back": xyz[:, 1] >= 0,
        "top": xyz[:, 2] > zmid,
        "bottom": xyz[:, 2] <= zmid,
        "even": ids % 2 == 0,
        "odd": ids % 2 == 1,
    }.get(scope, np.ones(len(ids), bool))


def apply_cues(base, t, xyz, resolved, extent):
    """base (n,3) RGB 0-255 at show time t, xyz (n,3) current world positions (2 columns allowed) -> (n,3)."""
    out, n = np.array(base, float), len(base)
    xyz = np.asarray(xyz, float)
    if xyz.shape[1] == 2:
        xyz = np.column_stack([xyz, np.zeros(n)])
    zr = (float(xyz[:, 2].min()), float(xyz[:, 2].max()))
    ids = np.arange(n)
    for c in resolved:
        if not (c["t0"] <= t < c["t1"]):
            continue
        m = _scope(c.get("scope", "all"), xyz, ids, zr)
        if not m.any():
            continue
        src = (
            np.tile(np.array(c["color"], float), (n, 1))
            if c.get("color")
            else np.array(base, float)
        )
        per = max(float(c.get("period", 1.0)), 1e-3)
        ph = t - c["t0"]
        e = c["effect"]
        if e == "off":
            res = np.zeros((n, 3))
        elif e in ("on", "color"):
            res = src
        elif e == "blink":
            res = src * (1.0 if (ph / per) % 1.0 < float(c.get("duty", 0.5)) else 0.0)
        elif e == "pulse":
            res = src * (0.5 - 0.5 * np.cos(2 * np.pi * ph / per))
        elif e == "rainbow":
            res = hsv(_axis(c.get("axis", "x"), xyz, ids, n, extent, zr) - ph / per)
        elif e == "wave":
            b = 0.5 + 0.5 * np.cos(
                2
                * np.pi
                * (_axis(c.get("axis", "x"), xyz, ids, n, extent, zr) - ph / per)
            )
            res = src * b[:, None]
        elif e == "fade_in":
            res = src * float(np.clip(ph / max(c["t1"] - c["t0"], 1e-9), 0, 1))
        else:
            res = src * (1.0 - float(np.clip(ph / max(c["t1"] - c["t0"], 1e-9), 0, 1)))
        out[m] = res[m]
    return out
