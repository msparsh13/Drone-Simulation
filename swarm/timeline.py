"""Nominal (planned) motion of the whole swarm as a function of show time."""

from __future__ import annotations
import numpy as np
from .animation import EASINGS
from .collision import positions

SMOOTH = EASINGS["smooth"]


class Timeline:
    """show time -> positions (n,3), base colours, phase. Phases: takeoff | show | landing."""

    def __init__(self, seq: dict, easing: str = "smooth"):
        self.seq, self.easing, self.n, self.hold = seq, easing, seq["n"], seq["hold"]
        self.tk, self.ld = seq.get("takeoff"), seq.get("landing")
        self.segs = [
            tuple(
                np.array(sg[k], float)
                for k in ("start", "target", "delay", "lift", "color")
            )
            + (sg["flight"], sg["total"])
            for sg in seq["segments"]
        ]
        self.t_tk = self.tk["duration"] if self.tk else 0.0
        self.offs, a = [], self.t_tk
        for sg in self.segs:
            self.offs.append(a)
            a += sg[6] + self.hold
        self.t_end_show, self.t_ld = a, (self.ld["duration"] if self.ld else 0.0)
        self.total = a + self.t_ld
        self.start_color = np.tile(np.array(seq["start_color"], float), (self.n, 1))
        self._tk_delay = np.array(self.tk["delay"], float) if self.tk else None
        self._ld_delay = np.array(self.ld["delay"], float) if self.ld else None

    def pos(self, t):
        return self._eval(t, False)[0]

    def at(self, t):
        """-> (positions (n,3), colours (n,3), phase, segment index)"""
        return self._eval(t, True)

    def _eval(self, t, colors):
        t = float(np.clip(t, 0.0, self.total))
        if self.tk and t < self.t_tk:
            s = self.segs[0][0]
            p = np.clip((t - self._tk_delay) / self.tk["climb_time"], 0, 1)
            return (
                np.column_stack([s[:, 0], s[:, 1], s[:, 2] * SMOOTH(p)]),
                self.start_color if colors else None,
                "takeoff",
                0,
            )
        if self.ld and t >= self.t_end_show:
            tg = self.segs[-1][1]
            p = np.clip(
                (t - self.t_end_show - self._ld_delay) / self.ld["climb_time"], 0, 1
            )
            return (
                np.column_stack([tg[:, 0], tg[:, 1], tg[:, 2] * (1 - SMOOTH(p))]),
                self.segs[-1][4] if colors else None,
                "landing",
                len(self.segs) - 1,
            )
        k = (
            max(i for i, o in enumerate(self.offs) if t >= o - 1e-12)
            if t >= self.offs[0]
            else 0
        )
        s, tg, dl, lf, col, T, total = self.segs[k]
        tau = t - self.offs[k]
        pos = positions(s, tg, dl, T, min(tau, total), self.easing, lf)
        c = None
        if colors:
            prev = self.segs[k - 1][4] if k > 0 else self.start_color
            c = prev + (col - prev) * np.clip((tau - dl) / T, 0, 1)[:, None]
        return pos, c, "show", k
