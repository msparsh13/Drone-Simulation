"""Local web app:  py -m swarm.server   ->  http://127.0.0.1:8000"""
from __future__ import annotations
import argparse
import base64
import json
import os
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np

from .imaging import load_image, decode_image, SUPPORTED
from .pipeline import formation_from_arrays
from .init import make_start
from .assignment import assign, travel_stats
from .sampling import evaluate_formation, nn_distances
from .sequence import build_sequence
from .export import build_export
from .video import extract_keyframes, encode_png, VIDEO_EXT

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "..", "samples")


def build_payload(bgr, alpha, *, n=500, threshold=127, invert=False, scale=1.0,
                  init="random", method="hungarian", seed=0, extent=100.0) -> dict:
    mask, f = formation_from_arrays(bgr, alpha, n, threshold, invert, scale, extent, seed=seed)
    n = len(f)
    starts = make_start(init, n, extent, seed)
    t0 = time.perf_counter()
    perm = assign(starts, f.positions, method)
    ms = (time.perf_counter() - t0) * 1000
    targets, colors = f.positions[perm], f.colors[perm]
    cov = evaluate_formation(f, mask)
    tri = 0.45 * float(np.median(nn_distances(f.positions))) if n > 1 else 2.0
    r = lambda a: np.round(a, 3).tolist()
    return dict(
        extent=extent, tri=tri, n=n, ids=list(range(n)),
        start=r(starts), target=r(targets),
        color=[f"rgb({a},{b},{c})" for a, b, c in colors.tolist()],
        metrics=dict(particles=n, method=method, assign_ms=ms,
                     **{f"travel_{k}": v for k, v in travel_stats(starts, targets).items()},
                     coverage_mean=cov["coverage_mean"], coverage_max=cov["coverage_max"],
                     min_spacing=cov["min_spacing"]))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "static", "index.html"), "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if path.startswith("/static/") and path.endswith(".js"):
            fp = os.path.join(ROOT, "static", os.path.basename(path))
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    return self._send(200, fh.read(), "text/javascript; charset=utf-8")
        if path == "/api/samples":
            names = sorted(x for x in os.listdir(SAMPLES) if x.lower().endswith(SUPPORTED))
            return self._send(200, json.dumps(names).encode())
        self._send(404, b'{"error":"not found"}')

    def _read_plan(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        images = []
        for st in body.get("stages", []):
            if st.get("data"):
                images.append(decode_image(base64.b64decode(st["data"])))
            else:
                images.append(load_image(os.path.join(SAMPLES, os.path.basename(st.get("sample", "")))))
        g = lambda k, d, c: c(body.get(k, d))
        kw = dict(n=g("n", 500, int), threshold=g("threshold", 127, int), invert=bool(body.get("invert")),
                  scale=g("scale", 1.0, float), init=body.get("init", "random"),
                  method=body.get("method", "hungarian_sq"), seed=g("seed", 0, int),
                  extent=g("extent", 100.0, float), min_sep=g("min_sep", 2.5, float),
                  avoid=bool(body.get("avoid", True)), flight_time=g("flight_time", 10.0, float),
                  easing=body.get("easing", "smooth"), vmax=g("vmax", 6.0, float), amax=g("amax", 2.0, float),
                  auto_time=bool(body.get("auto_time", True)), pos_error=g("pos_error", 0.3, float),
                  solid_color=body.get("solid_color"), hold=g("hold", 5.0, float),
                  min_brightness=g("min_brightness", 70.0, float), takeoff=bool(body.get("takeoff")),
                  landing=bool(body.get("landing")), base_alt=g("base_alt", 30.0, float),
                  climb_speed=g("climb_speed", 2.5, float), waves=g("waves", 4, int), wave_gap=g("wave_gap", 3.0, float))
        return body, build_sequence(images, **kw), kw["easing"]

    def _plan(self):
        try:
            self._send(200, json.dumps(self._read_plan()[1]).encode())
        except (ValueError, FileNotFoundError, KeyError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())

    def _export(self):
        try:
            body, seq, easing = self._read_plan()
            data = build_export(seq, easing=easing, rate_hz=float(body.get("rate_hz", 5)),
                                plane=body.get("plane", "horizontal"), base_alt=float(body.get("base_alt", 30)),
                                max_alt=float(body.get("max_alt", 120)), endurance_min=float(body.get("endurance_min", 15)),
                                cues=body.get("cues") or [], light_rate_hz=float(body.get("light_rate_hz", 20)))
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="swarm_trajectories.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        except (ValueError, FileNotFoundError, KeyError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())

    def _video(self, q):
        path, tmp = None, None
        try:
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if data:
                ext = q.get("ext", ".mp4").lower()
                if ext not in VIDEO_EXT:
                    raise ValueError(f"Unsupported video type (use {', '.join(VIDEO_EXT)})")
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
                    fh.write(data); path = tmp = fh.name
            else:
                path = os.path.join(SAMPLES, os.path.basename(q.get("sample", "")))
                if not path.lower().endswith(VIDEO_EXT) or not os.path.exists(path):
                    raise FileNotFoundError("Video not found.")
            frames, meta = extract_keyframes(path, count=int(q.get("count", 8)), mode=q.get("mode", "motion"),
                                             diff_thresh=int(q.get("thresh", 30)))
            label = q.get("name", "video")
            out = [dict(name=f"{label} @ {f['t']:.1f}s", png=base64.b64encode(encode_png(f["bgr"], f["alpha"])).decode())
                   for f in frames]
            self._send(200, json.dumps(dict(frames=out, **meta)).encode())
        except (ValueError, FileNotFoundError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())
        finally:
            if tmp and os.path.exists(tmp): os.remove(tmp)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/video":
            return self._video({k: v[0] for k, v in parse_qs(u.query).items()})
        if u.path == "/api/plan":
            return self._plan()
        if u.path == "/api/export":
            return self._export()
        if u.path != "/api/build":
            return self._send(404, b'{"error":"not found"}')
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if data:
                bgr, alpha = decode_image(data)
            else:
                name = os.path.basename(q.get("sample", ""))
                bgr, alpha = load_image(os.path.join(SAMPLES, name))
            payload = build_payload(
                bgr, alpha, n=int(q.get("n", 500)), threshold=int(q.get("threshold", 127)),
                invert=q.get("invert") == "1", scale=float(q.get("scale", 1.0)),
                init=q.get("init", "random"), method=q.get("method", "hungarian"),
                seed=int(q.get("seed", 0)))
            self._send(200, json.dumps(payload).encode())
        except (ValueError, FileNotFoundError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode())

    def log_message(self, *a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Swarm simulator running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
