import csv, io, os, sys, tempfile, unittest, zipfile
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.colors import lift_dark, WARM_WHITE
from swarm.video import extract_keyframes, encode_png
from swarm.sequence import build_sequence
from swarm.export import build_export
from swarm.lights import resolve_cues
from swarm.imaging import load_image, decode_image
from swarm.pipeline import formation_from_arrays
from swarm.server import SAMPLES

IMG = lambda *f: [load_image(os.path.join(SAMPLES, x)) for x in f]


class TestColors(unittest.TestCase):
    def test_black_becomes_warm_white_and_bright_untouched(self):
        out, k = lift_dark([[0, 0, 0], [255, 0, 0], [20, 10, 5]], 70)
        self.assertEqual(out[0].tolist(), list(WARM_WHITE))
        self.assertEqual(out[1].tolist(), [255, 0, 0])
        self.assertGreaterEqual(out[2] @ [0.2126, 0.7152, 0.0722], 60)
        self.assertEqual(k, 2)

    def test_hue_kept_when_brightened(self):
        out, _ = lift_dark([[40, 10, 10]], 70)
        self.assertGreater(out[0][0], out[0][1] * 3)  # still red-dominant

    def test_image_colours_used_by_default(self):
        seq = build_sequence(IMG("heart.png"), n=120)  # no solid_color -> image colours
        cols = np.array(seq["segments"][0]["color"])
        self.assertTrue(
            (cols[:, 0] > 200).all() and (cols[:, 1] < 80).all()
        )  # the heart is red

    def test_black_logo_is_lifted_and_noted(self):
        seq = build_sequence(IMG("star.png"), n=120)  # black star
        self.assertTrue(all(c == list(WARM_WHITE) for c in seq["segments"][0]["color"]))
        self.assertTrue(any("brightened" in n for n in seq["notes"]))


class TestVideo(unittest.TestCase):
    def test_demo_video_keyframes_have_shape_and_colour(self):
        frames, meta = extract_keyframes(
            os.path.join(SAMPLES, "video_demo.avi"), count=5, mode="motion"
        )
        self.assertEqual(len(frames), 5)
        self.assertAlmostEqual(meta["duration"], 4.0, delta=0.2)
        for f in frames:
            fg = (f["alpha"] > 0).mean()
            self.assertGreater(fg, 0.05)
            self.assertLess(fg, 0.5)
        first = frames[0]
        px = first["bgr"][first["alpha"] > 0].mean(axis=0)
        self.assertGreater(px[2], 200)  # orange circle: strong red channel

    def test_frames_become_formations(self):
        frames, _ = extract_keyframes(os.path.join(SAMPLES, "video_demo.avi"), count=4)
        imgs = [decode_image(encode_png(f["bgr"], f["alpha"])) for f in frames]
        for bgr, a in imgs:
            mask, form = formation_from_arrays(bgr, a, 200)
            self.assertEqual(len(form), 200)

    def test_threshold_mode_has_no_alpha(self):
        frames, _ = extract_keyframes(
            os.path.join(SAMPLES, "video_demo.avi"), count=3, mode="threshold"
        )
        self.assertTrue(all(f["alpha"] is None for f in frames))

    def test_bad_video(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
            fh.write(b"not a video")
            p = fh.name
        try:
            with self.assertRaises(ValueError):
                extract_keyframes(p)
        finally:
            os.remove(p)


class TestTakeoffLanding(unittest.TestCase):
    def setUp(self):
        self.seq = build_sequence(
            IMG("star.png", "diamond.png"),
            n=100,
            extent=150.0,
            min_sep=2.0,
            takeoff=True,
            landing=True,
            base_alt=30.0,
            climb_speed=2.5,
            waves=4,
            wave_gap=3.0,
            placement="horizontal",
        )

    def test_phase_data(self):
        tk, ld = self.seq["takeoff"], self.seq["landing"]
        self.assertLessEqual(tk["peak_climb_speed"], 2.5 + 1e-6)
        self.assertLessEqual(tk["peak_accel"], 2.0 + 1e-6)
        self.assertEqual(tk["duration"], tk["climb_time"] + 3 * 3.0)
        self.assertEqual(len(tk["delay"]), 100)
        self.assertEqual(self.seq["segments"][-1]["kind"], "landing_grid")
        self.assertEqual(len(self.seq["segments"]), 3)

    def test_export_starts_and_ends_on_ground(self):
        z = zipfile.ZipFile(io.BytesIO(build_export(self.seq, rate_hz=2)))
        rows = list(csv.DictReader(io.StringIO(z.read("trajectories.csv").decode())))
        zs = {}
        for r in rows:
            zs.setdefault(int(r["drone_id"]), []).append(float(r["z_m"]))
        self.assertTrue(
            all(abs(v[0]) < 1e-6 and abs(v[-1]) < 1e-6 for v in zs.values())
        )
        self.assertAlmostEqual(max(max(v) for v in zs.values()), 30.0, delta=2.0)
        self.assertIn("## Takeoff", z.read("REPORT.md").decode())
        self.assertIn("## Landing", z.read("REPORT.md").decode())

    def test_climb_is_vertical_and_collision_free(self):
        z = zipfile.ZipFile(io.BytesIO(build_export(self.seq, rate_hz=2)))
        rows = [
            r
            for r in csv.DictReader(io.StringIO(z.read("trajectories.csv").decode()))
            if int(r["drone_id"]) == 7
        ]
        takeoff_rows = [
            r for r in rows if float(r["t_s"]) < self.seq["takeoff"]["duration"]
        ]
        self.assertEqual(
            {(r["x_m"], r["y_m"]) for r in takeoff_rows}.__len__(), 1
        )  # one horizontal cell
        start = np.array(self.seq["segments"][0]["start"])
        from scipy.spatial import cKDTree

        self.assertGreaterEqual(
            cKDTree(start).query(start, k=2)[0][:, 1].min(), 2.0
        )  # cells >= min_sep apart

    def test_takeoff_works_with_vertical_wall(self):
        seq = build_sequence(
            IMG("star.png"),
            n=80,
            extent=100.0,
            min_sep=2.0,
            takeoff=True,
            base_alt=20.0,
            placement="vertical",
        )
        z = zipfile.ZipFile(io.BytesIO(build_export(seq, rate_hz=1)))
        rows = list(csv.DictReader(io.StringIO(z.read("trajectories.csv").decode())))
        zs = [float(r["z_m"]) for r in rows]
        self.assertAlmostEqual(min(zs), 0.0, places=3)
        self.assertGreater(max(zs), 60)  # wall rises above the hover grid

    def test_cue_anchors_include_takeoff_offset(self):
        tk, ld = self.seq["takeoff"]["duration"], self.seq["landing"]["duration"]
        r = resolve_cues(
            [
                {"effect": "on", "anchor": {"type": "takeoff"}},
                {"effect": "on", "anchor": {"type": "landing"}},
                {"effect": "on", "anchor": {"type": "hold", "stage": 0}},
                {"effect": "on"},
            ],
            self.seq["segments"],
            self.seq["hold"],
            tk,
            ld,
        )
        seg0 = self.seq["segments"][0]
        self.assertEqual((r[0]["t0"], r[0]["t1"]), (0, tk))
        self.assertAlmostEqual(r[2]["t0"], tk + seg0["total"])
        end = tk + sum(s["total"] + self.seq["hold"] for s in self.seq["segments"])
        self.assertAlmostEqual(r[1]["t0"], end)
        self.assertAlmostEqual(r[3]["t1"], end + ld)


if __name__ == "__main__":
    unittest.main()
