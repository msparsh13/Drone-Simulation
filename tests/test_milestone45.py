import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.init import random_scatter
from swarm.collision import scan
from swarm.planner import plan_segment
from swarm.sequence import build_sequence
from swarm.imaging import load_image
from swarm.server import SAMPLES


def A(x): return np.array(x, float)


class TestCollision(unittest.TestCase):
    def test_detects_crossing(self):
        s, t = A([[-5, 0], [0, -5]]), A([[5, 0], [0, 5]])
        r = scan(s, t, np.zeros(2), 4, 1.0, "linear", 100)
        self.assertEqual(r["pairs"], {(0, 1)}); self.assertLess(r["min_sep"], 1.0)

    def test_no_collision_far_apart(self):
        s, t = A([[-5, 0], [-5, 20]]), A([[5, 0], [5, 20]])
        r = scan(s, t, np.zeros(2), 4, 1.0, "cubic", 100)
        self.assertFalse(r["pairs"]); self.assertAlmostEqual(r["min_sep"], 20, places=3)

    def test_min_separation_matches_geometry(self):
        s, t = A([[0, 0], [0, 3]]), A([[10, 0], [10, 3]])   # parallel, constant 3 apart
        self.assertAlmostEqual(scan(s, t, np.zeros(2), 4, 1.0, "linear", 100)["min_sep"], 3.0)


class TestPlanner(unittest.TestCase):
    def test_stagger_resolves_crossing(self):
        s, t = A([[-5, 0], [0, -5]]), A([[5, 0], [0, 5]])
        p = plan_segment(s, t, 4, 1.0, "cubic", avoid=True, max_layers=1)  # stagger only
        self.assertEqual(p["collisions_naive"], 1); self.assertEqual(p["collisions_planned"], 0)
        self.assertGreater(p["delays"].max(), 0); self.assertGreater(p["total_time"], 4)
        self.assertEqual(p["warnings"], [])

    def test_layers_resolve_crossing_without_extra_time(self):
        s, t = A([[-5, 0], [0, -5]]), A([[5, 0], [0, 5]])
        p = plan_segment(s, t, 4, 1.0, "cubic", avoid=True, max_layers=3)
        self.assertEqual(p["collisions_planned"], 0); self.assertEqual(p["total_time"], 4)

    def test_no_avoid_keeps_collision(self):
        s, t = A([[-5, 0], [0, -5]]), A([[5, 0], [0, 5]])
        p = plan_segment(s, t, 4, 1.0, avoid=False)
        self.assertEqual(p["collisions_planned"], 1); self.assertTrue(p["warnings"])

    def test_head_on_swap_reported_not_hidden(self):
        s, t = A([[-5, 0], [5, 0]]), A([[5, 0], [-5, 0]])
        p = plan_segment(s, t, 4, 1.0, avoid=True, max_layers=1)   # no layering available
        self.assertGreater(p["collisions_planned"], 0)
        self.assertTrue(any("NOT satisfied" in w for w in p["warnings"]))

    def test_head_on_swap_resolved_by_layers(self):
        s, t = A([[-5, 0], [5, 0]]), A([[5, 0], [-5, 0]])
        p = plan_segment(s, t, 4, 1.0, avoid=True, max_layers=3)
        self.assertEqual(p["collisions_planned"], 0); self.assertGreater(p["layers_used"], 0)

    def test_clear_scene_has_no_delays(self):
        s, t = A([[-5, 0], [-5, 20]]), A([[5, 0], [5, 20]])
        p = plan_segment(s, t, 4, 1.0)
        self.assertEqual(p["delays"].max(), 0); self.assertEqual(p["total_time"], 4)

    def test_speed_limit_warning(self):
        p = plan_segment(A([[0, 0]]), A([[100, 0]]), 2, 1.0, "cubic", vmax=10)
        self.assertTrue(any("speed" in w.lower() for w in p["warnings"]))
        self.assertAlmostEqual(p["peak_speed"], 150.0)

    def test_static_overlap_reported(self):
        s, t = A([[0, 0], [0.2, 0]]), A([[10, 0], [10, 10]])
        self.assertTrue(any("cannot fix" in w or "timing" in w for w in plan_segment(s, t, 4, 1.0)["warnings"]))


class TestScatter(unittest.TestCase):
    def test_min_sep_honored(self):
        p = random_scatter(200, 100, 0, min_sep=3.0)
        from scipy.spatial import cKDTree
        d, _ = cKDTree(p).query(p, k=2)
        self.assertGreaterEqual(d[:, 1].min(), 3.0 - 1e-9); self.assertEqual(len(p), 200)


class TestSequence(unittest.TestCase):
    def test_two_stage_continuity(self):
        im = [load_image(os.path.join(SAMPLES, f)) for f in ("star.png", "diamond.png")]
        r = build_sequence(im, n=150, min_sep=1.5)
        a, b = r["segments"]
        np.testing.assert_allclose(a["target"], b["start"])        # particle identity continuity
        self.assertEqual(len(b["target"]), 150); self.assertEqual(len(b["delay"]), 150)
        self.assertEqual(len(r["segments"]), 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError): build_sequence([])


if __name__ == "__main__":
    unittest.main()


class TestSquaredAssignment(unittest.TestCase):
    def test_separation_bound_holds_and_no_collisions(self):
        from swarm.assignment import hungarian_sq
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(2)
        s = random_scatter(150, 100, 2, min_sep=4.0)
        t = rng.uniform(-40, 40, (150, 2))
        t = t[np.argsort(t[:, 0])]  # arbitrary targets; enforce spacing by thinning
        keep = [0]
        for i in range(1, len(t)):
            if np.min(np.linalg.norm(t[keep] - t[i], axis=1)) >= 4.0: keep.append(i)
        t = t[keep]; s = s[:len(t)]
        tg = t[hungarian_sq(s, t)]
        gap = lambda P: cKDTree(P).query(P, k=2)[0][:, 1].min()
        bound = min(gap(s), gap(tg)) / 2 ** 0.5
        r = scan(s, tg, np.zeros(len(s)), 5, 0.0001, "linear", 200)
        self.assertGreaterEqual(r["min_sep"], bound - 1e-6)

    def test_sequence_collision_free_default(self):
        im = [load_image(os.path.join(SAMPLES, f)) for f in ("star.png", "diamond.png")]
        r = build_sequence(im, n=300, min_sep=1.0)
        for s in r["segments"]:
            self.assertEqual(s["metrics"]["collisions_planned"], 0); self.assertEqual(s["warnings"], [])


class TestPhysicalLimits(unittest.TestCase):
    def test_smooth_easing_endpoints(self):
        from swarm.animation import EASINGS
        self.assertAlmostEqual(float(EASINGS["smooth"](np.float64(0))), 0); self.assertAlmostEqual(float(EASINGS["smooth"](np.float64(1))), 1)

    def test_auto_time_meets_speed_and_accel(self):
        p = plan_segment(A([[0, 0]]), A([[80, 0]]), 1, 1.0, "smooth", vmax=6, amax=2, auto_time=True)
        self.assertLessEqual(p["peak_speed"], 6 + 1e-9); self.assertLessEqual(p["peak_accel"], 2 + 1e-9)
        self.assertEqual(p["warnings"], []); self.assertGreater(p["flight_time"], 1)

    def test_accel_limit_warning_without_auto(self):
        p = plan_segment(A([[0, 0]]), A([[80, 0]]), 5, 1.0, "smooth", amax=2)
        self.assertTrue(any("acceleration" in w.lower() for w in p["warnings"]))

    def test_position_error_clearance_warning(self):
        p = plan_segment(A([[0, 0], [0, 3]]), A([[10, 0], [10, 3]]), 4, 1.0, "linear", pos_error=1.6)
        self.assertTrue(any("clearance" in w.lower() for w in p["warnings"]))


class TestExport(unittest.TestCase):
    def setUp(self):
        from swarm.export import build_export
        im = [load_image(os.path.join(SAMPLES, f)) for f in ("star.png", "diamond.png")]
        self.seq = build_sequence(im, n=100, extent=100.0, min_sep=2.0)
        self.build = build_export

    def _read(self, plane="horizontal", **kw):
        import io, zipfile, csv
        z = zipfile.ZipFile(io.BytesIO(self.build(self.seq, plane=plane, **kw)))
        rows = list(csv.DictReader(io.StringIO(z.read("trajectories.csv").decode())))
        return z, rows

    def test_package_contents_and_final_positions(self):
        z, rows = self._read(rate_hz=2)
        self.assertEqual(set(z.namelist()), {"trajectories.csv", "plan.json", "REPORT.md"})
        ids = {int(r["drone_id"]) for r in rows}; self.assertEqual(ids, set(range(100)))
        last = {int(r["drone_id"]): r for r in rows}                 # last row per drone = final formation
        tgt = np.array(self.seq["segments"][-1]["target"])
        for i in (0, 50, 99):
            self.assertAlmostEqual(float(last[i]["x_m"]), tgt[i][0], places=2)
            self.assertAlmostEqual(float(last[i]["y_m"]), tgt[i][1], places=2)
        self.assertIn("NOT A CERTIFIED FLIGHT PLAN", z.read("REPORT.md").decode())

    def test_vertical_plane_maps_image_y_to_altitude(self):
        _, rows = self._read(plane="vertical", rate_hz=2, base_alt=20)
        zs = [float(r["z_m"]) for r in rows]; self.assertGreaterEqual(min(zs), 20 - 1e-6)
        self.assertLessEqual(max(zs), 20 + 100 + 1e-6)

    def test_altitude_limit_flagged(self):
        z, _ = self._read(plane="vertical", rate_hz=2, base_alt=60, max_alt=120)
        self.assertIn("VIOLATION", z.read("REPORT.md").decode())
