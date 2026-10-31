import os, sys, unittest
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.placement import place_2d, shape_points, PLACEMENTS, SHAPES
from swarm.init import random_scatter3d, grid3d
from swarm.assignment import hungarian_sq
from swarm.collision import scan
from swarm.sequence import build_sequence
from swarm.timeline import Timeline
from swarm.failure import simulate, drain_rate
from swarm.imaging import load_image
from swarm.server import SAMPLES

IMG = lambda *f: [load_image(os.path.join(SAMPLES, x)) for x in f]
nn = lambda P: cKDTree(P).query(P, k=2)[0][:, 1]


class TestPlacement(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.p = np.column_stack([r.uniform(-50, 50, 200), r.uniform(-30, 30, 200)])

    def test_vertical_wall(self):
        w = place_2d(self.p, "vertical", 20)
        self.assertTrue((w[:, 1] == 0).all())
        self.assertAlmostEqual(w[:, 2].min(), 20)

    def test_horizontal_flat(self):
        self.assertTrue((place_2d(self.p, "horizontal", 33)[:, 2] == 33).all())

    def test_cylinder_keeps_arc_spacing(self):
        a, b = (
            np.median(nn(self.p)),
            np.median(nn(place_2d(self.p, "cylinder", 20)[:, :2])),
        )
        w = place_2d(self.p, "cylinder", 20)
        self.assertAlmostEqual(np.ptp(w[:, 1]) > 5, True)  # genuinely curved in depth
        self.assertGreater(np.median(nn(w)), 0.5 * a)

    def test_sphere_points_lie_on_a_sphere(self):
        w = place_2d(self.p, "sphere", 20)
        A = np.column_stack([2 * w, np.ones(len(w))])
        b = (w**2).sum(1)  # least-squares sphere fit
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        c = sol[:3]
        R = np.sqrt(sol[3] + c @ c)
        resid = np.abs(np.linalg.norm(w - c, axis=1) - R)
        self.assertLess(resid.max() / R, 0.02)

    def test_extrude_layers(self):
        w = place_2d(self.p, "extrude", 20, layers=4, gap=3.0)
        self.assertEqual(len(np.unique(w[:, 1])), 4)
        self.assertAlmostEqual(np.ptp(w[:, 1]), 9.0)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            place_2d(self.p, "blob")

    def test_shapes(self):
        for s in SHAPES:
            with self.subTest(shape=s):
                p, c = shape_points(s, 300, 30, 20)
                self.assertEqual(p.shape, (300, 3))
                self.assertEqual(c.shape, (300, 3))
                self.assertGreaterEqual(p[:, 2].min(), 20 - 1e-6)
                self.assertGreater(nn(p).min(), 0.15 * nn(p).mean())  # no clumping
        with self.assertRaises(ValueError):
            shape_points("pyramid", 10, 5)


class TestStart3D(unittest.TestCase):
    def test_scatter_spacing_and_box(self):
        p = random_scatter3d(200, 100, 20, 0, min_sep=3.5)
        self.assertGreaterEqual(nn(p).min(), 3.5 - 1e-9)
        self.assertGreaterEqual(p[:, 2].min(), 20)

    def test_grid(self):
        g = grid3d(100, 100, 25)
        self.assertTrue((g[:, 2] == 25).all())
        self.assertEqual(len(g), 100)

    def test_squared_assignment_bound_holds_in_3d(self):
        s = random_scatter3d(120, 100, 20, 1, min_sep=4.0)
        t = random_scatter3d(120, 100, 20, 2, min_sep=4.0)
        tg = t[hungarian_sq(s, t)]
        bound = min(nn(s).min(), nn(tg).min()) / 2**0.5
        self.assertGreaterEqual(
            scan(s, tg, np.zeros(120), 5, 0.01, "linear", 200)["min_sep"], bound - 1e-6
        )


class TestSequence3D(unittest.TestCase):
    def test_mixed_image_and_shape_stages(self):
        seq = build_sequence(
            [IMG("heart.png")[0], {"shape": "sphere"}],
            n=150,
            extent=100.0,
            min_sep=2.0,
            placement="vertical",
        )
        self.assertEqual(len(seq["segments"]), 2)
        self.assertEqual(np.array(seq["segments"][1]["target"]).shape, (150, 3))
        self.assertTrue(all(not s["warnings"] for s in seq["segments"]))

    def test_timeline_continuity_and_endpoints(self):
        seq = build_sequence(
            IMG("heart.png", "star_gold.png"), n=100, extent=100.0, min_sep=2.0
        )
        tl = Timeline(seq)
        np.testing.assert_allclose(
            tl.pos(0), np.array(seq["segments"][0]["start"]), atol=1e-6
        )
        np.testing.assert_allclose(
            tl.pos(tl.total), np.array(seq["segments"][-1]["target"]), atol=1e-6
        )
        k = tl.offs[1]
        self.assertLess(
            np.abs(tl.pos(k - 1e-6) - tl.pos(k + 1e-6)).max(), 0.01
        )  # no jump between stages


def _show():
    seq = build_sequence(
        IMG("heart.png", "star_gold.png"),
        n=300,
        extent=100.0,
        min_sep=2.0,
        takeoff=True,
        landing=True,
        base_alt=20.0,
    )
    tl = Timeline(seq)
    return (
        seq,
        tl,
        tl.offs[0] + seq["segments"][0]["total"] * 0.5,
        tl.offs[0] + seq["segments"][0]["total"] + 1.0,
    )


class TestFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seq, cls.tl, cls.mid, cls.hold = _show()

    def run_f(self, typ, t=None, drone=120, **fs):
        return simulate(
            self.seq,
            failures=[dict(drone=drone, type=typ, t=self.mid if t is None else t)],
            settings=fs,
        )

    def test_nominal_telemetry(self):
        r = simulate(self.seq)
        sp = np.array(r["speed"]) / 10
        bat = np.array(r["battery"]) / 10
        self.assertLessEqual(sp.max(), 6.0 * 1.05)
        self.assertTrue((np.diff(bat, axis=1) <= 1e-9).all())
        self.assertTrue((np.array(r["status"]) == 0).all())
        self.assertAlmostEqual(bat[:, 0].max(), 100.0, places=1)
        self.assertEqual(r["metrics"]["verdict"], "no failures injected")

    def test_motor_failure_free_fall_crashes(self):
        r = self.run_f("motor")
        f = r["failures"][0]
        self.assertEqual(f["state"], "CRASHED")
        self.assertGreater(f["impact"], 8.0)
        self.assertAlmostEqual(f["end"][2], 0.0)
        self.assertTrue(any("thrust was lost" in w for w in r["warnings"]))

    def test_battery_failure_lands_softly(self):
        f = self.run_f("battery")["failures"][0]
        self.assertEqual(f["state"], "LANDED")
        self.assertLess(f["impact"], 2.5)

    def test_empty_battery_falls(self):
        f = self.run_f("battery", battery_critical=0.3)["failures"][0]
        self.assertEqual(
            f["state"], "CRASHED"
        )  # ran out before the controlled descent finished

    def test_rtl_returns_near_launch_position(self):
        f = (
            self.run_f("battery", policy="rtl")["failures"][0]
            if False
            else simulate(
                self.seq,
                failures=[dict(drone=120, type="battery", t=self.mid, policy="rtl")],
            )["failures"][0]
        )
        home = np.array(self.seq["segments"][0]["start"])[120][:2]
        self.assertLess(np.linalg.norm(np.array(f["end"][:2]) - home), 2.5)
        self.assertEqual(f["state"], "LANDED")

    def test_link_loss_after_show_changes_nothing(self):
        r = simulate(
            self.seq, failures=[dict(drone=5, type="link", t=self.tl.total - 3)]
        )
        self.assertEqual(r["failures"][0]["state"], "LANDED")  # plan landed it normally

    def test_led_failure_has_no_motion_effect(self):
        r = self.run_f("led")
        self.assertEqual(r["override"], {})
        self.assertEqual(r["metrics"]["evaders"], 0)
        self.assertEqual(r["failures"][0]["state"], "LED FAULT")

    def test_reaction_reduces_contacts_during_hold(self):
        r = self.run_f("motor", t=self.hold)["metrics"]
        self.assertGreater(r["contacts_without_reaction"], 0)
        self.assertLess(r["contacts_with_reaction"], r["contacts_without_reaction"])

    def test_reaction_off_reports_baseline(self):
        r = self.run_f("motor", t=self.hold, reaction=False)["metrics"]
        self.assertEqual(r["contacts_with_reaction"], r["contacts_without_reaction"])
        self.assertEqual(r["evaders"], 0)

    def test_validation(self):
        for bad in (
            dict(drone=9999, type="motor", t=1),
            dict(drone=1, type="laser", t=1),
            dict(drone=1, type="motor", t=1e6),
            dict(drone=1, type="battery", t=5, policy="teleport"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                simulate(self.seq, failures=[bad])

    def test_battery_model(self):
        self.assertAlmostEqual(
            float(drain_rate(0, 0, 15)) * 900, 100.0
        )  # hover for the full endurance drains 100%
        self.assertGreater(float(drain_rate(6, 0, 15)), float(drain_rate(0, 0, 15)))


if __name__ == "__main__":
    unittest.main()
