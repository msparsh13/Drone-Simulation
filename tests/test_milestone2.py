import itertools, os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.init import random_scatter, grid, make_start
from swarm.assignment import greedy, hungarian, assign, travel_stats
from swarm.animation import positions_at
from swarm.imaging import load_image
from swarm.server import build_payload, SAMPLES


class TestInit(unittest.TestCase):
    def test_random_in_bounds_and_count(self):
        p = random_scatter(300, 100, seed=1)
        self.assertEqual(p.shape, (300, 2)); self.assertLessEqual(np.abs(p).max(), 50)

    def test_grid_count_unique(self):
        p = grid(250, 100)
        self.assertEqual(len(p), 250); self.assertEqual(len({tuple(r) for r in p}), 250)

    def test_bad_mode(self):
        with self.assertRaises(ValueError): make_start("spiral", 10)


class TestAssignment(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.s, self.t = rng.uniform(-50, 50, (200, 2)), rng.uniform(-50, 50, (200, 2))

    def test_is_permutation(self):
        for m in ("greedy", "hungarian"):
            self.assertEqual(sorted(assign(self.s, self.t, m).tolist()), list(range(200)))

    def test_hungarian_not_worse_than_greedy(self):
        g = travel_stats(self.s, self.t[greedy(self.s, self.t)])["total"]
        h = travel_stats(self.s, self.t[hungarian(self.s, self.t)])["total"]
        self.assertLessEqual(h, g + 1e-9); self.assertLess(h, g)

    def test_hungarian_matches_brute_force(self):
        rng = np.random.default_rng(5)
        s, t = rng.uniform(-10, 10, (6, 2)), rng.uniform(-10, 10, (6, 2))
        best = min(np.linalg.norm(s - t[list(p)], axis=1).sum() for p in itertools.permutations(range(6)))
        got = np.linalg.norm(s - t[hungarian(s, t)], axis=1).sum()
        self.assertAlmostEqual(got, best)

    def test_unknown_method_and_size_mismatch(self):
        with self.assertRaises(ValueError): assign(self.s, self.t, "magic")
        with self.assertRaises(ValueError): assign(self.s, self.t[:10], "greedy")


class TestAnimation(unittest.TestCase):
    def test_endpoints_exact(self):
        s, t = np.random.rand(20, 2), np.random.rand(20, 2)
        for e in ("linear", "cubic", "out"):
            np.testing.assert_allclose(positions_at(s, t, 0, 5, e), s)
            np.testing.assert_array_equal(positions_at(s, t, 5, 5, e), t)
            np.testing.assert_array_equal(positions_at(s, t, 9, 5, e), t)

    def test_monotonic_progress(self):
        s, t = np.zeros((1, 2)), np.array([[10.0, 0]])
        xs = [positions_at(s, t, x, 5, "cubic")[0, 0] for x in np.linspace(0, 5, 20)]
        self.assertTrue(all(b >= a for a, b in zip(xs, xs[1:])))


class TestPayload(unittest.TestCase):
    def test_payload_consistent(self):
        bgr, a = load_image(os.path.join(SAMPLES, "star.png"))
        p = build_payload(bgr, a, n=200, init="grid", method="hungarian")
        self.assertEqual(len(p["start"]), 200); self.assertEqual(len(p["target"]), 200)
        self.assertEqual(len(p["color"]), 200)
        self.assertGreater(p["metrics"]["travel_total"], 0)

    def test_hungarian_payload_shorter_than_greedy(self):
        bgr, a = load_image(os.path.join(SAMPLES, "star.png"))
        g = build_payload(bgr, a, n=300, method="greedy")["metrics"]["travel_total"]
        h = build_payload(bgr, a, n=300, method="hungarian")["metrics"]["travel_total"]
        self.assertLess(h, g)


if __name__ == "__main__":
    unittest.main()
