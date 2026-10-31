import glob, os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.cli import build_formation
from swarm.sampling import evaluate_formation

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


class TestAllSamples(unittest.TestCase):
    def test_every_sample_builds_evenly(self):
        files = sorted(glob.glob(os.path.join(SAMPLES, "*.png")))
        self.assertGreaterEqual(len(files), 18)
        for f in files:
            with self.subTest(sample=os.path.basename(f)):
                mask, form = build_formation(f, n=300)
                m = evaluate_formation(form, mask)
                self.assertEqual(len(form), 300)
                self.assertGreater(
                    m["min_spacing"], 0.3 * m["mean_spacing"]
                )  # no clumping
                self.assertLess(m["coverage_max"], 3.0 * m["mean_spacing"])  # no holes


if __name__ == "__main__":
    unittest.main()
