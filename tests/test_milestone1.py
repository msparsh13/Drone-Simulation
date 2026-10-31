import os, sys, tempfile, unittest
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.imaging import load_image, extract_mask, crop_to_content
from swarm.sampling import sample_targets, evaluate_formation
from swarm.render import render_formation
from swarm.cli import build_formation


def rect_img(w, h, pad=10):
    img = np.full((h + 2 * pad, w + 2 * pad, 3), 255, np.uint8)
    img[pad : pad + h, pad : pad + w] = 0
    return img


class TestImaging(unittest.TestCase):
    def test_dark_on_light_mask(self):
        self.assertEqual(extract_mask(rect_img(60, 40)).sum(), 60 * 40)

    def test_invert(self):
        self.assertEqual(
            extract_mask(255 - rect_img(60, 40), invert=True).sum(), 60 * 40
        )

    def test_alpha_used_when_transparent(self):
        rgba = np.zeros((50, 50, 4), np.uint8)
        rgba[10:30, 10:30] = (0, 0, 255, 255)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.png")
            cv2.imwrite(p, rgba)
            bgr, a = load_image(p)
        self.assertEqual(extract_mask(bgr, a).sum(), 400)

    def test_bad_extension(self):
        with self.assertRaises(ValueError):
            load_image("x.gif")

    def test_empty_mask_raises(self):
        with self.assertRaises(ValueError):
            crop_to_content(np.zeros((5, 5), bool), np.zeros((5, 5, 3), np.uint8))


class TestSampling(unittest.TestCase):
    def setUp(self):
        bgr = rect_img(200, 100)
        self.mask, self.bgr = crop_to_content(extract_mask(bgr), bgr)

    def test_count_ids_unique(self):
        f = sample_targets(self.mask, 250, self.bgr)
        self.assertEqual(len(f), 250)
        self.assertEqual(len(set(f.ids.tolist())), 250)
        self.assertEqual(f.positions.shape, (250, 2))
        self.assertEqual(f.colors.shape, (250, 3))

    def test_aspect_ratio_preserved(self):
        f = sample_targets(self.mask, 1000, self.bgr)
        ext = np.ptp(f.positions, axis=0)
        self.assertAlmostEqual(ext[0] / ext[1], 2.0, delta=0.2)

    def test_scale_bounds(self):
        f = sample_targets(self.mask, 300, self.bgr, extent=100, scale=0.5)
        self.assertLessEqual(np.abs(f.positions[:, 0]).max(), 25.01)
        self.assertLessEqual(np.abs(f.positions[:, 1]).max(), 12.51)

    def test_even_spacing_and_coverage(self):
        m = evaluate_formation(sample_targets(self.mask, 400, self.bgr), self.mask)
        self.assertGreater(m["min_spacing"], 0.5 * m["mean_spacing"])  # no clumps
        self.assertLess(m["coverage_max"], 2.5 * m["mean_spacing"])  # no holes

    def test_lloyd_beats_random(self):
        r = evaluate_formation(sample_targets(self.mask, 300, iterations=0), self.mask)
        l = evaluate_formation(sample_targets(self.mask, 300, iterations=30), self.mask)
        self.assertGreater(l["min_spacing"], r["min_spacing"])
        self.assertLess(l["coverage_max"], r["coverage_max"])

    def test_deterministic_seed(self):
        a = sample_targets(self.mask, 100, seed=3).positions
        b = sample_targets(self.mask, 100, seed=3).positions
        np.testing.assert_array_equal(a, b)

    def test_too_many_particles(self):
        with self.assertRaises(ValueError):
            sample_targets(self.mask, int(self.mask.sum()) + 1)

    def test_colors_follow_image(self):
        bgr = np.zeros((40, 80, 3), np.uint8)
        bgr[:, :40] = (0, 0, 255)
        bgr[:, 40:] = (255, 0, 0)
        f = sample_targets(np.ones((40, 80), bool), 200, bgr)
        self.assertTrue(
            (f.colors[f.positions[:, 0] < -5][:, 0] > 200).all()
        )  # red (RGB) left
        self.assertTrue(
            (f.colors[f.positions[:, 0] > 5][:, 2] > 200).all()
        )  # blue right


class TestPipeline(unittest.TestCase):
    def test_end_to_end_and_render(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            cv2.imwrite(p, rect_img(120, 80))
            mask, f = build_formation(p, n=200)
        img = render_formation(f, size=400)
        self.assertEqual(img.shape, (400, 400, 3))
        self.assertLess(img.mean(), 255)


if __name__ == "__main__":
    unittest.main()
