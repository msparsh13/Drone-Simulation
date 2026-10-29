import io, json, os, shutil, subprocess, sys, unittest, zipfile, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swarm.lights import resolve_cues, apply_cues, check_cues, hsv
from swarm.sequence import build_sequence
from swarm.export import build_export
from swarm.imaging import load_image
from swarm.server import SAMPLES

SEGS = [dict(total=10.0), dict(total=20.0)]
WHITE = np.full((4, 3), 255.0)
XY = np.array([[-30, 10], [30, 10], [-30, -10], [30, -10]], float)


def one(cue, t, base=WHITE, xy=XY):
    return apply_cues(base, t, xy, resolve_cues([cue], SEGS, 5.0), 100.0)


class TestCues(unittest.TestCase):
    def test_anchor_intervals(self):
        r = resolve_cues([{"effect": "on", "anchor": {"type": "hold", "stage": 0}},
                          {"effect": "on", "anchor": {"type": "flight", "stage": 1}},
                          {"effect": "on"}], SEGS, 5.0)
        self.assertEqual((r[0]["t0"], r[0]["t1"]), (10, 15))
        self.assertEqual((r[1]["t0"], r[1]["t1"]), (15, 35))
        self.assertEqual((r[2]["t0"], r[2]["t1"]), (0, 40))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): resolve_cues([{"effect": "laser"}], SEGS, 5)
        with self.assertRaises(ValueError): resolve_cues([{"effect": "on", "anchor": {"type": "hold", "stage": 5}}], SEGS, 5)

    def test_blink_on_off(self):
        c = {"effect": "blink", "period": 1.0, "duty": 0.5, "anchor": {"type": "abs", "t0": 0, "t1": 10}}
        self.assertTrue((one(c, 0.25) == 255).all()); self.assertTrue((one(c, 0.75) == 0).all())
        self.assertTrue((one(c, 11.0) == 255).all())           # outside window: untouched

    def test_off_and_scope(self):
        c = {"effect": "off", "scope": "left"}
        out = one(c, 1.0)
        self.assertTrue((out[[0, 2]] == 0).all()); self.assertTrue((out[[1, 3]] == 255).all())

    def test_color_override_and_base_kept(self):
        out = one({"effect": "blink", "color": [255, 0, 0], "period": 2, "duty": 1.0}, 0.5)
        self.assertTrue((out == [255, 0, 0]).all())

    def test_fade_out_half(self):
        c = {"effect": "fade_out", "anchor": {"type": "abs", "t0": 0, "t1": 10}}
        self.assertAlmostEqual(float(one(c, 5.0)[0, 0]), 127.5)

    def test_rainbow_varies_with_x(self):
        out = one({"effect": "rainbow", "axis": "x", "period": 4}, 0.0)
        self.assertFalse(np.allclose(out[0], out[1]))

    def test_later_cue_overrides(self):
        r = resolve_cues([{"effect": "off"}, {"effect": "on"}], SEGS, 5.0)
        self.assertTrue((apply_cues(WHITE, 1.0, XY, r, 100) == 255).all())

    def test_flash_rate_warning(self):
        self.assertTrue(check_cues([{"effect": "blink", "period": 0.2}]))
        self.assertFalse(check_cues([{"effect": "blink", "period": 1.0}]))


class TestParity(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_js_matches_python(self):
        rng = np.random.default_rng(1)
        cues = [{"effect": "rainbow", "axis": "x", "period": 3, "scope": "left"},
                {"effect": "wave", "axis": "radial", "period": 2.5, "color": [10, 200, 90], "scope": "top"},
                {"effect": "blink", "period": 0.7, "duty": 0.3, "scope": "even"},
                {"effect": "pulse", "period": 1.9, "scope": "odd", "anchor": {"type": "abs", "t0": 1, "t1": 20}},
                {"effect": "fade_out", "anchor": {"type": "hold", "stage": 1}}]
        cues.append({"effect": "blink", "period": 1.3, "anchor": {"type": "takeoff"}})
        cues.append({"effect": "pulse", "period": 2.0, "anchor": {"type": "landing"}})
        rc = resolve_cues(cues, SEGS, 5.0, 12.0, 20.0)
        n = 40; xy = rng.uniform(-50, 50, (n, 2)); base = rng.uniform(0, 255, (n, 3)); ts = [0.3, 4.2, 12.0, 18.5, 33.0, 60.0, 75.0]
        exp = [apply_cues(base, t, xy, rc, 100.0).tolist() for t in ts]
        js = ("const L=require(process.argv[1]);const d=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "const rc=L.resolve(d.cues,d.segs,d.hold,12,20);"
              "console.log(JSON.stringify(d.ts.map(t=>d.base.map((b,i)=>L.color(b,i,d.n,d.xy[i][0],d.xy[i][1],t,rc,100)))));")
        payload = json.dumps(dict(cues=cues, segs=SEGS, hold=5.0, ts=ts, base=base.tolist(), xy=xy.tolist(), n=n))
        p = subprocess.run(["node", "-e", js, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "swarm", "static", "lights.js"))],
                           input=payload, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        np.testing.assert_allclose(np.array(json.loads(p.stdout)), np.array(exp), atol=1e-6)


class TestExportLights(unittest.TestCase):
    def test_lights_csv_blink(self):
        im = [load_image(os.path.join(SAMPLES, "star.png"))]
        seq = build_sequence(im, n=60, extent=150.0, min_sep=2.0, solid_color=[255, 244, 214], hold=4.0)
        cues = [{"effect": "blink", "period": 1.0, "duty": 0.5, "anchor": {"type": "hold", "stage": 0}}]
        z = zipfile.ZipFile(io.BytesIO(build_export(seq, rate_hz=2, cues=cues, light_rate_hz=10)))
        self.assertIn("lights.csv", z.namelist())
        rows = list(csv.DictReader(io.StringIO(z.read("lights.csv").decode())))
        hold_start = seq["segments"][0]["total"]
        on = [r for r in rows if r["drone_id"] == "0" and abs(float(r["t_s"]) - (hold_start + 0.2)) < 1e-6]
        off = [r for r in rows if r["drone_id"] == "0" and abs(float(r["t_s"]) - (hold_start + 0.7)) < 1e-6]
        self.assertEqual(int(on[0]["r"]), 255); self.assertEqual(int(off[0]["r"]), 0)
        self.assertIn("Lighting", z.read("REPORT.md").decode())

    def test_solid_color_applied(self):
        im = [load_image(os.path.join(SAMPLES, "star.png"))]
        seq = build_sequence(im, n=60, solid_color=[10, 20, 30])
        self.assertTrue(all(c == [10, 20, 30] for c in seq["segments"][0]["color"]))

    def test_no_cues_no_lights_file(self):
        im = [load_image(os.path.join(SAMPLES, "star.png"))]
        z = zipfile.ZipFile(io.BytesIO(build_export(build_sequence(im, n=60), rate_hz=1)))
        self.assertNotIn("lights.csv", z.namelist())


if __name__ == "__main__":
    unittest.main()
