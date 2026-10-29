"""Runs the real page JavaScript in Node (stub DOM) against a live server: catches runtime errors in the UI."""
import os, shutil, socket, subprocess, sys, time, unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestUiSmoke(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_page_runs_all_phases(self):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        srv = subprocess.Popen([sys.executable, "-m", "swarm.server", "--no-browser", "--port", str(port)],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(2.5)
            r = subprocess.run(["node", os.path.join(os.path.dirname(__file__), "ui_smoke.js")], capture_output=True,
                               text=True, timeout=150, env=dict(os.environ, PORT=str(port)))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("ALL OK", r.stdout)
        finally:
            srv.kill()


if __name__ == "__main__":
    unittest.main()
