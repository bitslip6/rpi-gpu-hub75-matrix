"""Import ABI guard and offscreen affinity exclusion (no GPIO/real affinity)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import hub75gpu as h


@unittest.skipUnless(shutil.which("cc"), "C compiler needed for import interposition test")
class RuntimeABI(unittest.TestCase):
    def preload(self, source, code):
        with tempfile.TemporaryDirectory() as td:
            c = Path(td) / "guard.c"
            so = Path(td) / "guard.so"
            c.write_text(source)
            subprocess.run(["cc", "-shared", "-fPIC", str(c), "-o", str(so)], check=True,
                           capture_output=True)
            env = dict(os.environ, LD_PRELOAD=str(so))
            return subprocess.run([sys.executable, "-c", code], env=env,
                                  capture_output=True, text=True, timeout=30)

    def test_matching_abi(self):
        self.assertEqual(h.capabilities()["abi"], 3)
        self.assertIn("hub75-runtime-3", h.capabilities()["build"])

    def test_abi_2_rejected_at_import(self):
        result = self.preload("unsigned hub75_runtime_abi(void) { return 2; }", "import hub75gpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ABI", result.stderr)

    def test_import_cpu_and_media_offscreen_never_request_affinity(self):
        # Any accidental call to the shared helper kills only this subprocess.
        # This never invokes a real affinity syscall, even on CPU3-less hosts.
        source = """
#include <unistd.h>
#include <stddef.h>
int h75_gpio_affinity_require(char *message, size_t size) { _exit(77); }
"""
        code = """
import hub75gpu as h
for mode in h.OutputMode:
    with h.Display(h.DisplayConfig(width=16, height=16, offscreen=True, output_mode=mode)) as d:
        with d.frame() as f: f.clear((3,17,91))
        assert d.snapshot() == bytes((3,17,91,255))*256
"""
        # Reuse the real shader/video offscreen coverage under the same guard.
        # EGL stays opt-in via HUB75_TEST_GPU; no test starts physical output.
        code += f"""
import sys, unittest
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import test_content
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_content))
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
        result = self.preload(source, code)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
