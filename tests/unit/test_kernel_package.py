"""Packaging script checks using synthetic build files/tools; no kernel or GPIO."""
from pathlib import Path
import os
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "kernel/package-hub75-nohz.sh"
KREL = "6.18.54-v8-16k-hub75-nohz+"
COMMIT = "703bc4e92a9f98d797f0289bf2adb1e8b04c4bbb"
TOOLS = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
name = Path(sys.argv[0]).name
args = sys.argv[1:]
release = "6.18.54-v8-16k-hub75-nohz+"
if name == "git":
    if "rev-parse" in args:
        print("703bc4e92a9f98d797f0289bf2adb1e8b04c4bbb")
    elif "status" in args and os.environ.get("FAKE_DIRTY"):
        print(" M drivers/misc/rp1-pio.c")
elif name == "aarch64-linux-gnu-gcc":
    print("aarch64-linux-gnu" if "-dumpmachine" in args else "fake packaging compiler")
elif name == "make":
    if "kernelrelease" in args:
        print(release)
    elif "modules_install" in args:
        stage = Path(next(x.split("=", 1)[1] for x in args if x.startswith("INSTALL_MOD_PATH=")))
        directory = stage / "lib/modules" / release
        directory.mkdir(parents=True)
        (directory / "rp1-pio.ko").write_bytes(b"synthetic module")
        (directory / "build").symlink_to("/nonexistent/desktop/build")
    else:
        raise SystemExit("unexpected make target")
elif name == "modinfo":
    if os.environ.get("FAKE_MODINFO_ERROR"):
        raise SystemExit(1)
    if "-F" in args:
        print(release + " SMP preempt modversions aarch64")
    elif os.environ.get("FAKE_PIO_PATCH"):
        print("heavy_tx_burst4_sm:experimental")
'''


@unittest.skipIf(os.geteuid() == 0, "packaging intentionally refuses root")
class KernelPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "workspace with spaces"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ("git", "make", "modinfo", "aarch64-linux-gnu-gcc"):
            path = self.bin / tool
            path.write_text(TOOLS)
            path.chmod(0o755)
        for name in ("src/Makefile", "out/Module.symvers", "out/System.map"):
            path = self.work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture\n")
        image = self.work / "out/arch/arm64/boot/Image"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"\x00Linux version " + KREL.encode() + b" synthetic build\n")
        release = self.work / "out/include/config/kernel.release"
        release.parent.mkdir(parents=True)
        release.write_text(KREL + "\n")
        self.config = self.work / "out/.config"
        self.config.write_text("\n".join((
            'CONFIG_CC_VERSION_TEXT="synthetic build compiler"',
            "CONFIG_NO_HZ_FULL=y", "CONFIG_RCU_NOCB_CPU=y",
            "CONFIG_VIRT_CPU_ACCOUNTING_GEN=y", "CONFIG_CONTEXT_TRACKING_USER=y",
            "CONFIG_CPU_ISOLATION=y", "CONFIG_ARM64_16K_PAGES=y",
            "CONFIG_PREEMPT=y", "CONFIG_HZ=250", "CONFIG_RP1_PIO=m", "CONFIG_MODVERSIONS=y",
        )) + "\n")
        self.archive = self.work / (KREL + ".tar.gz")

    def run_script(self, **overrides):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   CROSS_COMPILE="aarch64-linux-gnu-")
        env.update(overrides)
        return subprocess.run(["bash", str(SCRIPT), str(self.work)],
                              env=env, text=True, capture_output=True, timeout=30)

    def test_package_and_checksum(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.archive.is_file())
        check = subprocess.run(["sha256sum", "-c", self.archive.name + ".sha256"],
                               cwd=self.work, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        with tarfile.open(self.archive) as archive:
            names = archive.getnames()
            self.assertIn("boot/kernel-hub75-nohz.img", names)
            self.assertIn("SHA256SUMS", names)
            self.assertNotIn(f"lib/modules/{KREL}/build", names)
            self.assertEqual(archive.extractfile("provenance/source-commit.txt").read(),
                             (COMMIT + "\n").encode())
        # Existing complete packages must never be replaced.
        original = self.archive.read_bytes()
        retry = self.run_script()
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("refusing to overwrite", retry.stderr)
        self.assertEqual(self.archive.read_bytes(), original)

    def test_reject_dirty_source(self):
        result = self.run_script(FAKE_DIRTY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not clean", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_reject_wrong_configuration(self):
        self.config.write_text(self.config.read_text().replace("CONFIG_NO_HZ_FULL=y", "CONFIG_NO_HZ_FULL=n"))
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Configuration check failed", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_reject_patched_pio_module(self):
        result = self.run_script(FAKE_PIO_PATCH="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Experimental PIO parameter", result.stderr)
        self.assertFalse(self.archive.exists())

    def test_modinfo_failure_is_not_accepted(self):
        result = self.run_script(FAKE_MODINFO_ERROR="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("packaging stopped", result.stderr)
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
