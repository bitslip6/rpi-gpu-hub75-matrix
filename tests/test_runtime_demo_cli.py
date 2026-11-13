"""Offscreen CLI regressions. Build first with `make example-runtime`."""
import fcntl
import json
import os
import pty
import re
import signal
import struct
import termios
import threading
import tempfile
import subprocess
import time
import unittest
from pathlib import Path

BINARY = Path(__file__).resolve().parents[1] / "build/release/example_runtime"


class RuntimeDemoCLI(unittest.TestCase):
    def run_demo(self, *args):
        return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=10)

    def test_frame_limit_and_inferred_topology(self):
        result = self.run_demo("--width", "192", "--height", "192",
                               "--warmup", "0", "--frames", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads(result.stdout)
        self.assertEqual(stats["frames"], 2)
        self.assertEqual((stats["width"], stats["height"]), (192, 192))

    def test_final_panel_size_used_regardless_of_order(self):
        result = self.run_demo("--width", "96", "--panel-width", "32",
                               "--height", "32", "--panel-height", "32",
                               "--warmup", "0", "--frames", "1")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nondivisible_inference_rejected(self):
        result = self.run_demo("--width", "65", "--frames", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("multiples of panel size", result.stderr)

    def test_explicit_topology_not_inferred(self):
        result = self.run_demo("--width", "65", "--chains", "1",
                               "--warmup", "0", "--frames", "1")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_seconds_limit_and_bcm_alias(self):
        start = time.monotonic()
        result = self.run_demo("--seconds", "1", "--warmup", "0", "--output-mode", "bcm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(time.monotonic() - start, 0.9)

    def test_forever_until_signal(self):
        for args in ([], ["--seconds", "-1"], ["--seconds", "forever"]):
            with self.subTest(args=args):
                process = subprocess.Popen([str(BINARY), *args], stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)
                try:
                    time.sleep(0.3)
                    self.assertIsNone(process.poll())
                    process.send_signal(signal.SIGINT)
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, stderr)
                    json.loads(stdout)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

    def test_verbose_redirected_snapshots_and_clean_stdout(self):
        for mode in ([], ["--retained"]):
            with self.subTest(mode=mode):
                result = self.run_demo("--verbose", "--seconds", "1", "--warmup", "0", *mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                json.loads(result.stdout)
                self.assertNotIn("\x1b", result.stderr)
                self.assertGreaterEqual(result.stderr.count("HUB75  /  LIVE RUNTIME"), 2)
                for metric in ("Content FPS", "target 60", "Panel refresh     n/a",
                               "CPU", "2s avg; 100% = 1 core", "RSS", "Input queue", "Latency avg", "Stage ms/frame",
                               "Totals:", "weighted-bcm", "curve CIE", "UI 500 ms", "Brightness 200/254", "playback speed 1x"):
                    self.assertIn(metric, result.stderr)

    def test_brightness_values(self):
        for brightness in (0, 32, 200, 254):
            with self.subTest(brightness=brightness):
                result = self.run_demo("--brightness", str(brightness), "--verbose",
                                       "--frames", "1", "--warmup", "0")
                self.assertEqual(result.returncode, 0, result.stderr)
                json.loads(result.stdout)
                self.assertIn(f"Brightness {brightness}/254", result.stderr)

    def test_invalid_brightness_reports_error(self):
        for args in (["--brightness"], ["--brightness", "-1"],
                     ["--brightness", "255"], ["--brightness", "bad"],
                     ["--brightness", "4294967296"]):
            with self.subTest(args=args):
                result = self.run_demo(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--brightness", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_legacy_dither_options(self):
        for quant in ("--quant-dither", "-q"):
            for spatial in ("--dither", "-l"):
                with self.subTest(quant=quant, spatial=spatial):
                    # Output-mode follows the flags: validation must use the
                    # final configuration, not argument order.
                    result = self.run_demo(quant, spatial, "2", "--output-mode", "equal-slot",
                                           "--frames", "2", "--warmup", "0", "--verbose")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    json.loads(result.stdout)
                    self.assertIn("Dither temporal on  |  spatial 2", result.stderr)
                    self.assertIn("offscreen: not applied", result.stderr)

    def test_spatial_dither_endpoints(self):
        for level in (0, 10):
            result = self.run_demo("--output-mode", "equal-slot", "--dither", str(level),
                                   "--frames", "1", "--warmup", "0", "--verbose")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Dither temporal off  |  spatial {level}", result.stderr)

    def test_invalid_spatial_dither(self):
        for option in ("--dither", "-l"):
            for value in ("-1", "11", "0.5", "nan", "inf", "bad"):
                with self.subTest(option=option, value=value):
                    result = self.run_demo("--output-mode", "equal-slot", option, value)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(option, result.stderr)
            result = self.run_demo(option)
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing value", result.stderr)

    def test_weighted_dither_not_silently_ignored(self):
        for flags in (["-q"], ["--quant-dither"]):
            with self.subTest(flags=flags):
                result = self.run_demo(*flags, "--output-mode", "bcm", "--frames", "1")
                self.assertEqual(result.returncode, 2)
                self.assertIn("requires --output-mode equal-slot", result.stderr)
                self.assertEqual(result.stdout, "")
        # Weighted BCM supports fixed spatial dithering independently of -q.
        for option in ("--dither", "-l"):
            for level in (0, 1, 2, 10):
                result = self.run_demo(option, str(level), "--frames", "1", "--warmup", "0",
                                       "--verbose")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Dither temporal n/a  |  spatial {level}", result.stderr)
                self.assertIn("offscreen: not applied", result.stderr)

    def test_file_help(self):
        result = self.run_demo("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--file PATH", result.stdout)
        self.assertIn("case-insensitive", result.stdout)
        self.assertIn("--loop", result.stdout)
        self.assertNotIn("--shader", result.stdout)
        self.assertNotIn("--video", result.stdout)

    def test_file_requires_path(self):
        for args in (["--file"], ["--file", "--frames", "1"]):
            with self.subTest(args=args):
                result = self.run_demo(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("missing value for --file", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_unknown_file_types(self):
        for path in ("", "clip", "clip.", "clip.txt", "clip.png", "clip.glsl.bak",
                     "directory.glsl/clip", "directory.mp4/", "clip.MP4.unknown"):
            with self.subTest(path=path):
                result = self.run_demo("--file", path, "--frames", "1")
                self.assertEqual(result.returncode, 2)
                self.assertIn("unsupported file type for --file", result.stderr)
                self.assertIn(repr(path), result.stderr)
                self.assertIn(".glsl", result.stderr)
                self.assertIn(".mp4", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_file_dispatch_by_extension_case_insensitively(self):
        # Missing files exercise the selected adapter without needing EGL or
        # media fixtures. CLI errors return 2; adapter errors return 1.
        with tempfile.TemporaryDirectory() as directory:
            for extensions, diagnostic in (
                (("glsl", "frag", "fs"), "opening shader"),
                (("mp4", "m4v", "mkv", "webm", "mov", "avi", "mpg", "mpeg", "ts"),
                 "video must be an existing local regular file"),
            ):
                for extension in extensions:
                    for case in (extension, extension.upper(), extension.title()):
                        with self.subTest(extension=case):
                            path = str(Path(directory) / f"missing.{case}")
                            result = self.run_demo("--file", path, "--frames", "1",
                                                   "--warmup", "0")
                            self.assertEqual(result.returncode, 1, result.stderr)
                            self.assertIn(diagnostic, result.stderr)
                            self.assertEqual(result.stdout, "")

    def test_old_media_options_rejected(self):
        for option, path in (("--shader", "unused.glsl"), ("--video", "unused.mp4")):
            with self.subTest(option=option):
                result = self.run_demo(option, path, "--frames", "1")
                self.assertEqual(result.returncode, 2)
                self.assertIn(option, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_upload_file_conflict(self):
        for flags in (("--upload", "--file", "unused.glsl"),
                      ("--file", "unused.mp4", "--upload")):
            with self.subTest(flags=flags):
                result = self.run_demo(*flags, "--frames", "1")
                self.assertEqual(result.returncode, 2)
                self.assertIn("--upload cannot be combined with --retained or --file", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_file_video_playback_with_loop_and_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.Mp4"
            # FFmpeg probes the stream independently of its extension. A tiny
            # Y4M stream avoids an external encoder/ffmpeg CLI dependency while
            # exercising mixed-case extension dispatch into the video adapter.
            # Two frames: consuming five proves --loop survives auto-selection.
            frame = b"FRAME\n" + bytes([81]) * 256 + bytes([90]) * 64 + bytes([240]) * 64
            video.write_bytes(b"YUV4MPEG2 W16 H16 F10:1 Ip A1:1 C420jpeg\n" + frame * 2)
            result = self.run_demo("--file", str(video), "--loop", "--speed", "2",
                                   "--brightness", "32", "--output-mode", "equal-slot",
                                   "--quant-dither", "--dither", "2", "--verbose",
                                   "--fps", "10", "--frames", "5", "--warmup", "0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreaterEqual(json.loads(result.stdout)["frames"], 5)
            for metric in (f"Source: {video}", "playback speed 2x", "Brightness 32/254",
                           "Dither temporal on  |  spatial 2", "offscreen: not applied"):
                self.assertIn(metric, result.stderr)

    def test_invalid_speed_reports_error(self):
        values = ("0", "-1", "0.0001", "100.1", "nan", "-nan", "inf", "-inf",
                  "1e100", "1e-100", "bad", "1x", "")
        for value in values:
            with self.subTest(value=value):
                result = self.run_demo("--speed", value, "--file", "unused.glsl")
                self.assertEqual(result.returncode, 2)
                self.assertIn("--speed", result.stderr)
                self.assertEqual(result.stdout, "")
        result = self.run_demo("--speed")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing value for --speed", result.stderr)

    def test_speed_requires_media(self):
        result = self.run_demo("--speed", "0.5", "--frames", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--speed requires --file", result.stderr)

    def test_valid_speed_parses_before_missing_media_error(self):
        # A nonexistent shader fails in the runtime, not the CLI. This checks
        # accepted endpoints without requiring an EGL-capable test host.
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.glsl")
            for value in ("0.001", "0.5", "1", "2", "100"):
                with self.subTest(value=value):
                    result = self.run_demo("--speed", value, "--file", missing,
                                           "--frames", "1", "--warmup", "0")
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn("runtime:", result.stderr)
                    self.assertNotIn("--speed", result.stderr)

    @unittest.skipUnless(os.environ.get("H75_TEST_GPU") == "1", "set H75_TEST_GPU=1 for EGL tests")
    def test_speed_scales_shader_time_and_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            shader = Path(directory) / "speed.GlSl"
            dump = Path(directory) / "rgba.bin"
            shader.write_text("void mainImage(out vec4 c, in vec2 p) {\n"
                              "    c = vec4(iTime, iTimeDelta, 0.0, 1.0);\n}\n")
            pixels = []
            for speed in ("0.5", "2"):
                result = self.run_demo("--file", str(shader), "--speed", speed,
                                       "--fps", "10", "--frames", "4", "--warmup", "0",
                                       "--dump", str(dump), "--verbose")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"playback speed {speed}x", result.stderr)
                pixels.append(dump.read_bytes()[:4])
            # At 10 FPS, delta ~= 0.1 s. Startup may consume one frame before
            # the CLI captures its frame-count baseline; don't assume the dump
            # is always exactly the fourth generated frame (time 0.3 vs 0.4 s).
            self.assertGreaterEqual(pixels[0][0], 25)
            self.assertLessEqual(pixels[0][0], 70)
            self.assertAlmostEqual(pixels[0][1], 13, delta=10)
            for slow, fast in zip(pixels[0][:2], pixels[1][:2]):
                self.assertGreater(fast, slow * 2.5)
                self.assertLess(fast, slow * 6)

    def test_unknown_option_reports_error(self):
        result = self.run_demo("--not-an-option", "32")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option '--not-an-option'", result.stderr)
        self.assertIn("--help", result.stderr)

    def test_nonverbose_stays_quiet(self):
        result = self.run_demo("--frames", "1", "--warmup", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_verbose_terminal_restored_on_interrupt(self):
        self.check_terminal(100, 18)

    def test_narrow_terminal_clipping(self):
        self.check_terminal(40, 8)

    def test_no_color_preserves_terminal_dashboard(self):
        self.check_terminal(100, 18, no_color=True)

    def test_fixed_colors_with_truecolor_environment(self):
        self.check_terminal(100, 18, truecolor=True)

    def check_terminal(self, width, height, no_color=False, truecolor=False):
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        process = subprocess.Popen([str(BINARY), "-v", "--retained"],
                                   stdout=subprocess.PIPE, stderr=slave, text=True,
                                   env={**os.environ, "TERM": "xterm-256color",
                                        "NO_COLOR": "1" if no_color else "",
                                        "COLORTERM": "truecolor" if truecolor else ""})
        os.close(slave)
        chunks = []

        def read_terminal():
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break  # Linux PTYs return EIO after the slave closes.
                if not chunk:
                    break
                chunks.append(chunk)

        # Styled frames can exceed the PTY buffer; drain while the child runs.
        reader = threading.Thread(target=read_terminal, daemon=True)
        reader.start()
        try:
            time.sleep(0.7)
            process.send_signal(signal.SIGINT)
            stdout, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0)
            json.loads(stdout)
            reader.join(timeout=2)
            self.assertFalse(reader.is_alive())
            screen = b"".join(chunks).decode()
            self.assertIn("\x1b[?1049h\x1b[?25l", screen)
            self.assertIn("\x1b[H\x1b[2K", screen)
            self.assertTrue(screen.endswith("\x1b[?25h\x1b[?1049l"))
            self.assertIn("Content FPS", screen)
            if no_color:
                self.assertNotRegex(screen, r"\x1b\[[0-9;]*m")
            else:
                self.assertIn("\x1b[0;1;36mHUB75", screen)
                # Label and value have separate SGR spans, not a whole-row tint.
                self.assertIn("\x1b[0;94mRUNNING", screen)
                self.assertIn("Content FPS \x1b[0;94m", screen)
                self.assertIn("\x1b[0m", screen)
                self.assertNotIn("38;5;", screen)
                self.assertNotIn("38;2;", screen)
            self.assertNotIn("Values:", screen)
            for snapshot in screen.split("\x1b[H")[1:]:
                plain = re.sub(r"\x1b\[[?0-9;]*[A-Za-z]", "", snapshot)
                self.assertTrue(all(len(row) < width for row in plain.splitlines()))
                self.assertLessEqual(len(plain.splitlines()), height - 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            reader.join(timeout=2)
            os.close(master)

    def test_invalid_seconds(self):
        for value in ("0", "-2", "nope"):
            self.assertEqual(self.run_demo("--seconds", value).returncode, 2)


if __name__ == "__main__":
    unittest.main()
