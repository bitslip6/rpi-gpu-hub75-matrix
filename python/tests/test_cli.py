"""Demo argument checks and native video playback, always without GPIO."""
import contextlib
import io
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import hub75gpu as h
from hub75gpu import __main__ as demo


def video_fixture(path: Path) -> None:
    # Three real YUV4MPEG2 frames, no ffmpeg executable or encoder dependency.
    # Wide source -> square display allows verification of aspect-fit bars.
    frame_pixels = 8 * 4
    path.write_bytes(b"YUV4MPEG2 W8 H4 F10:1 Ip A1:1 C444\n" + b"".join(
        b"FRAME\n" + bytes((y,)) * frame_pixels + bytes((128,)) * (2 * frame_pixels)
        for y in (64, 128, 192)
    ))


class DemoArguments(unittest.TestCase):
    def invoke(self, args, *, video=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        caps = dict(h.capabilities(), video=video)
        with mock.patch.object(sys, "argv", ["hub75gpu", *args]), \
             mock.patch.object(h, "capabilities", return_value=caps), \
             mock.patch.object(h, "Display") as display, \
             mock.patch.object(demo.time, "sleep"), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = 0
            try:
                demo.main()
            except SystemExit as exc:
                code = exc.code
        return code, display, stdout.getvalue(), stderr.getvalue()

    def test_video_shortcut_and_output_settings(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clip.y4m"
            video_fixture(path)
            for output_mode in ("weighted-bcm", "equal-slot"):
                with self.subTest(output_mode=output_mode):
                    code, display, _, error = self.invoke([
                        "--video", str(path), "--seconds", "5", "--brightness", "127",
                        "--preserve-aspect", "--output-mode", output_mode,
                        "--pwm-bits", "11", "--pwm-lsb-ns", "130", "--color-curve", "cie",
                    ])
                    self.assertEqual(code, 0, error)
                    config = display.call_args.args[0]
                    self.assertEqual(config.brightness, 127)
                    self.assertTrue(config.offscreen)
                    self.assertTrue(config.video_preserve_aspect)
                    expected_mode = h.OutputMode.WEIGHTED_BCM if output_mode == "weighted-bcm" else h.OutputMode.EQUAL_SLOT
                    self.assertEqual(config.output_mode, expected_mode)
                    instance = display.return_value.__enter__.return_value
                    instance.play_video.assert_called_once_with(path, loop=True)
                    instance.play.assert_not_called()
                    instance.play_video.return_value.ready.assert_called_once_with(10)
                    instance.play_video.return_value.stop.assert_called_once_with(5)

    def test_existing_mode_file_syntax(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clip.y4m"
            video_fixture(path)
            code, display, _, error = self.invoke(["--mode", "video", "--file", str(path)])
            self.assertEqual(code, 0, error)
            config = display.call_args.args[0]
            self.assertEqual(config.brightness, 64)
            self.assertEqual(config.gpio_cpu, -1)
            self.assertFalse(config.video_preserve_aspect)
            display.return_value.__enter__.return_value.play_video.assert_called_once_with(path, loop=True)

    def test_gpio_cpu_forwarded_without_changing_other_settings(self):
        # Display is mocked: no affinity changes or GPIO activation in this test.
        for cpu in (-1, 0, 3):
            with self.subTest(cpu=cpu):
                code, display, _, error = self.invoke([
                    "--gpio", "--acknowledge-wiring", "--gpio-cpu", str(cpu),
                    "--output-mode", "weighted-bcm", "--width", "192", "--height", "192",
                    "--chains", "3", "--ports", "3",
                ])
                self.assertEqual(code, 0, error)
                config = display.call_args.args[0]
                self.assertEqual(config.gpio_cpu, cpu)
                self.assertFalse(config.offscreen)
                self.assertEqual(config.output_mode, h.OutputMode.WEIGHTED_BCM)
                self.assertEqual((config.width, config.height, config.chains, config.ports),
                                 (192, 192, 3, 3))

    def test_missing_native_video_support_before_gpio_startup(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clip.y4m"
            video_fixture(path)
            code, display, _, error = self.invoke([
                "--video", str(path), "--gpio", "--acknowledge-wiring",
            ], video=False)
            self.assertEqual(code, 2)
            self.assertIn("video support is not compiled", error)
            self.assertIn("HUB75_BACKEND=gpu", error)
            display.assert_not_called()

    def test_invalid_arguments_never_construct_display(self):
        cases = [
            (["--video", "/nonexistent/hub75-clip.mp4"], "existing local file"),
            (["--mode", "video"], "requires --file"),
            (["--video", "a.mp4", "--mode", "cpu"], "different --mode"),
            (["--video", "a.mp4", "--file", "b.mp4"], "same file"),
            (["--video", "a.mp4", "--gpio"], "--acknowledge-wiring"),
            (["--brightness", "-1"], "--brightness"),
            (["--brightness", "255"], "--brightness"),
            (["--gpio-cpu", "-2"], "--gpio-cpu"),
            (["--gpio-cpu", "invalid"], "--gpio-cpu"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                code, display, _, error = self.invoke(args)
                self.assertEqual(code, 2)
                self.assertIn(message, error)
                display.assert_not_called()

    def test_help_describes_video_controls(self):
        code, display, output, _ = self.invoke(["--help"])
        self.assertEqual(code, 0)
        for option in ("--video", "--brightness", "--preserve-aspect", "--output-mode", "--gpio-cpu"):
            self.assertIn(option, output)
        display.assert_not_called()


@unittest.skipUnless(h.capabilities()["video"], "FFmpeg-enabled build required")
class DemoNativeVideo(unittest.TestCase):
    def test_multiframe_video_in_both_output_modes_offscreen(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clip.y4m"
            video_fixture(path)
            for mode in ("weighted-bcm", "equal-slot"):
                with self.subTest(output_mode=mode):
                    output = Path(td) / f"{mode}.rgba"
                    result = subprocess.run([
                        sys.executable, "-X", "faulthandler", "-m", "hub75gpu",
                        "--video", str(path), "--seconds", "0.35", "--fps", "10",
                        "--width", "16", "--height", "16", "--preserve-aspect",
                        "--output-mode", mode, "--brightness", "64", "--output", str(output),
                    ], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    submitted = re.search(r"'submitted': (\d+)", result.stdout)
                    self.assertIsNotNone(submitted, result.stdout)
                    self.assertGreaterEqual(int(submitted.group(1)), 2)
                    self.assertIn("'scanned': 0", result.stdout)  # never opened GPIO
                    pixels = output.read_bytes()
                    self.assertEqual(len(pixels), 16 * 16 * 4)
                    self.assertEqual(pixels[3::4], bytes((255,)) * 256)
                    self.assertEqual(pixels[:16 * 4], bytes((0, 0, 0, 255)) * 16)
                    center = (8 * 16 + 8) * 4
                    self.assertGreater(pixels[center], 0)
                    self.assertEqual(pixels[center], pixels[center + 1])
                    self.assertEqual(pixels[center], pixels[center + 2])


if __name__ == "__main__":
    unittest.main()
