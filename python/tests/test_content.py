"""Native text/video tests; EGL tests require explicit HUB75_TEST_GPU=1.

Fixtures are generated in temporary directories: no bundled font/media licenses
or external NumPy/Pillow/ffmpeg command-line dependencies are needed.
"""
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
import zlib

import hub75gpu as h


def png_gray(path, width=8, height=8):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    pixels = b"".join(b"\0" + bytes(255 if 1 <= x < width - 1 and 1 <= y < height - 1 else 0
                                   for x in range(width)) for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def font_fixture(path):
    png_gray(path / "A.png")
    (path / "metrics.csv").write_text("char,advance_width,bearing_x,bearing_y,width,height,path\nA,8,0,0,8,8,A.png\n")


def display():
    return h.Display(h.DisplayConfig(width=32, height=16, offscreen=True, fps=120))


@unittest.skipUnless(h.capabilities().get("text"), "build has no PNG/SDF font loader")
class TextTests(unittest.TestCase):
    def test_retained_font_and_cached_text_colors(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            font_fixture(path)
            font = h.Font(path)
            text = h.Text(font, "AA", color=(220, 30, 75), position=(2, 1))
            font.close()
            self.assertEqual(text.position, (2, 1))
            self.assertEqual(text.speed, 0)
            self.assertGreater(text.dimensions[0], 8)
            batch = h.DrawList().clear().text(text)
            with display() as d:
                with d.frame() as frame:
                    frame.draw(batch)
                pixels = d.snapshot()
                lit = [tuple(pixels[i:i+4]) for i in range(0, len(pixels), 4) if pixels[i]]
                self.assertTrue(lit)
                self.assertTrue(all(r > b > g for r, g, b, a in lit))
                # Rendering must not overwrite the allocation dimensions used
                # for subsequent cached blits and scrolling wrap timing.
                with d.frame() as frame:
                    frame.draw(batch)
                self.assertEqual(pixels, d.snapshot())
                text.set_text("A")
                text.color = (10, 240, 40, 255)
                with d.frame() as frame:
                    frame.draw(batch)
                self.assertNotEqual(pixels, d.snapshot())

    def test_native_scroll_sine_and_live_updates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            font_fixture(path)
            text = h.Text(h.Font(path), "AAAA", speed=24)
            text.sine = (2, 2, 1)
            batch = h.DrawList().clear().text(text)
            with display() as d:
                playback = d.play(batch)
                playback.ready(2)
                time.sleep(.04)
                before = d.snapshot()
                text.set_text("AA")
                text.weight = 1.1
                time.sleep(.08)
                playback.stop(2)
                self.assertNotEqual(before, d.snapshot())
                self.assertGreater(d.stats()["submitted"], 2)

    def test_extreme_metrics_layout_fails_without_integer_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            font_fixture(path)
            metrics = path / "metrics.csv"
            metrics.write_text(metrics.read_text().replace("A,8,", "A,4096,"))
            font = h.Font(path, size=256)
            text = h.Text(font, "A" * 65535)
            d = display().start()
            p = d.play(h.DrawList().text(text))
            with self.assertRaises(MemoryError):
                p.wait(2)
            self.assertIsInstance(p.error, MemoryError)
            self.assertIsInstance(d.error, MemoryError)
            with self.assertRaises(MemoryError):
                d.close()
            self.assertTrue(d.closed)
            self.assertIsInstance(d.error, MemoryError)

    def test_invalid_font_and_text(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            with self.assertRaises(OSError):
                h.Font(path / "missing")
            font_fixture(path)
            metrics = path / "metrics.csv"
            original = metrics.read_text()
            metrics.write_text(original.replace("A,8,", "A,nan,"))
            with self.assertRaises(OSError):
                h.Font(path, size=16)
            metrics.write_text(original)
            font = h.Font(path)
            for text in ("\0", "snowman \u2603"):
                with self.assertRaises(ValueError):
                    h.Text(font, text)


@unittest.skipUnless(h.capabilities()["video"], "CPU-only installation")
class VideoTests(unittest.TestCase):
    def fixture(self, path):
        # image2/PPM is a real FFmpeg video stream with one frame and EOF. This
        # checks drain/EOF without an external video encoder or ffmpeg binary.
        path.write_bytes(b"P6\n8 8\n255\n" + bytes((120, 40, 200)) * 64)

    def test_local_decode_rgba_eof_and_mode_transition(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "frame.ppm"
            self.fixture(path)
            with display() as d:
                p = d.play_video(path)
                p.ready(2)
                p.wait(3)
                self.assertFalse(p.playing)
                pixels = d.snapshot()
                self.assertEqual(pixels[:4], bytes((120, 40, 200, 255)))
                self.assertEqual(d.stats()["submitted"], 1)
                with d.frame() as f:
                    f.clear((1, 2, 3))
                self.assertEqual(d.snapshot()[:4], bytes((1, 2, 3, 255)))

    def test_preserve_aspect_centers_opaque_black_bars(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "frame.ppm"
            self.fixture(path)
            for preserve in (False, True):
                config = h.DisplayConfig(width=32, height=16, offscreen=True,
                                         video_preserve_aspect=preserve)
                with h.Display(config) as d:
                    p = d.play_video(path)
                    p.ready(2)
                    p.wait(3)
                    pixels = d.snapshot()
                    for y in range(16):
                        for x in range(32):
                            expected = (0, 0, 0, 255) if preserve and (x < 8 or x >= 24) else (120, 40, 200, 255)
                            offset = (y * 32 + x) * 4
                            self.assertEqual(pixels[offset:offset + 4], bytes(expected))

    def test_loop_and_stop(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "frame.ppm"
            self.fixture(path)
            with display() as d:
                p = d.play_video(path, loop=True)
                p.ready(2)
                time.sleep(.16)
                p.stop(2)
                self.assertGreaterEqual(d.stats()["submitted"], 2)

    def test_invalid_file_and_network_are_exceptions(self):
        for path in ("/nonexistent/hub75.mp4", "https://example.invalid/video.mp4"):
            d = display().start()
            try:
                p = d.play_video(path)
                with self.assertRaises(OSError):
                    p.wait(3)
            finally:
                try:
                    d.close(3)
                except OSError:
                    pass
            self.assertTrue(d.closed)


@unittest.skipUnless(h.capabilities()["shader"] and os.environ.get("HUB75_TEST_GPU") == "1",
                     "EGL tests require GPU build and explicit HUB75_TEST_GPU=1 (no GPIO)")
class ShaderTests(unittest.TestCase):
    def test_shader_and_cpu_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "solid.glsl"
            path.write_text("void mainImage(out vec4 c,in vec2 p){c=vec4(1.,0.,0.,1.);}")
            overlay = h.DrawList().pixel(0, 0, (0, 255, 0))
            with display() as d:
                p = d.play_shader(path, overlay=overlay)
                p.ready(10)
                time.sleep(.05)
                p.stop(3)
                pixels = d.snapshot()
                self.assertEqual(pixels[:8], bytes((0, 255, 0, 255, 255, 0, 0, 255)))

    def test_missing_and_fourth_channel_are_safe(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "channels.glsl"
            path.write_text("void mainImage(out vec4 c,in vec2 p){c=texture(iChannel3,vec2(.5));}")
            with display() as d:
                p = d.play_shader(path)
                p.ready(10)
                time.sleep(.03)
                p.stop(2)
                self.assertEqual(d.snapshot()[:4], bytes((0, 0, 0, 255)))
                png_gray(path.with_suffix(".channel3"))
                p = d.play_shader(path)
                p.ready(10)
                time.sleep(.03)
                p.stop(2)
                self.assertEqual(d.snapshot()[:4], bytes((255, 255, 255, 255)))

    def test_egl_exclusivity_is_recoverable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "solid.glsl"
            path.write_text("void mainImage(out vec4 c,in vec2 p){c=vec4(1.);}")
            with display() as first:
                p = first.play_shader(path)
                p.ready(10)
                second = display().start()
                try:
                    q = second.play_shader(path)
                    with self.assertRaises(h.BusyError):
                        q.wait(10)
                finally:
                    try:
                        second.close(3)
                    except h.BusyError:
                        pass
                self.assertTrue(p.playing)
                p.stop(3)

    def test_bad_shader_never_exits_process(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.glsl"
            path.write_text("this is not GLSL")
            d = display().start()
            try:
                p = d.play_shader(path)
                with self.assertRaises((ValueError, OSError)):
                    p.wait(10)
            finally:
                try:
                    d.close(3)
                except (ValueError, OSError):
                    pass
            self.assertTrue(d.closed)
