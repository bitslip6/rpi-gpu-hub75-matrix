"""Hardware-free binding/runtime tests. No test activates GPIO."""
from array import array
import gc
import importlib.util
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import hub75gpu as h


def display(width=16, height=16, **kwargs):
    return h.Display(h.DisplayConfig(width=width, height=height, offscreen=True, **kwargs))


class LifecycleTests(unittest.TestCase):
    def test_timeout_cleanup_does_not_take_blocking_stats_lock(self):
        d = display()
        runtime = d._runtime
        class Pending:
            def close(self, timeout):
                raise TimeoutError("simulated uninterruptible worker")
            def state(self):
                return h.DisplayState.STOPPING
            def stats(self):
                raise AssertionError("close timeout must not reacquire a busy render mutex")
        d._runtime = Pending()
        try:
            with self.assertRaises(TimeoutError):
                d.close(0)
            self.assertFalse(d.closed)
            self.assertEqual(d.state, h.DisplayState.STOPPING)
        finally:
            d._runtime = runtime
            d.close()

    def test_stop_request_is_observable_before_join(self):
        d = display()
        d.request_stop()
        self.assertEqual(d.state, h.DisplayState.STOPPING)
        d.wait(0)
        self.assertEqual(d.state, h.DisplayState.STOPPED)
        d.close()
        self.assertEqual(d.state, h.DisplayState.CLOSED)

    def test_wait_before_start_and_immutable_config(self):
        d = display()
        d.wait(0)
        self.assertEqual(d.state, h.DisplayState.CREATED)
        with self.assertRaises(AttributeError):
            d.config = h.DisplayConfig()
        d.start().close()
        self.assertEqual(d.state, h.DisplayState.CLOSED)

    def test_import_does_not_replace_signals(self):
        original = signal.getsignal(signal.SIGINT)
        import importlib
        importlib.reload(h)
        self.assertIs(signal.getsignal(signal.SIGINT), original)

    def test_never_started_close(self):
        d = display()
        d.close()
        d.close()
        self.assertTrue(d.closed)
        with self.assertRaises(h.StateError):
            d.start()

    def test_stop_before_first_frame(self):
        d = display().start()
        d.request_stop()
        d.wait(1)
        d.close(1)

    def test_start_one_shot(self):
        with display() as d:
            with self.assertRaises(h.StateError):
                d.start()
        with self.assertRaises(h.StateError):
            d.snapshot()

    def test_cancel_and_submit(self):
        with display() as d:
            with self.assertRaisesRegex(RuntimeError, "application"):
                with d.frame() as f:
                    f.clear((255, 0, 0))
                    raise RuntimeError("application")
            self.assertEqual(d.stats()["submitted"], 0)
            with d.frame() as f:
                f.clear((1, 2, 3))
            self.assertEqual(d.snapshot(), bytes((1, 2, 3, 255)) * 256)
            self.assertEqual(d.stats()["submitted"], 1)
            with self.assertRaises(h.StateError):
                f.clear()
            with self.assertRaises(h.StateError):
                f.__enter__()

    def test_nested_frame_and_double_finish(self):
        with display() as d:
            with d.frame() as frame:
                with self.assertRaises(h.BusyError):
                    with d.frame():
                        pass
                frame.submit()
                with self.assertRaises(h.StateError):
                    frame.submit()

    def test_wrong_thread(self):
        failures = []
        with display() as d:
            with d.frame() as frame:
                def wrong():
                    for op in (lambda: frame.clear(), lambda: d._runtime.submit(),
                               lambda: d._runtime.cancel()):
                        try:
                            op()
                        except h.StateError:
                            failures.append(True)
                t = threading.Thread(target=wrong)
                t.start()
                t.join(2)
                self.assertFalse(t.is_alive())
        self.assertEqual(len(failures), 3)

    def test_close_revokes_frame(self):
        d = display().start()
        frame = d.frame().__enter__()
        d.close(1)
        with self.assertRaises(h.StateError):
            frame.submit()
        frame.__exit__(RuntimeError, RuntimeError(), None)

    def test_concurrent_close_and_native_wait(self):
        d = display().start()
        errors = []
        def wait():
            try:
                d.wait(2)
            except Exception as exc:
                errors.append(exc)
        t = threading.Thread(target=wait)
        t.start()
        time.sleep(0.01)
        d.close(2)
        t.join(3)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])

    def test_sequential_sessions_and_descriptors(self):
        gc.collect()
        before = len(os.listdir("/proc/self/fd"))
        threads = len(os.listdir("/proc/self/task"))
        for size in (8, 64, 32, 128, 16) * 6:
            with display(size, size) as d:
                with d.frame() as frame:
                    frame.clear((size, 22, 33))
                self.assertEqual(d.snapshot()[:4], bytes((size, 22, 33, 255)))
        gc.collect()
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        self.assertEqual(len(os.listdir("/proc/self/task")), threads)

    def test_frame_gc_cancels(self):
        with display() as d:
            frame = d.frame().__enter__()
            del frame
            gc.collect()
            with d.frame() as f:
                f.clear()


class ValidationTests(unittest.TestCase):
    def test_config_ranges(self):
        cases = [dict(width=0), dict(height=-1), dict(width=2**64), dict(width=4097),
                 dict(bit_depth=8), dict(bit_depth=31), dict(bit_depth=65), dict(fps=0),
                 dict(brightness=255), dict(queue_depth=3), dict(ports=4), dict(chains=17),
                 dict(panel_order=8), dict(gamma=float("nan")), dict(dither=float("inf")),
                 dict(offscreen=1), dict(gpio_cpu=-2), dict(mapping=99)]
        for case in cases:
            with self.subTest(case=case):
                values = dict(offscreen=True)
                values.update(case)
                with self.assertRaises((ValueError, TypeError, OverflowError)):
                    h.Display(h.DisplayConfig(**values))

    def test_invalid_physical_topology_before_activation(self):
        with self.assertRaises(ValueError):
            h.Display(h.DisplayConfig(width=65))

    def test_interruption_after_acquire_returns_lease(self):
        with display() as d:
            original = h._wait
            def interrupted(call, timeout):
                original(call, timeout)
                raise KeyboardInterrupt
            h._wait = interrupted
            try:
                with self.assertRaises(KeyboardInterrupt):
                    d.frame().__enter__()
            finally:
                h._wait = original
            with d.frame(0) as frame:
                frame.clear((7, 8, 9))
            self.assertEqual(d.snapshot(), bytes((7, 8, 9, 255)) * (16 * 16))

    def test_bad_timeout(self):
        with display() as d:
            for timeout in (-1, math.nan, math.inf):
                with self.assertRaises(ValueError):
                    d.frame(timeout)

    def test_bad_colors_coords_polygons(self):
        with display() as d:
            with d.frame() as f:
                for color in ((256, 0, 0), (-1, 0, 0), (1.5, 0, 0), (1, 2), (2**100, 0, 0)):
                    with self.subTest(color=color), self.assertRaises((ValueError, TypeError, OverflowError)):
                        f.clear(color)
                for xy in ((-1, 1), (65536, 0), (16, 0), (math.nan, 0), (1.5, 0)):
                    with self.assertRaises((ValueError, TypeError, OverflowError)):
                        f.pixel(*xy, (255, 0, 0))
                for points in ([(0, 0)] * 2, [(0, 0)] * 33, [(0, 0), (math.nan, 1), (1, 0)]):
                    with self.assertRaises(ValueError):
                        f.polygon(points, (255, 0, 0))

    def test_wrong_resources_and_properties(self):
        with self.assertRaises((TypeError, ValueError)):
            h.Scene3D().add(h.Camera())
        t = h.Transform()
        for value in ((0, 1, 1), (math.nan, 1, 1), (1, 1), (1e100, 1, 1), (1e-30, 1, 1), (1e5, 1, 1)):
            with self.assertRaises(ValueError):
                t.scale = value
        t.close()
        with self.assertRaises(h.StateError):
            t.position = (0, 0, 0)


class ImageTests(unittest.TestCase):
    def test_rgba_round_trip(self):
        pixels = bytes((i % 256 for i in range(16 * 16 * 4)))
        # Force opaque alpha so source-over is a byte-identical upload.
        pixels = bytearray(pixels)
        pixels[3::4] = bytes([255]) * 256
        im = h.Image(16, 16, pixels)
        self.assertEqual(im.tobytes(), pixels)
        with display() as d:
            with d.frame() as f:
                f.blit(im)
            self.assertEqual(d.snapshot(), pixels)

    def test_native_image_transforms(self):
        raw = bytes((1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255))
        image = h.Image(2, 2, raw)
        self.assertEqual(image.transformed(h.Mapping.NONE).tobytes(), raw)
        self.assertEqual(image.transformed(h.Mapping.FLIP).tobytes()[::4], bytes((3, 4, 1, 2)))
        self.assertEqual(image.transformed(h.Mapping.MIRROR).tobytes()[::4], bytes((2, 1, 4, 3)))
        self.assertEqual(image.transformed(h.Mapping.MIRROR_FLIP).tobytes()[::4], bytes((4, 3, 2, 1)))
        with self.assertRaises(ValueError):
            image.transformed(h.Mapping.U)

    def test_full_frame_upload_preserves_transparent_bytes(self):
        raw = bytes((i % 256 for i in range(16 * 16 * 4)))
        with display() as d:
            with d.frame() as f:
                f.upload(raw)
            self.assertEqual(d.snapshot(), raw)

    def test_rgb_and_padded_rows(self):
        im = h.Image(2, 2)
        im.upload(bytes((1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12)), format="RGB", row_stride=8)
        self.assertEqual(im.tobytes(), bytes((1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255)))

    def test_raw_blit_copies_input_synchronously(self):
        pixels = bytearray((42, 43, 44, 255) * 256)
        with display() as d:
            with d.frame() as f:
                f.blit(pixels, width=16, height=16)
                pixels[:] = bytes(len(pixels))
            self.assertEqual(d.snapshot(), bytes((42, 43, 44, 255)) * 256)

    def test_alpha(self):
        im = h.Image(1, 1, bytes((200, 40, 80, 128)))
        with display() as d:
            with d.frame() as f:
                f.clear((0, 0, 0, 0)).blit(im)
            self.assertEqual(d.snapshot()[:4], bytes((200, 40, 80, 128)))
            with d.frame() as f:
                f.clear((0, 0, 0)).blit(im)
            self.assertEqual(d.snapshot()[:4], bytes((100, 20, 40, 255)))

    def test_views_pin_allocation_and_block_native_access(self):
        im = h.Image(2, 2)
        view = im.view()
        child = view[4:]
        view.release()
        with self.assertRaises(BufferError):
            im.close()
        with self.assertRaises(h.BusyError):
            im.upload(bytes(16))
        with self.assertRaises(h.BusyError):
            im.tobytes()
        with display() as d:
            with d.frame() as f:
                with self.assertRaises(h.BusyError):
                    f.blit(im)
        child[:] = bytes((1, 2, 3, 4)) * 3
        child.release()
        self.assertEqual(im.tobytes()[4:], bytes((1, 2, 3, 4)) * 3)
        im.close()
        im.close()

    def test_bad_buffers(self):
        im = h.Image(2, 2)
        for data in (bytes(15), array("f", [1, 2, 3, 4]), memoryview(bytes(32))[::2], memoryview(bytes(16))[::-1]):
            with self.subTest(data=type(data)), self.assertRaises(ValueError):
                im.upload(data)
        with self.assertRaises(ValueError):
            im.upload(bytes(16), row_stride=1)
        with self.assertRaises(ValueError):
            im.upload(bytes(16), format="BGRA")

    def test_native_retention_after_wrapper_close(self):
        im = h.Image(16, 16, bytes((200, 201, 202, 255)) * 256)
        batch = h.DrawList().blit(im)
        im.close()
        del im
        gc.collect()
        with display() as d:
            with d.frame() as f:
                f.draw(batch)
            self.assertEqual(d.snapshot()[:4], bytes((200, 201, 202, 255)))

    def test_hwc_buffer(self):
        buf = memoryview(bytes((1, 2, 3, 255)) * 4).cast("B", (2, 2, 4))
        im = h.Image(2, 2, buf)
        self.assertEqual(im.tobytes(), buf.tobytes())
        with self.assertRaises(ValueError):
            h.Image(1, 4, buf)


class RetainedTests(unittest.TestCase):
    def test_autonomous_and_single_producer(self):
        batch = h.DrawList().clear((11, 22, 33))
        with display() as d:
            p = d.play(batch)
            batch.close()  # C worker retains the list, not a Python callback.
            time.sleep(0.08)
            self.assertGreaterEqual(d.stats()["submitted"], 2)
            with self.assertRaises(h.BusyError):
                with d.frame():
                    pass
            with self.assertRaises(h.BusyError):
                d.play(h.DrawList())
            p.stop(1)
            self.assertEqual(d.snapshot()[:4], bytes((11, 22, 33, 255)))
            with d.frame() as f:
                f.clear()
            p2 = d.play(h.DrawList().clear())
            with self.assertRaises(h.StateError):
                p.stop()
            p2.stop(1)

    def test_gil_released_during_single_native_wait(self):
        with display() as d:
            p = d.play(h.DrawList().clear())
            counter = [0]
            stop = threading.Event()
            ready = threading.Event()
            def work():
                ready.set()
                while not stop.is_set():
                    counter[0] += 1
            t = threading.Thread(target=work)
            t.start()
            ready.wait(1)
            before = counter[0]
            with self.assertRaises(TimeoutError):
                d._runtime.wait_content(150)
            after = counter[0]
            stop.set()
            t.join(2)
            self.assertGreater(after - before, 100)
            p.stop(1)

    def test_mutation_at_frame_boundaries(self):
        batch = h.DrawList().clear((1, 2, 3))
        with display() as d:
            p = d.play(batch)
            batch.clear((4, 5, 6))
            time.sleep(0.06)
            p.stop(1)
            self.assertEqual(d.snapshot()[:4], bytes((4, 5, 6, 255)))

    def test_primitives_and_scene_retention(self):
        for name in ("cube", "tetrahedron", "octahedron", "pyramid", "cylinder", "sphere", "torus", "plane"):
            with self.subTest(name=name):
                scene = h.Scene3D(ambient=(1, 1, 1))
                mesh = h.Mesh.primitive(name)
                instance = scene.add(mesh, angular_velocity=(0, 0.5, 0))
                camera = h.Camera(position=(3, 2, 5))
                light = scene.directional_light(shadows=name == "cube")
                light.set(direction=(-2, -3, -4), intensity=0.7)
                batch = h.DrawList().clear().render(scene, camera)
                mesh.close()
                instance.transform.close()
                scene.close()
                camera.close()
                with display(64, 64) as d:
                    with d.frame() as f:
                        f.draw(batch, time=0.5)
                    self.assertGreater(sum(d.snapshot()[0::4]), 0)

    def test_bulk_geometry_and_nonblocking_submission(self):
        batch = h.DrawList().clear().lines_buffer(array("f", [0, 1, 15, 1]), (12, 34, 56))
        with display() as d:
            self.assertTrue(d.try_submit(batch))
            self.assertEqual(d.snapshot()[(16+5)*4:(16+6)*4], bytes((12, 34, 56, 255)))
            batch.points_buffer(array("f", [2, 3]), (90, 80, 70), replace=True)
            with d.frame() as f:
                f.draw(batch)
            self.assertEqual(d.snapshot()[(3*16+2)*4:(3*16+3)*4], bytes((90, 80, 70, 255)))
            batch.reset().clear((1, 2, 3))
            with d.frame() as f:
                f.draw(batch)
            self.assertEqual(d.snapshot()[:4], bytes((1, 2, 3, 255)))
            p = d.play(batch)
            with self.assertRaises(h.BusyError):
                d.try_submit(batch)
            p.stop(1)
        for data in (array("f", [math.nan, 0]), array("i", [0, 1]), array("f", [0])):
            with self.assertRaises(ValueError):
                h.DrawList().points_buffer(data, (1, 2, 3))

    def test_camera_matrix_and_degenerate_view(self):
        camera = h.Camera(aspect=1)
        matrix = camera.project(h.Transform())
        self.assertEqual(len(matrix), 16)
        self.assertTrue(all(math.isfinite(v) for v in matrix))
        camera.target = camera.position
        with self.assertRaises(ValueError):
            camera.project(h.Transform())
        scene = h.Scene3D()
        scene.add(h.Mesh.cube())
        with display() as d:
            with self.assertRaises(ValueError):
                with d.frame() as f:
                    f.render(scene, camera)

    def test_bulk_transforms(self):
        scene = h.Scene3D()
        instance = scene.add(h.Mesh.cube())
        scene.set_transforms(array("f", [1, 2, 3, 0, .5, 0, 2, 2, 2]))
        self.assertEqual(instance.transform.position, (1, 2, 3))
        self.assertEqual(instance.transform.scale, (2, 2, 2))
        with self.assertRaises(ValueError):
            scene.set_transforms(array("f", [1, 2, 3, 0, .5, 0, 0, 2, 2]))
        with self.assertRaises(ValueError):
            scene.set_transforms(array("f", [1, 2, 3]))

    def test_pause_resume_and_stop_while_paused(self):
        with display(fps=120) as d:
            p = d.play(h.DrawList().clear())
            p.ready(1)
            p.pause()
            time.sleep(.03)  # allow any in-flight boundary to complete
            before = d.stats()["submitted"]
            time.sleep(.04)
            self.assertEqual(d.stats()["submitted"], before)
            self.assertTrue(d.stats()["paused"])
            p.resume()
            time.sleep(.03)
            self.assertGreater(d.stats()["submitted"], before)
            p.pause()
            p.stop(1)

    def test_bounded_camera_lights_and_geometry(self):
        camera = h.Camera()
        for name, value in (("fov_y", 1e-30), ("aspect", 1e-30), ("z_near", 1e-30)):
            with self.assertRaises(ValueError):
                setattr(camera, name, value)
        light = h.Scene3D().directional_light()
        for kwargs in ({"direction": (0, 0, 0)}, {"color": (2, 0, 0)}, {"shadow_zoom": 1e-30}):
            with self.assertRaises(ValueError):
                light.set(**kwargs)
        for data in (array("f", [-1, 0]), array("f", [.5, 0]), array("f", [65536, 0])):
            with self.assertRaises(ValueError):
                h.DrawList().points_buffer(data, (1, 2, 3))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional NumPy not installed")
    def test_numpy_uploads_and_descendant_views_pin_storage(self):
        import numpy as np
        source = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
        image = h.Image(2, 2, source)
        mapped = image.numpy()
        mapped[0, 0] = (1, 2, 3, 255)
        descendant = mapped[0]
        del mapped
        gc.collect()
        with self.assertRaises(BufferError):
            image.close()
        del descendant
        gc.collect()
        self.assertEqual(image.tobytes()[:4], bytes((1, 2, 3, 255)))
        image.close()
        with display(2, 2) as d:
            with d.frame() as frame:
                frame.upload(source)
            self.assertEqual(d.snapshot(), source.tobytes())
        with self.assertRaises((ValueError, BufferError)):
            h.Image(2, 2, source[:, ::-1])
        with self.assertRaises((TypeError, ValueError)):
            h.Image(2, 2, source.astype(np.uint16))
        h.Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
               faces=np.array([[0, 1, 2]], dtype=np.uint32)).close()

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional Pillow not installed")
    def test_pillow_adapter_is_an_independent_copy(self):
        image = h.Image(2, 2, bytes((13, 27, 91, 255)) * 4)
        copy = image.pillow()
        image.upload(bytes((1, 2, 3, 4)) * 4)
        image.close()
        self.assertEqual(copy.getpixel((0, 0)), (13, 27, 91, 255))
        copy.close()

    def test_misaligned_native_buffers_rejected(self):
        data = memoryview(bytearray(13))[1:].cast("f")
        with self.assertRaises(ValueError):
            h.Mesh(data)

    def test_embedded_nul_image_path_rejected(self):
        with self.assertRaises(ValueError):
            h.Image.load("/tmp/hub75-nonexistent\0ignored")

    def test_mesh_buffer_validation(self):
        vertices = array("f", [-1, -1, 0, 1, -1, 0, 0, 1, 0])
        mesh = h.Mesh(vertices, faces=array("I", [0, 1, 2]))
        mesh.set_color((200, 30, 50))
        with self.assertRaises(ValueError):
            h.Mesh(vertices, faces=array("I", [0, 1, 3]))
        with self.assertRaises(ValueError):
            h.Mesh(array("d", vertices), faces=array("I", [0, 1, 2]))
        with self.assertRaises(ValueError):
            h.Mesh(array("f", [float("nan"), 0, 0]))
        with self.assertRaises(ValueError):
            h.Mesh.primitive("sphere", subdivisions=30)

    def test_cpu_unsupported_media(self):
        if h.capabilities()["shader"]:
            self.skipTest("GPU build")
        with display() as d:
            with self.assertRaises(h.UnsupportedError):
                d.play_shader("missing.glsl")
            with self.assertRaises(h.UnsupportedError):
                d.play_video("missing.mp4")


class ReentrantArgumentTests(unittest.TestCase):
    def test_conversion_closes_draw_list(self):
        batch = h.DrawList()
        class Number:
            def __float__(self):
                batch.close()
                return 0.0
        with self.assertRaises(h.StateError):
            batch.line(Number(), 0, 1, 1, (1, 2, 3))

    @unittest.skipIf(sys.version_info < (3, 12), "Python-level buffer exporters need 3.12+")
    def test_buffer_callback_cannot_leave_a_dangling_native_pointer(self):
        image = h.Image(2, 2)
        scene = h.Scene3D()
        batch = h.DrawList()
        for resource, data, upload in (
            (image, bytes(16), image.upload),
            (scene, array("f", [1]*9), scene.set_transforms),
            (batch, array("f", [1]*4), lambda data: batch.lines_buffer(data, (1, 2, 3))),
        ):
            class Exporter:
                def __buffer__(self, flags):
                    resource.close()
                    return memoryview(data)
            with self.assertRaises(h.StateError):
                upload(Exporter())


class SubprocessTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "requires compiler for interrupted-start shim")
    def test_interrupted_start_cleans_up_before_enter_fails(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "slow.c"
            library = Path(td) / "slow.so"
            source.write_text('''
#include <dlfcn.h>
#include <unistd.h>
#include <stdlib.h>
int h75_runtime_start(void *r, void *e) {
    void *handle=dlopen(getenv("H75_START_SHIM_LIBRARY"),RTLD_NOW|RTLD_NOLOAD);
    int (*next)(void*,void*)=handle?dlsym(handle,"h75_runtime_start"):NULL;
    if(!next)return 1;
    usleep(100000);int result=next(r,e);dlclose(handle);return result;
}
''')
            subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library), "-ldl"], check=True)
            code = '''
import hub75gpu as h, os, signal, threading, time
original=signal.getsignal(signal.SIGINT)
with open('/proc/self/maps') as maps:
    os.environ['H75_START_SHIM_LIBRARY']=next(line.split(None,5)[-1].strip() for line in maps if '/librpihub75' in line)
d=h.Display(h.DisplayConfig(offscreen=True))
t=threading.Thread(target=lambda: (time.sleep(.02),os.kill(os.getpid(),signal.SIGINT)))
t.start()
try:
    d.start()
except KeyboardInterrupt:
    pass
else:
    raise AssertionError("startup did not deliver KeyboardInterrupt")
t.join()
assert d.closed and d.state==h.DisplayState.CLOSED
assert signal.getsignal(signal.SIGINT) is original
'''
            subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                           env=dict(os.environ, LD_PRELOAD=(os.environ.get("LD_PRELOAD", "")+":"+str(library)).strip(":")),
                           check=True, timeout=10)

    def run_script(self, source, timeout=10):
        return subprocess.run([sys.executable, "-X", "faulthandler", "-c", source],
                              text=True, capture_output=True, timeout=timeout)

    def test_native_heap_types_participate_in_module_gc(self):
        result = self.run_script('''
import hub75gpu as h, gc, sys, weakref
native=h._native
native.cycle=native.Runtime({'offscreen':True})
native.cycle.start()
native.image_cycle=native.image(2,2)
assert gc.is_tracked(native.cycle) and gc.is_tracked(native.image_cycle)
ref=weakref.ref(native)
sys.modules.pop('hub75gpu._native')
del h._native, native
gc.collect()
assert ref() is None, 'module/type/instance cycle leaked a native session'
print('collected')
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("collected", result.stdout)

    def test_bad_config_cannot_exit_interpreter(self):
        result = self.run_script("import hub75gpu as h\ntry: h.Display(h.DisplayConfig(width=0))\nexcept ValueError: print('alive')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("alive", result.stdout)

    def test_keyboard_interrupt_closes_workers(self):
        source = '''
import hub75gpu as h, os, signal, threading, time
try:
    with h.Display(h.DisplayConfig(offscreen=True)) as d:
        p=d.play(h.DrawList().clear())
        def interrupt():
            time.sleep(.1)
            os.kill(os.getpid(), signal.SIGINT)
        t=threading.Thread(target=interrupt); t.start()
        p.wait()
except KeyboardInterrupt:
    t.join()
    assert d.closed
    print('clean interrupt')
'''
        result = self.run_script(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean interrupt", result.stdout)

    def test_subinterpreter_rejected(self):
        result = self.run_script('''
import hub75gpu as h
from pathlib import Path
try:
    import _interpreters as s
    modern=True
except ImportError:
    import _xxsubinterpreters as s
    modern=False
interpreter=s.create()
code='import sys;sys.path.insert(0,'+repr(str(Path(h.__file__).parent.parent))+');import hub75gpu'
try:
    if modern:
        error=s.run_string(interpreter, code)
        assert error is not None and error.type.__name__=='ImportError', error
        assert 'subinterpreters' in error.msg
    else:
        try:
            s.run_string(interpreter, code)
        except s.RunFailedError as error:
            assert 'subinterpreter' in str(error)
        else:
            raise AssertionError('subinterpreter unexpectedly supported')
    print('rejected')
finally:
    s.destroy(interpreter)
''')
        if "No module named '_xxsubinterpreters'" in result.stderr:
            self.skipTest("no subinterpreter test module")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rejected", result.stdout)


if __name__ == "__main__":
    unittest.main()
