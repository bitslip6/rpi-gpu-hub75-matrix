"""Native HUB75 rendering. Import and construction never activate GPIO.

Use ``with Display(config)`` (or explicit start/close), and ``with display.frame()``.
Rendering, retained animation, shader/video playback, pacing and output run in C.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math
import os
import threading
import time
from typing import Any, Iterable, Sequence

from . import _native
from ._native import Error, StateError, BusyError, UnsupportedError

__version__ = "0.3.0"
Color = tuple[int, int, int] | tuple[int, int, int, int]


class DrawMode(IntEnum):
    WIRE = 0
    FILLED = 1


class PanelOrder(IntEnum):
    RGB = 0
    RBG = 1
    GRB = 2
    GBR = 3
    BRG = 4
    BGR = 5


class Mapping(IntEnum):
    NONE = 0
    FLIP = 1
    MIRROR = 2
    MIRROR_FLIP = 3
    U = 4


class ToneMapping(IntEnum):
    NONE = 0
    ACES = 1
    REINHARD = 2
    HABLE = 3
    SATURATION = 4
    SIGMOID = 5


class GradientDirection(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2
    RADIAL = 3


class Easing(IntEnum):
    LINEAR = 0
    IN_QUAD = 1
    OUT_QUAD = 2
    IN_OUT_QUAD = 3


class DisplayState(IntEnum):
    CREATED = 0
    RUNNING = 1
    STOPPING = 2
    STOPPED = 3
    CLOSED = 4


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 64
    height: int = 64
    panel_width: int = 64
    panel_height: int = 64
    ports: int = 1
    chains: int = 1
    bit_depth: int = 64
    fps: int = 60
    brightness: int = 200
    queue_depth: int = 4
    panel_order: PanelOrder = PanelOrder.RGB
    mapping: Mapping = Mapping.NONE
    tone_mapping: ToneMapping = ToneMapping.NONE
    gamma: float = 1.0
    dither: float = 0.0
    tone_level: float = 0.0
    offscreen: bool = False
    quant_dither: bool = False
    jitter: bool = True
    video_preserve_aspect: bool = False
    gpio_cpu: int = -1


def capabilities() -> dict[str, Any]:
    """Compiled features, not a claim that a GPU/device is currently available."""
    return _native.capabilities()


def _rgba(color: Color) -> tuple[int, ...]:
    c = tuple(color)
    if len(c) == 3:
        c += (255,)
    return c


def _timeout(timeout: float | None) -> float | None:
    if timeout is not None:
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be nonnegative finite seconds, or None")
    return timeout


def _wait(call: Any, timeout: float | None) -> None:
    timeout = _timeout(timeout)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        ms = 50 if remaining is None else math.ceil(min(0.05, remaining) * 1000)
        try:
            call(ms)
            return
        except TimeoutError:
            if deadline is not None and time.monotonic() >= deadline:
                raise


def _property(name: str, cast: Any = None) -> property:
    def get(self: _Resource) -> Any:
        value = self._handle.get(name)
        return value if cast is None else cast(value)

    def set(self: _Resource, value: Any) -> None:
        if not isinstance(value, (tuple, list)):
            value = (value,)
        self._handle.set(name, value)
    return property(get, set)


class _Resource:
    _handle: _native.Resource

    @property
    def closed(self) -> bool:
        return getattr(self, "_closed", False)

    def close(self) -> None:
        """Close this wrapper. Native dependents keep their retained data alive."""
        self._handle.close()
        self._closed = True

    def __enter__(self):
        if self.closed:
            raise StateError("resource is closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Image(_Resource):
    """C-owned straight-alpha RGBA8 storage (top-down rows).

    ``view()`` exports a writable flat byte buffer. Release all views/NumPy slices
    before upload, drawing, playback or close; native access rejects live exports.
    Upload is synchronous: inputs may be reused when upload returns. Do not mutate
    the input from another thread while upload is executing.
    """
    def __init__(self, width: int, height: int, data: Any = None, *,
                 format: str = "RGBA", row_stride: int | None = None):
        self._handle = _native.image(width, height)
        if data is not None:
            self.upload(data, format=format, row_stride=row_stride)

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> Image:
        obj = cls.__new__(cls)
        obj._handle = _native.load_image(os.fspath(path))
        return obj

    @property
    def dimensions(self) -> tuple[int, int]:
        w, h = self._handle.get("dimensions")
        return int(w), int(h)

    @property
    def width(self) -> int:
        return self.dimensions[0]

    @property
    def height(self) -> int:
        return self.dimensions[1]

    def upload(self, data: Any, *, format: str = "RGBA", row_stride: int | None = None) -> None:
        if format not in ("RGB", "RGBA"):
            raise ValueError("format must be RGB or RGBA")
        w, h = self.dimensions
        channels = len(format)
        pitch = w * channels if row_stride is None else row_stride
        self._handle.upload(data, w, h, pitch, channels)

    def view(self) -> memoryview:
        return memoryview(self._handle)

    def tobytes(self) -> bytes:
        return self._handle.tobytes()

    def transformed(self, mapping: Mapping) -> Image:
        """Return an independent native flip/mirror copy; no Python pixel loop."""
        obj = type(self).__new__(type(self))
        obj._handle = _native.transform_image(self._handle, mapping)
        return obj

    def numpy(self):
        """Optional NumPy view. Release it and all slices before native use."""
        import numpy as np
        return np.frombuffer(self.view(), dtype=np.uint8).reshape(self.height, self.width, 4)

    def pillow(self):
        """Optional Pillow copy, independent of the native allocation."""
        from PIL import Image as PILImage
        return PILImage.frombytes("RGBA", self.dimensions, self.tobytes())


class Transform(_Resource):
    position = _property("position")
    rotation = _property("rotation")
    scale = _property("scale")

    def __init__(self, *, position: Sequence[float] = (0, 0, 0),
                 rotation: Sequence[float] = (0, 0, 0), scale: Sequence[float] = (1, 1, 1)):
        self._handle = _native.resource(2)
        self.position = tuple(position)
        self.rotation = tuple(rotation)
        self.scale = tuple(scale)


class Camera(_Resource):
    position = _property("position")
    target = _property("target")
    up = _property("up")
    fov_y = _property("fov_y")
    aspect = _property("aspect")
    z_near = _property("z_near")
    z_far = _property("z_far")

    def __init__(self, *, position: Sequence[float] = (0, 0, 5),
                 target: Sequence[float] = (0, 0, 0), up: Sequence[float] = (0, 1, 0),
                 fov_y: float = math.pi / 3, aspect: float = 1,
                 z_near: float = 0.1, z_far: float = 100):
        self._handle = _native.resource(1)
        self.position = tuple(position)
        self.target = tuple(target)
        self.up = tuple(up)
        self.fov_y = fov_y
        self.aspect = aspect
        # Permit any valid near/far pair, including pairs beyond native defaults.
        if z_far > 0.1:
            self.z_far = z_far
            self.z_near = z_near
        else:
            self.z_near = z_near
            self.z_far = z_far


    def project(self, transform: Transform) -> tuple[float, ...]:
        """Return the native column-major float32 model-view-projection matrix."""
        return _native.project(self._handle, transform._handle)


class Mesh(_Resource):
    draw_mode = _property("draw_mode", lambda v: DrawMode(int(v)))
    cull_backface = _property("cull_backface", bool)
    shadow_enabled = _property("shadow_enabled", bool)
    specular_strength = _property("specular_strength")
    specular_shininess = _property("specular_shininess")

    def __init__(self, vertices: Any, *, edges: Any = None, faces: Any = None):
        """Upload native-endian float32 Nx3 vertices and uint32 Nx2/Nx3 indices.

        ``array.array('f')``/``array.array('I')`` and compatible NumPy arrays work.
        Indices, finiteness, integer limits and layout are checked before upload.
        """
        self._handle = _native.mesh(vertices, edges, faces)

    @classmethod
    def primitive(cls, name: str, *, segments: int = 16, minor_segments: int = 8,
                  subdivisions: int = 2, width_segments: int = 8, height_segments: int = 8,
                  mode: DrawMode = DrawMode.FILLED, cull_backface: bool = True) -> Mesh:
        x, y = segments, minor_segments
        if name == "sphere":
            x = subdivisions
        elif name == "plane":
            x, y = width_segments, height_segments
        obj = cls.__new__(cls)
        obj._handle = _native.primitive(name, x, y, mode, cull_backface)
        return obj

    @classmethod
    def cube(cls, **kwargs: Any) -> Mesh:
        return cls.primitive("cube", **kwargs)

    def set_color(self, color: tuple[int, int, int]) -> None:
        self._handle.set("color", color)


@dataclass(frozen=True)
class Instance:
    scene: Scene3D
    id: int
    mesh: Mesh
    transform: Transform


class Light:
    """Stable ID-based reference; never a pointer into a reallocatable array."""
    def __init__(self, scene: Scene3D, index: int):
        self.scene, self.id = scene, index

    def set(self, *, direction: Sequence[float] | None = None,
            color: Sequence[float] | None = None, intensity: float | None = None,
            shadow_enabled: bool | None = None, shadow_zoom: float | None = None) -> None:
        for name, vector in (("direction", direction), ("color", color)):
            if vector is not None:
                self.scene._handle.light_set(self.id, name, tuple(vector))
        for name, scalar in (("intensity", intensity), ("shadow_enabled", shadow_enabled),
                             ("shadow_zoom", shadow_zoom)):
            if scalar is not None:
                self.scene._handle.light_set(self.id, name, (scalar,))


class Scene3D(_Resource):
    ambient = _property("ambient")

    def __init__(self, *, ambient: Sequence[float] = (0.2, 0.2, 0.2)):
        self._handle = _native.resource(4)
        self.ambient = tuple(ambient)

    def add(self, mesh: Mesh, transform: Transform | None = None, *,
            angular_velocity: Sequence[float] = (0, 0, 0)) -> Instance:
        if transform is None:
            transform = Transform()
        index = self._handle.add(mesh._handle, transform._handle, tuple(angular_velocity))
        return Instance(self, index, mesh, transform)

    def set_transforms(self, values: Any) -> None:
        """Bulk native float32 Nx9: position, Euler rotation, nonzero scale.

        A complete validated snapshot is applied between render frames. The row
        count must equal the number of scene instances.
        """
        self._handle.transforms(values)

    def directional_light(self, direction: Sequence[float] = (-1, -1, -1), *,
                          color: Sequence[float] = (1, 1, 1), intensity: float = 1,
                          shadows: bool = False) -> Light:
        index = self._handle.light(tuple(direction), tuple(color), intensity, shadows)
        return Light(self, index)


class Font(_Resource):
    def __init__(self, directory: os.PathLike[str] | str, *, size: float = 0):
        self._handle = _native.font(os.fspath(directory), size)


class Text(_Resource):
    position = _property("position")
    direction = _property("direction")
    speed = _property("speed")
    size = _property("size")
    tracking = _property("tracking")
    angle = _property("angle")
    wrap = _property("wrap", bool)
    kerning = _property("kerning", bool)
    sine = _property("sine")
    weight = _property("weight")
    outline_width = _property("outline_width")
    color = _property("color", lambda v: tuple(map(int, v)))
    outline_color = _property("outline_color", lambda v: tuple(map(int, v)))

    def __init__(self, font: Font, text: str, *, position: Sequence[float] = (0, 0),
                 color: Color = (255, 255, 255, 255), speed: float = 0,
                 direction: Sequence[float] = (-1, 0)):
        self._handle = _native.text(font._handle, text)
        self.position = tuple(position)
        self.color = _rgba(color)
        self.direction = tuple(direction)
        self.speed = speed

    def set_text(self, text: str) -> None:
        self._handle.text(text)

    @property
    def dimensions(self) -> tuple[int, int]:
        w, h = self._handle.get("dimensions")
        return int(w), int(h)


def _command(op: int, color: Color = (0, 0, 0, 255), params: Sequence[float] = (),
             points: Sequence[Sequence[float]] = (), gradient: Any = None,
             a: _Resource | None = None, b: _Resource | None = None) -> tuple[Any, ...]:
    return (op, _rgba(color), tuple(params), tuple(points), gradient,
            None if a is None else a._handle, None if b is None else b._handle)


class _Canvas:
    def _emit(self, command: tuple[Any, ...]) -> None:
        raise NotImplementedError

    def clear(self, color: Color = (0, 0, 0, 255)):
        self._emit(_command(0, color))
        return self

    def pixel(self, x: int, y: int, color: Color):
        """Convenience/debug API; prefer image uploads or reusable batches."""
        self._emit(_command(1, color, (x, y)))
        return self

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color, *, antialias: bool = False):
        self._emit(_command(3 if antialias else 2, color, (x0, y0, x1, y1)))
        return self

    def polygon(self, points: Sequence[Sequence[float]], color: Color):
        """Fill 3..32 normalized (0..1) polygon vertices with alpha blending."""
        self._emit(_command(4, color, points=points))
        return self

    def gradient(self, points: Sequence[Sequence[float]], start: tuple[int, int, int],
                 end: tuple[int, int, int], *, direction: GradientDirection = GradientDirection.HORIZONTAL,
                 easing: Easing = Easing.LINEAR):
        self._emit(_command(5, points=points, gradient=(tuple(start), tuple(end), direction, easing)))
        return self

    def blit(self, source: Image | Any, *, x: float = 0, y: float = 0,
             width: int | None = None, height: int | None = None,
             scale: float = 1, format: str = "RGBA", row_stride: int | None = None):
        """Composite an image or synchronously upload and composite a byte buffer.

        For raw buffers, width and height describe the input; scale describes the
        destination size. Persistent Image objects avoid repeated allocation.
        """
        if not isinstance(source, Image):
            if width is None or height is None:
                raise ValueError("buffer blit requires explicit width and height")
            source = Image(width, height, source, format=format, row_stride=row_stride)
        w, h = source.dimensions
        self._emit(_command(6, params=(x, y, w * scale, h * scale), a=source))
        return self

    def render(self, scene: Scene3D, camera: Camera):
        self._emit(_command(7, a=scene, b=camera))
        return self

    def text(self, text: Text):
        self._emit(_command(8, a=text))
        return self


class DrawList(_Resource, _Canvas):
    """Reusable C-owned batch. Native references retain images/text/scenes/cameras.

    Appending and property updates during playback are serialized at whole-frame
    boundaries. Replaying never iterates Python commands or invokes Python code.
    """
    def __init__(self):
        self._handle = _native.resource(7)

    def _emit(self, command: tuple[Any, ...]) -> None:
        self._handle.append(command)

    def reset(self) -> DrawList:
        """Clear retained commands/references at a frame boundary; reuse capacity."""
        self._handle.reset()
        return self

    def lines_buffer(self, xyxy: Any, color: Color, *, antialias: bool = False,
                     replace: bool = False) -> DrawList:
        """Copy a contiguous native float32 Nx4 batch in one C call."""
        self._handle.geometry(xyxy, 4, _rgba(color), antialias, replace)
        return self

    def points_buffer(self, xy: Any, color: Color, *, replace: bool = False) -> DrawList:
        """Copy a contiguous native float32 Nx2 batch in one C call."""
        self._handle.geometry(xy, 2, _rgba(color), False, replace)
        return self

    def lines(self, lines: Iterable[Sequence[int]], color: Color, *, antialias: bool = False) -> DrawList:
        for x0, y0, x1, y1 in lines:
            self.line(x0, y0, x1, y1, color, antialias=antialias)
        return self

    def points(self, points: Iterable[Sequence[int]], color: Color) -> DrawList:
        for x, y in points:
            self.pixel(x, y, color)
        return self


class Frame(_Canvas):
    """One-shot, thread-affine producer transaction. Never exports ring memory."""
    def __init__(self, display: Display, timeout: float | None):
        self.display = display
        self.timeout = _timeout(timeout)
        self._state = "new"
        self._owner: int | None = None

    def __enter__(self) -> Frame:
        if self._state != "new":
            raise StateError("frame transactions cannot be nested/reused")
        self.display._check()
        runtime = self.display._runtime
        expected = (runtime.lease_serial() + 1) & ((1 << 64) - 1)
        try:
            _wait(runtime.acquire, self.timeout)
            if runtime.state() != DisplayState.RUNNING:
                raise StateError("display stopped during frame acquisition")
            self._owner = threading.get_ident()
            self._state = "active"
            return self
        except BaseException:
            # Covers signals after C acquisition but before __enter__ returns.
            # The serial prevents revoking an outer/other producer's lease.
            runtime.cancel_serial(expected)
            self._state = "done"
            raise

    def _check(self) -> None:
        if self._state != "active" or self._owner != threading.get_ident():
            raise StateError("frame is inactive or belongs to another thread")
        self.display._check()

    def _emit(self, command: tuple[Any, ...]) -> None:
        self._check()
        self.display._runtime.command(command)

    def upload(self, data: Any, *, format: str = "RGBA", row_stride: int | None = None) -> Frame:
        """Replace the entire frame with one synchronous native copy/conversion.

        Unlike blit, this preserves RGBA bytes including transparent RGB values.
        The input may be reused after return, but must not be concurrently mutated.
        """
        self._check()
        if format not in ("RGB", "RGBA"):
            raise ValueError("format must be RGB or RGBA")
        pitch = self.display.config.width * len(format) if row_stride is None else row_stride
        self.display._runtime.upload(data, pitch, len(format))
        return self

    def draw(self, draw_list: DrawList, *, time: float = 0) -> Frame:
        self._check()
        self.display._runtime.draw(draw_list._handle, time)
        return self

    def submit(self) -> None:
        self._check()
        self.display._runtime.submit()
        self._state = "submitted"

    def cancel(self) -> None:
        self._check()
        self.display._runtime.cancel()
        self._state = "cancelled"

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._state != "active":
            return
        if exc_type is None:
            self.submit()
        else:
            try:
                self.cancel()
            except StateError:
                # Concurrent display.close already revoked/cancelled this lease.
                self._state = "cancelled"

    def __del__(self):
        if getattr(self, "_state", None) == "active":
            try:
                if self._owner == threading.get_ident():
                    self.cancel()
                else:
                    self.display.request_stop()
            except Exception:
                pass


class Playback:
    def __init__(self, display: Display, generation: int):
        self.display = display
        self._generation = generation

    def _check(self) -> None:
        self.display._check()
        if self._generation != self.display._generation:
            raise StateError("playback controller has been superseded")

    @property
    def playing(self) -> bool:
        self._check()
        return self.display.stats()["playing"]

    @property
    def error(self) -> Exception | None:
        self._check()
        return self.display.error

    def ready(self, timeout: float | None = None) -> None:
        """Wait for native initialization; report shader/decoder startup errors."""
        self._check()
        _wait(self.display._runtime.ready, timeout)

    def pause(self) -> None:
        self._check()
        self.display._runtime.pause(True)

    def resume(self) -> None:
        self._check()
        self.display._runtime.pause(False)

    def wait(self, timeout: float | None = None) -> None:
        self._check()
        _wait(self.display._runtime.wait_content, timeout)

    def stop(self, timeout: float | None = None) -> None:
        self._check()
        _wait(self.display._runtime.stop_content, timeout)

    def __enter__(self) -> Playback:
        self._check()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.stop()
        else:
            try:
                self.stop()
            except Exception:
                pass


class Display:
    def __init__(self, config: DisplayConfig | None = None):
        self._config = DisplayConfig() if config is None else config
        if not isinstance(self._config, DisplayConfig):
            raise TypeError("expected DisplayConfig")
        self._runtime = _native.Runtime(asdict(self.config))
        self._closed = False
        self._generation = 0

    def _check(self) -> None:
        if self._closed:
            raise StateError("display is closed")

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def error(self) -> Exception | None:
        """Latest asynchronous error, retained for diagnostics even after close."""
        return self._runtime.error()

    @property
    def config(self) -> DisplayConfig:
        return self._config

    @property
    def state(self) -> DisplayState:
        """Lock-free lifecycle state, also available after a close timeout."""
        return DisplayState(self._runtime.state())

    def start(self) -> Display:
        self._check()
        try:
            self._runtime.start()
        except BaseException:
            self._closed = self._runtime.state() == DisplayState.CLOSED
            raise
        return self

    def __enter__(self) -> Display:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                # Cleanup still ran; do not replace the application's exception
                # with a previously reported asynchronous worker failure.
                pass

    def close(self, timeout: float | None = None) -> None:
        if self._closed:
            return
        try:
            _wait(self._runtime.close, timeout)
        finally:
            self._closed = self._runtime.state() == DisplayState.CLOSED

    def request_stop(self) -> None:
        """Nonblocking. Use wait/close to join native workers."""
        self._runtime.stop()

    def wait(self, timeout: float | None = None) -> None:
        self._check()
        _wait(self._runtime.wait, timeout)

    def frame(self, timeout: float | None = None) -> Frame:
        self._check()
        return Frame(self, timeout)

    def try_submit(self, draw_list: DrawList, *, time: float = 0) -> bool:
        """Nonblocking/drop-incoming submission; False increments dropped frames.

        Render a retained batch only when a manual slot is immediately available.
        Native playback and explicit frame transactions remain mutually exclusive.
        """
        self._check()
        return self._runtime.try_draw(draw_list._handle, time)

    def flush(self, timeout: float | None = None) -> None:
        """Wait for submitted RGBA frames to be consumed, not electrical scanout."""
        self._check()
        _wait(self._runtime.flush, timeout)

    def snapshot(self, *, timeout: float | None = 1) -> bytes:
        """Top-down RGBA copy of the most recently consumed logical image."""
        self.flush(timeout)
        return self._runtime.read()

    def stats(self) -> dict[str, Any]:
        self._check()
        result = self._runtime.stats()
        result["state"] = DisplayState(result["state"])
        return result

    def play(self, draw_list: DrawList) -> Playback:
        self._check()
        self._runtime.play(draw_list._handle)
        self._generation += 1
        return Playback(self, self._generation)

    def _play_file(self, path: os.PathLike[str] | str, video: bool, loop: bool,
                   time_scale: float, overlay: DrawList | None) -> Playback:
        self._check()
        self._runtime.play_file(os.fspath(path), video, loop, time_scale,
                                None if overlay is None else overlay._handle)
        self._generation += 1
        return Playback(self, self._generation)

    def play_shader(self, path: os.PathLike[str] | str, *, time_scale: float = 1,
                    overlay: DrawList | None = None) -> Playback:
        return self._play_file(path, False, False, time_scale, overlay)

    def play_video(self, path: os.PathLike[str] | str, *, loop: bool = False,
                   time_scale: float = 1, overlay: DrawList | None = None) -> Playback:
        return self._play_file(path, True, loop, time_scale, overlay)


__all__ = ["DisplayConfig", "Display", "Frame", "Image", "DrawList", "Camera", "Transform",
           "Mesh", "Scene3D", "Instance", "Light", "Font", "Text", "Playback", "DrawMode",
           "PanelOrder", "Mapping", "ToneMapping", "GradientDirection", "Easing", "DisplayState",
           "Error", "StateError", "BusyError", "UnsupportedError", "capabilities"]
