# hub75gpu Python API

A thin, typed CPython interface to the native HUB75 renderer. Python describes
content; C owns rendering, animation, frame queues, RGB-to-BCM conversion and
GPIO. Shader rendering stays in OpenGL ES, video decoding stays in FFmpeg.
There are no Python callbacks on native playback threads.

**Status:** implementation/validation preview, not a hardware-qualified production
release. Supported build target: regular GIL-enabled CPython 3.10+ on Linux,
initially Pi 5/aarch64. The pin-free suite also supports Linux development without
a panel. Pi 3/4, armv7, other Python versions and electrical timing require their
own validation. Free-threaded Python and subinterpreters are explicitly rejected.

## Installation

From the repository root, in a virtual environment:

```sh
sudo apt install build-essential pkg-config python3-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./python                         # CPU, default
python -c 'import hub75gpu as h; print(h.capabilities())'
```

Optional native image/font dependencies: `libpng-dev`, `libjpeg-dev`. Capabilities
report whether PNG, JPEG and PNG-based SDF text were compiled. NumPy and Pillow
are optional Python integrations, not installation requirements.

For EGL/GLES shader rendering and FFmpeg video:

```sh
sudo apt install libegl-dev libgles-dev libgbm-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
HUB75_BACKEND=gpu python -m pip install --no-cache-dir --force-reinstall ./python
```

Select **one** backend per installed extension/process. The CPU package neither
links nor imports GPU/FFmpeg libraries. `shader`/`video` capabilities describe
compiled support, not whether a compatible render node/driver is accessible.
GPU methods on a CPU installation raise `UnsupportedError`.

Setuptools/PEP 517 invokes the root Makefile. The selected renderer is bundled in
`hub75gpu/.libs/`; the extension has `$ORIGIN/.libs` RUNPATH. Import works outside
the checkout without `LD_LIBRARY_PATH`. This bundles our renderer, **not** its
system PNG/JPEG/GPU/FFmpeg dependencies. Platform wheels are not claimed to be
manylinux-portable. Verify system dependencies and redistribution licenses before
publishing wheels; no fonts, shaders or large media assets are bundled.

Build options:

- `HUB75_BACKEND=cpu|gpu` (CPU by default).
- `HUB75_BUILD_JOBS=2` controls native build concurrency.
- `HUB75_CFLAGS_CPU=...` opts into target-specific native flags. The default avoids
  blindly using `-march=native` in distributable wheels; normal native optimization
  and architecture-supported NEON paths remain enabled.
- `HUB75_SYSTEM_LIBDIR=/absolute/path` selects explicitly managed matching native
  libraries instead of bundling them. `HUB75_BACKEND` still selects the library.
  The path becomes RUNPATH. ABI is checked before any legacy by-value API table
  could be used. Use `make install-cpu` or `make install` to install matching
  native headers and `rpihub75.pc` / `rpihub75_gpu.pc` metadata. Do not mix independently built same-symbol backends.

```sh
python -m pip install build
python -m build python                               # sdist and platform wheel
python -m pip install python/dist/hub75gpu-*.whl
```

The sdist includes native source/headers/Makefile; rebuilding does not require a
checkout. `make -C python` is an in-place development convenience and requires
setuptools; the obsolete capsule extension and distutils fallback were removed.

## Safe first frame (no GPIO)

```python
import hub75gpu as h

config = h.DisplayConfig(width=64, height=64, offscreen=True)
with h.Display(config) as display:
    with display.frame(timeout=0.1) as frame:
        frame.clear((0, 0, 0))
        frame.line(0, 0, 63, 63, (255, 80, 20), antialias=True)
    rgba = display.snapshot()  # top-down, tightly packed RGBA8 copy
    print(display.stats())
```

Import and construction do not touch GPIO, install signals, lock process memory,
change governors or disable RT throttling. `start()`/context entry activates the
selected backend. For actual hardware, explicitly set `offscreen=False`, configure
the topology, arrange device permissions, and verify wiring **before** starting:

```python
config = h.DisplayConfig(width=128, height=128, panel_width=64, panel_height=64,
                        ports=2, chains=2, bit_depth=48, fps=120,
                        brightness=200, offscreen=False)
```

Physical dimensions must match panel dimensions × chains/ports. Panels have
32/64 rows; bit depth is 32–64 and divisible by eight; FPS is 1–200; brightness is
0–254. Queue depth is a power of two (1–32 offscreen, at least 2 physical).
Float settings must be finite; invalid integer ranges are rejected, not truncated.
`Mapping.U` is reserved: the unsafe legacy U mapper is not supported for physical
runtime output. Flip/mirror mappings and panel color order are supported.

### Optional weighted BCM (Pi 5 RIO)

Equal-slot output remains the default. To select weighted output, add these
settings to your topology configuration:

```python
config = h.DisplayConfig(
    output_mode=h.OutputMode.WEIGHTED_BCM,
    pwm_bits=11, pwm_lsb_ns=130, color_curve=h.ColorCurve.CIE,
    queue_depth=4, offscreen=True,  # change only after verifying physical wiring
)
```

`ColorCurve.GAMMA` uses the existing `gamma` parameter; CIE is the weighted
default. `pwm_bits` accepts 1–11; it does not change the meaning of `bit_depth`.
Brightness 0–254 scales linear drive after the curve. Tone mapping, jitter and
legacy dithering settings are not applied in weighted mode.

With `offscreen=False`, weighted output requires Pi 5 RIO and queue depth at
least 2. Unsupported models raise `UnsupportedError` at start/context entry,
not a fallback to another scanner. Offscreen weighted sessions work on any
host but still produce RGBA snapshots, not simulated panel exposure. Physical
frame replacements happen only at full-cycle boundaries; held frames repeat
when no successor is available. The initial scanner blanks during shifting
and latching, with best-effort counter-based exposure delays; overlap and
on-display color/timing evaluation remain follow-up work.

The demo accepts `--output-mode weighted-bcm --pwm-bits 11 --pwm-lsb-ns 130
--color-curve cie`; it remains offscreen unless `--gpio --acknowledge-wiring`
is explicitly supplied.

Only one active physical runtime session is allowed per process. Cooperating
processes also use the advisory lock `/tmp/rpi-hub75-rio-scope.lock`; this cannot
protect against programs that ignore that lock. Multiple isolated offscreen CPU
sessions are allowed. At most one EGL shader session runs per process.
`gpio_cpu=-1` leaves affinity alone; a nonnegative CPU opts into scanner affinity.
The demo exposes this as `--gpio-cpu CPU` (default -1). It pins only the physical
GPIO scanner thread, not the shader/decoder or mapper. With an already isolated
CPU3, use `--gpio-cpu 3`; optionally launch with
`taskset -c 0-2 env PYTHONPATH=python python3 -m hub75gpu ... --gpio-cpu 3`
to keep the other threads on CPUs0–2. The scanner explicitly selects CPU3;
this requires the session's cgroup/cpuset to permit CPU3. Do not pin the whole
application to CPU3. Offscreen sessions do not start a GPIO thread.

Physical startup waits for scanner initialization/blanking; affinity failures are
recoverable startup errors, not ignored warnings. Affinity reduces migrations
and userspace contention but cannot mask hard IRQs or guarantee PWM deadlines.
The tested kernel lacks `CONFIG_NO_HZ_FULL`, so periodic timer interrupts remain
possible even on isolated CPUs. No realtime scheduling or host-wide tuning is
performed by the runtime. Do not inherit live native handles
across `fork()`; use multiprocessing `spawn` and construct resources in the child.

## Lifecycle, backpressure and errors

- `Display` is one-shot: created → running → stopping/stopped → closed. To restart,
  create another display. Repeated create/start/close with different dimensions
  is supported. `close()` is idempotent. `display.config` is read-only and
  `display.state` provides lock-free lifecycle observation, including after a
  timeout or closure.
- `request_stop()` is nonblocking. `wait(timeout)` joins output workers after a
  stop request. `close(timeout)` requests stop and joins before revoking leases.
- Timeouts are nonnegative seconds or `None`. Waits poll in native slices of at
  most 50 ms and check Python signals. They release the GIL. `KeyboardInterrupt`
  plus a `with Display(...)` block is the normal shutdown mechanism.
- A close timeout leaves a valid, stopping object and live resources. Retry close;
  do not assume a timeout freed a worker's memory. Fallback destruction can wait
  for native operations: always prefer explicit/context-managed shutdown.
- A frame is a one-shot, thread-affine producer transaction. Successful context
  exit submits exactly once; exceptional exit cancels. `submit()`/`cancel()` can
  also be explicit. Nested frames and manual drawing during playback are rejected.
  No ring pointer or writable frame memoryview is exported.
- FIFO queues are bounded. `frame(timeout=0)` rejects immediately if unavailable;
  positive timeouts block with a deadline. `display.try_submit(batch)` explicitly
  drops the incoming frame and returns `False` when no slot is immediately
  available, incrementing `dropped`. It never overwrites a scanner-owned slot.
- `flush()` waits for submitted RGBA frames to be consumed. It does **not** prove
  physical scanout or electrical settling. `snapshot()` flushes and copies the
  latest consumed logical RGBA image, before physical mapping/tone/BCM conversion.
- Native statuses map to `ValueError`, `TimeoutError`, `OSError`, `MemoryError`,
  `StateError`, `BusyError` and `UnsupportedError`. Playback `ready()`/`wait()` and
  synchronization calls expose asynchronous initialization/render/decoder errors.
  `playback.error` / `display.error` inspect the latest asynchronous exception
  without raising it; `display.error` remains available after close. A newly
  started content producer resets that stored error.

Rendering and content/property mutations release the GIL, pin native resources,
and serialize at whole render-frame boundaries. The GPIO hot loop takes no
resource mutex. Synchronous rendering cost still depends on content complexity;
50 ms signal polling applies to waits, not a promise to preempt an arbitrary
large native draw operation. Buffers supplied by callers must not be concurrently
mutated during synchronous native reads.

## Bulk images and native-owned views

```python
raw = bytes((10, 20, 30, 255)) * (64 * 64)
image = h.Image(64, 64, raw)
with h.Display(config) as display:
    with display.frame() as frame:
        frame.upload(raw)                # full-frame copy, preserves all RGBA bytes
        frame.blit(image, x=0, y=0)       # source-over composition
        frame.blit(raw, width=64, height=64)  # upload + composite convenience
```

Preferred format is native byte-order RGBA8, channels R,G,B,A. RGB24 is supported
with `format="RGB"`, converting to opaque RGBA in C. Supply a flat contiguous
byte buffer or C-contiguous `height × width × channels` uint8 array. A flat padded
buffer may use explicit positive `row_stride`; shaped arrays must be tightly
packed. Exact byte length/shape/type is checked. Negative/broadcast/arbitrary
strides and non-uint8 image buffers are rejected. Source-over uses straight alpha;
full-frame upload preserves transparent RGB bytes. Physical BCM uses RGB values.

```python
with h.Image(64, 64) as image:
    with image.memoryview() as view:       # flat writable uint8 allocation
        view[:] = raw
    flipped = image.transformed(h.Mapping.MIRROR_FLIP)
    copy = image.tobytes()
    # image.numpy() -> optional zero-copy uint8 H×W×4 ndarray
    # image.pillow() -> optional independent Pillow RGBA copy
```

Views pin image storage. Release **all** memoryviews/NumPy views before upload,
readback, composition/playback use or close. Incompatible operations fail instead
of racing exported writable memory; `close()` raises `BufferError` while pinned.
No resize operation is provided. `Image.load(path)` uses compiled native loaders.

## Reusable drawing and bulk geometry

```python
from array import array

batch = h.DrawList().clear((0, 0, 0))
batch.lines_buffer(array("f", [0, 0, 63, 63, 0, 63, 63, 0]), (255, 100, 20))
batch.points_buffer(array("f", [5, 6, 7, 8]), (0, 255, 80))
batch.polygon(((0.1, 0.1), (0.4, 0.1), (0.2, 0.4)), (60, 100, 255))

with h.Display(config) as display:
    with display.frame() as frame:
        frame.draw(batch)                  # replay in C, not Python tuple iteration
```

Bulk geometry uses aligned C-contiguous native float32 Nx4 lines or Nx2 points
(flat arrays are also accepted; max 65,535 rows per buffer). Coordinates must be
integer-valued in `[0, 65535]`, matching individual pixel/line calls. `replace=True`
atomically replaces an existing list at a frame boundary. `reset()` releases
retained commands while keeping allocation capacity. `lines()`/`points()` accept
Python iterables for convenient setup; the buffer methods avoid per-element
Python marshalling. Pixel/individual line calls are conveniences, not the
recommended dense-rendering path. Canvas operations also include clear,
antialiased lines and polygon/gradient fills using normalized vertices and
`GradientDirection`/`Easing` enums (use four vertices for a rectangle).

## 3D, transforms, materials and lights

```python
scene = h.Scene3D(ambient=(0.2, 0.2, 0.2))
mesh = h.Mesh.cube()
mesh.set_color((255, 120, 30))
mesh.specular_strength = 0.4
mesh.specular_shininess = 32
instance = scene.add(mesh, angular_velocity=(0.3, 0.7, 0))
light = scene.directional_light((-1, -1, -1), shadows=True)
camera = h.Camera(aspect=1)
batch = h.DrawList().clear().render(scene, camera)

with h.Display(config) as display:
    playback = display.play(batch)         # native animation, no Python frame loop
    playback.ready(2)
    # Application work; updates serialize at complete-frame boundaries.
    instance.transform.position = (0.2, 0, 0)
    light.set(intensity=0.8, shadow_zoom=1.2)
    playback.stop(2)
```

Primitive names: cube, tetrahedron, octahedron, pyramid, cylinder, sphere, torus,
plane. `Mesh.primitive(...)` exposes segment/subdivision limits, wire/filled mode
and backface culling. `Mesh(vertices, edges=..., faces=...)` copies contiguous
native float32 Nx3 vertices and uint32 Nx2/Nx3 indices; indices and uint16 native
limits are validated. `scene.set_transforms(float32_Nx9)` atomically updates every
instance's position, Euler rotation and nonzero scale. `camera.project(transform)`
returns the native column-major 16-float matrix. Degenerate views are rejected
when projected/rendered. Custom vertex components are limited to ±10,000; scale
magnitudes to `[0.0001, 10000]`. Camera FOV is `[0.001, 3.14)` radians, aspect
`[0.0001, 10000]`, near plane at least `0.0001` and far-minus-near at least
`0.0001`. Ambient/light RGB components are `[0, 1]`; intensity is `[0, 10000]` and
shadow zoom `[0.0001, 10000]`. These bounds protect native float32 calculations.

`Instance` and `Light` are owner-retaining ID handles, not pointers into reallocating
arrays. Scenes retain meshes/transforms; draw lists retain scene/camera/image/text;
text retains its font. Explicitly closing a Python wrapper prevents further use
of that wrapper but does not destroy data still retained by native dependents.
Native playback never owns or finalizes Python objects.

## Autonomous shader, video and scrolling SDF text

```python
with h.Display(config) as display:
    playback = display.play_shader("shaders/cartoon.glsl", time_scale=0.5)
    playback.ready(10)                     # report EGL/compiler initialization errors
    # playback.pause(); playback.resume()
    playback.wait()                       # interruptible until stop/error
```

Shader files use the project's Shadertoy-style `mainImage`, mediump float precision,
`iResolution`, `iTime`, `iGlobalTime`, `iTimeDelta`, `iFrame` and four `iChannel`
samplers. Optional sibling `name.channel0` … `name.channel3` image files supply
textures; absent channels are safe black textures. `iChannelResolution[4]` is set.
Readback is top-down RGBA. Errors are recoverable; no legacy fatal shader loop is
called. EGL objects live and are destroyed on the native producer thread.

```python
with h.Display(config) as display:
    video = display.play_video("local.mp4", loop=False, time_scale=1)
    video.ready(5)
    video.wait()                          # EOF drains delayed decoder frames
```

Only regular local files are supported; URLs/network/device streams are deferred.
To fit video without stretching, set `video_preserve_aspect=True` in
`DisplayConfig`. The original display aspect ratio (including non-square pixels)
is preserved with centered, opaque black letterbox/pillarbox bars. The default is
`False`, which stretches video to fill the display. Native C users can set the
same field in `h75_config`; the legacy example CLI exposes this as `-A`.

Video decodes/scales to the common RGBA queue with native presentation timestamps,
EOF/loop handling and interruptible FFmpeg reads. `fps` paces retained/shader
content; video uses its timestamps (modified by `time_scale`, range .001–100).
Stop/join a controller before changing producer modes. A controller from an older
playback generation cannot control a replacement.

### Video comparison with the demo

The demo supports `--video FILE` (equivalent to `--mode video --file FILE`).
Playback is native FFmpeg video-only decoding; it loops from the beginning until
`--seconds` expires, then stops and blanks on context exit. It does not use a
Python per-frame callback or open an EGL context for video.

Video requires the FFmpeg-enabled build, currently selected by
`HUB75_BACKEND=gpu`. If you run from the checkout with `PYTHONPATH=python`, rebuild
the **in-place** extension, not just a separately installed package. From the
repository root, after installing the optional development dependencies above:

```sh
(cd python && HUB75_BACKEND=gpu python3 setup.py build_ext --inplace)
PYTHONPATH=python python3 -c 'import hub75gpu as h; print(h.capabilities())'
```

Verify `video: True`. For a verified Pi 5 / single 64×64 panel setup:

```sh
PYTHONPATH=python python3 -m hub75gpu \
  --video /path/to/clip.mp4 --seconds 30 \
  --gpio --acknowledge-wiring \
  --output-mode weighted-bcm --pwm-bits 11 --pwm-lsb-ns 130 \
  --color-curve cie --brightness 64 --preserve-aspect
```

Run the same command again with `--output-mode equal-slot` to compare the same
clip from its beginning. Change the width/height/panel/ports/chains options for
other topologies. `--brightness` accepts 0–254 and defaults to 64, retaining the
original demo brightness. `--preserve-aspect` fits with black bars; omit it to
stretch across the entire display. Video timing follows the file timestamps;
`--fps` does not override its frame rate. Equal numeric brightness with CIE and
the legacy curve is not a brightness-matched quality comparison.

The default remains offscreen: omit `--gpio --acknowledge-wiring` for a pin-free
playback/readback check. Missing video support, invalid file paths, and invalid
brightness are reported before constructing a display or activating GPIO.

### Text overlays

```python
font = h.Font("assets/roboto", size=16)     # external metrics.csv + glyph PNGs
text = h.Text(font, "NATIVE SCROLL", position=(64, 4), speed=20,
              direction=(-1, 0), color=(255, 80, 20))
text.sine = (2, 3, 1)                     # frequency, amplitude, phase speed
text.outline_width = 0.1
text.outline_color = (0, 0, 0, 255)
overlay = h.DrawList().text(text)
with h.Display(config) as display:
    playback = display.play_shader("shaders/cartoon.glsl", overlay=overlay)
    playback.ready(10)
    text.set_text("LIVE UPDATE")
    playback.stop(2)
```

The same overlay works for video; `display.play(DrawList().clear().text(text))`
is CPU-only autonomous text. Layout/rasterization is cached until content/style
changes; native scrolling, wrapping, compositing and sine effects reuse the cache.
Position/direction/speed updates do not rerasterize. Weight, outline, size, tracking,
kerning, wrap, foreground and outline colors are exposed. Text currently supports
ASCII glyph atlases, not general Unicode shaping. Nonzero text rotation raises
`UnsupportedError` rather than silently pretending the legacy renderer supports it.

## Statistics, testing and migration

`stats()` provides submitted/consumed/mapped/full-scan counters, incoming drops,
approximate bounded queue occupancy, accumulated queue-wait/render nanoseconds,
state/readiness/playing/paused flags, and last/total submit-to-logical-consumption
latency. Full BCM scan count is not content FPS.
Use warm-up/repeated wall-clock measurements; enqueue speed alone is not display
throughput. See `benchmarks/benchmark.py` and `VALIDATION.md`.

```sh
make test-runtime test-sampler                # native, no GPIO
PYTHONPATH=python python -m unittest discover -s python/tests -v
HUB75_TEST_GPU=1 PYTHONPATH=python python -m unittest discover -s python/tests -v
python tests/clean_install.py                 # wheel + sdist, fresh venvs
```

EGL tests are explicitly opted in and do not activate GPIO. Hardware output tests
live in `tests/hardware_smoke.py` and require an explicit opt-in plus acknowledged
panel topology; they are never discovered by the normal unit suite. Offscreen
checks do not validate electrical timing or panel refresh stability.

The old `scene_new`, raw capsules, module-global `frame_begin/end`, and procedural
`hub75gpu_py.c` extension are not retained. Replace them with `DisplayConfig`,
`with Display(...)`, `with display.frame()`, `Mesh`/`Scene3D`, and resource methods.
`import hub75gpu` remains unchanged. Native C's legacy table layout remains additive;
`include/hub75_runtime.h` is the supported embedding/lifecycle API. Try
`make example-runtime && build/release/example_runtime --offscreen`, or
`./example --runtime --offscreen` after building the existing GPU example.
The old CLI path still has CLI-specific fatal errors/tuning; Python does not call it.
