# Implementation and validation record

This is an implementation preview, not production or electrical qualification.
All local validation below is **pin-free**. No GPIO session has been activated.
The phase-by-phase design and proposed release gates are in `../python-plan.md`
in the checkout (that planning document is not the installed API reference).

## Implemented scope

| Plan phase | Implementation |
| --- | --- |
| 0: inventory/baseline | Native ownership/lifecycle audit; additive runtime contract; C reference/demo and matched benchmark tooling. |
| 1: safe runtime | Recoverable one-shot lifecycle, tracked native workers, cancellation/deadlines, bounded queues, retained resources, cache cleanup, hardware exclusivity and opt-in activation. Both scanner generations retain slots through complete scans. |
| 2: Python vertical slice | Module-local, GC-aware heap types; immutable validated configuration; display/frame context managers; native exception translation; synchronous RGB/RGBA uploads; CPU-default packaging. |
| 3: performance API | Native draw lists and bulk geometry/transform/image operations; scene/camera/mesh/material/light wrappers; image mappings and pinned writable buffers; nonblocking submission; coarse stage/queue statistics. |
| 4: autonomous content | Native retained/scene/text animation and playback controls, local video timestamp pacing and looping, recoverable shaders with four texture channels, native CPU overlays, serialized live updates. |
| 5: release tooling | Private/system linkage, ABI/build/capability checks, wheel/sdist checks, stubs, CLI/examples/migration documentation, opt-in hardware smoke test, benchmarks, sanitizer CI and offscreen soak tooling. Hardware/release acceptance remains pending below. |

### Ownership and isolation

- A `Display` owns native workers/queues; native worker graphs contain no Python
  references or callbacks. Python waits and substantial native calls release the GIL.
- Frame slots are thread-affine transactions, not exported writable buffers.
  Submission/cancellation returns each lease once. Close timeout preserves live
  resources for retry; lock-free state observation does not undo that timeout.
- Scenes retain meshes/transforms; text retains fonts; batches/playback retain
  their dependencies. Closing a Python wrapper cannot invalidate those graphs.
- Image memoryviews retain the exporter/allocation; exported images reject close
  and incompatible rendering/upload until all views are released.
- One coarse resource lock serializes native rendering and property updates at
  frame boundaries. GPIO does not acquire that lock. Fine-grained lock-free
  mutation and multiple physical/EGL sessions are not promised.
- Imports/construction do not install signal handlers, open GPIO, set RT policies,
  lock process memory, or tune host-wide settings. Legacy CLI RT tuning now needs
  `HUB75_LEGACY_RT`; it is not used by the new runtime.

## Local environment

Linux aarch64, Raspberry Pi 5-class host; GCC 14; regular CPython 3.13.5.
Python development headers and build tools were staged under `/tmp` without
installing system packages. GPU/FFmpeg development libraries were already
available. Headless EGL tests require the explicit `HUB75_TEST_GPU=1` opt-in and
are separate from GPIO tests.

## Reproducible checks

```sh
# Never opens GPIO. Includes scanners operating against anonymous RAM only.
make test-runtime test-sampler example-runtime
UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1 \
make BUILD=debug BUILDDIR=build/runtime-asan \
  LIB_NO_GPU=build/runtime-asan/librpihub75_dbg.so \
  EXTRA_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-fsanitize=address,undefined' test-runtime test-sampler

python -m pip install ./python                         # CPU default
python -X faulthandler -m unittest discover -s python/tests -v
HUB75_TEST_GPU=1 python -X faulthandler -m unittest discover -s python/tests -v
python python/tests/clean_install.py --backend cpu
python python/tests/clean_install.py --backend gpu
mypy python/hub75gpu/__init__.pyi python/hub75gpu/__main__.py --ignore-missing-imports

python python/benchmarks/build_reference.py
python python/benchmarks/benchmark.py --workload lines --compare-unbatched
python python/benchmarks/benchmark.py --workload upload
python python/benchmarks/benchmark.py --workload scene
python python/benchmarks/benchmark.py --workload text --font assets/roboto
python python/benchmarks/soak.py --seconds 60 --output /tmp/hub75-soak.json
# Proposed long-duration gate, NOT claimed to have run:
python python/benchmarks/soak.py --seconds 28800 --output /tmp/hub75-eight-hours.json
```

Use a **GPU installation** for the opt-in GPU suite and media benchmarks. The
reference builder links the exact library loaded by the Python package, not a
potentially differently optimized checkout library. Benchmark dimensions,
compiler flags, queue depth, bit depth and workload must match. Offscreen
`map_ns`/`scan_ns` are zero: the offscreen consumer does not run electrical output
or physical RGB-to-BCM mapping. Mapper/scanner correctness is tested separately.

## Results

Verified locally on 2026-10-21 (full environment and raw measurements:
[`benchmarks/results/local-aarch64-cp313.json`](benchmarks/results/local-aarch64-cp313.json)):

- Native ASan+UBSan runtime/queue/mapper/sampler tests pass. Failure injection
  covers native allocation/worker startup, pre-hardware failure, and anonymous
  RAM mapping/scanner startup/affinity; timeout tests retain and retry live resources.
  Physical startup waits for scanner readiness and surfaces initialization errors.
  Stop requests publish `STOPPING` immediately, including during startup.
- Packed BCM words are checked for 1–3 ports and all bit planes, including
  alternating full-intensity channels. Pi 3/4/5 scanner paths run against
  anonymous RAM, including repeated activation and stop before the first frame.
  This tests software ownership/layout, **not** GPIO or electrical timing.
- GPU-enabled Python suite: **63 tests, 1 skipped**, passing with explicit EGL
  opt-in. The skip is the CPU-only unsupported-media case on a GPU installation.
  CPU suite: **63 tests, 7 skipped**, passing (GPU/video cases skipped). NumPy and
  Pillow adapters are tested locally, including descendant-view storage pinning.
  Subinterpreter rejection is verified using Python 3.13's `_interpreters` helper
  (the suite also supports the older `_xxsubinterpreters` helper).
- CPU Python extension/native-library ASan+UBSan run: **63 tests, 7 skipped**,
  passing with `PYTHONMALLOC=malloc` and halt-on-error. Leak detection is disabled
  for the host CPython process; the separate native ASan/LSan run enables it.
- Clean CPU and GPU wheel **and sdist** builds/installations pass in fresh external
  venvs. Checks cover installed-package tests, private-library contents, RUNPATH,
  incompatible-ABI rejection and CPU import without mapped GPU/FFmpeg libraries.
  Optional NumPy/Pillow cases skip in dependency-minimal fresh venvs.
- CPU → GPU → CPU and sanitizer → release backend/profile switches work. Native
  build outputs are fingerprinted by effective flags/dependencies, preventing
  stale instrumented/target-tuned libraries from being reused in another profile.
  Explicit managed-system linkage is also checked, followed by restoration of
  the CPU-default private-library build.
- Staged `install-cpu`, pkg-config metadata, public/transitive header compilation,
  external C demo linkage and uninstall pass. Prefix changes regenerate metadata.
  Native legacy `example` and the new runtime/reference demo build; runtime demos
  render offscreen. A dependency-minimal `PKG_CONFIG=false` CPU build links only
  libc/libm/libmvec on this host (no PNG/JPEG/GPU/FFmpeg dependencies).
- ThreadSanitizer was attempted, but this custom kernel exposes a 47-bit VMA
  layout unsupported by the installed TSan runtime (`Supported 39, 42 and 48`).
  **No TSan pass is claimed**, and host ASLR/settings were not changed.
- The suite includes native heap-type/module-cycle GC, reentrant buffer/argument
  callbacks, misaligned buffers, interruption during startup, asynchronous error
  inspection, extreme font/layout values, and exported-storage ownership.
- Stub/CLI mypy check passes. Other Python/platform combinations remain CI targets,
  not locally verified support claims.

### Matched offscreen measurements

These are recorded build-snapshot observations (library hash in the raw record),
not contractual performance qualification. All runs use the same GPU-profile
package native library, portable CPU flags,
queue depth 4, bit depth 64 and target FPS 120. Each entry is a median of three
matched C/Python runs after warmup. Manual cases are unpaced. Text uses the
checkout's Roboto SDF assets; media cases use generated constant-color shader/PPM
fixtures. These are content/queue measurements, **not panel refresh rates**.

| Workload | Dimensions | C frames/s | Python frames/s | Median regression |
| --- | --- | ---: | ---: | ---: |
| 2,048-line reusable batch | 64×64 | 1,904 | 1,837 | 3.55% |
| Whole RGBA upload (tiny) | 64×64 | 328,180 | 132,635 | 59.58% |
| Filled cube (tiny) | 64×64 | 76,937 | 60,962 | 21.07% |
| Cached scrolling text | 64×64 | 26,807 | 24,282 | 9.33% |
| Whole RGBA upload | 512×512 | 2,510 | 2,122 | **15.45%** |
| Filled cube with shadow mapping | 128×128 | 224.54 | 224.15 | 0.18% |
| Native retained lines | 64×64 | 119.05 | 119.00 | 0.04% |
| Native shader | 64×64 | 118.80 | 118.79 | ≈0% |
| Native looping video | 64×64 | 24.58 | 24.59 | −0.04% |

All manual output hashes match. Reusable line batches are **7.93×** faster than
individual Python line calls in the accompanying convenience-path comparison.
For the tiny upload, wall time is approximately **3.05 µs C vs 7.54 µs Python**;
for the tiny cube, **13.00 vs 16.40 µs**. Percentages exaggerate those small
absolute differences. However, the **512×512 upload misses the proposed 10% gate**;
it needs further memory/pipeline profiling, and the release performance gate is
not declared passed. The raw record includes CPU time, call-latency percentiles,
logical-consumption latency and close timings. Median runtime close measurements
were approximately **9–21 ms** across these local workloads; this is not a bound
for arbitrary media/drivers or physical sessions.

### Short resource soaks

- CPU profile: **332 cycles / 20.03 s**, peak sampled RSS growth ≈0.78 MiB.
- GPU profile with CPU retained content: **333 cycles / 20.08 s**, ≈0.78 MiB.
- GPU profile mixing shader, looping video, uploads and retained scene/text:
  **293 cycles / 20.04 s**, ≈2.86 MiB.

All finished with **4 descriptors and 1 thread**, matching their post-warmup
baselines. The script samples after each session closes; these counts are not the
number of active worker threads during rendering. Native resources/dependencies
are closed/recreated and shadow rendering is included. These short runs show
bounded local behavior only, **not** an eight-hour leak/deadlock qualification.

## Remaining release gates

- Explicitly authorized panel/HAT wiring/topology validation; output blanking,
  brightness, color order, all ports/chains and long-running physical sessions.
- Scope/logic-analyzer measurements for electrical timing, refresh jitter and
  tearing under CPU/I/O load. Offscreen throughput is not panel refresh rate.
- An eight-hour mixed-workload soak on the agreed deployment configuration.
  Short offscreen checks cannot substitute for this gate. The soak script records
  bounded RSS, descriptor/thread counts and samples; inspect trends as well as
  its configured growth limit.
- Agreement and measurements for the proposed ≤5% autonomous / ≤10% substantive
  manual throughput gates. Tiny workload percentages alone are not useful.
- Observed CPython 3.10/3.13 CI and other supported desktop Linux builds; Pi 3/4,
  armv7 and alternate driver/FFmpeg combinations need their own validation.
- Wheel redistribution, architecture/runtime dependency and asset license checks
  before publishing platform wheels. No manylinux claim is made.

Unsupported scope remains explicit: free-threaded Python, subinterpreters,
fork-inherited live native handles, multiple simultaneous physical sessions,
multiple EGL shader sessions, physical
U mapping, non-ASCII shaping, nonzero text rotation and network/device media.
