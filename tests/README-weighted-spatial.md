# Fixed spatial dithering in weighted BCM

## Usage

```sh
make example-runtime
./build/release/example_runtime --gpio --width 192 --height 192 --fps 48 \
  --brightness 32 --file shaders/grafeq.glsl --dither 2 --verbose
```

`--dither N` / `-l N` accepts integer strength **0..10**. Zero (the default) keeps
it disabled; start with 1 or 2. No `--output-mode equal-slot` switch is required.
Temporal quantization dithering (`--quant-dither` / `-q`) remains equal-slot-only.
Use GPIO only after verifying wiring and scanner CPU-affinity requirements.

## Implementation

This intentionally matches the legacy spatial **effect**, rather than adding a
new PWM-fraction rounding algorithm:

- Same fixed coordinate hash, using the full mapped-image x/y coordinates.
- Separate six-bit hash slices decorrelate R, G and B.
- Same amplitude scaling/rounding and channel cutoff: only source bytes below 80
  receive offsets; the adjusted byte is clamped to 0..254.
- No time, frame counter or random generator. Identical input/settings produce
  identical bitplanes indefinitely; repeated encoding does not advance a phase.
- As in the legacy algorithm, stronger settings can lift source black. Overall
  brightness zero still produces zero PWM for every channel.

The mapper caches three signed-byte offsets per pixel, built only when geometry
or strength changes. A 192x192 image uses **108 KiB** for this optional pattern.
It is allocated lazily, owned by the mapper thread, and freed by
`hub75_mapper_cache_clear()`. Input remapping/panel calibration runs before the
spatial effect, as it does for the legacy mapper. Cache keys include width,
height and strength; even geometry/stride changes with the same input byte
count are handled.

At frame encoding, offsets are fused into the existing channel lookup/bitplane
packing loop. The corrected CIE/gamma curve and linear brightness still use the
final-precision PWM LUT: there is no corrected 8-bit image or extra image pass.
Pattern generation is not repeated for each source frame. Applying the pattern
occurs once per encoded source frame; subsequent GPIO scans reuse those planes.
There is **no change to the GPIO scanner**, no per-scan dithering, and no temporal
accumulator or per-frame allocation.

The original `h75_weighted_encode()` remains the plain non-dithered entry point.
For direct encoder users, `h75_weighted_spatial_build()` fills caller-owned RGB
int8 offsets and `h75_weighted_encode_spatial()` applies them. Buffers/patterns
remain caller-owned and immutable during encoding. The normal runtime/legacy
mapper selects the new encoder when `display->dither` is nonzero. No existing
struct layout or runtime ABI is changed.

## Pin-free tests

```sh
make test-weighted-bcm test-weighted-delivery test-weighted-rio
make example-runtime
python3 tests/test_runtime_demo_cli.py
# Optional GPU shader-time regression (still no GPIO):
H75_TEST_GPU=1 python3 tests/test_runtime_demo_cli.py
```

Coverage includes independent reconstruction/reference matching, repeatability,
strength 0/1/2/10, source/alpha immutability, RGB/RGBA, all channel orders, multiple
ports and panel heights, brightness-zero black, invalid parameters/capacities,
mapper preprocessing, strength/precision changes, queue backpressure, and cache
resize/re-keying for different geometry with the same input byte count.

Offscreen runtime mode bypasses the BCM encoder; its FPS and RGBA dump cannot
measure or demonstrate this effect. Use the encoder tests/benchmark instead.

## Steady-state encoder benchmark

```sh
make build/release/bench_weighted_spatial
# Use one producer CPU for repeatable comparisons; do not use the scanner CPU.
taskset -c 1 ./build/release/bench_weighted_spatial
# Convenience build-and-run target:
make bench-weighted-spatial
```

The standalone benchmark encodes actual 192x192 RGBA data, three ports/chains,
11 PWM bits, CIE and brightness 32. It reports median and p95 over 300 frames
following warmup. Pattern construction is excluded from steady-state timing.
It initializes no GPU, GPIO, library runtime or display device.

One local Cortex-A76 run, pinned to CPU1 with release optimization and strict
float semantics, compared the encoder from commit `10f7c62` with the new encoder:

| Input | Original median | New, disabled median | New, strength 2 median | Strength 2 p95 |
| --- | ---: | ---: | ---: | ---: |
| Mixed gradient | 2.295 ms | 2.317 ms | 2.653 ms | 2.669 ms |
| Flat dark (16) | 1.099 ms | 1.065 ms | 1.379 ms | 1.383 ms |
| Flat bright (128) | 1.185 ms | 1.235 ms | 1.363 ms | 1.367 ms |

Enabled overhead relative to the new disabled path was about 0.13..0.34 ms/frame
in these samples. For the gradient, this is about **15% of encoder time**, but
about **2% of a 16.67 ms/60 FPS frame budget**, or 1.6% at 48 FPS. Dark input adds
about 30% encoder time while remaining a similar absolute cost. Disabled timings
were close to the baseline, not guaranteed cycle-identical. These numbers are
local steady-state samples, not hardware refresh guarantees or full GPU-to-panel
latency measurements. Actual GPIO refresh, contention and visual quality still
require on-panel validation.

The benchmark can also be built with `BENCH_BASELINE` against an older
`src/weighted_bcm.c`; that excludes calls to the new APIs and measures only the
original encoder. Match compiler options, CPU affinity and system load when
comparing versions.
