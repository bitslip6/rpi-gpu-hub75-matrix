# Weighted RIO shift/exposure overlap

## Implementation

`src/weighted_rio.c:h75_weighted_rio_cycle()` is shared by the native runtime
and legacy weighted entry point. Overlap is now the weighted default; no new
public configuration field, ABI change or CLI option was added. Equal-slot
output is unchanged. The retained implementation is the fixed-eight-pixel
overlap baseline from commit `9ffa8d8`; the later adaptive-burst experiment was
rolled back at the user's request after no meaningful refresh improvement.

A cycle now performs:

1. Start blank and shift the first row/plane.
2. With OE blank, set the current address, finish CLK low, pulse latch.
3. Enable OE and establish one counter-based exposure start.
4. If enough exposure remains, shift NEXT row/plane data in eight-pixel bursts.
   These writes retain the CURRENT address, OE active, and preserved unowned
   pins. Only data comes from the next staged row/plane. There are two direct
   volatile RIO Out stores per pixel, and no timer/atomic/mode/topology decisions
   inside an eight-pixel burst.
5. Between bursts, check stop and remaining exposure. Drain Device writes and
   sample elapsed burst time after each illuminated burst.
6. Wait only for the remainder of the ORIGINAL exposure, then blank. A short
   exposure is not deliberately extended to finish a whole serial row.
7. While blank, set the next address and shift any not-yet-loaded pixels,
   continuing at the first unshifted x. Then latch/expose that row/plane.
8. Drain the final exposure and return blank. Never preload the first plane of
   another cycle/frame while the last exposure is active.

Compact plane-major queue payloads and the private aligned row-major staging
buffer are unchanged. Staged words still contain their blank/address prefix.
Illuminated bursts mask out that prefix and apply the current active prefix;
blank bursts retain the minimal unchanged-word kernel. No extra framebuffer,
per-pixel callback, lock, log or timestamp array is added to production.

## Short exposures and timing limitations

At cycle startup, the first eight actual pixels are shifted while blank using
the masked burst kernel. Its measured counter cost seeds an admission margin of
twice that cost (at least one tick). Each illuminated burst can increase the
margin for the rest of that cycle. A new burst starts only when remaining
exposure exceeds the margin. Short planes therefore stay on the counter-wait
then blank-shift path; near-expiry planes leave their remaining shifting blank.
The margin uses saturating arithmetic. The calibration emits no extra clocks
or pixels.

The margin is conservative software admission control, NOT a maximum electrical
burst-time guarantee. An unexpected stall or Linux preemption can extend an
exposure; at most one already-admitted eight-pixel burst completes before the
next check. Preemption can also extend the original busy-wait exposures. There is a Device
completion barrier per illuminated burst, never per pixel. The blank kernel
continues to use barriers at control boundaries, not per burst.

The CIE/gamma LUT, brightness drive scaling, plane count/durations, row address
helper, scan order and full-cycle frame ownership are unchanged. However,
reducing blank time can increase average brightness at the same brightness
setting. Matching old apparent brightness is not automatic. No brightness or
color retuning is part of this change.

The overlap work itself made no CPU/IRQ affinity, kernel, governor, watchdog or
boot policy changes. The subsequent CPU3 lifecycle update requires native scanner
affinity exactly `{3}` and runtime ABI 3; no CPU-selection flag remains. It does
not change this fixed-eight-pixel kernel or configure system isolation/IRQs.
No GPIO output is run automatically by tests or by the implementation agent.

## Validation

`tests/unit/test_weighted_rio.c` decodes CLK/LATCH/OE transitions from a simulated
RIO register trace with a synthetic clock. It checks:

- Exactly one correct serial stream per plane/row, no duplicate/skipped x.
- All staged data lanes, addresses and unowned outputs; current address remains
  stable during active shifting, including next-row preloads.
- Latching/address changes only while blank; OE enabled with CLK/latch low.
- Full overlap, partial overlap with blank continuation, and short-plane fallback.
- One exposure budget, not a restarted full wait after shifting.
- Slower synthetic writes, 54 MHz tick conversion, unsigned counter wrap and
  an injected stall during an illuminated burst.
- Stop mid-burst and during the final drain, with both run-flag polarities.
- Immutable generation across successor publication during active shifting,
  first exposure and final exposure; blank-only adoption, repeats, backpressure
  and the existing head-release-before-sem_post publication-gap handling.

These are RAM/control-order tests, not GPIO clock/setup/pulse validation or
proof of a refresh improvement. The optical test remains user-operated.

Build and test (preserve GPU-enabled private Python renderer):

```bash
make test libgpu example example-runtime CFLAGS_CPU= -j2
(cd python && HUB75_BACKEND=gpu python3 setup.py build_ext --inplace)
PYTHONPATH=python python3 -c 'import hub75gpu as h; print(h.capabilities())'
make test-python
taskset -c 0-2 env PYTHONPATH=python HUB75_TEST_GPU=1 \
  python3 -m unittest discover -s python/tests -p test_content.py -v
```

Recorded implementation validation (rerun after rollback to fixed-eight-pixel overlap):
- Native suite plus GPU library and both examples: built/passed.
- GPU in-place Python rebuild: completed; `/proc/self/maps` confirms the loaded
  renderer is `python/hub75gpu/.libs/librpihub75_gpu.so` (shader/video enabled).
- Full Python suite: 74 run, passed with 8 skipped.
- Explicit offscreen EGL/content suite: 13 passed.
- Scanner tests compiled/run with AddressSanitizer and UndefinedBehaviorSanitizer:
  passed (scanner/encoder instrumented; linked support library not instrumented).
- Pixel kernel retains sixteen direct scalar Out stores per eight-pixel burst;
  counter reads, Device completion and stop checks are outside the burst.

## User-operated display comparison

Use the unchanged topology/color/affinity command in `MOTION_COHERENCE.md` with
`shaders/motion_coherence.glsl`. No new option is needed to enable overlap.
Compare the same requested content FPS, brightness and exposure settings against
the pre-overlap observations. Report hold/direction/eye-tracking behavior and
final statistics. Before overlap, the reported active-cycle estimate was about
12.15 ms / 82.3 Hz. The user subsequently reported it looks noticeably better.

### First user-reported overlap result

Counters: submitted=consumed=mapped=1601, scanned=1814, dropped=0,
queue_occupancy=0, scan_ns=19861630323, map_ns=1983679404,
queue_wait_ns=13156139, latency_total_ns=2026811960.

| Derived metric | Before | With overlap |
| --- | ---: | ---: |
| Active-cycle scan rate | 82.32 Hz | 91.33 Hz |
| Mean active cycle | 12.148 ms | 10.949 ms |
| Mean submit-to-consume latency | 47.850 ms | 1.266 ms |
| Mean mapper-call duration (including output-queue wait) | 12.080 ms | 1.239 ms |
| Accumulated producer queue wait | 19.147 s | 0.013 s |
| RGBA queue occupancy at sampled stats | 4 | 0 |

Active-cycle rate increased about 10.95%; mean cycle duration decreased about
9.87%. These estimates exclude frame-adoption gaps and may include a partial
stop cycle in scan_ns but not scanned. The latency/queue figures are consistent
with greatly reduced backpressure, not a tenfold encoder CPU speedup. Exact
commands/content FPS were not supplied alongside these two counter sets, so do
not attribute all queue/latency differences to overlap without that control.
More scan cycles than mapped frames is expected from repeating the held frame.
This is user-reported optical improvement plus software runtime timing, not
scope/electrical validation; synthetic-clock tests remain separate evidence.

### First adaptive-burst runtime result (experiment; subsequently rolled back)

User-supplied counters: submitted=1155, consumed=mapped=1154, scanned=1819,
dropped=0, queue_occupancy=1, scan_ns=19886542655, render_ns=19992774214,
map_ns=3037876091, queue_wait_ns=9011019, latency_total_ns=3083238348.
Exact command/content was not supplied with these counters.

| Derived metric | Fixed-eight-pixel overlap | Adaptive 32/16/8 |
| --- | ---: | ---: |
| Active-cycle scan rate | 91.332 Hz | 91.469 Hz |
| Mean active cycle | 10.949 ms | 10.933 ms |
| Mean submit-to-consume latency | 1.266 ms | 2.672 ms |
| Accumulated render time | 0.463 s | 19.993 s |

The active-rate difference is only about +0.15%, or 16.4 microseconds/cycle,
which is not a demonstrated meaningful gain from a single uncontrolled pair
of runs. Reducing control boundaries may save work already hidden by exposure;
these counters cannot identify the remaining scan bottleneck. Render time is
very different and the producer submitted fewer frames, so confirm exact
content/FPS/settings before attributing changes in producer or mapper behavior
to the burst optimization. No adaptive-burst visual comparison was supplied.
Keep the optical/runtime evidence separate from synthetic sequencing tests.

### Diagnostic rerun with exact command/settings

User reran `shaders/motion_coherence.glsl` with content FPS 80, duration 20 s,
192x192, 64x64 panels, three chains/three ports, weighted BCM, 11 bits,
130 ns LSB, CIE, brightness 64, GPIO thread CPU3 and process taskset CPUs0–2.

Counters: submitted=consumed=mapped=1601, scanned=1817, dropped=0,
queue_occupancy=0, scan_ns=19877579542, render_ns=470569317,
map_ns=1983345964, queue_wait_ns=14122592, latency_total_ns=2030790331.

Derived active scan rate is 91.410 Hz, mean active cycle 10.940 ms, mean
submit-to-consume latency 1.268 ms and mean mapper-call duration 1.239 ms.
Compared with fixed-eight-pixel overlap's 91.332 Hz / 10.949 ms / 1.266 ms,
this is effectively unchanged: +0.085% rate or about 9.3 microseconds/cycle.
Render totals and submission counts now closely match that diagnostic baseline.
There is no demonstrated useful refresh improvement from adaptive bursts.

The current cycle exceeds requested exposure-only time (8.51552 ms) by about
2.42 ms, but that difference includes actual exposure overshoot as well as
blank shifting and control overhead. Frame-adoption/staging gaps are excluded
from scan_ns. Before further changes, distinguish those active-cycle components;
fewer checks in already-hidden shifting need not improve complete-cycle rate.
No production code or system setting was changed at that point.

### Rollback decision

The user subsequently requested reverting the adaptive 32/16/8-pixel burst and
shared-counter-sample optimization: the approximately +0.085% diagnostic change
was not a demonstrated useful gain and did not justify its added complexity.

`src/weighted_rio.c` and `include/weighted_rio.h` were restored exactly to commit
`9ffa8d8`, retaining the successful fixed-eight-pixel overlap. Adaptive-only test
instrumentation and assertions were removed; the existing overlap, exposure,
row-delivery, ownership, stall and cancellation regression coverage remains.
The comparison results above are retained as experiment history, not as a
statement that adaptive bursts are still enabled. CIE/brightness, row addressing,
frame ownership, affinity and kernel settings were not altered by the rollback.
