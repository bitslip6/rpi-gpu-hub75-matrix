# Motion-dependent 32-row discontinuity investigation

## Preserved baseline (initial diagnostic-only phase)

The initial diagnostic-only phase made no production scanner, LUT/CIE,
row-address, queue ownership, GPIO affinity, kernel or system policy changes. Existing untracked work
was retained. No GPIO output was started by the agent.

Read-only system verification:
- Running kernel: `6.18.54-v8-16k-hub75-nohz+`.
- Boot args include `isolcpus=domain,managed_irq,3 irqaffinity=0,1,2 nohz_full=3 rcu_nocbs=3`.
- `/sys/devices/system/cpu/nohz_full` contains `3`.
- `rpihub75-isolation.service` is enabled, active/exited successfully;
  `irqbalance.service` is inactive. This does not measure actual tick suppression.
- At that historical diagnostic baseline, the Python renderer reported GPU backend,
  shader/video enabled, ABI 2. Current builds require runtime ABI 3 and mandatory
  native CPU3 scanner affinity; rebuild the GPU extension before current tests.

Source review confirms `gpio_weighted_loop()` adopts/stages before calling one
complete `h75_weighted_rio_cycle()`. The cycle uses private immutable staging;
no producer queue reads occur during it. The RGBA sink retains its input lease
through mapping; synchronous shader readback completes before publication.
No new ownership defect was identified by this review.

At this topology, logical rows `r + {0,32,64,96,128,160}` share the same clock,
latch and OE exposure. Ordinary scan-address progression is not sufficient to
explain relative offsets between these six banks. Exact observed geometry
(alternating vs cumulative, etc.) is still unconfirmed.

## Controlled visual comparison (user-operated GPIO only)

The original `frame_coherence.glsl` advances two pixels per generated frame:
changing FPS also changes speed. Keep it intact for reproducing the old test.
New `shaders/motion_coherence.glsl` uses elapsed `iTime`, not `iFrame`, and has
no y-dependent content, textures or interpolation. Position is pixel-quantized.

| Shader time | Bar behavior |
| --- | --- |
| 0–4 seconds | Stationary at x=48 on a 192-wide image |
| 4–8 seconds | Right at 20 pixels/second (80 pixels total) |
| 8–12 seconds | Stationary at x=128 |
| 12–16 seconds | Left at 20 pixels/second |
| 16+ seconds | Stationary at x=48 |

Holds still generate and publish new frames at the selected FPS. This isolates
motion from frame-publication activity, not publication activity from scanning.
Keep default shader time scale 1. Timing starts after shader initialization,
so wall-clock intervals from command launch will include startup delay.

From the repository root, run the following separately at FPS=10, 60 and 100.
This is the current ABI-3 command: the historical `--gpio-cpu 3` flag has been
removed and must not be supplied. CPU3 is mandatory; taskset keeps other threads
on CPUs0–2. Isolation/IRQ policy remains separate deployment configuration.

```bash
FPS=100
taskset -c 0-2 env PYTHONPATH=python python3 -m hub75gpu \
  --mode shader --file shaders/motion_coherence.glsl \
  --gpio --acknowledge-wiring \
  --output-mode weighted-bcm \
  --pwm-bits 11 --pwm-lsb-ns 130 --color-curve cie \
  --brightness 64 \
  --width 192 --height 192 \
  --panel-width 64 --panel-height 64 \
  --chains 3 --ports 3 \
  --fps "$FPS" --seconds 24 \
  --output "/tmp/hub75-motion-${FPS}.rgba"
```

Report for each run:
1. Does the 32-row displacement disappear in the holds, including the middle
   hold while frames are still being published rapidly?
2. Does displacement reverse when the bar moves left, or remain tied to a bank?
3. Are rows 0–31 / 32–63 / 64–95 / 96–127 / 128–159 / 160–191 offset
   alternately, cumulatively, or in some other pattern? A sketch is enough.
4. Is this observed by eye, with eyes following the bar, or only in a camera?
   A camera can introduce rolling-shutter artifacts of its own.
5. Paste the final statistics and exact FPS for each run.

Content FPS is not physical scan refresh. `scanned / elapsed_seconds` includes
startup/adoption gaps if elapsed covers them. `scanned * 1e9 / scan_ns` estimates
active scanning rate, excluding staging/adoption gaps; scan_ns may include a
partial cancelled cycle whereas scanned counts only completed cycles. Neither
is an electrical timing measurement.

Disappearance during holds or reversal with motion would support a temporal
mechanism but would not identify its cause. Persistence while held would justify
revisiting static lane/physical mapping or sampling hypotheses. Do not compensate
with bank X offsets or assume slower GPIO is a fix without the observations.

## Follow-up optical observations and statistics

User reports with the fixed-speed diagnostic:
- No persistent pixel offset; the held bar is fully vertical.
- Motion discontinuities remain at 32-row boundaries and are more noticeable
  at lower content FPS.
- Apparent displacement reverses with leftward motion.
- Much more noticeable when the eyes follow the bar than with fixed gaze.

These observations favor scan/eye-tracking temporal integration over a fixed
bank coordinate error. They do not prove the absence of every delivery defect.
The scan-address timeline repeats across all six simultaneously emitted banks:
address 31 immediately above a bank boundary and address 0 below it have different
exposure times, even though matching addresses in different banks are simultaneous.
Following the bar can turn that exposure-time difference into a sawtooth-like
apparent horizontal displacement, reversing with tracking direction.

Reported counters: submitted=1642, consumed=mapped=1638, scanned=1635,
dropped=0, queue_occupancy=4, scan_ns=19862099896,
map_ns=19787222974, latency_total_ns=78379100091.
Exact command/requested FPS not supplied with these counters.

Derived active-cycle average: approximately 12.15 ms, or 82.32 cycles/s,
excluding frame-adoption gaps and with the partial-stop-cycle caveat above.
At the diagnostic's nominal 20 px/s, one cycle corresponds to about 0.24 px of
tracking motion; this is a scale estimate, not a measured bank displacement.
Mean reported submit-to-consume latency is about 47.85 ms. Mapper time averages
12.08 ms/frame but includes waiting up to 200 ms for an encoded queue slot in
`hub75_display_map_image_to_weighted_bcm()`, so it is not encoder CPU time.
A full RGBA queue and producer wait time show backpressure, not a torn frame.
Zero dropped does not imply zero blocking or perfectly regular presentation.

These observations alone did not justify a bank-coordinate correction or
ownership rewrite. The user subsequently requested shift/exposure overlap to
reduce blank time. That implementation is now in the shared weighted scanner;
see `README-weighted-overlap.md` for design, tests and timing caveats. LUT/color,
row addressing, scan order and complete-cycle ownership remain unchanged.
The user subsequently reported noticeably better motion. First post-overlap
runtime counters imply about 10.95 ms / 91.33 active cycles/s; see
`README-weighted-overlap.md` for the full comparison and limitations.

## Software validation

- `make test CFLAGS_CPU= -j2`: passed, pin-free native tests, including existing
  weighted mapper, row-address, queue and scanner-boundary tests.
- `taskset -c 0-2 env PYTHONPATH=python HUB75_TEST_GPU=1 python3 -m unittest discover -s python/tests -p test_content.py -v`:
  13 passed, EGL/FFmpeg/offscreen only, no GPIO.
- `make test-python`: 74 tests run, passed with 8 skipped (including opt-in EGL).
- New opt-in EGL test samples multiple moving frames from the actual private
  renderer through shader/readback, RGBA queue and snapshot. At 192x192 it checks
  all rows byte-identical, exactly six white columns, opaque binary pixels and
  multiple distinct positions. Accelerated shader time keeps the test short.

The dynamic test checks sampled RGBA deliveries, not every produced frame,
weighted runtime mapping, physical GPIO clocks or panel behavior. Existing
separate native tests cover mapper and staged scanner behavior; this is not a
single end-to-end proof of the complete physical pipeline. No native rebuild
was necessary for these diagnostic/test-only additions.
