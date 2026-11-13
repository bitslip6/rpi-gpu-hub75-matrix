# Plan A — take userspace RIO as far as possible

## 1. Goal, constraints, and honest guarantees

Retain CPU-driven RP1 RIO output. Use existing standard or PREEMPT_RT kernels,
boot parameters, affinity, scheduling, and userspace changes only. No new kernel
build, kernel patch, or additional custom module in this plan.

Aim for 20–25 million correctly timed fresh pixel updates/s inside a shift
sequence, stable complete scan cycles, and intentional housekeeping at a blanked
end-of-cycle boundary. Move all other work off the output CPU.

**Userspace cannot mask local hard interrupts or force all interrupt delivery to
frame boundaries.** PREEMPT_RT can defer/move much interrupt *thread work*, but
primary handlers, local timers, IPIs and exceptional events can still interrupt
output. The achievable contract here is “minimize unsolicited interruptions and
schedule voluntary work at frame boundaries,” not “interrupts only once/frame.”
A production user process also cannot guarantee blanking after SIGKILL, kernel
panic or hardware failure. Panels remain disconnected during scope development.

The reported 20–25 MHz RIO capability is the starting hypothesis to reproduce
with new RGB data per edge, not a clock-only guarantee. Physical setup/hold and
RGB correctness remain unverified with only GPIO17 probed.

## 2. Define a frame before setting a 200–1000 Hz service rate

Track these separately:

1. Content frame: a renderer/video update (e.g. 60/120 Hz).
2. Plane sweep: one pass over rows for one value of `pwm`.
3. Full scan cycle: all rows for all `bit_depth` values in the current loop.
4. Service boundary: a safe blanked point where the emitter may enter the kernel.

The requested “once/frame” boundary is initially a **full scan cycle**, not a
content update. The existing Pi5 `frame_count` increments once per plane sweep,
so its printed panel refresh rate is NOT the full-cycle rate. Fix/report both
before making comparisons. Current Pi5 frame acquisition can occur at pwm=0,16,
32,...; make generation changes only at full-cycle boundaries.

For the inspected loop, excluding control/dwell/service costs:

    Npixel = width * (panel_height/2) * bit_depth
    Cshift = Npixel / pixel_rate
    Fcycle <= pixel_rate / Npixel

Three parallel RGB ports share a pixel clock and do not multiply Npixel.
Any additional dwell, mapping topology or protocol timing must be included in
the actual measured cycle cost, not omitted from this lower bound.

Examples at 20 Mword/s, 32 scanned rows:

| Shift width | Planes | Shift-only time | Maximum full cycles/s |
| ---: | ---: | ---: | ---: |
| 64 | 8 | 0.8192 ms | 1221 |
| 64 | 16 | 1.6384 ms | 610 |
| 64 | 32 | 3.2768 ms | 305 |
| 128 | 32 | 6.5536 ms | 153 |

At 25 Mword/s these times are 20% shorter and rates 25% higher. Therefore 1000
full cycles/s is impossible for some configurations even with no interrupts;
128x32x32 does not reach 200 at either 20 or 25 Mword/s. Resolve this by reducing
work, accepting a lower full-cycle rate, or defining additional safe service
boundaries inside a cycle. Do not silently relabel plane sweeps as full frames.

Choose a feasible target period P: 1ms at 1000Hz through 5ms at 200Hz. Require
measured worst observed emit time plus planned boundary work and margin < P.
This is an empirical acceptance test, not a mathematical WCET proof.

## 3. Starting evidence and code issues to address

Relevant code: `src/rpihub75.c::hub75_display_run`, `src/lowlevel.c`,
`src/spsc.c`, `src/pixels.c`, `src/util.c::map_gpio/configure_gpio`.

- CPU3 is currently selected for output; the shader renderer selects CPU2.
- Current RT helper tries FIFO99, locks memory and changes governor/global RT
  throttling. These changes need explicit configuration, failure checks and
  rollback rather than hidden process-wide/system-wide side effects.
- Recorded experimental kernel config lacks NO_HZ_FULL. Boot logs explicitly
  reject `nohz_full=3 rcu_nocbs=3`. Do not describe those runs as fully tickless.
- A stock RT kernel is installed (`6.18.54-v8-rt+`); its actual config/behavior
  must be checked before selecting it. PREEMPT_RT alone does not imply NO_HZ_FULL.
- Current emitter does per-pixel address/jitter ORs, queue/semaphore work during
  plane processing, periodic time/logging calls and a cycle-end `sched_yield`.
- Current queue pointer ownership needs correction before performance tuning:
  initial pop-begin retains a slot; later another begin can return that same
  uncommitted tail; commit then releases memory while `bcm_signal` still uses it.
  Producer reuse can cause tearing independently of interrupt latency.
- `sched_yield` is not a promise to run lower-priority housekeeping. FIFO yield
  only yields to runnable same-priority peers; it is not a defined service window.
- RIO register comments are misleading: Out is a full output-latch write, while
  SET/CLR aliases modify only selected bits. Audit GPIO0/1 and other unowned bits.
  `configure_gpio()` compares a pin number with PIN_CLK/PIN_LATCH bit masks,
  rather than ADDRESS_CLK/ADDRESS_STROBE pin numbers, so its intended clock/latch
  slew selection is not applied correctly. Correct and independently scope this
  electrical change before broad pin testing; do not combine it silently with a
  scheduler comparison.

## 4. Phase U0 — reproduce and attribute the baseline

Create a standalone RIO scope benchmark first, separate from shaders/video and
PIO tests. Do not change the existing production hot loop until a baseline is
saved. Use pin-free schedule/ownership unit tests before touching GPIO.

Benchmark modes:

- Clock-only: diagnostic only, not fresh pixel throughput.
- Prepared alternating RGB and CLK with OE high; panels disconnected.
- Deterministic nontrivial word sequence to expose repeated/skipped data when
  additional probes/logic analyzer are available.
- Real row/plane protocol, initially blanked, later a static test image.

Use identical MMIO mapping, pin mask, store ordering, delays and workload for
scheduler comparisons. Do not use trace-instrumented peak rates as final results.

Record kernel/config/cmdline, scheduler and priority, actual CPU placement,
frequency/temperature/throttling, memory faults, CPU3 interrupt/IPI deltas, and
scope burst/long-term rate and gap patterns. Add counters for content generations,
plane sweeps, full cycles, repeats, underruns, service time, deadline misses and
shutdown latency. Aggregate off-core; no per-pixel clocks or logs.

Use narrow irq/sched tracing on CPU3 in separate diagnostic runs to attribute
long gaps. Sample before/after counters where trace overhead is too high.
Frame-level timestamps alone cannot locate or bound every individual pulse gap.
Do not assume every remaining gap is an interrupt: retain PCIe/memory contention
and data/layout/cache effects as hypotheses.

Gate: repeated 30s runs plus a 30min loaded soak, with raw captures and correct
labels. No GPIO17-only claim of RGB setup/hold or flicker improvement.

## 5. Phase U1 — isolate a minimal output process

Prefer a dedicated emitter process over another thread sharing the renderer's
address space. This reduces allocator, signals, mapping changes and address-space
TLB-shootdown coupling. It does not prevent global shootdowns/IPIs or shared-cache
and memory/interconnect contention.

Suggested CPU partition (adjust to profiling):

| CPU | Work |
| --- | --- |
| 0 | Device IRQs, kernel housekeeping, controller, telemetry, shutdown |
| 1 | RGB-to-BCM mapping and output-schedule preparation |
| 2 | Shader/video/application rendering |
| 3 | Emitter only |

Set affinities before work starts, including mapper and FFmpeg/helper threads.
Do not assume the main thread's affinity governs all later thread placement.
Use cpusets/cgroup partitioning where supported; ensure the emitter is allowed
CPU3. Audit IRQ/RPS/XPS, irqbalance, unbound workqueues, timers and services.
Route movable device IRQs away from CPU3; managed/per-CPU IRQs require explicit
verification and cannot all be moved through smp_affinity. Keep at least one
working housekeeping CPU and retain an out-of-band/remote recovery path.

For each *already available* kernel:

- Obtain its exact config; check PREEMPT_RT, NO_HZ_FULL, RCU_NOCB_CPU and tracing.
- Enable supported isolation/nohz/RCU/IRQ-affinity settings at boot. Potential
  parameters include isolcpus flags, nohz_full, rcu_nocbs and irqaffinity, but use
  only those supported by that kernel and verify the resulting masks/logs.
- Exclude CPU3 from unbound work and watchdog activity where supported and safe.
- Verify the scheduler tick disappears during a single runnable userspace loop
  if full dynticks is supported. Perf events/CPU timers can keep it running.
- If no installed kernel has full dynticks/offload, record the limitation. Adding
  those options by rebuilding belongs to Plan B, not this userspace-only plan.

Lock/prefault all output buffers and stack before entering RT execution. Resolve
symbols/initialization ahead of time. Avoid fork, mmap/munmap, allocation/free,
page migration/automatic NUMA work and THP compaction affecting active data during
output; make any system-wide changes explicit and reversible. Keep source data
immutable and producer writes off the active cache lines. Partition/align queue
state to avoid false sharing; use ARM release/acquire publication, not volatile.

Route ordinary signals to a controller/sigwait thread on a housekeeping CPU.
Emitter checks stop requests once per service boundary. Do not run signal handlers
inside the shift loop. Fatal/nonmaskable events remain outside the guarantee.

## 6. Phase U2 — move protocol preparation off CPU3

Introduce immutable, owned scan schedules with explicit states:

    FREE -> BUILDING -> READY -> ACTIVE -> FREE

Use two/three or a small bounded number of buffers and one atomic publication
per full cycle. Hold ACTIVE ownership for every read; never release a live slot.
Latest-complete-frame selection may drop queued content generations, not read
partially written ones. If no new content is ready, repeat the last complete
scan schedule intentionally; report repeats separately from malformed/stale data.

On CPU1 precompute:

- Pixel low-state words with RGB, row address, OE/jitter policy and CLK low.
- Plane/row ordering, latch/address controls and validated dwell/delay descriptors.
- Width/plane constants, offsets, loop limits and any deterministic jitter phases.

Benchmark two layouts: compact one-word/pixel plus CLK OR, and fully expanded
low/high writes. Expansion reduces ALU work but doubles pixel-source storage and
can hurt caches; keep it only if measured faster/more regular. Precompute a finite
phase bank if jitter must evolve across cycles; do not accidentally freeze/change
brightness behavior merely to remove work. No floating point on CPU3.

Emitter inner loop should be loads, required integer loop/CLK operations, ordered
MMIO stores and unavoidable protocol delays. No queue semaphores, locks, syscalls,
logging, time reads, producer wakeups or per-pixel atomic polling. Publish counters
and ownership transitions once at the boundary; workers poll or are notified
there, not on every pixel/plane. Prefetch/unrolling are independent measured
variants, not assumed wins. Maintain exactly two ordered pixel stores where the
protocol permits; do not remove setup/hold or coalesce edges to inflate MHz.

Use only owned pins. If full-latch writes conflict with other GPIO users, evaluate
owned-bit XOR/SET/CLR sequences without corrupting other pins; assess their store
count/performance. Do not add per-pixel MMIO read-modify-write round trips.

## 7. Phase U3 — scheduler matrix, not “FIFO99 everywhere”

Test only after preparation/ownership correctness is established:

| Variant | Benefit | Limitation |
| --- | --- | --- |
| Standard kernel + FIFO | Low overhead, no intentional budget expiry | Hard IRQs still interrupt |
| PREEMPT_RT + FIFO | Most IRQ work threaded/movable; output can outrank selected local IRQ threads | Primary hard IRQs/IPIs remain; dependency starvation risk |
| Either + DEADLINE | Defined runtime/period; boundary sleep can align cycles | CBS depletion can throttle mid-frame; may need affinity/admission tuning |
| RR/OTHER | Diagnostic controls | Time slicing/fair scheduling generally unsuitable for best determinism |

Start with FIFO at a reviewed priority (e.g. 80), not an unconditional 99. Move
IRQ threads away first; raise priority only with known dependency/IRQ-thread
priorities. Keep recovery/controller runnable on another CPU. Higher userspace
priority does not mask hard IRQs. PREEMPT_RT is a measurement candidate, not an
automatic speed win; compare mapping/store cost and tail latency under load.

For DEADLINE use measured full-cycle CPU runtime plus margin and a feasible
period. Admission/bandwidth limits may require spare capacity; prefer a genuine
blocking boundary. A missed/exhausted budget inside visible output is a rejection
criterion. Do not claim DEADLINE creates a no-preemption transaction.

Handle RT throttling explicitly. An almost-100%-busy FIFO emitter can hit the
system RT runtime limit even on an isolated CPU. Either leave adequate blocked
budget or make a conscious reversible throttling change with recovery safeguards.
Never silently disable the global safety net in a library startup helper.

## 8. Phase U4 — a real blanked service window

At each feasible complete-cycle boundary:

1. Set OE high, CLK low and LAT inactive through the correct ordered MMIO path.
2. Where required for the timing experiment, perform one documented completion
   read/flush at the boundary so posted blanking is effective before service.
   A compiler fence or CPU store barrier alone does not prove remote completion.
3. Publish generation/counters and acquire next immutable schedule (constant work).
4. Block until the absolute next start time with a monotonic clock; do not rely
   on sched_yield. All potentially blocking work occurs after blanking.
5. Resume output; record any overrun and resynchronize at a safe boundary rather
   than shorten setup/hold or partially skip a cycle to catch up.

A blocked output task gives lower-priority local work a chance to run, but cannot
promise that all such work completes in that window or prevent hard IRQs later.
Bound the controller's work; if the window is insufficient, reduce refresh/work
or expand it. If peak-free-running operation is preferred, keep an explicit
minimum boundary block rather than pretending yield reserves housekeeping time.

Initial test budgets (not promises): boundary 10–50us at 1000Hz, or 50–250us at
200Hz, each 1–5% of the period. Absolute sleep/wakeup tail may require more.
20 Mword/s with 5% non-shifting time yields at most 19 Mword/s long-term; 25 yields
23.75. Include latch/dwell and other overhead separately. Fixed cycle periods
and fixed intended exposure make boundary pauses less visible than arbitrary
in-row pauses, but variable blanking can still modulate brightness/camera bands.

## 9. Estimated benefit and acceptance criteria

These are engineering hypotheses, not measured improvements. Establish actual
pause time first. If R is instantaneous pixel rate, J is total unwanted pause
per cycle and C is useful emitting time, ideal elimination gives speedup
(C+J)/C. Example: 5us lost per 1ms cycle is ~0.5%, not a 25% throughput win, even
though eliminating a single mid-exposure pause may greatly improve image quality.

| Change | Planning estimate | Determinism / flicker expectation |
| --- | --- | --- |
| Owned complete-cycle buffers | No promised clock gain | Removes producer-overwrite/mid-cycle-generation tearing, not scan flicker |
| Off-core preparation, compact hot loop | 0–10% peak-rate improvement to test; may be zero/negative | Fewer software-caused branches/cache effects; PCIe limits remain |
| Supported full isolation + RT/FIFO tuning | Usually 0–5% mean gain if pause duty is small; potentially larger only if baseline has substantial interference | Target >=90% fewer >1us unsolicited gaps, subject to measurement; no hard bound |
| Blanked scheduled boundaries | Costs chosen 1–5% period plus protocol overhead | Moves voluntary disturbances out of exposure; stable period may reduce brightness modulation |

Do not add percentages independently. Do not promise rates above validated panel
and RIO capabilities. Evaluate flicker with later OE/LAT/RGB probes plus a fixed
camera/photodiode setup; GPIO17 alone cannot quantify optical flicker reduction.
For one interrupted exposure, relative brightness error can be roughly J/T_on;
1us on a 10us exposure can be ~10% error even with <1% average throughput loss.
Actual jitter-OE behavior and exposure segmentation determine the real effect.

Acceptance:
- Correct data ownership; no unexpected generation changes within a cycle.
- Validated pixel setup/hold and no missing/extra edges.
- Target full-cycle/service rate feasible by measured work; explicit 200/1000Hz
  definitions; no misleading plane-count refresh rate.
- No allocations, faults, syscalls or logs inside emission; CPU3 thread/IRQ
  deltas and tracing substantiate isolation claims.
- >=30min representative GPU/video/network/storage stress with gap/cycle
  percentiles, maximum observed values, repeats and optical results. A measured
  maximum is not a guaranteed worst case.
- Record remaining IRQ/IPI/non-IRQ tails honestly. If userland cannot meet the
  accepted image/timing budget, proceed to Plan B without claiming hard isolation.

## 10. Implementation order / deliverables

1. Pin-free schedule/ownership tests and definitions/counters.
2. Standalone RIO scope benchmark + baseline collector (no automatic pin drive).
3. Minimal dedicated emitter + controller/preparer, active-buffer ownership fix.
4. Kernel capability/affinity audit and reversible tuning script.
5. Existing standard/RT scheduler matrix; blanked absolute-time service window.
6. Optimize hot-loop/layout variants only after correctness and isolation.
7. Publish A/B table and decide which measured tails require Plan B.

Keep existing RIO behavior available as a selectable baseline, preserve unrelated
worktree changes, and do not run PIO and RIO tests concurrently on the same pins.
