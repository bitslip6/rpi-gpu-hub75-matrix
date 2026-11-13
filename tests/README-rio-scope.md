# RIO userspace baseline — implementation phase 2

This is deliverable 2 in `RIO_USERSPACE_PLAN.md`: a standalone scope workload and
baseline collector. The production emitter, GPIO setup, SPSC queue, scheduler
helpers and refresh labels remain unchanged. This tool is **not** proof that RIO
meets 20–25 MHz, RGB setup/hold, interruption or optical/flicker requirements.

Optional Pi 5 environment setup is documented in
[`README-rio-environment.md`](README-rio-environment.md): explicit backed-up boot
changes and health-checked runtime isolation, never GPIO output. Apply/restore
only while collectors and GPIO workloads are stopped; do not silently combine
that profile with FIFO or the production helper's automatic global RT tuning.

## Start without GPIO

```sh
make test-rio-scan
make test-rio-scope
make rio-scope                         # builds only; NEVER runs GPIO
./build/release/rio_scope --cycles 3    # safe volatile-RAM smoke test

# 30 seconds per mode, three repeats, CPU3 emitter / CPU0 collector and sigwait.
python3 tests/rio_baseline.py
```

No `--output` means **no GPIO device opens, mmap, mux or pad writes**. The benchmark
uses volatile RAM stores; those rates are **not MMIO/RIO throughput**. `--cycles`
is a smoke-test cap, not a timed acceptance run. No root/GPU/FFmpeg dependency is
needed for the software tests. The collector uses Linux proc/sysfs and requires
separate allowed emitter/controller CPUs; change `--cpu` / `--controller-cpu` if
3/0 are unavailable in your cpuset. Its affinities affect only its own process
and child workers, not the system's IRQs or other tasks.

Modes (all hardware modes keep panel OE high):

| Mode | Work |
| --- | --- |
| `clock` | Diagnostic CLK toggles; OE high. Only GPIO17 and GPIO18 are owned. NOT fresh RGB throughput. |
| `alternating` | New prepared all-on/all-off RGB word per edge, across all 18 default RGB lines. No latch/address protocol. |
| `sequence` | Deterministic hashed RGB words, with adjacent duplicates removed. For later RGB/logic-analyzer missing/skipped/reordered-data checks. |
| `protocol` | Same sequence with row addresses, all planes/rows, and latch pulses. Always blanked; not a visible-image or dwell/brightness test. |

The fixed pinout is **default HZELLER_HAT only**, not ADA/custom wiring. RGB pins
match `include/rpihub75.h`; CLK=17, OE=18, LAT=4, A/B/C/D/E=22/23/24/25/15. Row
addresses use default offset -1 and A as LSB. Protocol rows must be powers of two,
1–32. Defaults are width 64, 32 scanned rows, 8 planes. Three parallel RGB ports
share one pixel clock: words/cycle = width × scanned rows × planes.

## Hardware is an explicit manual experiment

**Disconnect panels.** Disable conflicting GPIO/I2C/SPI/UART/PIO users first.
Keep a remote/out-of-band recovery path. Do not run PIO and RIO tools, the example,
or another GPIO program concurrently. An advisory lock excludes another instance
of this benchmark only; existing PIO tools do not participate in that lock.

Start with CLK on GPIO17 (physical pin 11), OE on GPIO18 (pin 12), and a short Pi
ground connection. Use a 10x probe with matching scope attenuation. The program
checks the Pi 5 model and RP1 gpiochip identity, rejects kernel-owned selected
lines, active RIO outputs, alternate-function muxes, and (by default) pad
output-disable. It fails closed rather than forcibly taking pins from another
driver. Reboot/release
previous GPIO users deliberately if they left outputs enabled; do not defeat the
checks while another program is active.

```sh
# Interactive collector asks for YES, then prompts before EACH pin-driving run.
python3 tests/rio_baseline.py --output --modes clock --seconds 30 --repeats 3

# Only after reviewing clock/OE captures and the recorded pad settings:
python3 tests/rio_baseline.py --output --modes alternating sequence protocol \
    --seconds 30 --repeats 3 --notes 'scope/probe/ground/sample-rate/timebase; idle load'
```

If a run refuses a pad such as GPIO17 `pad=0x96`, bit 7 is output-disable (OD).
This is not a successful clock baseline: no pixels were emitted. For a separate,
explicit clock/OE scope experiment, opt into clearing OD only:

```sh
# Panels disconnected; no other GPIO/PIO users; scope CLK/OE before broader pins.
python3 tests/rio_baseline.py --output --enable-pad-output \
    --modes clock --seconds 30 --repeats 3
```

`--enable-pad-output` requires `--output`, is forwarded to the child and recorded
in the manifest and benchmark JSON. It clears OD only on selected pads where OD
was set, e.g. `0x96 -> 0x16`. Slew, drive, pulls and receiver settings are preserved.
Changed OD bits are restored during cleanup, including partial-setup failure.
After clock/OE validation, use the same explicit option for any broader baseline
that needs pad enabling. This is not an automatic correction of the production
slew-selection bug. Do not silently mix pad-enable policies in scheduler A/B runs.

The collector's `--output` requires a terminal. A direct benchmark run requires
BOTH `--output --panels-disconnected`; this is a deliberate operator acknowledgement,
not a physical disconnect detector. Direct invocations are useful for matching
scope capture intervals, but the collector is preferred for provenance.

There is **no unblank/static-visible-image option** in this stage. Missing/extra
edge and RGB setup/hold validation must precede later visible-image work. GPIO17
alone cannot verify RGB correctness, latch/address timing, or optical flicker.

### Invalid MMIO readback versus a genuine output conflict

Preflight logs raw Out/RIO_OE and selected CTRL/PAD values before interpreting
ownership. All-ones CTRL or PAD (`0xffffffff`, including reserved bits) is invalid
RP1 readback, not proof that another process enabled every output. The tool
refuses GPIO writes in this case; `--enable-pad-output` does not bypass it.
Actual RIO-enable conflicts remain rejected and now include raw/conflicting masks.

If this happens, stop hardware tests and inspect `before/after/dmesg.json`,
`pci.json`, and `throttled.json`. An RP1 PCI configuration reported as `Unknown
header type 7f`, together with all-ones MMIO, indicates an access/device problem
rather than a usable GPIO configuration. Mapping/access or PCIe/device state
needs investigation; do not clear enables or force a takeover using those reads.

Repeated kernel `Undervoltage detected!` messages are a separate power warning.
For example, `get_throttled=0x50000` records historical undervoltage and throttling;
it does NOT say undervoltage is active now, or prove it caused the RP1 failure.
Check the Pi supply/cable and probe/panel wiring, shut down cleanly before wiring
changes, and arrange a controlled power cycle once power is reliable. Keep panels
disconnected and remote recovery available. This collector never resets RP1,
unbinds drivers, changes PCI power policy, or reboots automatically. Verify sane
register/PCI reads before attempting another scope baseline.

A recurrence in `tests/results/20261019T153207.496183Z-rio/` had all-ones
CTRL/PAD and `Unknown header type 7f` **with `throttled=0x0` before and after**.
The captured kernel log did not change during the failed run. The PCI error was
also already present in the earlier `20261019T151847.190510Z-rio` lock-failure
capture, before the lock fix or any emission by either failed run. Do not label
this recurrence undervoltage, a lock-fix regression, or an isolation-caused
failure without further evidence. Live sysfs reported RP1 runtime PM `active`,
`control=on`, D0 and zero suspended milliseconds despite invalid configuration
reads; cached driver/BAR/power metadata does not prove accessible hardware.

If runtime isolation was recently applied, preserve its private state and failed
captures, then use the documented same-boot restore with GPIO stopped and recovery
access ready. Recheck PCI configuration without starting a GPIO workload. This
is a controlled rollback diagnostic, not proof that affinity caused the failure.
If PCI remains invalid, stop retries and arrange a deliberate clean shutdown and
power-cycle/recovery inspection. After reboot, never replay the old journal.
Verify PCI health before any new isolation apply and immediately afterward,
before GPIO output; preserve both observations to narrow the failure interval.

### Mixing normal-user and sudo runs: benchmark lock

GPIO hardware requires permissions, but running the collector as root is not
otherwise required. Earlier successful non-root commands can still be used after
root applied the separate runtime isolation settings; affinity settings persist
until restored or independently changed.

The benchmark keeps `/tmp/rpi-hub75-rio-scope.lock` for mutual exclusion between
instances. Existing files are now opened read-only **without O_CREAT**, then
exclusively locked. On Linux local filesystems an exclusive flock does not need
a writable descriptor. This avoids `fs.protected_regular=2` rejecting a sudo
O_CREAT reopen of an existing user-owned file in sticky `/tmp`.

Only a missing file is created with O_EXCL; new empty lock files get mode 0644,
even under a restrictive umask, so root-first and user-first runs can share them.
Existing permissions/content are never changed. A legacy root-owned 0600 file
can still require root; the tool does not silently relax it. Symlinks, hardlinks,
nonregular objects and group/world-writable lock files are rejected. Creation
races are bounded; a held lock remains a hard refusal. No alternate lock path,
automatic unlink, ownership change or sysctl relaxation is used. Do not delete a
lock file to clear an error: another process may still hold its inode. The lock
is advisory and does not stop unrelated PIO tools or a malicious file owner.

### Mapping, electrical settings and store paths

- Hardware opens `/dev/gpiomem0` with `O_SYNC` and maps exactly 0x30000 bytes at
  offset zero, the RP1 IO_BANK0/RIO/PADS window. The IO/RIO/PAD byte offsets match
  the production code's RP1 register layout; the production code's much larger
  mmap and offset arguments are not copied. There is no `/dev/mem` fallback or
  arbitrary physical-address override. Only aarch64 Pi 5 hardware is accepted.
- Selected muxes are changed to RIO function 5; output latch values are initialized
  with OE high before RIO output enables are set. Pad values are logged and
  **not written by default**. `--enable-pad-output` permits clearing/restoring only
  owned OD bits, with readback checks; it never selects new slew, drive or pull
  settings. This avoids silently combining the production clock/latch slew bug
  fix with a scheduler experiment. Any electrical retuning must be a separate
  reviewed/scoped experiment, not an inferred fix here.
- Default `--store xor`: low-state XOR deltas are precomputed, then each pixel
  uses two separate volatile writes (delta, CLK bit) to the RIO XOR alias. Only
  owned bits can change, including at row/cycle boundaries; GPIO0/1 are untouched.
- Optional `--store out`: precompute low states, then write low and low|CLK to
  the full Out latch, like the production pixel store pair. Unowned latch bits
  are snapshotted and included, **not zeroed**, but concurrent changes to those
  bits would still be overwritten. Hardware additionally requires
  `--exclusive-bank`, and rejects any pre-enabled RIO output anywhere in the bank.
  The acknowledgement means ALL other bank users are quiescent, including GPIO0/1.
  Do not use this variant if you cannot establish that exclusivity.
- Protocol rows clear CLK before LAT, with configurable `--delay-nops` loop counts
  before, during and after LAT (default 8). These are uncalibrated instruction
  delays, not nanosecond guarantees or production `SLOW2` equivalence. OE stays high.
- Every cycle ends by SETting OE, CLRaring other owned bits, executing `dsb sy`,
  and reading back Out in the same RP1 block. Latch readback is checked; the
  read-after-write is a posted-write completion experiment. Neither a CPU barrier
  alone nor output-latch readback establishes physical pin setup/hold. Scope it.
- On normal exit/cooperative stop/error cleanup, pins are blanked, RIO enables
  disabled, owned latch bits restored and prior mux/output enables restored with
  readback checks. Any explicitly cleared pad OD bits are restored without
  clobbering other pad settings; unowned bits are not retuned. Cleanup releases OE to
  its prior configuration, which can be high impedance: keep panels disconnected.
  SIGKILL, panic and hardware failure cannot guarantee cleanup/blanking.

Keep store path, mode, geometry, delays, pad values, binary, sampling, completion
read, frequency and load unchanged within a scheduler comparison. XOR vs Out is a
**separate** A/B variable. This prepared workload is not the original production
hot loop; do not claim direct production improvement from its rate.

## Scheduling, signals and counters

`--scheduler other` is the default. Optional `--scheduler fifo --priority 80`
requires permission, a pinned emitter and a distinct housekeeping/sigwait CPU.
There is no priority fallback, DEADLINE admission, `sched_yield`, absolute-period
service sleep, global governor tuning or RT-throttle disabling here. Read the
recorded runtime/period limits: a continuously busy FIFO test can be throttled
mid-cycle. The collector now requires `--ack-rt-throttling` for an intentional
FIFO diagnostic; direct executable users must review the same risk themselves.
DEADLINE and real timed service windows remain later deliverables. PREEMPT_RT
does not imply full dynticks.

For supported runtime isolation controls, use the separate, default-read-only
`rio_isolate.py` tool described in [README-rio-isolate.md](README-rio-isolate.md).
Apply/restore is explicit, root-only and journaled; the collector never applies
those changes automatically. The audited kernel cannot suppress its busy-core
scheduler tick because NO_HZ_FULL is not compiled in.

`--mlock` explicitly requires `mlockall`; failure aborts, not a silent fallback.
Buffers, sample storage and 64 KiB of emitter stack are touched before emission.
This is not a claim of complete production initialization/prefaulting: check the
measured emitter minor/major fault deltas. No allocator, logs, semaphore, time
reads or atomic polling occur per pixel. Cycle-level timestamps and the boundary
acquire/counters do add overhead and cannot bound individual pulse gaps.

SIGINT/SIGTERM are blocked on the emitter and consumed by a SCHED_OTHER `sigwait`
worker pinned to the controller CPU. Stop is observed at a complete workload
boundary, not by a signal handler in the pixel loop. `shutdown_latency_ns` measures
sigwait's userspace request timestamp to emitter acknowledgement after blanking;
it excludes signal dispatch delay BEFORE sigwait runs. Interrupted runs exit 130
and the collector stops; partial runs are not accepted baselines.

JSON stdout includes:

- `workload_cycles`, `issued_pixel_edges`, `words_per_cycle`, wall elapsed time and
  `software_issued_mword_s`. These count issued software work, **not scoped edges**.
- `full_cycles` / `plane_sweeps` are nonzero **only for protocol** mode. For other
  modes the geometry is just a workload grouping, not a physical full scan.
  Rendered content updates are zero; one immutable prepared generation is reused
  and `repeats` counts subsequent workload cycles. No producer is running here.
- `emit_total_ns`, min/max observed emission duration, sampled emission p50/p99,
  `service_total_ns`, boundary count, and fault deltas. Emission duration includes
  row controls/delays plus final blanking/readback, NOT only pixel shifting.
  Service time is measured wall time minus summed emission durations (includes
  boundary acquisition, timestamps, bookkeeping, initial/final boundary work).
- `--sample-every 64` retains one in 64 cycle durations, up to 65536 records. The
  sample count/drop count are explicit; sampled nearest-rank quantiles are not
  all-cycle or pulse-gap quantiles. `--samples NEW.csv` reserves a new file before pin setup
  and writes raw cycle/start-offset/emission durations only AFTER GPIO cleanup.
- `pad_output_enable_requested` records the explicit policy, and
  `pad_output_enable_applied_mask` identifies pads whose OD bits were actually
  cleared (before restoration). RAM runs always report false/zero.
- Deadline misses are `null` because no target period is enforced. Shutdown
  latency is `null` unless a cooperative request was actually observed. Scope
  and RGB verification remain false; dry-run blank readback is `null`, not success.

## Collector artifacts and acceptance gate

By default the collector forces a fresh local release build with the command and
compiler flags logged. `--skip-build --exe PATH` is explicit reuse, recorded in
the manifest. Each new directory contains the exact executable snapshot (the copy
is used for every run), hashes and source snapshots including untracked new
sources, git diff/status, compiler/firmware/kernel/config provenance, cmdline,
CPU isolation masks, governor/frequency/temperature/throttling observations,
RT limits, and per-run command/stdout/stderr/sample CSV. Missing/inaccessible
provenance is recorded explicitly, not treated as zero or a supported capability.

`before/` and `after/` preserve raw interrupt/IPI/softirq tables, kernel logs and
read-only PCI configuration diagnostics (`pci.json`, or an explicit unavailable error).
`isolation.json` adds a read-only IRQ/effective-mask, kernel-thread, steering and
kernel-capability audit (or explicit collection failure); its script/guide are
included in source snapshots. Exact config provenance also records HZ,
NO_HZ_IDLE, CPU_ISOLATION and IRQ_FORCED_THREADING. These observations cannot
establish IRQ-free output from the presence of command-line arguments.
`interrupt-deltas.json` includes per-CPU differences, changed/reset rows and errors.
Thread/priority snapshots also record competing tasks; a large wall-clock pause
can be scheduler starvation by an existing RT task, not necessarily an interrupt.
These snapshots surround process startup/shutdown and collection overhead, not
only the measured emission interval. They cannot attribute individual gaps.
`telemetry.jsonl` samples leader placement/affinity, frequency, temperature and
throttling off-core approximately once/second (plus collection time). Collection
itself can still cause shared-cache/memory/kernel interference. Review all kernel
messages; there is no claim of automated IRQ or kernel-error attribution.

The collector validates workload identity, placement, counters, rate arithmetic
and run status, and stops on failed/interrupted/malformed runs. Collector SIGINT,
SIGTERM or collection errors request child termination and wait for cooperative
cleanup; a child that fails to stop needs manual recovery, never an automatic
SIGKILL. On a nonzero benchmark exit, `summary.csv` and the terminal show the
actual stderr diagnostic even if no JSON summary was produced; full stderr remains
in `stderr.log`. Existing failed results are preserved, not rewritten. `summary.csv`
leaves scope rates/gaps/capture fields blank for manual captures; it never fills
them from software rates. Save matching raw scope screenshots/CSV and record:

- Scope model, sample rate/depth, trigger, timebase, probe/ground and capture time.
- CLK high/low widths, minimum setup/hold when RGB is probed, short-burst rate,
  long-term edge counts, p99 and maximum observed gaps with enough sample depth.
- OE/LAT/address/RGB probes when available; deterministic-sequence comparisons.
- Idle versus named GPU/video/network/storage load, temperature/throttle state,
  camera/photodiode settings and optical results only once visible tests exist.

After repeated 30-second runs, an explicit representative loaded soak is:

```sh
# Launch/describe the intended load separately, never another pin-driving tool.
python3 tests/rio_baseline.py --output --modes protocol --seconds 1800 --repeats 1 \
    --notes 'exact GPU/video/network/storage load and scope setup'
```

A 30-minute software run alone is not the gate: matching raw captures, correct
RGB/setup/hold/edge checks, gap attribution and later optical measurements remain
necessary. A measured maximum is not a guaranteed worst case. Save the baseline
before integrating/replacing the production emitter in the next phase.

## Pin-free validation

`make test-rio-scope` checks the same emission implementation through a software
MMIO trace model: two pixel stores/edge, stable RGB at CLK rise, exact edge/latch
counts, OE high, complete-cycle repeats, default row mapping and preservation of
unowned bits for every mode/path and row size. Python tests exercise RAM modes,
CLI safety guards, exclusive sample files, cooperative signal stop, config/IRQ
parsing, collector provenance, missing-summary diagnostics and malformed-run failure
handling. An additional anonymous-RAM setup/cleanup model executes the actual
benchmark setup functions to verify OD-only changes, zero default pad writes,
unowned-pin preservation, restoration, and recovery after a simulated partial
pad-write/readback failure. Preflight model cases also verify all-ones rejection
before ownership diagnosis, genuine conflicts, alternate functions and OD opt-in,
with zero GPIO stores on every inspection/rejection path. No GPIO discovery/device open occurs in this model. This is not a
hardware/electrical validation. A separate temporary-file lock test covers
existing-file read-only reopen, restrictive umasks, legacy-lock compatibility,
cross-process exclusion, unsafe file types and creation/permission races without
privilege changes, sysctl writes or GPIO access. The Python discovery also runs the isolation
suite using only temporary filesystem controls and fake task affinities, including
journal recovery, identity reuse, driver refusal, effective-mask failure and
conflicting-setting restoration. `make test-rio-isolate` runs that suite alone.

For sanitizers use a distinct directory, e.g.:

```sh
make BUILD=debug BUILDDIR=build/rio-scope-asan \
    EXTRA_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -Werror' test-rio-scope
```
