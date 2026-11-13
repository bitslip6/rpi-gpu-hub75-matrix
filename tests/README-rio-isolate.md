# Reversible RIO CPU isolation controls

`rio_isolate.py` implements the runtime portion of the CPU3 isolation proposal
in `RIO_CPU3_ISOLATION_AUDIT.md`. **Default operation is read-only.** It does not
access GPIO devices and must not be confused with the scope benchmark.

For automatic persistent setup and unified restore, see
[`README-configure-system.md`](README-configure-system.md) and the project-root
`configure-system.py`. Do not stack independent runtime journals.

For the backed-up Pi 5 boot profile and health-checked runtime wrapper, see
[`README-rio-environment.md`](README-rio-environment.md). That separate controller
can explicitly edit the next-boot command line; this lower-level tool still does
not edit boot files. Do not stack journals from the two entry points.

## What is implemented

- Audit exact running-kernel features, online/isolated CPUs, full-dynticks runtime membership,
  RT budget, IRQ requested/effective masks and available debug flags, kernel and
  userspace thread placement, global workqueues and network steering controls.
- Narrow eligible default/device IRQ, global unbound-workqueue, supported
  watchdog CPU, and nonzero RPS/XPS masks to their existing overlap with the
  housekeeping set. Never enable disabled steering or broaden an affinity mask.
- Narrow reviewed movable kernel-thread classes (RCU GP/task threads, watchdogd,
  selected filesystem/memory/display/helper threads). The reviewed expression is
  in `REVIEWED_THREADS` in the source. Also move eligible IRQ-thread work for
  read-only cascaded IRQ controls (e.g. HDMI), only with readable debug evidence
  excluding per-CPU/managed/non-balancing IRQs and a verified IRQ binding. This
  does not move the primary hard IRQ. Userspace is not automatically moved.
- Skip kernel-required/no-setaffinity/per-CPU tasks, unreviewed worker classes,
  read-only/managed/non-balancing primary IRQs and inactive IRQs. Threads of
  writable IRQs follow IRQ affinity; the read-only cascaded-thread fallback above
  has separate identity/eligibility checks and is never used for managed IRQs.
- Verify requested readback and, for IRQs, a nonempty effective affinity entirely
  inside the requested set. A driver refusal or ineffective IRQ restriction is
  recorded, and any corresponding change is rolled back locally.
- Save a private, durable write-ahead journal **before every change**, including
  pending operations. Restore only same-boot, same-online-CPU, same-identity
  objects that still have the setting this tool installed. Never force a restore
  over another actor's different setting.
- Integrate a complete read-only `isolation.json` audit into each collector
  before/after snapshot. Collector captures this script and guide with source
  hashes. Exact-kernel provenance now also includes HZ, NO_HZ_IDLE, CPU_ISOLATION
  and IRQ_FORCED_THREADING.
- Require explicit `--ack-rt-throttling` for collector FIFO experiments. It does
  not alter the RT runtime limit or turn a free-running loop into a service-window
  scheduler. Default OTHER runs are unaffected.

No boot edits, service stops, cpuset changes, kernel/module installation,
frequency/thermal changes, scheduler changes or RT-throttle disabling are done.

## 1. Run tests and review the plan

```bash
make test-rio-isolate        # temporary filesystem + fake affinities only
make test-rio-scope          # all isolation/Python tests + RAM/register models
python3 tests/rio_isolate.py
python3 tests/rio_isolate.py audit --json --report build/rio-isolation-before.json
```

The report filename must be new. `--cpu 3` is the default. Housekeeping defaults
to the other online CPUs; `--housekeeping 0-2` can select it explicitly. The
selection must be nonempty, online and exclude the output CPU. The tool does not
bring CPUs online/offline. A dry-run summary lists all proposed restrictions;
full JSON records unsupported/skipped controls as well.

Unprivileged audit may not read every debug control. Apply re-audits as root
rather than trusting an unprivileged/stale plan. No CLI fake-root option exists;
test backends are accessible only through the test module.

## 2. Apply explicitly, while the hardware benchmark is stopped

Keep remote recovery available and all GPIO/PIO programs stopped. The tool does
not stop any process for you; it rejects recognized active benchmark names on the
output CPU but cannot detect every possible GPIO user. `--yes` acknowledges the
global affinity experiment and recovery readiness.

Use a **new state directory**, with an existing trusted parent:

```bash
STATE="$PWD/build/rio-isolation-cpu3"
sudo python3 tests/rio_isolate.py apply --cpu 3 --housekeeping 0-2 \
  --state "$STATE" --yes
sudo python3 tests/rio_isolate.py audit --cpu 3 --json \
  --report "$PWD/build/rio-isolation-after.json"
```

The state directory is mode 0700, journal 0600, owned by the applying user
(normally root). It contains the original audit, each original/requested value,
operation status, refusal details and an after audit. Restoration also records
an after-restore audit. A global advisory lock
excludes concurrent tool apply/restore operations, not unrelated system tools.

Exit status:

- **0:** attempted operations verified (or successful restore).
- **2 from apply:** partial result; successful restrictions remain installed,
  while refusals/ineffective controls are recorded. Inspect the journal before
  benchmarking. Exit 2 from restore means some items could not be restored.
- **1/130:** failure/interruption; ordinary transactional failures attempt
  rollback, but always inspect the saved state. Parser errors also use exit 2.

```bash
sudo less "$STATE/journal.json"
```

Never interpret a partial result as successful full isolation. Known unsupported
controls omitted from the plan remain explicit skips; a zero exit code does not
mean all kernel work was removable. Do not reuse or delete the state directory
before restoration. Do not stack unrelated affinity tools/irqbalance changes,
CPU hotplug, driver reloads or cpuset edits during the experiment.

## 3. Collect the same hardware baseline manually

Only after reviewing apply/readback, with panels disconnected, no competing GPIO
users, scope/probe connections checked and stable power:

```bash
python3 tests/rio_baseline.py \
  --output --enable-pad-output --modes clock --seconds 30 --repeats 3 \
  --notes 'after runtime isolation; state build/rio-isolation-cpu3; OTHER unchanged'
```

The collector can run as the same normal user as the earlier baseline; root is
needed for the isolation apply/restore, not inherently for collection. Updated
benchmark locking also supports sudo reopening a user-owned lock without changing
`fs.protected_regular`. Do not delete the shared lock or reapply the same state
just because collection failed before emission.

Leave emitter stores, pad settings, frequency policy and scheduler unchanged for
this A/B comparison. Save actual scope BMP/CSV files, not only filenames in the
notes. Compare raw physical high/low gaps, per-source IRQ/IPI/softirq deltas and
scheduler activity, not just mean Mword/s. Confirm physical OE high separately.
A single-channel clock capture does not validate RGB/setup/hold or optical output.

If preflight instead returns all-ones RP1 CTRL/PAD or PCI diagnostics show
`Unknown header type 7f`, **stop GPIO collection**. Affinity readback success is
not a hardware-health check. Preserve the failed results and private state, then
use the same-boot restore below with all GPIO users stopped and recovery access
ready. Recheck `lspci -s 0002:01:00.0 -vv` without launching GPIO output. If PCI
remains invalid, arrange deliberate shutdown/power-cycle recovery; do not reset
or unbind RP1, force GPIO takeover, or replay the journal after reboot. A future
new isolation experiment should record sane PCI configuration immediately before
apply and check it immediately afterward, before any GPIO workload. This can
narrow the failure interval; neither temporal proximity nor `throttled=0x0`
alone proves a cause. See the scope guide's invalid-MMIO troubleshooting section.

New runs contain `before/isolation.json` and `after/isolation.json` with audit or
explicit collection errors. These are off-core snapshots, not a per-pulse trace;
thread affinity/presence is not proof of actual runtime. Diagnostic tracing must
be a separately labeled experiment because it perturbs timing.

## 4. Restore after testing

Stop the GPIO benchmark first, then use the exact state directory from apply:

```bash
sudo python3 tests/rio_isolate.py restore --state "$STATE" --yes
python3 tests/rio_isolate.py audit
```

Restore is idempotent for completed entries. Reboot/CPU-online changes or an
identity change prevent stale writes. Task start time, TID/TGID/name and kernel
flags guard task reuse; IRQ proc inode/chip/hwirq/actions/type guard IRQ reuse.
Network steering restoration checks interface ifindex and queue inode as well.
If an object exits/disappears or another actor changes a setting, restore reports
it instead of guessing or forcing. Concurrent lifecycle changes still cannot be
made atomic with numeric-TID affinity calls: run the experiment in a quiescent
configuration and inspect conflicts. New IRQs/tasks created after apply were not
part of the original snapshot and are not given invented original values.

A SIGKILL, power loss or filesystem failure cannot guarantee automatic cleanup.
Pending write-ahead records let an explicit same-boot restore recover a write
that completed before its success status was saved. Do not replay a journal after
reboot. Reboot normally discards these runtime settings but does not replace a
verification/recovery plan for drivers/services that independently reconfigure.

## Remaining improvements that require separate decisions

### Full dynticks and RCU callback offload

The audited kernel lacks NO_HZ_FULL and rejects `nohz_full=3 rcu_nocbs=3`.
The architecture timer supplied about 99.4% of counted CPU3 IRQ/IPIs in the latest
baseline. Runtime device affinity cannot move that per-CPU scheduler tick.
Moving RCU GP threads is not RCU callback offload.

Check exact config/boot logs of any **already available** alternative kernel
before selecting it. Even compiled-in support needs runtime CPU membership and
actual timer-delta measurements; mask membership alone does not prove no ticks.
PREEMPT_RT or an RT module directory is not evidence of
NO_HZ_FULL or a bootable matching image. No kernel build is performed here.
Without a suitable available kernel, retain this limitation honestly.

### Durable boot placement

After backing up the active boot files and arranging recovery, separate reviewed
boot experiments can consider `irqaffinity=0-2` and supported
`isolcpus=domain,managed_irq,3`. Managed-IRQ isolation is best effort. Do not append
duplicate/unsupported options, automatically edit boot files or reboot from the
tool. Full-dynticks/RCU arguments are meaningful only on a supporting kernel.

### cpusets and a dedicated emitter

PID1 affinity is already CPUs0-2 on this system. Strong cpuset isolation requires
a separate emitter subtree allowing CPU3 before ordinary work is restricted.
Blindly setting `AllowedCPUs=0-2` on user.slice would also block this user-launched
benchmark from selecting CPU3. Production preparer/emitter integration remains a
separate plan phase; the production emitter and installed API are unchanged.

### Scheduler/service windows

Keep OTHER for the first affinity comparison. This free-running benchmark can
hit the default 950000/1000000 us RT budget under FIFO, causing large pauses;
FIFO does not mask hard IRQs. An intentional collector diagnostic requires
`--scheduler fifo --priority 80 --ack-rt-throttling` and appropriate privileges,
but it is **not** the next default recommendation. First design/review a blanked
blocking service window/budget and recovery. Never silently disable RT throttling
or thermal/watchdog protection to chase sub-microsecond scope gaps.
