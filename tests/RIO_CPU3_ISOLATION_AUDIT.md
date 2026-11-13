# CPU3 isolation audit and proposed next steps

Read-only follow-up to `RIO_USERSPACE_PLAN.md` and the clock-only scope captures.
This is an audit/proposal, not an applied tuning configuration. Its runtime
controls are now implemented in `rio_isolate.py`; see
[README-rio-isolate.md](README-rio-isolate.md) for explicit apply/restore steps.
No GPIO output, live IRQ/thread affinity change, service stop, boot edit, reboot
or RT-limit change was performed by the assistant.

## Confirmed current state

Running kernel: `6.18.54-v8-16k-pioexp+`, standard preemptible kernel, not
PREEMPT_RT. Exact `/boot/config-6.18.54-v8-16k-pioexp+` agrees with the captured
baseline configuration:

```text
CONFIG_NO_HZ_IDLE=y
# CONFIG_NO_HZ_FULL is not set
CONFIG_HZ_250=y
CONFIG_HZ=250
CONFIG_IRQ_FORCED_THREADING=y
# CONFIG_PREEMPT_RT is not set
# CONFIG_RCU_EXPERT is not set
```

The command line includes `isolcpus=3 nohz_full=3 rcu_nocbs=3`, but boot logs say:

```text
Housekeeping: nohz unsupported. Build with CONFIG_NO_HZ_FULL
Unknown kernel command line parameters "nohz_full=3 rcu_nocbs=3", will be passed to user space.
```

Therefore the busy output core is NOT fully tickless and RCU callback offload is
NOT established by those parameters. `/sys/devices/system/cpu/nohz_full` is
absent. Adding the same arguments again cannot enable a missing kernel feature.

Other observations:

| Item | Observed | Meaning |
| --- | --- | --- |
| Scheduler isolation | `/sys/devices/system/cpu/isolated` = `3` | Domain isolation is present, not complete kernel isolation. |
| PID 1 affinity | `0-2` | Ordinary descendant userspace is already directed off CPU3. |
| Current userspace thread audit | None allowed CPU3 at audit time | No current userspace competitor found; this is not a during-emission trace. |
| Global unbound workqueue mask | `7`; requested `7`, isolated `8` | CPUs0-2 already selected globally. Individual queue mask `f` does not by itself prove execution is allowed on CPU3; the global restriction also matters. |
| Default IRQ mask | `f` | New eligible IRQs can still allow CPU3. |
| Existing IRQ affinity | Many configured `0-3`; most active device counts on CPU0 | Tighten eligible masks proactively; little device-IRQ traffic reached CPU3 in these idle baselines. |
| irqbalance | Inactive | Not currently undoing IRQ placements. |
| Network RPS | `0` on eth0/wlan0/lo RX queues | Software receive steering is disabled, not explicitly steering to CPU3. Receive processing still depends on its IRQ and execution context. |
| cgroup controllers | `cpuset cpu io pids` | cpuset partitioning is available for investigation; hierarchy and emitter placement must be designed before applying restrictions. |
| RT budget | 950000 us per 1000000 us | A continuously busy FIFO run can throttle; do not blindly replace OTHER with FIFO or disable the safety limit. |
| Watchdog CPU mask | Not exposed | No supported watchdog_cpumask runtime setting to apply here. Do not disable safety mechanisms as a substitute. |

In the latest 30-second collection window CPU3 received 7,711 counted IRQ/IPIs,
of which 7,665 were `arch_timer` (about 99.4%). These totals include snapshot
window overhead, not only the benchmark's emission interval. The architecture
timer is per-CPU; writing a device IRQ affinity mask cannot transfer CPU3's local
scheduler tick to CPU0.

Several kernel threads still allow `0-3`, including RCU threads, HDMI IRQ threads,
watchdogd, filesystem and memory-management threads. Per-CPU workers also exist
on CPU3. An allowed mask or a listed sleeping/idle thread is not proof that it
ran during an observed physical pulse stretch.

## Proposed staged changes

Each stage needs a saved pre-change state, explicit operator opt-in, readback,
a restoration path and a separate scope/counter comparison. Keep panels
disconnected, housekeeping CPUs available and remote recovery working.

### 1. Reversible runtime affinity cleanup

- Change the default IRQ mask from `f` to `7` for future eligible IRQs.
- Restrict reviewed movable device IRQs to CPUs0-2; inspect both requested and
  effective masks. Skip/report per-CPU, managed or driver-refused changes.
  Do not blanket-write every IRQ and silently ignore errors.
- Constrain reviewed movable kernel threads and IRQ threads to CPUs0-2 where
  supported. Moving RCU GP threads does not offload all per-CPU RCU callbacks.
- Leave the already-restricted global unbound workqueue mask alone. Identify
  bound workqueue activity rather than trying to move every `kworker/3:*`.
- Preserve irqbalance's existing state. If enabled later, configure exclusion
  explicitly rather than allowing it to undo the chosen IRQ placements.

Do not try to move `migration/3`, `cpuhp/3`, `ksoftirqd/3`, architecture timers or
other required per-CPU tasks as if they were ordinary movable processes.

### 2. Durable placement and boot policy

After reviewing backup/recovery, candidates include `irqaffinity=0-2` and a
supported `isolcpus=domain,managed_irq,3` policy. Managed IRQ isolation is best
effort and effective placement still needs checking. Boot changes must be a
separate explicit experiment, not automatic library startup behavior.

For cpusets, place the emitter in a separate subtree that actually permits CPU3
before constraining ordinary work to CPUs0-2. Putting `AllowedCPUs=0-2` on
`user.slice` would otherwise prevent the current user-launched benchmark from
pinning itself to CPU3. Ordinary affinity inheritance currently permits that
explicit pinning; a cpuset restriction is a stronger limit.

### 3. Resolve the dominant local-tick limitation

To suppress the periodic scheduler tick while CPU3 runs a single userspace
emitter, an already-available kernel must actually support `CONFIG_NO_HZ_FULL`.
RCU callback offload also needs the corresponding support. Verify its exact
configuration and boot logs, then measure the during-emission timer deltas.
Full dynticks does not suppress every timer, IPI or exceptional event.

The inspected versioned `/boot/config-*` files all lack NO_HZ_FULL. Firmware
kernel8/kernel_2712 images identify as 6.18.54 variants, but their matching
isolation configs were not established by this audit. An RT module directory
exists for `6.18.54-v8-rt+`; that alone is not a verified bootable RT image or
proof of NO_HZ_FULL. No kernel build/install is proposed within Plan A. If no
available kernel supports these features, retain and report that limitation.

`CONFIG_IRQ_FORCED_THREADING=y` permits considering `threadirqs` as a separate
boot experiment on the current kernel. It can thread eligible device IRQ work,
not the local scheduler tick or every primary handler. Given the current
arch_timer dominance, this is not the first expected source of a large gain.

### 4. Scheduler tests only with an explicit budget

A reviewed FIFO priority such as 80 can reduce ordinary task preemption, but
cannot mask local hard IRQs. The current free-running benchmark does not provide
a deliberate blocking service window; a near-100%-busy FIFO variant can hit the
95% RT budget and introduce large pauses (potentially around 50 ms per second).
Use an explicitly designed blanked blocking/budget policy before treating FIFO
as a determinism improvement. Do not silently set `sched_rt_runtime_us=-1`.

## Validation and remaining uncertainty

Compare the same workload, stores and pad policy before/after each isolation
stage. Collect CPU3 per-source IRQ/IPI/softirq deltas, scheduler activity, kernel
warnings and matched raw scope captures. Narrow tracing belongs in separately
labeled diagnostic runs because it adds overhead. A separate emitter address
space can reduce shared-mm/TLB disturbance but cannot eliminate global IPIs or
MMIO/interconnect stalls.

The supplied 100-500 ns pulse stretches are not yet attributed to interrupts.
Moving kernel work is a testable improvement attempt, not a promise to remove
those stretches or guarantee uninterrupted physical clocking.

## Suggested implementation deliverable

A default-read-only audit/dry-run helper, with explicit saved-state `apply` and
`restore` modes for reviewed runtime changes. It must distinguish requested and
effective masks, preserve nonempty housekeeping CPUs, report unsupported controls,
avoid blanket kernel-thread manipulation, and never drive GPIO, reboot, edit boot
files, stop services or change RT throttling automatically. This helper is now
implemented with pin-free tests; only its read-only live audit has been run.
Reviewed fallback IRQ-thread placement also handles non-managed cascaded IRQs
whose primary IRQ affinity cannot be changed, without moving the primary IRQ.
