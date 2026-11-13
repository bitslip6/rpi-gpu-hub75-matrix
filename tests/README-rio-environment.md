# Pi 5 RIO boot and runtime isolation profile

For the project-root installer that **persists runtime setup across boots**, use
[`../configure-system.py`](../configure-system.py) and read
[`README-configure-system.md`](README-configure-system.md). It installs root-owned
helpers and a systemd service with a single `--restore` operation. The lower-level
manual controller described below does not install that service; do not stack
its runtime ownership journals with the persistent installer.

`rio_environment.py` is an opt-in environment controller, **not a GPIO driver or
application launcher**. With no arguments it audits only. It wraps the existing
`rio_isolate.py` runtime implementation; do not stack the two tools' journals.
No new kernel, GPIO output, service installation, scheduling/governor changes,
RT-budget changes, driver reset, power-policy change or reboot is performed.
Run it from this trusted repository; invoking Python from a user-writable checkout
as root requires trusting that checkout. No root boot service is installed.

## Reproduction checklist

These steps reproduce the **isolation configuration**, not a proven minimum set
of causes for the operator's reported improvement. Application/scheduler and
power conditions must also be recorded when comparing results.

1. **Use a Pi 5 with CPUs0–3 online and a checkout containing both
   `tests/rio_environment.py` and `tests/rio_isolate.py`.** Python 3, root access
   for changes, readable RP1 PCI config and `vcgencmd` are needed by the runtime
   health gates. The tested kernel is `6.18.54-v8-16k-pioexp+`; no new kernel was
   built/installed for this isolation change. Do not copy this machine's custom
   PIO kernel/initramfs stanza onto another installation. Other kernel/layouts
   need capability review. Normal Pi 5 automatic `kernel_2712.img` selection is
   accepted only with a matching release banner/config; indirect or mismatched
   firmware layouts require manual review.
2. **Stop GPIO/PIO programs and pending collectors; retain recovery access.**
   The current wrapper refuses **any running irqbalance instance**, including
   an instance with a configured exclusion. Our setup has irqbalance inactive.
   Review that separately on distributions that enable it; the script does not
   stop/disable services. Do not bypass its guard or start a daemon that can undo
   the applied affinities.
3. **One-time boot setup**, preserving your own existing boot arguments:

   ```sh
   make test-rio-environment
   python3 tests/rio_environment.py boot-plan
   sudo python3 tests/rio_environment.py boot-apply --yes
   # Review/save the backup path; then reboot deliberately when safe.
   ```

   The required CPU-placement arguments for this tested kernel/profile are
   `isolcpus=domain,managed_irq,3 irqaffinity=0,1,2`. The file must remain ONE
   line. Do not copy another machine's root PARTUUID or replace its full command
   line. Remove unsupported nohz/RCU arguments only according to kernel capability;
   on this kernel they were ineffective and their removal is not itself a speedup.
   If the editor refuses a different boot layout, review/back up and edit that
   layout deliberately rather than forcing the script or switching kernels.
4. **After EVERY boot, before starting GPIO applications:**

   ```sh
   sudo python3 tests/rio_environment.py apply --yes
   python3 tests/rio_environment.py
   ```

   No automatic startup service is installed. Each boot gets its own journal;
   never copy/replay another boot's journal or stack this with a separate
   `rio_isolate.py apply`. The number of changed objects varies: after the new
   boot IRQ policy, many IRQ restrictions may already be satisfied.
5. **Put only the GPIO emitter on CPU3.** Keep rendering/preparation, collector,
   telemetry and other application work on CPUs0–2. The script never moves
   userspace for you. The existing Pi 5 emitter already pins itself to CPU3;
   the standalone collector defaults to emitter CPU3/controller CPU0. Do not pin
   an entire multithreaded renderer to CPU3 or put it in a CPUs0–2-only cpuset
   that prevents its emitter from selecting CPU3.
6. **Match application and scheduling policy.** The controlled scope A/B used
   `--scheduler other --cpu 3 --controller-cpu 0 --store xor`, with no new governor
   change or RT-throttle disabling. Production's legacy helper is DIFFERENT: it
   attempts FIFO99, memory locking, performance governor and global RT-throttle
   disabling when permitted. Do not treat these as extra required isolation
   tweaks, and do not claim the production and OTHER profiles are equivalent.
7. **Verify, don't count a hardcoded number of changes.** Confirm CPU3 in the
   isolated list, default IRQ mask `7` (CPUs0–2), reviewed work confined by the
   audit, matching current-boot journal settings and healthy RP1/power checks.
   Local/per-CPU kernel work still exists. For GPIO verification follow
   `README-rio-scope.md`: panels disconnected, no competing users, checked scope
   setup, fresh captures and explicit hardware/pad-output opt-ins. Environment
   configuration alone does not drive a waveform.

No custom kernel build, new module, full-dynticks support, forced threaded IRQs,
overclock/voltage change, new electrical pad tuning, disabled thermal protection
or `fs.protected_regular` change is part of this recipe. The clock loop and
production emitter were not rewritten for it. Full-dynticks is unavailable on
the tested kernel, not a required feature someone should try to enable by merely
adding an ignored boot argument.

## What changes

For the currently selected `kernel-pioexp.img` / `6.18.54-v8-16k-pioexp+`:

- `isolcpus=3` becomes `isolcpus=domain,managed_irq,3`.
- `irqaffinity=0,1,2` sets the default IRQ allocation affinity at boot.
- Unsupported `nohz_full=3` and `rcu_nocbs=3` are removed. This kernel explicitly
  disables NO_HZ_FULL and RCU_EXPERT, making RCU_NOCB_CPU unselectable/absent.
  The script records that dependency-based conclusion rather than inferring a
  disabled capability from an arbitrary missing config key.
- Root filesystem, serial console, regulatory and PIO settings are preserved.
  Token order outside the four managed keys is preserved; whitespace is normalized
  to a single line. `config.txt`, image selection, image and initramfs are unchanged.

`managed_irq` is best effort for eligible managed IRQs whose device masks include
housekeeping CPUs. It does not move every IRQ. Domain isolation is not a security
boundary or a ban on explicitly pinning work to CPU3. Runtime restrictions still
matter for existing eligible IRQs, reviewed kernel threads, unbound workqueues and
network steering. Disabled steering stays disabled; userspace is not migrated.

**This does not provide full dynticks.** A busy CPU3 can still take the local timer
tick, hard interrupts and IPIs; RP1/interconnect stalls are also possible. Existing
CSV data plus the operator's report of rarer high-pulse triggers justify an
experimental profile, not a guaranteed pulse-width bound or RGB acceptance.

## Inspect first — no changes

```sh
make test-rio-environment       # temporary boot files / fake task affinities only
python3 tests/rio_environment.py
python3 tests/rio_environment.py boot-plan
```

Boot edits are restricted to Pi 5, a simple explicit image selection (or normal
`kernel_2712.img` automatic selection) matching the running release, a readable
matching kernel config and an uncomplicated firmware configuration. Includes, custom command-line selection, prefix/tryboot
indirection, unknown filters, quoted/multiline command lines and unsafe files
are refused rather than guessed. Review manually if your layout differs.

## Stage the next boot — explicit global change

Stop GPIO/PIO applications and pending collectors first. Keep recovery/console
access. Panels stay disconnected during scope work.

```sh
sudo python3 tests/rio_environment.py boot-apply --yes
```

The JSON output identifies a NEW private backup under
`/var/lib/rpihub75/boot/<UTC timestamp>/`. It contains:

- `journal.json`: original/proposed command line, original firmware config,
  kernel/config hashes, capability evidence and transaction phase;
- `cmdline.txt.before`, `cmdline.txt.after`, `config.txt.before`: plain-text copies
  for recovery, written and fsynced before replacing the live command line.

Only `/boot/firmware/cmdline.txt` is replaced, atomically within its filesystem,
with durable write-ahead backup and readback verification. Ordinary errors/signals
attempt rollback; SIGKILL, filesystem failure and power loss cannot guarantee
cleanup or FAT crash consistency. Retain the backup outside the boot partition.
A repeated request for an already matching command line makes no new backup or
change. **No reboot happens.** Boot policy becomes active only after an
operator-controlled reboot. No alternative installed kernel has been selected.

To undo only this staged boot edit (also usable after reboot):

```sh
sudo python3 tests/rio_environment.py boot-restore --yes \
  --state /var/lib/rpihub75/boot/<exact-backup-directory>
```

Restore refuses changed firmware config/image or a third-party command line.
Unlike runtime journals, boot backups intentionally work across boots, but do
not change live affinities. If the system cannot boot, use recovery media to
restore `cmdline.txt.before` to the firmware partition's `cmdline.txt`; do not
blindly replace unrelated configuration files.

## Runtime — once per boot, after health checks

```sh
sudo python3 tests/rio_environment.py apply --yes
```

The default state is
`/var/lib/rpihub75/runtime/<CURRENT-BOOT-UUID>-cpu3/`. After reboot this is a new
journal, never a replay of an older boot. Explicit `--state /root/private/NEW`
is available for a fresh same-boot experiment; its parent must be private and
trusted. Save the returned state path if overriding the default.

The controller:

1. Requires a clear root-readable process/FD/maps scan, including pending known
   collectors even when their controller runs off CPU3. This cannot detect every
   application, race, or custom hardware user; the operator must ensure exclusivity.
2. Reads the exact RP1 endpoint's first 64 PCI config bytes, checking vendor/device
   and normal header. All-ones, short, missing or invalid config is not healthy.
   No MMIO or GPIO registers are accessed. Firmware active throttle bits must be
   clear; historical bits are recorded, not mistaken for current undervoltage.
3. Calls the existing narrowing-only, identity-checked isolation transaction.
   Any active irqbalance instance is refused, even with a configured exclusion;
   separate review is required and the daemon is not stopped by this tool.
4. Verifies every journaled identity/requested setting and nonempty effective IRQ
   subset. Checks RP1/power immediately afterward and again after one second.
   Any partial apply or failed health gate attempts guarded rollback of the NEW
   transaction, with conflicts retained in its journal rather than forced.
5. Leaves that verified profile active. A repeat apply verifies it without
   overwriting saved originals. Drift/new movable work requires review and a
   restore/fresh journal. A fully confined but unowned state is reported without
   inventing originals or claiming ownership.

Normal-user scope collection is still separate and requires its own hardware
opt-in. Do not invoke environment apply/restore during a pending or active
collector. Do not run GPIO from the environment script.

After GPIO stops, restore this boot's runtime changes with:

```sh
sudo python3 tests/rio_environment.py restore --yes
# If apply used --state, restore must use that exact --state too.
```

Restore checks boot, topology, identities and nonconflicting current settings,
then independently verifies saved originals. It deliberately does not require
healthy PCI first: CPU isolation rollback must remain available during a hardware
fault, but it cannot repair an inaccessible endpoint. Do not reset/unbind RP1 or
force GPIO takeover. A restored journal is preserved and cannot be reapplied;
use a NEW explicit state for another same-boot experiment. Runtime changes vanish
on reboot, and old runtime journals must NEVER be restored after reboot.

## Important production limitation

This script does not launch or change the production renderer. The existing
`enable_rt_fifo_and_lock_mem()` production helper still attempts FIFO99,
performance governor and RT-throttle disabling. Launching that path is **not**
the tested OTHER profile even if isolation was applied. Continue using OTHER in
the standalone baseline; production runtime-policy separation remains a separate
code change. Do not infer pulse-timing guarantees from software throughput or IRQ
mask changes. Physical OE, RGB setup/hold and full protocol validation remain open.
