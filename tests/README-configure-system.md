# Persistent Pi 5 display isolation

The project-root **`configure-system.py`** installs the CPU3 isolation profile and
reapplies its runtime restrictions automatically after boot. It does not emit
GPIO, install/switch kernels, alter pad settings, change governor/scheduler/RT
budgets, disable thermal protection, or reboot.

## Configure

Stop GPIO/PIO applications (including pending collectors), retain recovery access,
and run from a trusted checkout containing the two helper scripts in `tests/`:

```sh
sudo ./configure-system.py --dry-run
sudo ./configure-system.py
# Noninteractive alternative, after reviewing the plan:
# sudo ./configure-system.py --yes
```

The interactive command requires typing `YES`. The dry run performs read-only
preflight; it neither creates state nor changes services. Root is required even
for inspection because the tool must inspect all processes and private state.

Requirements: Pi 5, CPUs0–3 online, systemd, Python 3, `udevadm`, `vcgencmd`,
working RP1 access and a verified kernel configuration. Both the selected explicit
image and the normal Pi 5 `kernel_2712.img` automatic selection are supported only
when their Linux release banner matches the running kernel. Indirect/complex
firmware selections, missing/mismatched config evidence and unsafe files are
refused for manual review. No new kernel build or PIO module is needed.

### Changes made

1. Back up and update `/boot/firmware/cmdline.txt`, preserving unrelated argument
   tokens and keeping it one line. Establish:

   ```text
   isolcpus=domain,managed_irq,3 irqaffinity=0,1,2
   ```

   `nohz_full=3` / `rcu_nocbs=3` are retained/added only when supported by the
   verified kernel. On the tested kernel they are removed because they do not
   work. Full dynticks cannot be enabled just by adding an ignored boot argument.
2. Mask `irqbalance.service` with an owned `/etc/systemd/system` symlink. If its
   normal systemd-managed instance is active, stop it, recording its prior active
   state. A preexisting mask is preserved; a local unit override or unmanaged
   daemon requires manual review. Existing service enablement links are not
   disabled/replaced, so unmasking can restore the original policy later.
3. Install root-owned, non-user-writable copies of the installer and runtime
   helpers under `/usr/libexec/rpihub75-system/` with trusted root-owned ancestors.
   User-owned `/usr/local` is deliberately not used. Firmware commands resolve
   through a sanitized system PATH, not user-writable local directories.
   Boot does **not** execute a
   user-writable checkout or depend on its location, current directory, virtualenv
   or `PYTHONPATH`. Installed sources are hash-checked against private state and
   loaded with isolated Python imports. Moving the checkout is safe after setup.
4. Install and enable `rpihub75-isolation.service`, a systemd oneshot. It waits for
   udev completion, runs on CPUs0–2 and applies supported restrictions before
   `multi-user.target`. It does not launch a display application.
5. Apply the runtime profile immediately through the same checked helper. Every
   later boot gets a fresh runtime ownership journal. Checks include no detected
   GPIO/PIO users, valid exact RP1 PCI config, no active throttle flags, eligible
   narrowing-only changes, identity/value/effective IRQ verification and settled
   health. Partial applies attempt rollback; unmovable/per-CPU tasks are skipped.

Runtime is applied directly during installation; the enabled oneshot itself runs
at subsequent boots (it may therefore show inactive immediately after install).
The tool reports whether a reboot is needed for its boot edits, but never starts
one. CPU3 is for the GPIO emitter; rendering/preparation/collection should remain
on CPUs0–2. Do not pin an entire multithreaded application to CPU3. This tool
never migrates userspace or alters application/cpuset configuration.

For a custom auto-started display service, explicitly order it after the profile:

```ini
[Unit]
Requires=rpihub75-isolation.service
After=rpihub75-isolation.service
```

The installer does not edit arbitrary application units. `Before=multi-user.target`
alone does not serialize every other service wanted by that target. If a GPIO app
starts too early, the isolation hook refuses to adjust a live workload; it does
not kill the app or force a takeover. Later driver/task creation or third-party
policy changes can still require review; a oneshot is not a continuous enforcer.

## Inspect

```sh
sudo ./configure-system.py --status
systemctl is-enabled rpihub75-isolation.service
systemctl status rpihub75-isolation.service
journalctl -u rpihub75-isolation.service -b
# Full read-only live isolation audit from a checkout:
python3 tests/rio_environment.py
```

Private write-ahead backups and ownership records live at
`/var/lib/rpihub75/configure-system/`. They are retained after restoration. A later
configure cycle archives the restored state instead of overwriting originals.
Do not stack manual `rio_environment.py apply` / `rio_isolate.py apply` experiments
with the persistent tool. Existing restrictions can be left in place, but the
installer does not adopt their saved originals or claim ownership of them.

## Restore

With GPIO/PIO applications and collectors stopped:

```sh
sudo ./configure-system.py --restore
# Noninteractive:
# sudo ./configure-system.py --restore --yes
```

If the checkout was moved/deleted, the installed copy can restore itself:

```sh
sudo /usr/libexec/rpihub75-system/configure-system.py --restore
```

Restore disables/removes its startup service, restores ONLY its current-boot owned
runtime changes, restores boot arguments it changed, removes its irqbalance mask
and restarts the prior systemd irqbalance instance if it stopped an active one.
Its original enablement links and any preexisting mask are preserved. Installed
helper copies are removed on successful restoration; backup journals remain.
No shutdown hook replays runtime journals: old-boot journals are evidence only.
Removing boot isolation takes effect at the next operator-controlled reboot.

**Persistent restoration baselines are from before the first configure invocation,
not factory defaults. Runtime originals are from immediately before each boot's
owned apply.** On a machine already manually configured with the earlier tools,
preexisting boot arguments remain; current-boot affinities not owned by this tool
are not undone. Manual runtime settings from an older boot are not recreated.
Restore earlier experiments separately using their own valid backups and
same-boot guards; never replay an old runtime journal after reboot.

Foreign edits, reused identities, changed kernel/image/boot config, unknown
package files and conflicting masks are not overwritten/deleted. Partial restore
returns failure and retains recovery state; resolve the reported conflict and
retry `--restore`. There is no recursive deletion or SIGKILL/power-loss/FAT-crash
cleanup guarantee. Boot/image changes since setup may require manual recovery.
If the Pi cannot boot, restore the saved `boot/cmdline.txt.before` from recovery
media, preserving unrelated firmware settings.

## Scope and production limitations

No FIFO99, performance governor or RT-throttle disabling is added by this tool.
Production's existing `enable_rt_fifo_and_lock_mem()` helper can still change
those independently; a production run is not the controlled OTHER scope profile.
This script restores only its own settings, not later application tuning. Active
throttle flags fail health gates; latched historical flags are recorded, not
misrepresented as currently active or clean history. Check power separately.

The kernel can still interrupt CPU3, and RP1/MMIO stalls remain possible. This
isolation profile is not a guaranteed pulse-width bound, RGB/OE acceptance test
or a kernel/full-dynticks upgrade. See `README-rio-environment.md` for the lower-
level policy and `README-rio-scope.md` for explicitly guarded scope collection.

## Pin-free validation

```sh
make test-configure-system
make test-rio-environment
make test-rio-scope
```

Installer tests inject fake systemd and fake kernel-task affinities, and mutate
only temporary files. They do not install a service, edit the real boot partition,
change live controls or drive GPIO. Live installation/reboot verification is a
separate operator action.
