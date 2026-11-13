# RP1 PIO baseline collection (no kernel changes)

Run from the project root. No replacement kernel/module, watermark writes,
cyclic DMA, or PIO instruction changes are included in this stage.

## Automated pin-free baseline

```sh
python3 tests/pio_baseline.py
```

Builds the release executable locally and runs 30 seconds per case, three repeats
of heavy/default and forced-light DMA at dividers 8 and 4 with 256 KiB chunks.
Runs sequentially; do not run another GPIO/PIO application concurrently.
Results go to a new timestamped directory in `tests/results/`.

Each directory contains build output, source and executable snapshots/hashes,
worktree diff/status, kernel/module/firmware provenance where accessible,
kernel config and device tree where accessible, complete kernel logs after each
run, log deltas, per-run output/command lines, and `summary.csv` including CPU time.
Missing provenance is explicitly recorded; effective DMA burst programming is
**not verified** by DMACTRL alone. Matching kernel/controller source review is
still required. Requested duration excludes startup/drain; wall time includes them.
CPU times cover the test subprocess, including startup/drain.

The runner stops on test failure, missing summary/DMACTRL, known DMA error messages,
or loss of kernel-log continuity. Inspect *all* new kernel messages even when the
runner does not flag them. Old messages are saved but do not invalidate a new run.
No forced timeout is used: a stuck ioctl needs manual intervention and log review;
interrupted/aborted runs are not valid measurements. Do not blindly rerun a stalled
channel or unload a driver while it is in use.

Optional descriptor-boundary comparison (after the default suite succeeds):

```sh
python3 tests/pio_baseline.py --chunks 64 1024
```

Keep the kernel, PIO program, divider/mode and scope setup unchanged for A/B tests.

## Manual GPIO17 scope baseline

Only the scope is connected on physical pin 11 (GPIO17), referenced to Pi ground.
The panels must remain disconnected. **`--output` still drives all 18 RGB pins,
GPIO17 and OE18**, even though only GPIO17 is probed; ensure they have no conflicting
peripherals. Pins are released to high-impedance NULL mode, not their previous mux.
First complete the pin-free suite above, then run in an interactive terminal:

```sh
python3 tests/pio_baseline.py --output
```

The runner prompts before every run so you can prepare the scope, and afterward
for readings/capture filenames. Save raw screenshots/CSV alongside each run's logs.
Use a 10x probe, scope attenuation set to match, short ground, and record:

- Scope model, sample rate, timebase, trigger, probe/ground setup and capture time.
- Short timebase: pulse width, burst intervals, periodic gaps and worst observed gap.
- Long timebase: pauses and long-term edge rate. If possible, count all rising edges
  over a known capture interval; a scope frequency readout within a burst is not a
  sustained edge count. Label estimates as estimates.
- Minimum/99th/maximum intervals only when capture depth/sample count supports them;
  leave unavailable CSV fields blank rather than inventing distributions.
- CPU time, DMACTRL, submission throughput, and kernel messages from the runner.

The scope on GPIO17 alone **cannot establish RGB setup/hold, missing/repeated or
reordered data, or fresh-word correctness**. Submission Mword/s is not scoped MHz.
This baseline cannot satisfy the final 20 Mword/s success gate by itself.
At divider 4 the three-instruction program has a nominal 16.67 MHz ceiling; at
8 it is 8.33 MHz, assuming the recorded 200 MHz PIO clock. Divider <=3 is required
for eventual 20 MHz testing but is intentionally not added to this baseline matrix.

Before kernel implementation: back up the SD card, establish rollback, acquire
exact matching source/config, and review both the PIO and DMA controller drivers
as specified in `tests/KERNEL_DMA_HANDOFF.md`.
