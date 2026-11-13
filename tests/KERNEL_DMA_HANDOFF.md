# RP1 PIO → DMA kernel investigation: handoff plan

## Goal and hard constraint

Find out whether Raspberry Pi 5's RP1 PIO can sustain **20 million GPIO17 CLK rising edges/s, with one NEW 18-RGB-bit DMA word before every edge**, not just generate a free-running 20 MHz clock. The default HZELLER_HAT pinout uses 18 RGB pins, GPIO17 CLK and GPIO18 OE. Two 18-bit pixel updates cannot fit into one 32-bit DMA word: 20 MHz requires at least **20 Mword/s = 80 MB/s = 76.3 MiB/s**, plus reliable setup/hold and control timing. This is a timing test, not yet a full HUB75 driver.

Current PIO scope test executes `PULL block side 0; OUT PINS,28 side 0; NOP side 1` (three PIO cycles/word). At RP1 `clk_sys` = 200 MHz the instruction limit is `200 / (3 * divider)` MHz, so divider 4 caps *burst* CLK at 16.67 MHz, divider 3 at 22.22 MHz. Reducing the divider cannot increase sustained rate unless DMA keeps up.

## Known facts (verify afresh, not assumptions about the next boot)

- The machine was running `6.18.54-v8-16k+` when this plan was written, **not** the previously installed `+rpt-rpi-2712` kernel. `modinfo rp1_pio` showed an in-tree module `rp1-pio.ko.xz`; `/lib/modules/$(uname -r)/build` was **absent**. Obtain the *exact matching* kernel source/config, module version, 16 KiB page-size configuration, firmware and device tree before building a replacement. The file `/tmp/rp1pio.c` is downloaded **upstream reference only** and must not be assumed to match this booted kernel.
- The independent clock-only program `tests/pio_clock_scope.c` produced ~19.99 MHz on GPIO17 at divider 5, confirming the PIO/GPIO clock path but **not** DMA data bandwidth. The Siglent SDS1102CML scope is 100 MHz/1 GSa/s; use 10× probes, short grounds and check wide timebase for gaps.
- `tests/pio_hub75_scope.c` synchronizes data and clock in one SM. At divider 4: ~10.4 MHz scope long-term clock, eight-clock bursts with pauses (~100 ns and sometimes ~750 ns). At divider 8: ~7.54 MHz and ~100 ns pause every eight clocks. GPIO27 (physical header pin 13) is an example RGB probe, GPIO17 is header pin 11, Pi ground is pin 9. OE18 is held high (blanked); LAT/address are NOT generated.
- After upgrading to 6.18.54, the **default/heavy** channel `PIO_IOC_SM_GET_DMACTRL` returned `0xc0000100` (TX DREQ threshold 0); `--light-dma` via `PIO_IOC_SM_CONFIG_XFER_V2` with FORCE_LIGHT returned `0xc0000104` (threshold 4). Reference driver `drivers/misc/rp1-pio.c` computes `RP1_PIO_FIFO_DEPTH - config.dst_maxburst`, with joined TX FIFO depth 8; heavy advertises burst 8 and light burst 4. **Check the live kernel source and actual DMA descriptor programming** before treating this as a proven cause. The 8-clock periodicity is consistent with empty-FIFO DREQ refill, but correlation is not proof.
- Pin-free `pio_hub75_scope` at divider 8 and 4: heavy ~7.6 and ~10.5 Mword/s, respectively; forced light ~6.5 Mword/s at both dividers. Light refills earlier but has a lower measured ceiling. Benchmark output is submissions, not a pin measurement. No sustained 20 Mword/s has been demonstrated.
- Repeated interrupted runs have logged `dma2chan0 is non-idle`, `failed to stop`, and `rp1-pio: DMA bounce interrupted`. Both userspace DMA tests now give queued buffers time to drain on normal exit, but kernel stalls remain a possible separate issue. Check `dmesg` on every run; never treat a stalled/aborted run as data.
- **Worktree state:** `Makefile` is modified and `tests/` is currently untracked. Preserve these files/changes; do not inadvertently build against an older installed copy of the test. No unrelated shader/source changes are part of this investigation.

## Safety and setup (stage 0)

1. **Disconnect the HUB75 panels** before any PIO test. `--output` only drives 18 RGB pins, CLK17 and OE18 high; it does not generate valid row/latch signals. Plain `pio_hub75_scope` without `--output` muxes **no GPIO**. Never unbind/reload RP1 PIO/DMA while an application is using it. Do not change a live DMACTRL watermark independently of DMA burst length.
2. Back up the bootable SD/image and record a working kernel+modules/firmware boot option before loading any modified driver. Prefer a spare SD card or second Pi. Do not replace the running module in place if dependency/state makes that unsafe. Reboot between kernel/module variants and verify recovery.
3. Save provenance with every data set: `uname -a`, `modinfo rp1_pio`, `cat /sys/kernel/debug/clk/clk_sys/clk_rate`, `dmesg`, full source commit/diff, `/proc/config.gz` if available, device tree DMA channel names/IDs, compiler version, PIO test executable revision, scope setup, capture timestamp. Locate/build against **6.18.54-v8-16k+** source and config or choose a fully matching alternative kernel+modules; do not mix the older 6.18.50 headers with this kernel.

## Baseline (stage 1; do before patching)

Build with `make pio-hub75-scope`. Run pin-free for each divider/mode, 30 s, repeat at least three times; then repeat GPIO tests with panels disconnected and a scope:

```sh
./build/release/pio_hub75_scope 30 8 256
./build/release/pio_hub75_scope 30 4 256
./build/release/pio_hub75_scope 30 8 256 --light-dma
./build/release/pio_hub75_scope 30 4 256 --light-dma
# ONLY after panel disconnection, repeat with --output
```

Measure **total rising edges over elapsed time**, inter-edge interval distribution, long-tail/maximum gap, clock duty/pulse width, data stability before each rising edge (CH2 GPIO27), whether words/edges are ever missed or repeated, min/max throughput across runs, CPU use, and `dmesg`. Capture short and long scope timebases. Keep the 256 KiB chunk and PIO program constant for A/B comparisons. Separately try 64 KiB and 1024 KiB to discriminate descriptor-boundary pauses (a 256 KiB buffer at 10 Mword/s lasts ~26 ms) from **every-eight-word** gaps. Test no-pins first whenever changing code.

## Kernel experiment (stage 2): heavy DMA, smaller burst and EARLIER DREQ together

In the matching `drivers/misc/rp1-pio.c`, trace the TX path around `rp1_pio_sm_config_xfer_internal`: DMA channel selection (`txN` heavy versus `txNl` light), `dma_get_slave_caps()`, TX `config.dst_maxburst`, `dma->burst_bytes`, `dmaengine_slave_config()`, and the `SM_SET_DMACTRL` call. Inspect the DMA controller driver too: verify that `dst_maxburst=4` really programs a four-word burst, not merely metadata. Verify what low DMACTRL bits mean for RP1 firmware/hardware.

Add a **scoped, default-OFF** kernel option for *this PIO TX heavy channel* to cap the effective burst to **4 words**, and use that same effective burst value in ALL dependent fields: `config.dst_maxburst=4`, `dma->burst_bytes=4*sizeof(uint32_t)`, and DREQ threshold `8-4=4`. Preserve unrelated DMACTRL bits; ensure descriptor preparation and bounce recovery use the effective size too. Keep baseline heavy-8/threshold-0 selectable for an A/B comparison on the *same kernel*. Do not merely use `PIO_IOC_SM_SET_DMACTRL` to raise a heavy-8 channel's watermark: an eight-word burst into a half-full eight-word FIFO could overrun it or corrupt output. Confirm caps, alignment, and valid burst sizes before extending to heavy-2/threshold-6.

Build the matching module/kernel via that tree's documented procedure, install with rollback, reboot and re-run stage 1 unchanged. Record DMACTRL and verify on the scope that early refill removes the 8-word gaps **without missing/reordering data**. Check whether long-tail pauses/throughput improve or whether extra, smaller bursts increase arbitration overhead. Collect driver/engine trace counters for DREQ/IRQ/descriptor completion/FIFO stall if available; introduce targeted tracepoints only if measurements are ambiguous. Do not confuse the PIO *burst instruction frequency* with sustained output.

## Follow-ups only if the paired burst/watermark test warrants them (stage 3)

- If gaps persist, profile the RP1 DMA engine and RP1 PIO TX FIFO events. Correlate FIFO occupancy/DREQ and DMA bursts with GPIO17 edge gaps. Investigate kernel bounce-buffer copy, DMA descriptor preparation, IRQ latency, PCIe/RP1 contention, and whether DMA has enough pending descriptors. Record *which* stage causes ~100 ns versus longer gaps.
- On 6.18.54, `CONFIG_XFER_V2` works for forced light. Test **cyclic TX DMA** (`RP1_PIO_SM_CONFIG_XFER_FL_DMA_CYCLE`, at least three TX periods) in a *separate opt-in pin-free test first*. Review `rp1_pio_sm_tx_user_cyclic()` underflow handling; do not mistake a repeating stale test pattern for valid fresh data. Cyclic DMA may remove descriptor-boundary overhead, but cannot alone solve an eight-word FIFO DREQ gap. Ensure cyclic DMA is terminated cleanly; document kernel errors.
- If userspace bounce copies/descriptor churn dominate, prototype a **bounded kernel-resident, DMA-safe ring** or another supported zero/low-copy handoff, with strict pin-safe stop and watchdog behavior. Compare identical scope/program/data tests to baseline; no direct unreviewed `/dev/mem` DMA register pokes.
- A faster PIO program (`OUT + NOP side-set`, autopull) removes one instruction per word, but cannot overcome a <20 Mword/s DMA feed. Benchmark only after the DMA path is stable, and scope setup/hold plus pauses; do not present free-running 20 MHz CLK as proof of 20 MHz RGB transfer.

## Decision gates and handoff output

Each variant gets a table of: kernel/source revision, TX DMA mode, configured **and verified** burst and DREQ watermark, divider/PIO instruction count, chunk size, no-pin submission Mword/s, scoped **long-term** CLK MHz, minimum/99th/max edge interval or gap distribution, data errors, and kernel logs. Attach raw scope CSV/screenshots and relevant driver diff.

**Success:** reproducible ≥20 million fresh words/s and ≥20 million corresponding GPIO17 rising edges/s over ≥30 s, across multiple runs, no periodic starvation gaps/incorrect data/driver errors, safe OE blanking and acceptable setup/hold. With the present three-instruction program, divider must be ≤3 for 20 MHz; 18-bit per-word data still needs ≥76.3 MiB/s DMA. Only after meeting this gate should a complete LAT/OE/address/BCM panel protocol be attempted. **If paired burst/watermark and cyclic/low-copy tests remain below this**, report a measured achievable clock and consider another hardware data engine/architecture rather than speeding up an unsynchronized free-running clock.
