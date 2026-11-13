# RP1 PIO hardware-DMA TX benchmark (Pi 5)

## Clock-only 20 MHz scope test

With the HUB75 panels **disconnected**, run `make pio-clock-scope`, then:

```sh
./build/release/pio_clock_scope 10 5    # nominal 20 MHz on GPIO17 (physical header pin 11)
```

The program claims one RP1 PIO state machine, explicitly muxes GPIO17 to PIO function 7, and uses a two-instruction mandatory side-set loop (`NOP side 0; NOP side 1`). At the RP1's observed 200 MHz `clk_sys`, divider 5 yields 20 MHz. Only the project's CLK pin (GPIO17) is muxed/driven; there is **no DMA, pixel data, latch or OE timing**. It is not safe to use to drive a connected HUB75 panel or to infer 20 Mpixel/s data throughput. Probe GPIO17 relative to Pi ground with a high-bandwidth scope and a short ground connection. On exit GPIO17 is left as high-impedance NULL function, **not restored** to its previous mode. SIGKILL/power loss may prevent cleanup.

## Synchronized 18-data-line + CLK scope test

This test differs from the free-running clock above: one PIO state machine **PULLs one DMA word, outputs the project's 18 RGB pins, then raises GPIO17 CLK with side-set**. The clock cannot advance to the next pixel without a new word. The low word drives all 18 data pins low; the high word drives them high. OE (GPIO18) is held HIGH (blanked); LAT and row-address pins are not driven. **Disconnect the panels**: this is not a complete HUB75 scan program.

```sh
make pio-hub75-scope
./build/release/pio_hub75_scope 5 4 256             # safe pin-free DMA/PIO test
./build/release/pio_hub75_scope 30 4 256 --output  # panel DISCONNECTED: scope GPIO17 and GPIO27
./build/release/pio_hub75_scope 30 8 256 --light-dma --output  # compare light DMA (experimental)
```

Use scope CH1 GPIO17 (header pin 11) and CH2 GPIO27 (header pin 13), both referenced to ground. The PIO ceiling at divider 4 is 200/4/3 = **16.67 MHz** (three instructions per pixel). The actual long-term clock rate cannot exceed the DMA word rate (~10–12 Mword/s observed in the pin-free test); it may appear as faster bursts interrupted by stalls. Compare both pulse widths and the count of CLK edges over a longer interval, not just the scope's frequency between adjacent edges. OE stays high; GPIO17, GPIO18 and the 18 data GPIOs are left in high-impedance NULL mode on normal exit, not restored to their previous functions. GPIO0/1, address and latch pins are untouched. Run with the default HZELLER_HAT pinout only.

**DMA burst diagnostic (6.18.54):** The default heavy channel reports TX DMACTRL `0xc0000100` (TX DREQ watermark 0): it requests an eight-word DMA burst only when the joined TX FIFO empties. This is consistent with a short pause after each eight clocks. The optional `--light-dma` uses `CONFIG_XFER_V2` to *force* a different DMA channel with a four-word burst and watermark 4 (`0xc0000104`). It refills earlier but in pin-free tests reached only ~6.5 Mword/s, versus ~7.6 Mword/s for default DMA at divider 8 and ~10.5 Mword/s at divider 4. It might yield a more regular waveform but **does not raise throughput**. The ioctl fails cleanly if a different kernel does not support it. Do **not** change the heavy channel's watermark through `SET_DMACTRL` alone: an eight-word burst sent to a half-full eight-word FIFO could overflow it. A driver-level experiment would instead cap the heavy channel to four-word bursts and configure the corresponding watermark 4 together. Cyclic DMA is another experiment that may remove transfer-boundary overhead; it cannot by itself fix the eight-word FIFO refill pause.

**Kernel DMA caveat:** Repeated pin-free runs on this installation have sometimes stalled after a few queued transfers; `dmesg` has reported `dma2chan0 is non-idle` / `failed to stop`. Both DMA test programs now wait for in-flight buffers to drain on normal completion, but a kernel driver problem or an interrupted run may still require investigation before treating any timing number as reliable. If the program reports stalled DMA, stop it and inspect `dmesg`; do not interpret a burst frequency alone as sustained throughput.

## DMA data benchmark

Build with `make pio-dma-bench`, then run:

```sh
./build/release/pio_dma_bench 5 4 64              # no pins driven
./build/release/pio_dma_bench 5 32 256 --autopull  # test single-instruction DMA without GPIO
./build/release/pio_dma_bench 5 32 256 --gpio 8     # drive GPIO8..15 (two instructions/word)
./build/release/pio_dma_bench 5 32 256 --gpio 8 --autopull  # one instruction/word
./build/release/pio_dma_bench 5 4 256 --pack4             # four updates/word, NO GPIO
./build/release/pio_dma_bench 5 4 256 --gpio 8 --pack4    # four updates/word on GPIO8..15
# WITH PANELS DISCONNECTED, scope all default project output GPIOs:
./build/release/pio_dma_bench 5 4 256 --project-pins      # GPIO2..27, one 26-bit update/word
# arguments: seconds (1..3600), PIO clock divider (1..65535), transfer size in KiB (1..1024)
# --gpio BASE: BASE 0..20, uses exactly eight consecutive RP1 GPIOs
```

Needs `/dev/pio0` access (usually membership in the `gpio` group), a free RP1 PIO state machine, and a kernel with `PIO_IOC_SM_CONFIG_XFER32` and `PIO_IOC_SM_XFER_DATA32`. It uses the kernel's DMA-engine TX transfer interface and four coherent bounce buffers. By default, the single-instruction `PULL block` PIO program **configures/drives no GPIOs**. With `--gpio BASE`, the PIO program runs `PULL block; OUT PINS, 8`: the low byte of each DMA word alternates `0x00` and `0xff`, driving all eight selected pins together. Each word takes two PIO cycles. Add `--autopull` to replace these two instructions with one `OUT PINS, 8` instruction and autopull after eight bits. With `--pack4` (which implies `--autopull`), each word contains four bytes `00 ff 00 ff`; autopull happens after 32 bits, producing **four pin updates per DMA word**. With no GPIO mode, `--autopull` and `--pack4` use `OUT NULL, 8` instead of driving pins, so the PIO/DMA path can be tested safely. `--project-pins` instead uses `OUT PINS, 26` and 26-bit autopull, alternating all-low/all-high across GPIO2..27 (18 RGB, five address pins A–E, CLK, LAT, OE; from `include/rpihub75.h`'s default HZELLER_HAT pinout). **The project uses 26, not 28, RP1 GPIOs. GPIO0/1 are not touched.** This mode cannot pack multiple 26-bit updates into a single word and rejects `--pack4`. Consequently, expect a substantially lower square-wave frequency than eight-pin `--pack4` mode (at ~12 Mword/s, at most ~6 MHz square wave). No root or `/dev/mem` access is required.

**Disconnect the HUB75 panels before using `--project-pins`.** This mode drives CLK, LAT, OE, address and data together as a **square wave, not HUB75 protocol**. It is solely for probing with a scope. The Pi GPIOs are **3.3 V outputs**, not 5 V tolerant. Before using `--gpio`, choose eight pins that are free, not used by a connected panel or another peripheral/process; the PIO driver does not reserve GPIO lines against other users. The selected pins' previous mux and pull settings are **not restored**: on exit the tool disables its state machine, switches those pins to input, then to high-impedance NULL function. Reconfigure them yourself afterward if needed. Do not run alongside another program using the same pins or all PIO state machines.

The results are *sustained userspace DMA submission throughput* (including ioctl overhead and copies into coherent DMA buffers), not measured GPIO signal rate. Up to four buffers may still be queued when the timer expires; long runs reduce that effect. The PIO consumes one 32-bit word per PIO clock cycle in the one-instruction modes and one per two cycles in the default GPIO mode, subject to DMA and software limits. To verify actual pin timing use an oscilloscope and ensure any connected hardware tolerates the signal.

## Performance tuning (measured on this Pi 5, no GPIO driven)

The RP1 PIO clock is 200 MHz here (`/sys/kernel/debug/clk/clk_sys/clk_rate`). At divider 32, one-instruction PIO runs at 6.25 Mword/s (23.84 MiB/s theoretical); at divider 16, 12.5 Mword/s (47.68 MiB/s theoretical). The existing two-instruction GPIO loop runs at half that rate. For comparisons, keep the PIO program and transfer size identical and change just one setting at a time.

| Divider | KiB/transfer | Measured DMA submission (no GPIO, 1 instruction/word) |
| ---: | ---: | ---: |
| 32 | 256 | ~23.7 MiB/s (~6.2 Mword/s) |
| 16 | 256 | ~42.0 MiB/s (~11.0 Mword/s) |
| 4 | 256 | ~46.0 MiB/s (~12.1 Mword/s) |
| 1 | 256 | ~46.0 MiB/s (~12.1 Mword/s) |
| 4 | 4 | ~36 MiB/s (~9.4 Mword/s) |

Thus `--autopull` plus divider 32 should roughly double the output word rate *if GPIO itself is not limiting*; this prediction has not been verified at the pins. To target an **18.4 MHz alternating square wave** (36.8 million pin updates/s), use `--pack4` and divider **4** or less. At divider 4, the pin-free `OUT NULL,8` test submitted ~11.0 Mword/s = ~44 Mupdates/s, equivalent to a **22 MHz alternating square wave**; at divider 2, ~12.1 Mword/s = ~48 Mupdates/s (~24 MHz). Neither number is a measured GPIO rate. Since the pattern toggles every update, a MHz square wave has *twice* that many transitions/updates per second. Around 12 Mword/s, lowering the divider further does not help this kernel DMA/ioctl path. Use 64–256 KiB transfers to reduce syscall/descriptor overhead; 4 KiB transfers are significantly slower. This kernel supports only four non-cyclic DMA bounce buffers via the original ioctl. The new 6.18.54 kernel supports `CONFIG_XFER_V2` (tested with `--light-dma`); cyclic DMA is not yet benchmarked here. Further increases beyond the ~46 MiB/s measured here would likely require profiling or changing the kernel DMA path (bounce-buffer copies, DMA channel/descriptor overhead), not just changing the PIO clock divider. Increasing GPIO pin count does not increase bytes per DMA word, but can change electrical loading and signal quality.
