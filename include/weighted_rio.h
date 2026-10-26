#ifndef H75_WEIGHTED_RIO_H
#define H75_WEIGHTED_RIO_H

#include "weighted_bcm.h"
#include <stdatomic.h>

/* Internal Pi 5 RIO scanner. Caller owns GPIO configuration, exclusive GPIO
 * lock, SPSC reservations and lifetime. Nothing here opens devices, changes
 * scheduling, locks process memory or installs signals. Single scan-thread use.
 * Initialize/destroy only while blank; zero-initialize before init.
 *
 * Encoder/queue storage stays compact plane-major. Prepare a NEW held frame
 * once into this private, cache-line-aligned row-major staging buffer. Every
 * pixel word already contains address, OE and preserved non-HUB75 output bits.
 * Repeat cycles do not copy or touch the encoder LUT/image. This adds one
 * uint32_t per encoded word (not expanded clock-high/low pairs).
 */
typedef struct {
    uint32_t blank_address;
    uint64_t exposure_ticks;
} h75_weighted_rio_step;

typedef struct {
    volatile uint32_t *out, *set, *clear;
    const volatile atomic_bool *run_flag;
    bool run_value, ready;
    uint32_t width, rows, bits, steps;
    size_t word_count;
    uint64_t counter_hz;
    uint32_t *words;
    h75_weighted_rio_step schedule[32 * 11];
} h75_weighted_rio;

/* Pure integer duration conversion, rounding UP to counter ticks. Validates
 * bits 1..11, positive uint32 nanoseconds and counter frequency 1..UINT32_MAX.
 * No pulse-timing guarantee: MMIO, counter granularity and Linux preemption
 * add overhead. Exposed separately for hardware-free arithmetic tests. */
h75_status h75_weighted_rio_ticks(uint32_t bits, uint32_t lsb_ns, uint64_t counter_hz,
                                 uint64_t ticks[11]);

/* rio_out points at RIO Out, not the peripheral mapping base. Pi != 5 returns
 * UNSUPPORTED before touching registers. Production addresses use
 * h75_weighted_row_to_address (direct by default, with board wiring/explicit
 * offset overrides preserved). Width must be divisible by 16.
 * Stop is requested when *run_flag != run_value; polarity supports both entry
 * points without callbacks in the scan loop. Init allocates and pre-touches
 * staging pages and snapshots unowned output bits. No GPIO writes on success. */
h75_status h75_weighted_rio_init(h75_weighted_rio *scan, const hub75_display_t *display,
                                int pi_model, volatile uint32_t *rio_out,
                                const uint32_t *addresses,
                                const volatile atomic_bool *run_flag, bool run_value);
/* Prepare only at a full-cycle boundary while blank, with immutable held input.
 * Checks capacity/data-only words before marking ready. Errors leave OE blank.
 * Do not call again for a repeated held frame. */
h75_status h75_weighted_rio_prepare(h75_weighted_rio *scan,
                                   const uint32_t *planes, size_t words);
/* Call ONLY at a complete-cycle boundary (or startup), with a >=2-slot queue
 * of compact weighted planes. Acquire a first frame, or replace the held lease
 * only if a successor is published. Prepare only on adoption; otherwise repeat
 * the existing staging. TIMEOUT means no lease yet (startup or the transient
 * head-publication/sem_post gap); retry without reading any released slot.
 * The caller retains *held until a later boundary or blanked shutdown, then
 * commits it once. Private staging remains valid across acquisition retries. */
h75_status h75_weighted_rio_frame_begin(h75_weighted_rio *scan, spsc_semring_t *queue,
                                       const uint32_t **held, long timeout_ms);

/* One FULL cycle: for each row, all planes LSB-first. Pixel kernel is unrolled,
 * sequential loads + two Out stores per pixel, with no data-dependent branch,
 * timer, stop polling, allocation, lookup or function pointer call per pixel.
 * Stop is checked at plane boundaries and during exposure waits. Returns
 * CANCELLED for partial cycles; OK only after all rows/planes. Always blanks.
 * Initial path: blank -> shift -> latch -> expose, without shift overlap. */
h75_status h75_weighted_rio_cycle(h75_weighted_rio *scan);
void h75_weighted_rio_blank(h75_weighted_rio *scan);
void h75_weighted_rio_destroy(h75_weighted_rio *scan);

#endif
