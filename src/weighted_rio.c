#define _POSIX_C_SOURCE 200809L
#include "weighted_rio.h"
#include "rpihub75.h"
#include <time.h>

/* Tests replace writes/counter reads at compile time. The production object
 * has direct volatile stores and counter reads, not indirect callbacks. */
#ifdef H75_WEIGHTED_RIO_TESTING
extern void h75_weighted_rio_test_write(volatile uint32_t *, uint32_t);
extern uint64_t h75_weighted_rio_test_counter(void);
extern uint64_t h75_weighted_rio_test_frequency(void);
#define WRITE(reg, value) h75_weighted_rio_test_write((reg), (value))
#define counter_now() h75_weighted_rio_test_counter()
#define counter_frequency() h75_weighted_rio_test_frequency()
#else
#define WRITE(reg, value) (*(reg) = (value))
static inline uint64_t counter_now(void) {
#if defined(__aarch64__)
    uint64_t ticks;
    __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(ticks) :: "memory");
    return ticks;
#else
    /* Portable pin-free build fallback: busy wait, NEVER an OS sleep. */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec;
#endif
}
static uint64_t counter_frequency(void) {
#if defined(__aarch64__)
    uint64_t hz;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(hz));
    return hz;
#else
    return 1000000000u;
#endif
}
#endif

static inline void device_complete(void) {
#if defined(__aarch64__) || defined(__arm__)
    /* Completion at control boundaries only, NOT a barrier per pixel. MMIO
     * must be mapped as Device memory (the existing gpiomem mapping). */
    __asm__ __volatile__("dsb oshst" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

h75_status h75_weighted_rio_ticks(uint32_t bits, uint32_t lsb_ns, uint64_t hz,
                                 uint64_t ticks[11]) {
    if (!ticks || bits < 1 || bits > 11 || !lsb_ns || !hz || hz > UINT32_MAX)
        return H75_INVALID;
    for (unsigned plane = 0; plane < bits; plane++) {
        const uint64_t ns = (uint64_t)lsb_ns << plane;
        /* Split to avoid overflow in ns*hz, even at UINT32_MAX settings. */
        ticks[plane] = (ns / 1000000000u) * hz +
            ((ns % 1000000000u) * hz + 999999999u) / 1000000000u;
    }
    return H75_OK;
}

void h75_weighted_rio_blank(h75_weighted_rio *s) {
    if (s && s->set && s->clear) {
        WRITE(s->set, PIN_OE);
        WRITE(s->clear, PIN_CLK | PIN_LATCH);
        device_complete();
    }
}

h75_status h75_weighted_rio_init(h75_weighted_rio *s, const hub75_display_t *d,
                                int model, volatile uint32_t *out,
                                const uint32_t *addresses,
                                const volatile atomic_bool *flag, bool value) {
    if (model != 5)
        return H75_UNSUPPORTED;
    if (!s || !out || !addresses || !flag || s->words)
        return H75_INVALID;
    size_t words;
    h75_status status = h75_weighted_word_count(d, &words);
    if (status != H75_OK)
        return status;
    if (d->width % 16 || !d->pwm_lsb_ns)
        return H75_INVALID;
    uint64_t hz = counter_frequency(), ticks[11];
    status = h75_weighted_rio_ticks(d->pwm_bits, d->pwm_lsb_ns, hz, ticks);
    if (status != H75_OK)
        return status;
    const unsigned rows = d->panel_height / 2u;
    for (unsigned row = 0; row < rows; row++)
        if (addresses[row] & ~(uint32_t)MASK_ADDRESS)
            return H75_INVALID;
    /* 32 trailing words keep even the last software prefetch in allocation.
     * Page touch happens before the first exposure, on the GPIO worker. */
    if (words > SIZE_MAX / sizeof(uint32_t) - 32)
        return H75_INVALID;
    uint32_t *storage = NULL;
    const size_t bytes = (words + 32) * sizeof(uint32_t);
    if (posix_memalign((void **)&storage, 64, bytes))
        return H75_MEMORY;
    memset(storage, 0, bytes);
    *s = (h75_weighted_rio){.out = out, .set = out + 0x2000 / 4,
        .clear = out + 0x3000 / 4, .run_flag = flag, .run_value = value,
        .width = d->width, .rows = rows, .bits = d->pwm_bits,
        .steps = rows * d->pwm_bits, .word_count = words,
        .counter_hz = hz, .words = storage};
    const uint32_t preserved = *out & ~(uint32_t)(MASK_DATA | MASK_ADDRESS | MASK_CTRL);
    unsigned step = 0;
    for (unsigned row = 0; row < rows; row++)
        for (unsigned plane = 0; plane < d->pwm_bits; plane++)
            s->schedule[step++] = (h75_weighted_rio_step){
                .blank_address = preserved | addresses[row] | PIN_OE,
                .exposure_ticks = ticks[plane]};
    return H75_OK;
}

h75_status h75_weighted_rio_prepare(h75_weighted_rio *s,
                                   const uint32_t *planes, size_t count) {
    if (!s || !s->words)
        return H75_INVALID;
    h75_weighted_rio_blank(s);
    s->ready = false;
    if (!planes || count < s->word_count)
        return H75_INVALID;
    const size_t plane_words = (size_t)s->rows * s->width;
    uint32_t *dst = s->words;
    uint32_t invalid = 0;
    unsigned step = 0;
    for (size_t row = 0; row < s->rows; row++) {
        for (unsigned plane = 0; plane < s->bits; plane++) {
            const uint32_t *src = planes + plane * plane_words + row * s->width;
            const uint32_t prefix = s->schedule[step++].blank_address;
            for (unsigned x = 0; x < s->width; x++) {
                const uint32_t data = src[x];
                invalid |= data & ~(uint32_t)MASK_DATA;
                *dst++ = data | prefix;
            }
        }
    }
    if (invalid)
        return H75_INVALID;
    s->ready = true;
    return H75_OK;
}

h75_status h75_weighted_rio_frame_begin(h75_weighted_rio *s, spsc_semring_t *queue,
                                       const uint32_t **held, long timeout_ms) {
    if (!s || !s->words || !queue || !held || queue->capacity < 2 ||
        queue->item_size != s->word_count * sizeof(uint32_t)) {
        h75_weighted_rio_blank(s);
        return H75_INVALID;
    }
    bool changed = false;
    if (!*held) {
        *held = spsc_pop_ptr_begin(queue, timeout_ms);
        if (!*held)
            return H75_TIMEOUT;
        changed = true;
    }
    /* Count includes the held reservation. Producer cannot reclaim it until
     * commit, and a published successor is immutable until we acquire it. */
    if (spsc_count(queue) >= 2) {
        spsc_pop_ptr_commit(queue);
        *held = spsc_pop_ptr_begin(queue, 0);
        if (!*held) {
            /* SPSC publishes head before sem_post. A boundary can observe the
             * successor's head while its can_pop token is not posted yet.
             * Never treat that transient gap (or EINTR) as scanner corruption;
             * retry acquisition next time, retaining the private staging. */
            h75_weighted_rio_blank(s);
            return H75_TIMEOUT;
        }
        changed = true;
    }
    return changed ? h75_weighted_rio_prepare(s, *held, s->word_count) : H75_OK;
}

static inline bool running(const h75_weighted_rio *s) {
    return atomic_load_explicit(s->run_flag, memory_order_relaxed) == s->run_value;
}

static inline bool wait_exposure(uint64_t duration,
                                  const volatile atomic_bool *flag, bool run_value) {
    const uint64_t start = counter_now();
    while (counter_now() - start < duration)
        if (atomic_load_explicit(flag, memory_order_relaxed) != run_value)
            return false;
    return true;
}

static inline void shift_row(volatile uint32_t *restrict out,
                             const uint32_t *restrict src, uint32_t width) {
    const uint32_t *end = src + width;
    /* Width validated once. No scalar tail, mode/port/brightness check, lookup,
     * counter read, atomic, or per-pixel branch. Each eight-pixel chunk is 32 B.
     * One cache line ahead (plus hardware sequential prefetch) hides some load
     * latency; this is a hint, not a promise of cache residency under Linux. */
    do {
        __builtin_prefetch(src + 16, 0, 3);
        /* Load a half-cache-line before hitting Device stores. This allows
         * RAM reads to complete ahead of the GPIO burst, rather than serially
         * waiting on a new RAM load between every clock pulse. */
        const uint32_t v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
        const uint32_t v4 = src[4], v5 = src[5], v6 = src[6], v7 = src[7];
        /* Compiler-only register operands force these loads ahead of the MMIO
         * burst. No Device barrier or emitted instruction per eight pixels. */
        __asm__ __volatile__("" : : "r"(v0), "r"(v1), "r"(v2), "r"(v3),
                             "r"(v4), "r"(v5), "r"(v6), "r"(v7) : "memory");
#define PIXEL(v) do { WRITE(out, (v)); WRITE(out, (v) | PIN_CLK); } while (0)
        PIXEL(v0); PIXEL(v1); PIXEL(v2); PIXEL(v3);
        PIXEL(v4); PIXEL(v5); PIXEL(v6); PIXEL(v7);
#undef PIXEL
        src += 8;
    } while (src != end);
}

h75_status h75_weighted_rio_cycle(h75_weighted_rio *s) {
    if (!s || !s->words || !s->ready) {
        h75_weighted_rio_blank(s);
        return H75_INVALID;
    }
    h75_weighted_rio_blank(s);
    const uint32_t *src = s->words;
    const h75_weighted_rio_step *step = s->schedule, *end = step + s->steps;
    for (; step != end; step++, src += s->width) {
        if (!running(s))
            goto cancelled;
        /* Last exposure ended blank. Change address while OE is high. */
        WRITE(s->out, step->blank_address);
        device_complete();
        shift_row(s->out, src, s->width);
        WRITE(s->clear, PIN_CLK);
        device_complete();
        WRITE(s->set, PIN_LATCH);
        device_complete();
        WRITE(s->clear, PIN_LATCH);
        device_complete();
        if (!running(s))
            goto cancelled;
        WRITE(s->clear, PIN_OE);
        device_complete();
        /* Dedicated counter-based busy wait. Stop remains responsive even for
         * deliberately very long exposures. No ordinary OS sleeps. */
        if (!wait_exposure(step->exposure_ticks, s->run_flag, s->run_value))
            goto cancelled;
        h75_weighted_rio_blank(s);
    }
    return H75_OK;
cancelled:
    h75_weighted_rio_blank(s);
    return H75_CANCELLED;
}

void h75_weighted_rio_destroy(h75_weighted_rio *s) {
    if (!s)
        return;
    h75_weighted_rio_blank(s);
    free(s->words);
    *s = (h75_weighted_rio){0};
}
