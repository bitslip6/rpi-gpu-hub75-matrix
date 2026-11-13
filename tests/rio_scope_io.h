#ifndef RIO_SCOPE_IO_H
#define RIO_SCOPE_IO_H

/* Standalone benchmark only: fixed default HZELLER_HAT pinout from rpihub75.h.
 * No ADA/custom pinouts. Register offsets are bytes relative to RP1 IO_BANK0.
 * Tests replace RIO_STORE/RIO_LOAD to check the identical emission algorithm.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RIO_CLK (UINT32_C(1) << 17)
#define RIO_OE (UINT32_C(1) << 18)
#define RIO_LAT (UINT32_C(1) << 4)
#define RIO_RGB (UINT32_C(0x0fffffff) & ~(UINT32_C(3) | RIO_CLK | RIO_OE | RIO_LAT | RIO_ADDR))
#define RIO_ADDR ((UINT32_C(15) << 22) | (UINT32_C(1) << 15))
#define RIO_MAP_BYTES 0x30000u
#define RIO_REG_OUT 0x10000u
#define RIO_REG_OE 0x10004u
#define RIO_REG_SET 0x12000u
#define RIO_REG_CLR 0x13000u
#define RIO_REG_XOR 0x11000u
#define RIO_PAD_BASE 0x20004u
#define RIO_PAD_OD (UINT32_C(1) << 7)

/* Setup/cleanup only. Never alter slew, drive, pulls or receiver configuration. */
static inline uint32_t rio_scope_pad_output_enabled(uint32_t pad)
{
    return pad & ~RIO_PAD_OD;
}

static inline uint32_t rio_scope_pad_restore_od(uint32_t current, uint32_t saved)
{
    return (current & ~RIO_PAD_OD) | (saved & RIO_PAD_OD);
}

#ifndef RIO_STORE
#define RIO_STORE(reg, value) (*(reg) = (value))
#endif
#ifndef RIO_LOAD
#define RIO_LOAD(reg) (*(reg))
#endif

typedef enum { RIO_CLOCK, RIO_ALTERNATING, RIO_SEQUENCE, RIO_PROTOCOL } rio_scope_mode;

typedef struct {
    volatile uint32_t *out, *pixel, *set, *clr;
    uint32_t owned;
} rio_scope_io;

static inline uint32_t rio_scope_owned(rio_scope_mode mode)
{
    return RIO_CLK | RIO_OE | (mode == RIO_CLOCK ? 0 : RIO_RGB) |
           (mode == RIO_PROTOCOL ? RIO_LAT | RIO_ADDR : 0);
}

static inline uint32_t rio_scope_address(uint32_t row, uint32_t rows)
{
    /* Matches default row offset -1, ADDRESS_LSB_IS_A=1. */
    uint32_t r = (row + rows - 1) % rows;
    return ((r & 15u) << 22) | ((r & 16u) << 11);
}

static inline uint32_t rio_scope_data(rio_scope_mode mode, size_t index)
{
    if (mode == RIO_CLOCK)
        return 0;
    if (mode == RIO_ALTERNATING)
        return index & 1u ? 0 : RIO_RGB;
    uint32_t x = (uint32_t)index + UINT32_C(0x9e3779b9);
    x ^= x >> 16;
    x *= UINT32_C(0x85ebca6b);
    x ^= x >> 13;
    x *= UINT32_C(0xc2b2ae35);
    return (x ^ (x >> 16)) & RIO_RGB;
}

/* Prepared compact words, with clock-high OR for Out, or low-state XOR delta
 * followed by constant clock XOR. Every delta contains ONLY owned bits.
 * Boundaries start with OE high and all other owned bits low. */
static inline void rio_scope_prepare(uint32_t *words, uint32_t width,
                                      uint32_t rows, uint32_t planes,
                                      rio_scope_mode mode, bool use_xor,
                                      uint32_t unowned)
{
    uint32_t previous = RIO_OE, previous_rgb = 0;
    size_t index = 0;
    for (uint32_t p = 0; p < planes; ++p) {
        for (uint32_t y = 0; y < rows; ++y) {
            uint32_t address = mode == RIO_PROTOCOL ? rio_scope_address(y, rows) : 0;
            for (uint32_t x = 0; x < width; ++x, ++index) {
                uint32_t rgb = rio_scope_data(mode, index);
                if ((mode == RIO_SEQUENCE || mode == RIO_PROTOCOL) && rgb == previous_rgb)
                    rgb ^= UINT32_C(1) << 2;
                previous_rgb = rgb;
                uint32_t low = RIO_OE | address | rgb;
                words[index] = use_xor ? previous ^ low : unowned | low;
                previous = low | RIO_CLK;
            }
            if (mode == RIO_PROTOCOL)
                previous &= ~RIO_CLK; /* Protocol clears clock before latching. */
        }
    }
}

static inline void rio_scope_barrier(void)
{
#if defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

static inline void rio_scope_delay(unsigned nops)
{
    for (unsigned i = 0; i < nops; ++i)
        __asm__ volatile("nop" ::: "memory");
}

/* OE never goes low. Read back the same RP1 block after ordered blanking;
 * this is a completion experiment, NOT proof of physical pin timing. */
static inline bool rio_scope_blank(rio_scope_io *io)
{
    RIO_STORE(io->set, RIO_OE);
    RIO_STORE(io->clr, io->owned & ~RIO_OE);
    rio_scope_barrier();
    uint32_t latched = RIO_LOAD(io->out);
    return (latched & io->owned) == RIO_OE;
}

/* Two separate ordered volatile pixel stores, without per-pixel branches,
 * clocks, atomics, queue operations or function-pointer callbacks. */
#define RIO_EMIT_BODY(high_word) \
    size_t index = 0; \
    for (uint32_t p = 0; p < planes; ++p) { \
        for (uint32_t y = 0; y < rows; ++y) { \
            for (uint32_t x = 0; x < width; ++x) { \
                uint32_t word = words[index++]; \
                RIO_STORE(io->pixel, word); \
                RIO_STORE(io->pixel, (high_word)); \
            } \
            if (protocol) { \
                RIO_STORE(io->clr, RIO_CLK); \
                rio_scope_delay(nops); \
                RIO_STORE(io->set, RIO_LAT); \
                rio_scope_delay(nops); \
                RIO_STORE(io->clr, RIO_LAT); \
                rio_scope_delay(nops); \
            } \
        } \
    }

static inline void rio_scope_emit_xor(rio_scope_io *io, const uint32_t *words,
                                      uint32_t width, uint32_t rows, uint32_t planes,
                                      bool protocol, unsigned nops)
{
    RIO_EMIT_BODY(RIO_CLK)
}

static inline void rio_scope_emit_out(rio_scope_io *io, const uint32_t *words,
                                      uint32_t width, uint32_t rows, uint32_t planes,
                                      bool protocol, unsigned nops)
{
    RIO_EMIT_BODY(word | RIO_CLK)
}
#undef RIO_EMIT_BODY
#endif
