#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void trace_store(volatile uint32_t *reg, uint32_t value);
static uint32_t trace_load(volatile uint32_t *reg);
#define RIO_STORE(reg, value) trace_store((reg), (value))
#define RIO_LOAD(reg) trace_load(reg)
#include "rio_scope_io.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); abort(); \
} } while (0)

/* Independent software register model with GPIO0/1 sentinel bits. */
static volatile uint32_t regs[4];
static uint32_t state, owned, rising[10000];
static size_t edges, stores, latches;
static bool use_xor;

static void trace_store(volatile uint32_t *reg, uint32_t value)
{
    uint32_t old = state;
    ++stores;
    if (reg == &regs[1]) {
        if (use_xor) {
            CHECK(!(value & ~owned));
            state ^= value;
        } else {
            state = value;
        }
    } else if (reg == &regs[2]) {
        CHECK(!(value & ~owned));
        state |= value;
    } else {
        CHECK(reg == &regs[3]);
        CHECK(!(value & ~owned));
        state &= ~value;
    }
    CHECK(state & RIO_OE);
    CHECK((state & ~owned) == (UINT32_C(0xa0000003) & ~owned));
    if (!(old & RIO_CLK) && (state & RIO_CLK)) {
        /* A rising edge changes ONLY CLK, not RGB/address/blanking. */
        CHECK((old ^ state) == RIO_CLK);
        CHECK(edges < sizeof(rising) / sizeof(rising[0]));
        rising[edges++] = state;
    }
    if (!(old & RIO_LAT) && (state & RIO_LAT)) {
        CHECK(!(state & RIO_CLK));
        ++latches;
    }
}

static uint32_t trace_load(volatile uint32_t *reg)
{
    CHECK(reg == &regs[0]);
    return state;
}

static void run_case(rio_scope_mode mode, bool xor_path, uint32_t rows)
{
    const uint32_t width = 17, planes = 3;
    size_t count = (size_t)width * rows * planes;
    uint32_t *words = malloc(count * sizeof(*words));
    CHECK(words != NULL);
    use_xor = xor_path;
    owned = rio_scope_owned(mode);
    uint32_t unowned = UINT32_C(0xa0000003) & ~owned;
    state = unowned | RIO_OE;
    stores = edges = latches = 0;
    rio_scope_io io = {&regs[0], &regs[1], &regs[2], &regs[3], owned};
    rio_scope_prepare(words, width, rows, planes, mode, use_xor, unowned);
    for (unsigned cycle = 0; cycle < 2; ++cycle) {
        if (use_xor)
            rio_scope_emit_xor(&io, words, width, rows, planes, mode == RIO_PROTOCOL, 0);
        else
            rio_scope_emit_out(&io, words, width, rows, planes, mode == RIO_PROTOCOL, 0);
        CHECK(rio_scope_blank(&io));
        CHECK(state == (unowned | RIO_OE));
    }
    CHECK(edges == count * 2);
    CHECK(latches == (mode == RIO_PROTOCOL ? (size_t)rows * planes * 2 : 0));
    CHECK(stores == count * 4 + 4 + (mode == RIO_PROTOCOL ? (size_t)rows * planes * 6 : 0));
    for (size_t i = 0; i < count; ++i) {
        CHECK(rising[i] == rising[i + count]); /* Complete cycles repeat exactly. */
        if (mode == RIO_CLOCK) CHECK(!(rising[i] & RIO_RGB));
        if (mode == RIO_ALTERNATING) CHECK((rising[i] & RIO_RGB) == (i & 1u ? 0 : RIO_RGB));
        if (mode == RIO_SEQUENCE || mode == RIO_PROTOCOL) {
            if (i) CHECK((rising[i] & RIO_RGB) != (rising[i - 1] & RIO_RGB));
        }
        if (mode == RIO_PROTOCOL) {
            /* Independent expected -1 row offset, default A..E significance. */
            uint32_t row = (uint32_t)(i / width % rows);
            uint32_t address_row = row ? row - 1 : rows - 1;
            unsigned pins[] = {22, 23, 24, 25, 15};
            uint32_t address = 0;
            for (unsigned bit = 0; bit < 5; ++bit)
                if (address_row & (1u << bit)) address |= 1u << pins[bit];
            CHECK((rising[i] & RIO_ADDR) == address);
        }
    }
    free(words);
}

int main(void)
{
    CHECK((RIO_RGB | RIO_ADDR | RIO_CLK | RIO_OE | RIO_LAT) == UINT32_C(0x0ffffffc));
    CHECK(__builtin_popcount(RIO_RGB) == 18);
    for (rio_scope_mode mode = RIO_CLOCK; mode <= RIO_PROTOCOL; ++mode)
        for (unsigned path = 0; path < 2; ++path)
            for (uint32_t rows = 1; rows <= 32; rows *= 2)
                run_case(mode, path != 0, rows);
    puts("RIO scope software MMIO trace tests passed (not electrical validation).");
    return 0;
}
