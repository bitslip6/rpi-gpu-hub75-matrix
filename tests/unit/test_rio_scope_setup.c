/* Execute the benchmark's actual setup/cleanup against anonymous RAM only.
 * GPIO discovery/open is NEVER called. Override MMIO access to model aliases.
 */
#define _GNU_SOURCE
#include <stdint.h>
static void setup_store(volatile uint32_t *reg, uint32_t value);
static uint32_t setup_load(volatile uint32_t *reg);
#define RIO_STORE(reg, value) setup_store((reg), (value))
#define RIO_LOAD(reg) setup_load(reg)
#define main rio_scope_unused_benchmark_main
#include "../rio_scope.c"
#undef main

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); abort(); \
} } while (0)

static uint32_t *model;
static uint32_t final_state[RIO_MAP_BYTES / sizeof(uint32_t)];
static unsigned pad_writes[28];
static int fail_pad;
static uint32_t owned;
static size_t writes;

static void setup_store(volatile uint32_t *reg, uint32_t value)
{
    ++writes;
    size_t offset = (size_t)((uintptr_t)reg - (uintptr_t)model);
    CHECK(offset < RIO_MAP_BYTES && offset % 4 == 0);
    if (offset == RIO_REG_SET || offset == RIO_REG_CLR ||
        offset == RIO_REG_SET + 4 || offset == RIO_REG_CLR + 4) {
        CHECK(!(value & ~owned));
        size_t target = (offset == RIO_REG_SET || offset == RIO_REG_CLR) ? RIO_REG_OUT : RIO_REG_OE;
        if (offset == RIO_REG_SET || offset == RIO_REG_SET + 4)
            model[target / 4] |= value;
        else
            model[target / 4] &= ~value;
    } else if (offset >= RIO_PAD_BASE && offset < RIO_PAD_BASE + 28 * 4) {
        unsigned pin = (unsigned)((offset - RIO_PAD_BASE) / 4);
        CHECK(owned & (UINT32_C(1) << pin));
        CHECK(((model[offset / 4] ^ value) & ~RIO_PAD_OD) == 0);
        ++pad_writes[pin];
        if ((int)pin != fail_pad || (value & RIO_PAD_OD))
            model[offset / 4] = value;
    } else {
        CHECK(offset < 28 * 8 && offset % 8 == 4);
        unsigned pin = (unsigned)(offset / 8);
        CHECK(owned & (UINT32_C(1) << pin));
        model[offset / 4] = value;
    }
}

static uint32_t setup_load(volatile uint32_t *reg)
{
    size_t offset = (size_t)((uintptr_t)reg - (uintptr_t)model);
    CHECK(offset < RIO_MAP_BYTES && offset % 4 == 0);
    /* Keep the final state inspectable after hardware_close unmaps the model. */
    memcpy(final_state, model, RIO_MAP_BYTES);
    return model[offset / 4];
}

static void run_preflight(unsigned scenario)
{
    model = mmap(NULL, RIO_MAP_BYTES, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(model != MAP_FAILED);
    owned = RIO_CLK | RIO_OE;
    writes = 0;
    hardware hw = {.map = model, .lock_fd = -1};
    for (unsigned pin = 0; pin < 28; ++pin) {
        model[(pin * 8 + 4) / 4] = 0x9f;
        model[(RIO_PAD_BASE + pin * 4) / 4] = 0x96;
    }
    if (scenario == 0) memset(model, 0xff, RIO_MAP_BYTES);
    if (scenario == 1) model[(RIO_PAD_BASE + 17 * 4) / 4] = UINT32_MAX;
    if (scenario == 2) model[RIO_REG_OE / 4] = owned;
    if (scenario == 5 || scenario == 6) model[RIO_REG_OE / 4] = 1u;
    if (scenario == 7) model[(17 * 8 + 4) / 4] = 7; /* PIO function: still refused. */
    rio_scope_io io = {reg_at(model, RIO_REG_OUT), reg_at(model, RIO_REG_XOR),
                       reg_at(model, RIO_REG_SET), reg_at(model, RIO_REG_CLR), owned};
    uint32_t *before = malloc(RIO_MAP_BYTES);
    CHECK(before != NULL);
    memcpy(before, model, RIO_MAP_BYTES);
    FILE *log = tmpfile();
    CHECK(log != NULL);
    int saved_stderr = dup(STDERR_FILENO);
    CHECK(saved_stderr >= 0);
    CHECK(dup2(fileno(log), STDERR_FILENO) >= 0);
    bool accepted = hardware_preflight(&hw, &io, scenario != 6, scenario != 3);
    fflush(stderr);
    CHECK(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);
    rewind(log);
    char diagnostic[4096];
    size_t n = fread(diagnostic, 1, sizeof(diagnostic) - 1, log);
    diagnostic[n] = '\0';
    fclose(log);
    CHECK(accepted == (scenario == 4 || scenario == 5));
    CHECK(writes == 0 && !hw.changed);
    CHECK(memcmp(before, model, RIO_MAP_BYTES) == 0);
    free(before);
    CHECK(strstr(diagnostic, "RP1 preflight:") != NULL);
    if (scenario <= 1) {
        CHECK(strstr(diagnostic, "Invalid RP1 MMIO readback") != NULL);
        CHECK(strstr(diagnostic, "already enabled") == NULL);
    } else if (scenario == 2 || scenario == 6) {
        CHECK(strstr(diagnostic, "RIO outputs already enabled") != NULL);
        CHECK(strstr(diagnostic, "conflicts=0x") != NULL);
    } else if (scenario == 3) {
        CHECK(strstr(diagnostic, "--enable-pad-output") != NULL);
    } else if (scenario == 7) {
        CHECK(strstr(diagnostic, "alternate function") != NULL);
    } else {
        CHECK(hw.pads_to_enable == owned);
    }
    CHECK(hardware_close(&hw, &io));
    CHECK(writes == 0); /* Reject/inspection paths cannot silently reset the bank. */
    model = NULL;
}

static void run_setup(uint32_t enable_mask, bool inject_failure)
{
    model = mmap(NULL, RIO_MAP_BYTES, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(model != MAP_FAILED);
    owned = RIO_CLK | RIO_OE;
    fail_pad = inject_failure ? 18 : -1;
    memset(pad_writes, 0, sizeof(pad_writes));
    hardware hw = {.map = model, .lock_fd = -1, .pads_to_enable = enable_mask};
    for (unsigned pin = 0; pin < 28; ++pin) {
        model[(pin * 8 + 4) / 4] = hw.saved_ctrl[pin] = 0x9f;
        model[(RIO_PAD_BASE + pin * 4) / 4] = hw.saved_pad[pin] = 0x96;
    }
    model[RIO_REG_OUT / 4] = hw.saved_out = UINT32_C(0x80000003) | RIO_CLK;
    model[RIO_REG_OE / 4] = hw.saved_oe = UINT32_C(2); /* Unowned sentinel. */
    rio_scope_io io = {reg_at(model, RIO_REG_OUT), reg_at(model, RIO_REG_XOR),
                       reg_at(model, RIO_REG_SET), reg_at(model, RIO_REG_CLR), owned};
    CHECK(hardware_enable(&hw, &io) == !inject_failure);
    CHECK(hw.pads_changed == enable_mask);
    for (unsigned pin = 0; pin < 28; ++pin) {
        uint32_t expected = 0x96;
        if ((enable_mask & (UINT32_C(1) << pin)) && (int)pin != fail_pad)
            expected = 0x16;
        CHECK(model[(RIO_PAD_BASE + pin * 4) / 4] == expected);
    }
    CHECK(hardware_close(&hw, &io));
    for (unsigned pin = 0; pin < 28; ++pin) {
        CHECK(final_state[(RIO_PAD_BASE + pin * 4) / 4] == 0x96);
        CHECK(final_state[(pin * 8 + 4) / 4] == 0x9f);
        CHECK(pad_writes[pin] == (enable_mask & (UINT32_C(1) << pin) ? 2u : 0u));
    }
    CHECK(final_state[RIO_REG_OUT / 4] == hw.saved_out);
    CHECK(final_state[RIO_REG_OE / 4] == hw.saved_oe);
    model = NULL;
}

int main(void)
{
    CHECK(rio_scope_pad_output_enabled(0x96) == 0x16);
    for (uint32_t pad = 0; pad < 1024; ++pad) {
        uint32_t enabled = rio_scope_pad_output_enabled(pad);
        CHECK(!(enabled & RIO_PAD_OD));
        CHECK(((enabled ^ pad) & ~RIO_PAD_OD) == 0);
        CHECK(rio_scope_pad_restore_od(enabled, pad) == pad);
        /* Restoration must not clobber a hypothetical unrelated slew change. */
        CHECK(rio_scope_pad_restore_od(enabled ^ 1u, pad) == (pad ^ 1u));
    }
    for (unsigned scenario = 0; scenario < 8; ++scenario)
        run_preflight(scenario);
    run_setup(0, false); /* No opt-in: setup/cleanup never writes pads. */
    run_setup(RIO_CLK, false);
    run_setup(RIO_CLK | RIO_OE, false);
    fputs("Expected simulated pad readback failure (RAM model only):\n", stderr);
    run_setup(RIO_CLK | RIO_OE, true); /* Partial setup must restore both pads. */
    puts("RIO scope setup/cleanup model tests passed (anonymous RAM, no GPIO devices).");
    return 0;
}
