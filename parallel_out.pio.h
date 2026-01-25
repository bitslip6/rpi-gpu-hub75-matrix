// -------------------------------------------------- //
// Parallel output PIO program for HUB75 control     //
// Outputs to 28 GPIO pins (0-27) simultaneously     //
// -------------------------------------------------- //

#pragma once

#if !PICO_NO_HARDWARE
#include "hardware/pio.h"
#endif

// ------------ //
// parallel_out //
// ------------ //
// 0x80a0, //  0: pull   block           side 0
// sm_config_set_sideset(&c, 1, false, false);

#define parallel_out_wrap_target 0
#define parallel_out_wrap 0

static const uint16_t parallel_out_program_instructions[] = {
            //     .wrap_target
    0x601c, //  1: out    pins, 28        side 0  (was 27, now 28 to include GPIO 27)
            //     .wrap
};

#if !PICO_NO_HARDWARE
static const struct pio_program parallel_out_program = {
    .instructions = parallel_out_program_instructions,
    .length = 1,
    .origin = -1,
};

static inline pio_sm_config parallel_out_program_get_default_config(uint offset) {
    pio_sm_config c = pio_get_default_sm_config();
    sm_config_set_wrap(&c, offset + parallel_out_wrap_target, offset + parallel_out_wrap);
    return c;
}
#endif
