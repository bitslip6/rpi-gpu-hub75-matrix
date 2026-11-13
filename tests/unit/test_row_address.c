/* Actual production address policies, without GPIO, runtime or library loading. */
#include "rpihub75.h"
#include <limits.h>

#ifndef TEST_LEGACY_OFFSET
#define TEST_LEGACY_OFFSET -1
#endif
#ifndef TEST_WEIGHTED_OFFSET
#define TEST_WEIGHTED_OFFSET 0
#endif
#ifndef TEST_REVERSED
#define TEST_REVERSED 0
#endif

static uint32_t expected(int y, unsigned rows, int offset, bool legacy) {
    int64_t row = (int64_t)y % rows;
    if (row < 0) row += rows;
    row = (row + offset) % rows;
    if (legacy)
        row = (uint16_t)row; /* Preserve the existing equal-slot wrap quirk. */
    else if (row < 0)
        row += rows;
    const unsigned pins[] = {ADDRESS_A, ADDRESS_B, ADDRESS_C, ADDRESS_D, ADDRESS_E};
    uint32_t mask = 0;
    for (unsigned b = 0; b < 5; b++)
        if ((uint64_t)row & (1u << b))
            mask |= 1u << pins[TEST_REVERSED ? 4 - b : b];
    return mask;
}

int main(void) {
    assert(row_to_address(0, 0) == 0);
    assert(h75_weighted_row_to_address(0, 0) == 0);
    for (unsigned rows = 16; rows <= 32; rows += 16) {
        /* Every row, both halves, negative indices and several full wraps. */
        for (int y = -(int)rows * 3; y < (int)rows * 4; y++) {
            assert(row_to_address(y, (uint16_t)rows) == expected(y, rows, TEST_LEGACY_OFFSET, true));
            assert(h75_weighted_row_to_address(y, (uint16_t)rows) == expected(y, rows, TEST_WEIGHTED_OFFSET, false));
            assert(h75_weighted_row_to_address(y, (uint16_t)rows) ==
                   h75_weighted_row_to_address(y + (int)rows, (uint16_t)rows));
        }
        const int extremes[] = {INT_MIN, INT_MAX};
        for (unsigned i = 0; i < 2; i++) {
            assert(row_to_address(extremes[i], (uint16_t)rows) == expected(extremes[i], rows, TEST_LEGACY_OFFSET, true));
            assert(h75_weighted_row_to_address(extremes[i], (uint16_t)rows) == expected(extremes[i], rows, TEST_WEIGHTED_OFFSET, false));
        }
    }
    puts("Production row-address policies passed (pin-free).");
    return 0;
}
