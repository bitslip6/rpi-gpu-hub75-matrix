/* Pure row-address policies shared by runtime and legacy entry points. */
#include "rpihub75.h"

/* An explicitly supplied global offset still applies to both modes. A
 * weighted-specific override can opt out of it. Resolve this BEFORE assigning
 * the historical equal-slot default below. */
#ifndef WEIGHTED_ROW_ADDRESS_Y_OFFSET
#ifdef ROW_ADDRESS_Y_OFFSET
#define WEIGHTED_ROW_ADDRESS_Y_OFFSET ROW_ADDRESS_Y_OFFSET
#else
#define WEIGHTED_ROW_ADDRESS_Y_OFFSET 0
#endif
#endif
#ifndef ROW_ADDRESS_Y_OFFSET
#define ROW_ADDRESS_Y_OFFSET -1
#endif
#ifndef ADDRESS_LSB_IS_A
#define ADDRESS_LSB_IS_A 1
#endif

static uint32_t address_bits(uint16_t row) {
    const uint8_t order[5] =
#if ADDRESS_LSB_IS_A
        { ADDRESS_A, ADDRESS_B, ADDRESS_C, ADDRESS_D, ADDRESS_E };
#else
        { ADDRESS_E, ADDRESS_D, ADDRESS_C, ADDRESS_B, ADDRESS_A };
#endif
    uint32_t mask = 0;
    for (unsigned bit = 0; bit < 5; bit++)
        if (row & (1u << bit))
            mask |= 1u << order[bit];
    return mask;
}

uint32_t row_to_address(int y, uint16_t half_height) {
    /* Equal-slot exposes the previously latched data during the next shift.
     * Preserve its historical arithmetic too, including uint16 wrap after a
     * negative offset (row 0 on a 16-row scan historically sets E as well). */
    if (!half_height)
        return 0;
    int norm = y % half_height;
    if (norm < 0)
        norm += half_height;
    return address_bits((uint16_t)((norm + ROW_ADDRESS_Y_OFFSET) % half_height));
}

uint32_t h75_weighted_row_to_address(int y, uint16_t half_height) {
    /* Weighted sequencing sets address, shifts, latches, THEN exposes this row.
     * Normalize signed offsets within the scan half, never across halves. */
    if (!half_height)
        return 0;
    int64_t row = ((int64_t)y + WEIGHTED_ROW_ADDRESS_Y_OFFSET) % half_height;
    if (row < 0)
        row += half_height;
    return address_bits((uint16_t)row);
}
