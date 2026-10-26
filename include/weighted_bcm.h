#ifndef H75_WEIGHTED_BCM_H
#define H75_WEIGHTED_BCM_H

#include <stddef.h>
#include "hub75_runtime.h"

/* Caller-owned, immutable during encoding. No globals, allocations or GPIO I/O.
 * Rebuild when pwm_bits, color_curve, gamma or brightness changes.
 * Each logical channel has its own byte -> final-precision PWM table. */
typedef struct {
    uint32_t pwm_bits;
    uint16_t channel[3][256]; /* logical R, G, B; values <= (1 << pwm_bits)-1 */
} h75_weighted_lut;

/* Uses only pwm_bits, color_curve, gamma and brightness from display.
 * CIE interprets byte/255 as normalized perceptual L* (inverse CIE L*).
 * GAMMA uses powf(byte/255, gamma). Brightness is linear drive * brightness/254
 * AFTER the curve; clamp and round (nearest, ties up) only at PWM precision.
 * Tone mapping, jitter and all dithering are intentionally ignored. */
h75_status h75_weighted_lut_build(const hub75_display_t *display, h75_weighted_lut *lut);

/* Geometry: width=panel_width*num_chains; height=panel_height*num_ports;
 * 1..3 ports, 1..16 chains, 32/64-row panels, stride 3 (RGB) or 4 (RGBA).
 * Returns pwm_bits * (panel_height/2) * width uint32_t words, checked for
 * size_t overflow. No equal-slot alignment requirement is imposed here. */
h75_status h75_weighted_word_count(const hub75_display_t *display, size_t *words);

/* Input is tightly packed, already image/panel-mapped RGB(A), with logical
 * port images stacked vertically and chained panels laid out left-to-right.
 * Alpha is ignored, as in the legacy encoder. Image mapping/panel calibration
 * may be done by existing preprocessing before this call; no curve is applied
 * to an intermediate 8-bit image here. No input or display state is mutated.
 * Output: [plane LSB first][scanned row][x], containing DATA bits ONLY.
 * LUT must match pwm_bits and contain in-range values. Buffers must not overlap.
 * Byte/word capacities are checked before any output is written. */
h75_status h75_weighted_encode(const hub75_display_t *display,
                               const h75_weighted_lut *lut,
                               const uint8_t *image, size_t image_bytes,
                               uint32_t *output, size_t output_words);

#endif
