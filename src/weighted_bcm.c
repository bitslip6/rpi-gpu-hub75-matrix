#include "weighted_bcm.h"
#include "rpihub75.h" /* same compile-time GPIO wiring as the legacy encoder */
#include <math.h>

static bool valid_bits(uint32_t bits) {
    return bits >= 1 && bits <= 11;
}

h75_status h75_weighted_lut_build(const hub75_display_t *d, h75_weighted_lut *lut) {
    if (!d || !lut || !valid_bits(d->pwm_bits) || d->brightness > 254 ||
        (d->color_curve != H75_COLOR_CIE && d->color_curve != H75_COLOR_GAMMA) ||
        !isfinite(d->gamma) || d->gamma <= 0.0f || d->gamma > 8.0f)
        return H75_INVALID;

    const float max_pwm = (float)((1u << d->pwm_bits) - 1u);
    const float drive = (float)d->brightness / 254.0f;
    lut->pwm_bits = d->pwm_bits;
    for (unsigned i = 0; i < 256; i++) {
        const float input = (float)i / 255.0f;
        float mapped;
        if (d->color_curve == H75_COLOR_CIE) {
            const float lightness = input * 100.0f;
            /* Exact CIE constants: epsilon=216/24389, kappa=24389/27.
             * The linear branch meets the cubic branch at L*=8. */
            if (lightness <= 8.0f) {
                mapped = lightness * (27.0f / 24389.0f);
            } else {
                const float t = (lightness + 16.0f) / 116.0f;
                mapped = t * t * t;
            }
        } else {
            mapped = powf(input, d->gamma);
        }
        const float scaled = fminf(1.0f, fmaxf(0.0f, mapped * drive)) * max_pwm;
        const uint16_t pwm = (uint16_t)floorf(scaled + 0.5f);
        for (unsigned channel = 0; channel < 3; channel++)
            lut->channel[channel][i] = pwm;
    }
    return H75_OK;
}

static bool multiply_size(size_t a, size_t b, size_t *result) {
    if (a && b > SIZE_MAX / a)
        return false;
    *result = a * b;
    return true;
}

h75_status h75_weighted_word_count(const hub75_display_t *d, size_t *words) {
    if (!words)
        return H75_INVALID;
    *words = 0;
    if (!d || !valid_bits(d->pwm_bits) || !d->width || !d->panel_width ||
        d->num_ports < 1 || d->num_ports > 3 ||
        d->num_chains < 1 || d->num_chains > 16 ||
        (d->panel_height != 32 && d->panel_height != 64) ||
        (uint32_t)d->width != (uint32_t)d->panel_width * d->num_chains ||
        (uint32_t)d->height != (uint32_t)d->panel_height * d->num_ports ||
        (d->stride != 3 && d->stride != 4) ||
        (unsigned)d->panel_order > PANEL_BGR)
        return H75_INVALID;
#ifdef ADA_HAT
    /* This wiring only defines a real set of data pins for port 0. */
    if (d->num_ports != 1)
        return H75_UNSUPPORTED;
#endif
    size_t count, bytes;
    if (!multiply_size(d->width, d->panel_height / 2u, &count) ||
        !multiply_size(count, d->pwm_bits, &count) ||
        !multiply_size(count, sizeof(uint32_t), &bytes))
        return H75_INVALID;
    *words = count;
    return H75_OK;
}

h75_status h75_weighted_encode(const hub75_display_t *d, const h75_weighted_lut *lut,
                               const uint8_t *image, size_t image_bytes,
                               uint32_t *output, size_t output_words) {
    size_t words, pixels, bytes;
    h75_status status = h75_weighted_word_count(d, &words);
    if (status != H75_OK)
        return status;
    if (!lut || !image || !output || lut->pwm_bits != d->pwm_bits ||
        !multiply_size(d->width, d->height, &pixels) ||
        !multiply_size(pixels, d->stride, &bytes) ||
        image_bytes < bytes || output_words < words)
        return H75_INVALID;
    const uint32_t max_pwm = (1u << d->pwm_bits) - 1u;
    for (unsigned c = 0; c < 3; c++)
        for (unsigned i = 0; i < 256; i++)
            if (lut->channel[c][i] > max_pwm)
                return H75_INVALID;

    /* source[order][wire] matches legacy build_idx_remap(), including the
     * non-self-inverse GBR/BRG permutations. No shared mutable LUT cache. */
    static const uint8_t source[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
    };
    static const uint8_t pin[3][2][3] = {
        {{ADDRESS_P0_R1, ADDRESS_P0_G1, ADDRESS_P0_B1},
         {ADDRESS_P0_R2, ADDRESS_P0_G2, ADDRESS_P0_B2}},
        {{ADDRESS_P1_R1, ADDRESS_P1_G1, ADDRESS_P1_B1},
         {ADDRESS_P1_R2, ADDRESS_P1_G2, ADDRESS_P1_B2}},
        {{ADDRESS_P2_R1, ADDRESS_P2_G1, ADDRESS_P2_B1},
         {ADDRESS_P2_R2, ADDRESS_P2_G2, ADDRESS_P2_B2}}
    };
    const size_t rows = d->panel_height / 2u;
    const size_t plane_words = rows * d->width;
    for (size_t y = 0; y < rows; y++) {
        for (size_t x = 0; x < d->width; x++) {
            uint32_t packed[11] = {0};
            for (unsigned port = 0; port < d->num_ports; port++) {
                for (unsigned half = 0; half < 2; half++) {
                    const size_t image_y = (size_t)port * d->panel_height + half * rows + y;
                    const uint8_t *pixel = image + (image_y * d->width + x) * d->stride;
                    for (unsigned wire = 0; wire < 3; wire++) {
                        const unsigned channel = source[d->panel_order][wire];
                        const uint16_t value = lut->channel[channel][pixel[channel]];
                        const uint32_t mask = 1u << pin[port][half][wire];
                        for (unsigned plane = 0; plane < d->pwm_bits; plane++)
                            if (value & (1u << plane))
                                packed[plane] |= mask;
                    }
                }
            }
            for (unsigned plane = 0; plane < d->pwm_bits; plane++)
                output[plane * plane_words + y * d->width + x] = packed[plane];
        }
    }
    return H75_OK;
}
