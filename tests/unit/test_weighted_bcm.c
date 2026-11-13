/* Standalone, hardware-free weighted encoder tests. */
#include "weighted_bcm.h"
#include "rpihub75.h"
#include <math.h>
#include <stdio.h>

static hub75_display_t config(void) {
    return (hub75_display_t){.width = 32, .height = 32, .panel_width = 16,
        .panel_height = 32, .num_ports = 1, .num_chains = 2, .stride = 4,
        .pwm_bits = 11, .brightness = 254, .gamma = 1.0f,
        .color_curve = H75_COLOR_CIE, .panel_order = PANEL_RGB};
}

static void unexpected_tone(const RGBF *in, RGBF *out, float level) {
    (void)in; (void)out; (void)level;
    assert(!"weighted LUT must not invoke the legacy tone mapper");
}

static void curves(void) {
    hub75_display_t d = config();
    h75_weighted_lut lut, full;
    for (unsigned bits = 1; bits <= 11; bits++) {
        d.pwm_bits = bits;
        for (unsigned curve = 0; curve < 2; curve++) {
            d.color_curve = (h75_color_curve)curve;
            d.gamma = 2.2f;
            assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
            for (unsigned c = 0; c < 3; c++) {
                assert(lut.channel[c][0] == 0);
                assert(lut.channel[c][255] == (1u << bits) - 1);
                for (unsigned i = 1; i < 256; i++)
                    assert(lut.channel[c][i] >= lut.channel[c][i-1]);
            }
        }
    }
    d = config();
    assert(h75_weighted_lut_build(&d, &full) == H75_OK);
    const unsigned input[] = {0, 1, 2, 3, 4, 10, 20, 21, 64, 128, 192, 255};
    const unsigned expected[] = {0, 1, 2, 3, 4, 9, 18, 19, 91, 380, 998, 2047};
    for (unsigned c = 0; c < 3; c++)
        for (unsigned i = 0; i < sizeof(input)/sizeof(*input); i++)
            assert(full.channel[c][input[i]] == expected[i]);
    /* Inputs 1..4 remain distinct: no corrected 8-bit intermediate. */
    for (unsigned i = 1; i <= 4; i++)
        assert(full.channel[0][i] == i);
    for (unsigned brightness = 0; brightness <= 254; brightness++) {
        d.brightness = (uint8_t)brightness;
        assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
        assert(lut.channel[0][255] == (uint16_t)floorf(2047.0f * (float)brightness / 254.0f + 0.5f));
        for (unsigned i = 0; i < 256; i++) {
            assert(lut.channel[0][i] <= full.channel[0][i]);
            /* brightness scales the unrounded mapped drive, not rounded PWM */
            const double l = (double)i / 255.0 * 100.0;
            const double drive = l <= 8 ? l * 27.0 / 24389.0 : pow((l + 16.0) / 116.0, 3);
            assert(lut.channel[0][i] == (unsigned)floor(drive * brightness / 254.0 * 2047.0 + 0.5));
        }
    }
    d = config();
    d.color_curve = H75_COLOR_GAMMA;
    d.gamma = 2.0f;
    assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
    assert(lut.channel[0][128] == 516);
    d.gamma = 1.0f;
    assert(h75_weighted_lut_build(&d, &full) == H75_OK);
    /* Legacy-only settings/callbacks have no effect, nor does output_mode. */
    d.jitter_brightness = true; d.quant_dither = true; d.dither = 10;
    d.tone_level = 99; d.tone_mapper = unexpected_tone;
    d.output_mode = H75_OUTPUT_WEIGHTED_BCM;
    assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
    assert(!memcmp(&lut, &full, sizeof(lut)));
}

static const unsigned pins[3][2][3] = {
    {{ADDRESS_P0_R1, ADDRESS_P0_G1, ADDRESS_P0_B1}, {ADDRESS_P0_R2, ADDRESS_P0_G2, ADDRESS_P0_B2}},
    {{ADDRESS_P1_R1, ADDRESS_P1_G1, ADDRESS_P1_B1}, {ADDRESS_P1_R2, ADDRESS_P1_G2, ADDRESS_P1_B2}},
    {{ADDRESS_P2_R1, ADDRESS_P2_G1, ADDRESS_P2_B1}, {ADDRESS_P2_R2, ADDRESS_P2_G2, ADDRESS_P2_B2}}
};
/* Independent wire-name strings, not encoder source-index tables. */
static const char *orders[] = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"};
static unsigned logical_channel(char color) {
    return color == 'R' ? 0u : color == 'G' ? 1u : 2u;
}
static unsigned reconstruct(const hub75_display_t *d, const uint32_t *output,
                            size_t pos, unsigned pin) {
    unsigned value = 0;
    const size_t plane_words = (size_t)d->width * (d->panel_height / 2u);
    for (unsigned plane = 0; plane < d->pwm_bits; plane++)
        value |= ((output[plane * plane_words + pos] >> pin) & 1u) << plane;
    return value;
}

static void layout(void) {
    /* All permutations, 1..3 ports, 32/64 rows, RGB/RGBA, 1 and 3 chains.
     * Every pixel/channel is reconstructed independently, including boundaries. */
#ifdef ADA_HAT
    const unsigned max_ports = 1;
#else
    const unsigned max_ports = 3;
#endif
    for (unsigned ports = 1; ports <= max_ports; ports++)
    for (unsigned height = 32; height <= 64; height += 32)
    for (unsigned stride = 3; stride <= 4; stride++)
    for (unsigned chains = 1; chains <= 3; chains += 2)
    for (unsigned order = 0; order < 6; order++) {
        hub75_display_t d = config();
        d.num_ports = (uint8_t)ports; d.panel_height = (uint16_t)height;
        d.height = (uint16_t)(height * ports); d.stride = (uint8_t)stride;
        d.num_chains = (uint8_t)chains; d.width = (uint16_t)(16 * chains);
        d.panel_order = (panel_order_t)order;
        size_t words;
        assert(h75_weighted_word_count(&d, &words) == H75_OK);
        assert(words == (size_t)11 * (height / 2) * d.width);
        size_t bytes = (size_t)d.width * d.height * stride;
        uint8_t *image = malloc(bytes), *saved = malloc(bytes);
        uint32_t *output = malloc((words + 1) * sizeof(*output));
        assert(image && saved && output);
        for (size_t i = 0; i < bytes; i++) image[i] = (uint8_t)(i * 71u + i / 19u);
        memcpy(saved, image, bytes);
        output[words] = 0xdeadbeef;
        h75_weighted_lut lut;
        assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
        /* Distinguish channel-specific LUT addressing as well as ordering. */
        for (unsigned c = 0; c < 3; c++)
            for (unsigned i = 0; i < 256; i++)
                lut.channel[c][i] = (uint16_t)((lut.channel[c][i] + c * 157u) & 2047u);
        assert(h75_weighted_encode(&d, &lut, image, bytes, output, words) == H75_OK);
        uint32_t active = 0;
        for (unsigned port = 0; port < ports; port++)
        for (unsigned half = 0; half < 2; half++)
        for (unsigned wire = 0; wire < 3; wire++) {
            active |= 1u << pins[port][half][wire];
            const unsigned c = logical_channel(orders[order][wire]);
            for (unsigned y = 0; y < height / 2; y++)
            for (unsigned x = 0; x < d.width; x++) {
                size_t image_y = port * height + half * height / 2 + y;
                uint8_t input = image[(image_y * d.width + x) * stride + c];
                assert(reconstruct(&d, output, (size_t)y * d.width + x,
                                   pins[port][half][wire]) == lut.channel[c][input]);
            }
        }
        for (size_t i = 0; i < words; i++) assert(!(output[i] & ~active));
        assert(output[words] == 0xdeadbeef && !memcmp(image, saved, bytes));
        free(output); free(saved); free(image);
    }
}

/* Independent reference for the existing legacy hash/noise convention. */
static uint8_t spatial_reference(uint8_t input, unsigned x, unsigned y,
                                  unsigned channel, unsigned strength) {
    if (input >= 80) return input;
    uint32_t h = (uint32_t)x * 0x9E3779B1u ^ ((uint32_t)y + 0x7F4A7C15u);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    int noise = (int)((h >> (channel * 6)) & 63u) - 31;
    int offset = (int)((float)noise * ((float)strength / 31.f) + .5f);
    int adjusted = input + offset;
    return (uint8_t)(adjusted < 0 ? 0 : adjusted > 254 ? 254 : adjusted);
}

static void spatial(void) {
#ifdef ADA_HAT
    const unsigned max_ports = 1;
#else
    const unsigned max_ports = 3;
#endif
    const unsigned strengths[] = {0, 1, 2, 10};
    const uint8_t inputs[] = {0, 1, 5, 16, 32, 64, 79, 80, 81, 128, 254, 255};
    bool changed = false;
    for (unsigned ports = 1; ports <= max_ports; ++ports)
    for (unsigned height = 32; height <= 64; height += 32)
    for (unsigned stride = 3; stride <= 4; ++stride)
    for (unsigned order = 0; order < 6; ++order) {
        hub75_display_t d = config();
        d.num_ports = (uint8_t)ports;
        d.panel_height = (uint16_t)height;
        d.height = (uint16_t)(height * ports);
        d.stride = (uint8_t)stride;
        d.panel_order = (panel_order_t)order;
        d.brightness = 32;
        size_t words;
        assert(h75_weighted_word_count(&d, &words) == H75_OK);
        size_t pixels = (size_t)d.width * d.height, bytes = pixels * stride;
        size_t pattern_bytes = pixels * 3;
        uint8_t *image = malloc(bytes), *saved = malloc(bytes);
        int8_t *pattern = malloc(pattern_bytes + 1), *second = malloc(pattern_bytes);
        uint32_t *plain = malloc(words * sizeof(*plain));
        uint32_t *output = malloc((words + 1) * sizeof(*output));
        assert(image && saved && pattern && second && plain && output);
        for (size_t p = 0; p < pixels; ++p)
            for (unsigned c = 0; c < stride; ++c)
                image[p * stride + c] = inputs[(p + c * 3) % (sizeof(inputs) / sizeof(*inputs))];
        memcpy(saved, image, bytes);
        h75_weighted_lut lut;
        assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
        assert(h75_weighted_encode(&d, &lut, image, bytes, plain, words) == H75_OK);
        for (unsigned s = 0; s < sizeof(strengths) / sizeof(*strengths); ++s) {
            unsigned strength = strengths[s];
            d.dither = (float)strength;
            pattern[pattern_bytes] = 99;
            output[words] = 0xdeadbeef;
            assert(h75_weighted_spatial_build(&d, pattern, pattern_bytes) == H75_OK);
            assert(h75_weighted_spatial_build(&d, second, pattern_bytes) == H75_OK);
            assert(!memcmp(pattern, second, pattern_bytes));
            assert(pattern[pattern_bytes] == 99);
            assert(h75_weighted_encode_spatial(&d, &lut, image, bytes, output, words,
                                               pattern, pattern_bytes) == H75_OK);
            if (!strength) assert(!memcmp(plain, output, words * sizeof(*plain)));
            else if (memcmp(plain, output, words * sizeof(*plain))) changed = true;
            for (unsigned port = 0; port < ports; ++port)
            for (unsigned half = 0; half < 2; ++half)
            for (unsigned wire = 0; wire < 3; ++wire)
            for (unsigned y = 0; y < height / 2; ++y)
            for (unsigned x = 0; x < d.width; ++x) {
                unsigned image_y = port * height + half * height / 2 + y;
                unsigned c = logical_channel(orders[order][wire]);
                uint8_t input = image[((size_t)image_y * d.width + x) * stride + c];
                uint8_t adjusted = spatial_reference(input, x, image_y, c, strength);
                assert(reconstruct(&d, output, (size_t)y * d.width + x, pins[port][half][wire]) ==
                       lut.channel[c][adjusted]);
            }
            /* Encoding never advances the pattern and never mutates RGBA. */
            assert(h75_weighted_encode_spatial(&d, &lut, image, bytes, plain, words,
                                               pattern, pattern_bytes) == H75_OK);
            assert(!memcmp(plain, output, words * sizeof(*plain)));
            assert(!memcmp(image, saved, bytes) && !memcmp(pattern, second, pattern_bytes));
            assert(output[words] == 0xdeadbeef);
            /* The old direct entry point remains non-dithered even with the flag set. */
            assert(h75_weighted_encode(&d, &lut, image, bytes, plain, words) == H75_OK);
        }
        d.brightness = 0;
        assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
        assert(h75_weighted_encode_spatial(&d, &lut, image, bytes, output, words,
                                           pattern, pattern_bytes) == H75_OK);
        for (size_t i = 0; i < words; ++i) assert(!output[i]);
        free(output); free(plain); free(second); free(pattern); free(saved); free(image);
    }
    assert(changed);
}

static void single_bits(void) {
    hub75_display_t d = config();
    size_t words;
    assert(h75_weighted_word_count(&d, &words) == H75_OK);
    uint32_t *output = malloc(words * sizeof(*output));
    size_t bytes = (size_t)d.width * d.height * d.stride;
    uint8_t *image = calloc(bytes, 1);
    assert(output && image);
    /* Synthetic LUT lets each individual PWM bit be checked even when no
     * input byte happens to map to that exact power of two under the curve. */
    for (unsigned bits = 1; bits <= 11; bits++) {
        d.pwm_bits = bits;
        for (unsigned plane = 0; plane < bits; plane++) {
            h75_weighted_lut lut = {.pwm_bits = bits};
            lut.channel[0][1] = (uint16_t)(1u << plane);
            image[0] = 1;
            assert(h75_weighted_encode(&d, &lut, image, bytes, output, words) == H75_OK);
            const size_t plane_words = (size_t)d.width * d.panel_height / 2;
            for (size_t i = 0; i < bits * plane_words; i++)
                assert(output[i] == (i == plane * plane_words ? 1u << ADDRESS_P0_R1 : 0));
        }
    }
    free(image); free(output);
}

static void invalid(void) {
    hub75_display_t d = config();
    h75_weighted_lut lut;
    assert(h75_weighted_lut_build(NULL, &lut) == H75_INVALID);
    assert(h75_weighted_lut_build(&d, NULL) == H75_INVALID);
    const float bad_gamma[] = {NAN, INFINITY, -1, 0, 9};
    for (unsigned i = 0; i < sizeof(bad_gamma)/sizeof(*bad_gamma); i++) {
        d.gamma = bad_gamma[i];
        assert(h75_weighted_lut_build(&d, &lut) == H75_INVALID);
    }
    d = config(); d.color_curve = (h75_color_curve)-1;
    assert(h75_weighted_lut_build(&d, &lut) == H75_INVALID);
    d = config(); d.brightness = 255;
    assert(h75_weighted_lut_build(&d, &lut) == H75_INVALID);
    d = config();
    size_t words;
    assert(h75_weighted_word_count(NULL, &words) == H75_INVALID && words == 0);
    assert(h75_weighted_word_count(&d, NULL) == H75_INVALID);
#define BAD(field, value) do { \
    d = config(); d.field = (value); \
    assert(h75_weighted_word_count(&d, &words) == H75_INVALID && words == 0); \
} while (0)
    BAD(pwm_bits, 0); BAD(pwm_bits, 12); BAD(width, 0); BAD(width, 31);
    BAD(height, 64); BAD(panel_height, 16); BAD(num_ports, 0); BAD(num_ports, 4);
    BAD(num_chains, 0); BAD(num_chains, 17); BAD(stride, 2); BAD(panel_order, PANEL_BGR+1);
#undef BAD
    d = config();
    assert(h75_weighted_word_count(&d, &words) == H75_OK);
    size_t bytes = (size_t)d.width * d.height * d.stride;
    uint8_t *image = calloc(bytes, 1);
    uint32_t *output = malloc(words * sizeof(*output));
    assert(image && output);
    for (size_t i = 0; i < words; i++) output[i] = 0xdeadbeef;
    assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
    assert(h75_weighted_encode(&d, &lut, image, bytes-1, output, words) == H75_INVALID);
    assert(h75_weighted_encode(&d, &lut, image, bytes, output, words-1) == H75_INVALID);
    assert(h75_weighted_encode(&d, NULL, image, bytes, output, words) == H75_INVALID);
    assert(h75_weighted_encode(&d, &lut, NULL, bytes, output, words) == H75_INVALID);
    assert(h75_weighted_encode(&d, &lut, image, bytes, NULL, words) == H75_INVALID);
    lut.pwm_bits = 10;
    assert(h75_weighted_encode(&d, &lut, image, bytes, output, words) == H75_INVALID);
    lut.pwm_bits = 11; lut.channel[2][255] = 2048;
    assert(h75_weighted_encode(&d, &lut, image, bytes, output, words) == H75_INVALID);
    for (size_t i = 0; i < words; i++) assert(output[i] == 0xdeadbeef);
    /* Malformed patterns/strengths fail before any output is touched. */
    size_t pattern_bytes = (size_t)d.width * d.height * 3;
    int8_t *pattern = malloc(pattern_bytes);
    assert(pattern);
    memset(pattern, 42, pattern_bytes);
    const float bad_strengths[] = {NAN, INFINITY, -1, 11};
    for (unsigned i = 0; i < sizeof(bad_strengths) / sizeof(*bad_strengths); ++i) {
        d.dither = bad_strengths[i];
        assert(h75_weighted_spatial_build(&d, pattern, pattern_bytes) == H75_INVALID);
        for (size_t j = 0; j < pattern_bytes; ++j) assert(pattern[j] == 42);
    }
    d.dither = 2;
    assert(h75_weighted_spatial_build(NULL, pattern, pattern_bytes) == H75_INVALID);
    assert(h75_weighted_spatial_build(&d, NULL, pattern_bytes) == H75_INVALID);
    assert(h75_weighted_spatial_build(&d, pattern, pattern_bytes - 1) == H75_INVALID);
    assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
    assert(h75_weighted_encode_spatial(&d, &lut, image, bytes, output, words,
                                       NULL, pattern_bytes) == H75_INVALID);
    assert(h75_weighted_encode_spatial(&d, &lut, image, bytes, output, words,
                                       pattern, pattern_bytes - 1) == H75_INVALID);
    for (size_t i = 0; i < words; ++i) assert(output[i] == 0xdeadbeef);
    free(pattern); free(output); free(image);
}

int main(void) {
    curves(); layout(); spatial(); single_bits(); invalid();
    puts("Weighted BCM LUT/plane tests passed (no GPIO).");
    return 0;
}
