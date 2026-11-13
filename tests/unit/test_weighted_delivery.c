/* Shared runtime/legacy mapper path; no GPIO/device initialization. */
#include "weighted_bcm.h"
#include "pixels.h"
#include "rpihub75.h"
#include "transformers.h"
#include <assert.h>
#include <math.h>

static void unexpected_tone(const RGBF *in, RGBF *out, float level) {
    (void)in; (void)out; (void)level;
    assert(!"weighted delivery invoked legacy tone mapping");
}

/* Reference is the real legacy spatial pass, followed by plain weighted
 * encoding: the fused implementation must match without modifying its input. */
extern void dither_spatial_hash_low(uint8_t *image, int width, int height,
                                    uint8_t stride, int cutoff, int strength);

static void expected_encode(const hub75_display_t *d, const uint8_t *image, size_t bytes,
                             uint32_t *expected, size_t words) {
    h75_weighted_lut lut;
    assert(h75_weighted_lut_build(d, &lut) == H75_OK);
    if (d->dither > 0) {
        uint8_t *reference = malloc(bytes);
        assert(reference);
        memcpy(reference, image, bytes);
        dither_spatial_hash_low(reference, d->width, d->height, d->stride, 80, (int)d->dither);
        assert(h75_weighted_encode(d, &lut, reference, bytes, expected, words) == H75_OK);
        free(reference);
    } else assert(h75_weighted_encode(d, &lut, image, bytes, expected, words) == H75_OK);
}

static void mapper_layout(unsigned ports, unsigned bits, unsigned stride) {
    hub75_display_t *d = hub75_display_new();
    assert(d);
    d->width = 32; d->panel_width = 16; d->num_chains = 2;
    d->panel_height = 32; d->height = (uint16_t)(32 * ports); d->num_ports = (uint8_t)ports;
    d->output_mode = H75_OUTPUT_WEIGHTED_BCM;
    d->pwm_bits = bits; d->stride = (uint8_t)stride; d->brightness = 254;
    d->tone_mapper = unexpected_tone; d->quant_dither = true;
    d->jitter_brightness = true; d->dither = 10.0f;
    d->gamma = 2.2f;
    assert(!d->accum && !d->quant_errors_lut);
    size_t words;
    assert(h75_weighted_word_count(d, &words) == H75_OK);
    d->ring_buf_renderer = spsc_create(2, words * sizeof(uint32_t));
    assert(d->ring_buf_renderer);
    const size_t bytes = (size_t)d->width * d->height * stride;
    uint8_t *image = malloc(bytes), *preprocessed = malloc(bytes), *saved = malloc(bytes);
    uint32_t *expected = malloc(words * sizeof(*expected));
    assert(image && preprocessed && saved && expected);
    for (size_t i = 0; i < bytes; i++) image[i] = (uint8_t)(i * 71 + i / 17);
    memcpy(saved, image, bytes);
    d->frame_buffer.data = (RGBA *)image;
    for (unsigned variant = 0; variant < 6; variant++) {
        const float strengths[] = {0, 1, 2, 10, 2, 0};
        d->dither = strengths[variant];
        d->image_mapper = variant == 1 ? mirror_mapper : NULL;
        d->num_panel_types = variant == 2 ? 2 : 0;
        d->color_curve = variant >= 3 ? H75_COLOR_GAMMA : H75_COLOR_CIE;
        d->brightness = variant == 4 ? 0 : variant == 5 ? 127 : 254;
        d->gamma = variant == 5 ? 1.0f : 2.2f;
        d->panel_order = (panel_order_t)variant;
        memcpy(preprocessed, image, bytes);
        if (variant == 1) {
            for (unsigned y = 0; y < d->height; y++)
                for (unsigned x = 0; x < d->width; x++)
                    memcpy(preprocessed + ((size_t)y * d->width + x) * stride,
                           image + ((size_t)y * d->width + d->width - 1 - x) * stride, stride);
        }
        if (variant == 2) {
            d->panel_scale[0] = (panel_rgb_scale){0, 0, 0};
            d->panel_scale[1] = (panel_rgb_scale){255, 255, 255};
            for (unsigned p = 0; p < ports; p++) {
                d->panel_types[p * 2] = 0; d->panel_types[p * 2 + 1] = 1;
                for (unsigned y = 0; y < 32; y++)
                    for (unsigned x = 0; x < 16; x++)
                        memset(preprocessed + (((size_t)p * 32 + y) * 32 + x) * stride, 0, 3);
            }
        }
        expected_encode(d, preprocessed, bytes, expected, words);
        /* Exercise the public dispatcher and the explicit worker-selected path,
         * including frame_buffer fallback and dynamic curve/brightness changes. */
        assert((variant & 1 ? hub75_display_map_image_to_weighted_bcm(d, image)
                           : hub75_display_map_image_to_bcm(d, NULL)) == HUB75_OK);
        const uint32_t *held = spsc_pop_ptr_begin(d->ring_buf_renderer, 0);
        assert(held && !memcmp(held, expected, words * sizeof(*held)));
        assert(!memcmp(image, saved, bytes));
        spsc_pop_ptr_commit(d->ring_buf_renderer);
    }
    /* Backpressure must not overwrite a held/published frame or strand a push. */
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_OK);
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_OK);
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_ERR_RING_BUFFER_FULL);
    assert(spsc_count(d->ring_buf_renderer) == 2);
    for (unsigned i = 0; i < 2; i++) {
        assert(spsc_pop_ptr_begin(d->ring_buf_renderer, 0));
        spsc_pop_ptr_commit(d->ring_buf_renderer);
    }
    /* Wrong mode-sized payload and invalid LUT controls cannot be published. */
    d->pwm_bits = bits == 11 ? 1 : 11;
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_ERR_INVALID_COORDS);
    d->pwm_bits = bits;
    d->gamma = NAN;
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_ERR_INVALID_COORDS);
    assert(spsc_count(d->ring_buf_renderer) == 0);
    d->gamma = 1;
    const float bad_strengths[] = {NAN, INFINITY, -1, 11};
    for (unsigned i = 0; i < sizeof(bad_strengths) / sizeof(*bad_strengths); ++i) {
        d->dither = bad_strengths[i];
        assert(hub75_display_map_image_to_bcm(d, image) == HUB75_ERR_INVALID_COORDS);
        assert(!spsc_count(d->ring_buf_renderer));
    }
    d->dither = 2;
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_OK);
    assert(spsc_pop_ptr_begin(d->ring_buf_renderer, 0));
    spsc_pop_ptr_commit(d->ring_buf_renderer);
    /* Same display/cache, new mode-sized queue: precision changes rebuild LUTs. */
    spsc_destroy(d->ring_buf_renderer);
    d->pwm_bits = bits == 11 ? 1 : 11;
    assert(h75_weighted_word_count(d, &words) == H75_OK);
    d->ring_buf_renderer = spsc_create(2, words * sizeof(uint32_t));
    assert(d->ring_buf_renderer);
    free(expected);
    expected = malloc(words * sizeof(*expected));
    assert(expected);
    expected_encode(d, image, bytes, expected, words);
    assert(hub75_display_map_image_to_bcm(d, image) == HUB75_OK);
    const uint32_t *changed_frame = spsc_pop_ptr_begin(d->ring_buf_renderer, 0);
    assert(changed_frame && !memcmp(changed_frame, expected, words * sizeof(*expected)));
    spsc_pop_ptr_commit(d->ring_buf_renderer);
    hub75_mapper_cache_clear();
    spsc_destroy(d->ring_buf_renderer);
    free(expected); free(saved); free(preprocessed); free(image); free(d);
}

static void same_bytes_new_geometry(void) {
    hub75_display_t *d = hub75_display_new();
    assert(d);
    d->num_ports = 1; d->panel_width = 16;
    d->output_mode = H75_OUTPUT_WEIGHTED_BCM;
    d->pwm_bits = 11; d->brightness = 32; d->gamma = 1;
    d->color_curve = H75_COLOR_CIE; d->dither = 2;
    const size_t bytes = 12288;
    uint8_t *image = malloc(bytes);
    assert(image);
    for (size_t i = 0; i < bytes; ++i) image[i] = (uint8_t)(i % 80);
    for (unsigned shape = 0; shape < 2; ++shape) {
        /* Same display and input byte count, but a larger RGB pattern on the
         * second call: the cache must re-key geometry AND grow its storage. */
        d->width = shape ? 128 : 48;
        d->height = d->panel_height = shape ? 32 : 64;
        d->num_chains = shape ? 8 : 3;
        d->stride = shape ? 3 : 4;
        assert((size_t)d->width * d->height * d->stride == bytes);
        size_t words;
        assert(h75_weighted_word_count(d, &words) == H75_OK);
        d->ring_buf_renderer = spsc_create(2, words * sizeof(uint32_t));
        uint32_t *expected = malloc(words * sizeof(*expected));
        assert(d->ring_buf_renderer && expected);
        expected_encode(d, image, bytes, expected, words);
        assert(hub75_display_map_image_to_weighted_bcm(d, image) == HUB75_OK);
        const uint32_t *held = spsc_pop_ptr_begin(d->ring_buf_renderer, 0);
        assert(held && !memcmp(held, expected, words * sizeof(*held)));
        spsc_pop_ptr_commit(d->ring_buf_renderer);
        spsc_destroy(d->ring_buf_renderer);
        free(expected);
    }
    hub75_mapper_cache_clear();
    free(image); free(d);
}

int main(void) {
#ifdef ADA_HAT
    const unsigned max_ports = 1;
#else
    const unsigned max_ports = 3;
#endif
    for (unsigned ports = 1; ports <= max_ports; ports++)
        for (unsigned bits = 1; bits <= 11; bits += 10)
            for (unsigned stride = 3; stride <= 4; stride++)
                mapper_layout(ports, bits, stride);
    same_bytes_new_geometry();
    puts("Weighted frame-delivery tests passed (no GPIO).");
    return 0;
}
