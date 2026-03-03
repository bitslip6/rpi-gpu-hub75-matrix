#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hub75gpu.h"
#include "mymath.h"
#include "pixels.h"

/**
 * @brief Apply a vertical sinusoidal offset to create a sine scroller effect
 *
 * @param dst The destination buffer to render into (assumed pre-allocated)
 * @param src The source image buffer to read from
 * @param amplitude The vertical amplitude of the sine wave in pixels
 * @param frequency The number of complete sine waves across the buffer width
 * @param time_offset Time value for animation (in radians, typically 2π * time_seconds * speed)
 *
 * Each vertical column is shifted up or down based on a sine wave pattern.
 * Pixels shifted out of bounds are clipped, and empty space becomes transparent.
 * The destination buffer must be pre-allocated with matching dimensions.
 */
void apply_sine_wave_vertical(image_buffer_t *dst, const image_buffer_t *src, const float amplitude, const float frequency, const float time_offset) {
    if (!dst || !dst->data || !src || !src->data) {
        return;
    }

    const int32_t width = src->dimensions.x;
    const int32_t height = src->dimensions.y;

    if (width == 0 || height == 0) {
        return;
    }

    // Verify destination buffer has matching dimensions
    if (dst->dimensions.x != width || dst->dimensions.y != height) {
        return;
    }

    // Clear destination buffer to transparent
    memset(dst->data, 0, (size_t)(width * height) * sizeof(RGBA));

    // Process each column
    for (int32_t x = 0; x < width; x++) {
        // Calculate sine offset for this column
        // Phase = 2 * pi * frequency * (x / width) + time_offset
        float phase = 2.0f * (float)M_PI * frequency * ((float)x / (float)width) + time_offset;
        float sine_value = sinf(phase);
        float y_off = roundf(amplitude * sine_value);
        int32_t y_offset = (int32_t)y_off;

        // Copy pixels from source column to destination column with offset
        for (int32_t src_y = 0; src_y < height; src_y++) {
            int32_t dst_y = src_y + y_offset;

            // Only copy if destination is within bounds
            if (dst_y >= 0 && dst_y < height) {
                int32_t src_idx = src_y * width + x;
                int32_t dst_idx = dst_y * width + x;
                dst->data[dst_idx] = src->data[src_idx];
            }
            // Pixels out of bounds are not copied (dst remains transparent)
        }
    }
}


/**
 * @brief interpolate between two colors
 * 
 */
__attribute__((hot))
void interpolate_rgb(RGB* result, const RGB* start, const RGB* end, const Normal ratio) {
    result->r = (uint8_t)(start->r + (end->r - start->r) * ratio);
    result->g = (uint8_t)(start->g + (end->g - start->g) * ratio);
    result->b = (uint8_t)(start->b + (end->b - start->b) * ratio);
}

__attribute__((hot))
void composite_rgba(RGBA* result, const RGBA* src, const RGBA* dst) {
    // Standard straight-alpha "over": out = src*alpha + dst*(1 - alpha)
    const uint8_t inv_a = 255 - src->a;
    const Normal bg_percent = Normal_clamp((float)((float)inv_a / 255.0f));
    const Normal fg_percent = Normal_clamp(1.0f - bg_percent);
    result->r = (uint8_t)(src->r * fg_percent + dst->r * bg_percent);
    result->g = (uint8_t)(src->g * fg_percent + dst->g * bg_percent);
    result->b = (uint8_t)(src->b * fg_percent + dst->b * bg_percent);
    result->a = (uint8_t)((float)src->a + (float)dst->a * bg_percent);
}

/**
 * Composite src onto dst with sub pixel sampling and scaling.
 *
 * src_quad and dst_quad are in float image space, [x0, y0, x1, y1).
 * The area src_quad is mapped onto dst_quad.
 *
 * Example: to scale a src rect 100x50 into a dst rect 200x100,
 * pass src_quad = {0,0,100,50}, dst_quad = {dx,dy,dx+200,dy+100}.
 */
void composite_rgba_over_rgba(image_buffer_t       *dst,
                               const image_buffer_t *src,
                               const vec4                  dst_quad,
                               const vec4                  src_quad) {
    if (!dst || !src || !dst->data || !src->data) {
        printf("no data!\n");
        return;
    }

    /* compute integer dst bounds from float quad, preserving coverage */
    int32_t dst_x0 = (int32_t)floorf(dst_quad.x);
    int32_t dst_y0 = (int32_t)floorf(dst_quad.y);
    int32_t dst_x1 = (int32_t)ceilf(dst_quad.z);
    int32_t dst_y1 = (int32_t)ceilf(dst_quad.w);

    /* clamp dst rect to dst image bounds */
    dst_x0 = clamp_int(dst_x0, 0, dst->dimensions.x);
    dst_y0 = clamp_int(dst_y0, 0, dst->dimensions.y);
    dst_x1 = clamp_int(dst_x1, 0, dst->dimensions.x);
    dst_y1 = clamp_int(dst_y1, 0, dst->dimensions.y);

    const int32_t dst_w = dst_x1 - dst_x0;
    const int32_t dst_h = dst_y1 - dst_y0;

    if (dst_w <= 0 || dst_h <= 0) {
        return;
    }

    /* precompute src extents and sizes in float space */
    const float src_x0 = src_quad.x;
    const float src_y0 = src_quad.y;
    const float src_x1 = src_quad.z;
    const float src_y1 = src_quad.w;

    const float src_w  = src_x1 - src_x0;
    const float src_h  = src_y1 - src_y0;

    if (src_w <= 0.0f || src_h <= 0.0f) {
        printf("src hw < 0\n");
        return;
    }

    const float dst_quad_w = dst_quad.z - dst_quad.x;
    const float dst_quad_h = dst_quad.w - dst_quad.y;

    if (dst_quad_w <= 0.0f || dst_quad_h <= 0.0f) {
        printf("dquad < 0\n");
        return;
    }

    /* ratios: how dst float positions map into src float positions */
    const float inv_dst_quad_w = 1.0f / dst_quad_w;
    const float inv_dst_quad_h = 1.0f / dst_quad_h;

    const float src_w_over_dst = src_w;
    const float src_h_over_dst = src_h;

    /* iterate over integer dst pixels in the clipped rectangle */
    for (int32_t dy = dst_y0; dy < dst_y1; ++dy) {
        RGBA *dst_row = buffer_get_px(dst, dst_x0, dy);

        for (int32_t dx = dst_x0; dx < dst_x1; ++dx) {
            /* compute normalized position of this dst pixel center inside dst_quad */
            const float dst_fx = (float)dx + 0.5f;
            const float dst_fy = (float)dy + 0.5f;

            const float u = clampf((dst_fx - dst_quad.x) * inv_dst_quad_w, 0.0f, 1.0f);
            const float v = clampf((dst_fy - dst_quad.y) * inv_dst_quad_h, 0.0f, 1.0f);

            /* map normalized coords into src_quad space (sub pixel) */
            const float sx = src_x0 + u * src_w_over_dst;
            const float sy = src_y0 + v * src_h_over_dst;

            RGBA src_sample;
            sample_rgba_bilinear(&src_sample, src, sx, sy);

            RGBA *dst_px = dst_row + (dx - dst_x0);
            composite_rgba(dst_px, &src_sample, dst_px);
        }
    }
}