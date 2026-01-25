#ifndef COMPOSITOR_H
#define COMPOSITOR_H

#include "hub75gpu.h"

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
void apply_sine_wave_vertical(image_buffer_t *dst, const image_buffer_t *src, const float amplitude, const float frequency, const float time_offset);

/**
 * @brief Composite src pixel over dst pixel using standard "over" alpha blending.
 * dst and result may be the same buffer
 * @param result Pointer to the output pixel (modified in place)
 * @param src Pointer to the source pixel
 * @param dst Pointer to the destination pixel
 */
void composite_rgba(RGBA* result, const RGBA* src, const RGBA* dst);

/**
 * @brief Composite src onto dst with sub pixel sampling and scaling.
 * @param dst The destination image buffer to render into (modified in place)
 * @param src The source image buffer to read from
 * @param dst_quad The destination rectangle in dst to map to (x0, y0, x1, y1)
 * @param src_quad The source rectangle in src to read from (x0, y0, x1, y1)
 */
void composite_rgba_over_rgba(image_buffer_t       *dst,
                               const image_buffer_t *src,
                               const vec4                  dst_quad,
                               const vec4                  src_quad);


#endif // COMPOSITOR_H
