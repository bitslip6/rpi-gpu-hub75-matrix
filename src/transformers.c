#include <stdint.h>

#include "rpihub75.h"
#include "util.h"

/**
 * @brief map the lower half of the image to the front of the image. this allows connecting
 * panels in a left, left, down, right pattern (or right, right, down, left) if the image is
 * mirrored.
 * 
 * NOTE: This code is un-tested. If you have the time, please send me an implementation of U and V
 * mappers
 * 
 * NOTE: we need to expand the scene to support render width hieght and display with height
 *  then we can transform the image from 128x128 to 64x256 for a 1x4 panel setup
 * 
 * 
 * @param image - input buffer to map
 * @param output_image - if NULL, the output buffer will be allocated for you
 * @param scene - the scene information
 * @return uint8_t* - pointer to the output buffer
 */
uint8_t *u_mapper_impl(const uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene) {
    static uint8_t *output_image = NULL;
    if (output_image == NULL) {
        debug("Allocating memory for u_mapper\n"); 
        size_t total = (size_t)scene->width * (size_t)scene->height * (size_t)scene->stride;
        output_image = (uint8_t*)aligned_alloc(64, total);
        if (output_image == NULL) {
            die("Failed to allocate memory for u_mapper image\n");
        }
    }
    if (image_out == NULL) {
        debug("output image is NULL, using allocated memory\n");
        image_out = output_image;
    }


    // Split image into top and bottom halves
    const uint8_t *bottom_half = image_in + (scene->width * (scene->height / 2) * scene->stride);  // Last 64 rows
    const uint32_t row_length = scene->width * scene->stride;

    debug("width: %d, stride: %d, row_length: %d", scene->width, scene->stride, row_length);
    // Remap bottom half to the first part of the output
    for (int y = 0; y < (scene->height / 2); y++) {
        // Copy each row from bottom half
        debug ("  Y: %d, offset: %d", y, y * scene->width * scene->stride);
        memcpy(output_image + (y * scene->width * scene->stride), bottom_half + (y * scene->width * scene->stride), row_length);
    }

    // Remap top half to the second part of the output
    for (int y = 0; y < (scene->height / 2); y++) {
        // Copy each row from top half
        memcpy(output_image + ((y + (scene->width / 2)) * scene->width * scene->stride), image_in + (y * scene->width * scene->stride), row_length);
    }

    return output_image;
}




/**
 * @brief invert the image vertically
 */
__attribute__((hot, flatten))
uint8_t *flip_mapper(const uint8_t *image,
                             uint8_t *image_out,
                             const scene_info *scene)
{
    const size_t row_sz   = (size_t)scene->width * (size_t)scene->stride;
    const size_t height   = (size_t)scene->height;

    if (UNLIKELY(row_sz == 0 || height == 0)){
        return image_out;
    }

    // Fast path, out-of-place: one memcpy per row, sequential IO, best for bandwidth
    if (image_out && image_out != image) {
        uint8_t *__restrict dst = image_out;
        const uint8_t *__restrict src = image;
        // iterate top->bottom on dst, read bottom->top on src
        for (size_t y = 0; y < height; ++y) {
            const uint8_t *src_row = src + (height - 1 - y) * row_sz;
            uint8_t       *dst_row = dst + y * row_sz;

            // prefetch next rows to hide latency on A72/A76
            __builtin_prefetch(src_row - row_sz, 0, 3);
            __builtin_prefetch(dst_row + row_sz, 1, 3);

            memcpy(dst_row, src_row, row_sz);
        }
        return image_out;
    }

    printf("in place flip not supported any longer\n");
    return image_out;
}





#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
static inline uint8x16_t reverse16_u8(uint8x16_t v) {
    // reverse bytes within 64-bit lanes, then swap halves
    uint8x16_t r = vrev64q_u8(v);
    return vextq_u8(r, r, 8);
}
#endif


__attribute__((hot, flatten))
uint8_t *mirror_mapper(const uint8_t *image,
                       uint8_t *image_out,
                       const struct scene_info *scene)
{
    const size_t w   = (size_t)scene->width;
    const size_t h   = (size_t)scene->height;
    const size_t bpp = (size_t)scene->stride;   // RGB => 3
    const size_t row_sz = w * bpp;

    if (UNLIKELY(!image || !image_out || w == 0 || h == 0 || bpp == 0)) {
        return image_out;
    }

    for (size_t y = 0; y < h; ++y) {
        const uint8_t *src_row = image     + y * row_sz;
        uint8_t       *dst_row = image_out + y * row_sz;

#if (defined(__ARM_NEON) || defined(__aarch64__))
        if (LIKELY(bpp == 3)) {
            // NEON path: process 16 RGB pixels (48 bytes) per chunk
            const size_t px_block = 16;
            const size_t blk_bytes = px_block * 3;   // 48
            const size_t blocks = w / px_block;
            const size_t tail_px = w - blocks * px_block;

            // write blocks into destination from left after reserving space for tail
            // first fill the vectorized part: dest base after tail
            uint8_t *dst_blocks_base = dst_row + tail_px * 3;

            for (size_t i = 0; i < blocks; ++i) {
                const uint8_t *s = src_row + i * blk_bytes;

                // deinterleave 16 RGB pixels
                uint8x16x3_t rgb = vld3q_u8(s);

                // reverse pixel order within each channel
                rgb.val[0] = reverse16_u8(rgb.val[0]);
                rgb.val[1] = reverse16_u8(rgb.val[1]);
                rgb.val[2] = reverse16_u8(rgb.val[2]);

                // destination block position (mirror): place from right to left
                uint8_t *d = dst_blocks_base + (blocks - 1 - i) * blk_bytes;

                // interleaved store
                vst3q_u8(d, rgb);
            }

            // tail pixels (remaining leftmost in dest): scalar copy, mirrored
            for (size_t t = 0; t < tail_px; ++t) {
                const size_t src_x  = w - 1 - t;        // rightmost going left
                const size_t dst_x  = t;                // leftmost going right
                const uint8_t *sp = src_row + src_x * 3;
                uint8_t       *dp = dst_row + dst_x * 3;
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
            }
            continue;
        }
#endif
        // Scalar fallback (any bpp), still cache-friendly
        // Copy each row reversed in pixels, preserving per-pixel byte order
        size_t left  = 0;
        size_t right = w - 1;
        while (left < right) {
            uint8_t *dl = dst_row + left  * bpp;
            uint8_t *dr = dst_row + right * bpp;
            const uint8_t *sl = src_row + (w - 1 - left)  * bpp;
            const uint8_t *sr = src_row + (w - 1 - right) * bpp;

            // write two pixels per iteration to reduce loop overhead
            memcpy(dl, sr, bpp);
            memcpy(dr, sl, bpp);

            ++left;
            --right;
        }
        if (left == right) {
            memcpy(dst_row + left * bpp, src_row + (w - 1 - left) * bpp, bpp);
        }
    }

    return image_out;
}




__attribute__((hot, flatten))
uint8_t *mirror_flip_mapper(const uint8_t *image,
                            uint8_t *image_out,
                            const struct scene_info *scene)
{
    const size_t w   = (size_t)scene->width;
    const size_t h   = (size_t)scene->height;
    const size_t bpp = (size_t)scene->stride;         // RGB => 3
    const size_t row_sz = w * bpp;

    if (UNLIKELY(!image || !image_out || w == 0 || h == 0 || bpp == 0)) {
        return image_out;
    }

    for (size_t y = 0; y < h; ++y) {
        // vertical flip selects source row from bottom
        const uint8_t *src_row = image     + (h - 1 - y) * row_sz;
        uint8_t       *dst_row = image_out + y * row_sz;

#if (defined(__ARM_NEON) || defined(__aarch64__))
        if (LIKELY(bpp == 3)) {
            // process 16 RGB pixels (48 bytes) per block with deinterleave/reverse/interleave
            const size_t px_block  = 16;
            const size_t blk_bytes = px_block * 3;  // 48
            const size_t blocks    = w / px_block;
            const size_t tail_px   = w - blocks * px_block;

            // vectorized blocks: dst left→right, src right→left in block-sized chunks
            for (size_t i = 0; i < blocks; ++i) {
                // source block starts px_block pixels from the right edge, moving left
                const size_t src_px   = w - (i + 1) * px_block;
                const uint8_t *s      = src_row + src_px * 3;
                uint8_t       *d      = dst_row + i * blk_bytes;

                __builtin_prefetch(s - 96, 0, 3);
                __builtin_prefetch(d + 96, 1, 3);

                // deinterleave 16 pixels of RGB
                uint8x16x3_t rgb = vld3q_u8(s);

                // reverse pixel order within each channel to mirror horizontally
                rgb.val[0] = reverse16_u8(rgb.val[0]);
                rgb.val[1] = reverse16_u8(rgb.val[1]);
                rgb.val[2] = reverse16_u8(rgb.val[2]);

                // interleaved store to destination
                vst3q_u8(d, rgb);
            }

            // tail pixels at the left of dst, sourced from the left of src but reversed
            for (size_t t = 0; t < tail_px; ++t) {
                const size_t src_x = tail_px - 1 - t; // right-to-left within the remaining tail
                const uint8_t *sp  = src_row + src_x * 3;
                uint8_t       *dp  = dst_row + (blocks * px_block + t) * 3;
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
            }
            continue;
        }
#endif
        // scalar fallback: generic bpp, still sequential writes and reads
        const uint8_t *sp = src_row + (w - 1) * bpp; // start at rightmost pixel of source row
        uint8_t       *dp = dst_row;                 // start at leftmost pixel of dest row

        // copy w pixels, mirrored horizontally
        for (size_t x = 0; x < w; ++x) {
            __builtin_prefetch(sp - 3 * bpp, 0, 1);
            // copy one pixel of bpp bytes
            switch (bpp) {
                case 3:
                    dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                    break;
                case 4:
                    // common case for RGBA buffers
                    ((uint32_t *)dp)[0] = ((const uint32_t *)sp)[0];
                    break;
                default:
                    memcpy(dp, sp, bpp);
                    break;
            }
            dp += bpp;
            sp -= bpp;
        }
    }

    return image_out;
}