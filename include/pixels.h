#include <stdint.h>
#include "rpihub75.h"
#include "hub75gpu.h"

#ifndef _HUB75_PIXELS_H
#define _HUB75_PIXELS_H 1

// ACES tone mapping constants
#define ACES_A 2.51f
#define ACES_B 0.03f
#define ACES_C 2.43f
#define ACES_D 0.59f
#define ACES_E 0.14f

// Constants for Uncharted 2 tone mapping
#define UNCHART_A 0.15f
#define UNCHART_B 0.50f
#define UNCHART_C 0.10f
#define UNCHART_D 0.20f
#define UNCHART_E 0.02f
#define UNCHART_F 0.30f
#define UNCHART_W 11.2f  // White point (adjust as needed)



// Helper functions
#define ipart(x) ((int)(x))      // Integer part of x
#define fpart(x) ((x) - floorf(x)) // Fractional part of x
#define rfpart(x) (1.0f - fpart(x)) // 1 - fractional part of x


void *main_thread_mapper(void *arg);

/**
 * @brief function definition to function that maps RGB image data
 * to BCM data for shifting out to GPIO
 */
typedef void (*update_bcm_signal_fn)(
    const hub75_display_t *scene,
    const void *bits,  // Use void* to handle both uint32_t* and uint64_t*
    uint32_t *bcm_signal,
    const uint8_t *image,
    const uint16_t *quant_err);

/**
 * @brief update_bcm_signal_fn implementation for up to 64 bit BCM data
 * 
 * @param scene  pointer to current scene info
 * @param void_bits RGB to BCM index data. Red 0-255, Green 256-511, Blue 512 - 768
 * @param bcm_signal the ouput buffer to write BCM data to
 * @param image the input RGB or RGBA image to render
 * @param offset offset into the bcm buffer
 */
__attribute__((hot))
void update_bcm_signal_64_rgb(
    const hub75_display_t *scene,
    const void *__restrict__ void_bits,
    uint32_t *__restrict__ bcm_signal,
    const uint8_t *__restrict__ image,
    uint8_t phase
);


void interpolate_rgb(RGB* result, const RGB* start, const RGB* end, const Normal ratio);

/**
 * helper to get a pixel from a buffer. compiler will inline
 */
RGBA *buffer_get_px(const image_buffer_t *buffer, int32_t x, int32_t y);

static inline uint8_t u8_mul_div255(uint16_t x) {
    // (x + 128) * 257 >> 16 is a common fast approximation.
    return (uint8_t)((x + 128) * 257u >> 16);
}


/**
 * @brief this function takes the image data and maps it to the bcm signal.
 *
 * if scene->tone_mapper is updated, new bcm bit masks will be created.
 *
 * @param scene the scene information
 * @param image the image to map to the scene bcm data. if NULL scene->image will be used
 */
hub75_error_t hub75_display_map_image_to_bcm(const hub75_display_t *scene, uint8_t *image);

/**
 * @brief Convert BCM data to PIO-ready format with pre-baked jitter and addressing
 *
 * @param scene Scene configuration with panel dimensions and rendering parameters
 * @param bcm_data Input BCM data (1 uint32_t per pixel with RGB bits)
 * @param pio_buffer Output buffer for PIO-ready data (must be pre-allocated)
 * @param addr_map Pre-computed address line mappings for each row
 * @param jitter_mask Pre-computed jitter pattern for brightness
 * @return uint32_t Number of words written to pio_buffer
 */
__attribute__((hot))
uint32_t bcm_to_pio_buffer(
    const hub75_display_t *scene,
    const uint32_t *bcm_data,
    uint32_t *pio_buffer,
    const uint32_t *addr_map,
    const uint32_t *jitter_mask);

/**
 * @brief normalize a uint8_t (0-255) to a float (0-1)
 * 
 * @param in byte to normalize
 * @return Normal a floating point value between 0-1
 */
__attribute__((pure))
Normal normalize_8(const uint8_t in);

/**
 * @brief take a normalized RGB value and return luminance as a Normal
 * 
 * @param in value to determine luminance for
 * @return Normal 
 */
__attribute__((pure))
Normal luminance(const RGBF *__restrict__ in);

/**
 * @brief adjust the contrast and saturation of an RGBF pixel
 * 
 * @param in this RGBF value will be adjusted in place. no new RGBF value is returned
 * @param contrast - contrast value 0-1
 * @param saturation - saturation value 0-1
 */
void adjust_contrast_saturation(RGBF *__restrict__ in, const Normal contrast, const Normal saturation);


/**
 * @brief  perform gamma correction on a single byte value (0-255)
 * 
 * @param x - value to gamma correct
 * @param gamma  - gamma correction factor, 1.0 - 2.4 is typical
 * @return uint8_t  - the gamma corrected value
 */
__attribute__((pure))
uint8_t byte_gamma_correct(const uint8_t x, const float gamma);


/**
 * @brief  perform gamma correction on a normalized value. 
 * this is faster than byte_gamma_correct which has to normalize the values then call this function
 * 
 * 
 * @param x - value to gamma correct
 * @param gamma  - gamma correction factor, 1.0 - 2.4 is typical
 * @return Normal  - the gamma corrected value
 */
__attribute__((pure))
Normal normal_gamma_correct(const Normal x, const float gamma);

/**
 * @brief  perform hable uncharted 2 tone mapping
 * @param color  - the color to tone map 
 * @return Normal  - the gamma corrected value
 */
__attribute__((pure))
Normal hable_tone_map(const Normal color);

/**
 * @brief  perform reinhard tone mapping
 * @param color  - the color to tone map 
 * @return Normal  - the gamma corrected value
 */
__attribute__((pure))
Normal reinhard_tone_map(const Normal color, const float level);

/**
 * @brief  perform aces tone mapping
 * @param color  - the color to tone map 
 * @return Normal  - the gamma corrected value
 */
__attribute__((pure))
Normal aces_tone_map(const Normal color);

/**
 * @brief  perform aces tone mapping on RGB values
 * @param in  - the color to tone map 
 * @return out  - the RGB pixel to store the result
 */
void aces_tone_mapper8(const RGB *__restrict__ in, RGB *__restrict__ out);


/**
 * @brief perform ACES tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
void aces_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level);

/**
 * @brief perform Saturation tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */

void saturation_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level);

void sigmoid_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) ;


/**
 * @brief perform hable Uncharted 2 tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
void hable_tone_mapper(const RGB *__restrict__ in, RGB *__restrict__ out);

/**
 * @brief perform hable Uncharted 2 tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
void hable_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level);

/**
 * @brief perform hable Uncharted 2 tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
void reinhard_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level);

/**
 * @brief perform ACES reinhard mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
void reinhard_tone_mapper(const RGB *__restrict__ in, RGB *__restrict__ out);

/**
 * @brief an empty tone mapper that does nothing
 * 
 * @param in 
 * @param out 
 */
void copy_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level);

/**
 * @brief Sample a tightly-packed 8-bit grayscale image at floating-point coordinates
 * using bilinear filtering. Pixel centers are at integer+0.5 in input space.
 * Coordinates are clamped to the valid [0..width-1] x [0..height-1] range.
 *
 * @param pixels  Pointer to grayscale buffer (stride == width)
 * @param width   Image width in pixels
 * @param height  Image height in pixels
 * @param fx      X coordinate in pixel space
 * @param fy      Y coordinate in pixel space
 * @return uint8_t Bilinearly filtered sample (0..255)
 */
uint8_t sdf_sample_gray8_bilinear_buf(const uint8_t *pixels, int width, int height, float fx, float fy);

/* bilinear sample at float (sx, sy) in src image space */
void sample_rgba_bilinear(RGBA *out,
                                        const image_buffer_t *src,
                                        float sx,
                                        float sy);

/**
 * @brief create a lookup table for the pwm values for each pixel value
 * applies gamma correction and tone mapping based on the settings passed in scene
 * scene->brightness, scene->gamma, scene->red_linear, scene->green_linear, scene->blue_linear
 * scene->tone_mapper
 * 
 */
__attribute__((cold))
void *tone_map_rgb_bits(const hub75_display_t *scene, const uint8_t num_bits, uint16_t *quant_errors);


/**
 * @brief draw a line using Bresenham's line drawing algorithm
 * 
 * @param scene 
 * @param x0 start pixel x location
 * @param y0 start pixel y location
 * @param x1 end pixel x
 * @param y1 end pixel y
 * @param color color to draw the line
 */

/**
 * @brief draw an anti-aliased line using Xiolin Wu's line drawing algorithm
 * 
 * @param scene 
 * @param x0 start pixel x location
 * @param y0 start pixel y location
 * @param x1 end pixel x
 * @param y1 end pixel y
 * @param color RGBA color to draw the line
 */
void draw_line_aa(hub75_display_t *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, const RGBA color);




/**
 * @brief helper method to set a pixel in a 24 bpp RGB image buffer
 * 
 * @param scene the scene to draw the pixel at
 * @param x horizontal position (starting at 0) clamped to scene->width
 * @param y vertical position (starting at 0) clamped to scene->height
 * @param pixel RGB value to set at pixel x,y
 */
void draw_pixel(hub75_display_t *scene, const int x, const int y, const RGBA pixel);

/**
 * @brief helper method to set a pixel in a 32 bit RGBA image buffer
 * NOTE: You probably wand draw_pixel_factor for most cases
 * 
 * @param scene the scene to draw the pixel at
 * @param x horizontal position (starting at 0)
 * @param y vertical position (starting at 0)
 * @param pixel RGB value to set at pixel x,y
 */
void draw_pixel_alpha(hub75_display_t *scene, const int x, const int y, const RGBA pixel);

/**
 * @brief fill in a rectangle from x1,y1 to x2,y2. x2,y2 do not need to be > x1,y1
 * 
 * all x,y values will WRAP if they exceed the scene->width or scene->height
 * 
 * @param scene 
 * @param x1
 * @param y1
 * @param x2 
 * @param y2 
 * @param color 
 */
void hub_fill(hub75_display_t *scene, const uint16_t x1, const uint16_t y1, const uint16_t x2, const uint16_t y2, const RGB color);

/**
 * @brief Draw an unfilled circle using Bresenham's algorithm
 * 
 * @param scene 
 * @param width 
 * @param height 
 * @param centerX 
 * @param centerY 
 * @param radius 
 * @param color 
 */
void hub_circle(hub75_display_t *scene, const uint16_t centerX, const uint16_t centerY, const uint16_t radius, const RGB color);

float gradient_horiz(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1);
float gradient_vert(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1);
float gradient_min(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1);
float gradient_max(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1);
float gradient_quad(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1);

uint8_t sample_gray8_bilinear_buf(const uint8_t *__restrict__ pixels,
                                             int width, int height,
                                             float fx, float fy);
/**
 * Stride-aware variant of gray8 bilinear sampling. Rows are separated by 'stride' bytes.
 * DEBUG
 */
uint8_t sdf_sample_gray8_bilinear_stride(const uint8_t *pixels, int width, int height, int stride, float fx, float fy);
 

#endif

