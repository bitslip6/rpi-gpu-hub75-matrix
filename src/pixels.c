#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <sched.h>


#include "rpihub75.h"
#include "util.h"
#include "pixels.h"
#include "mymath.h"





/**
 * @brief interpolate between two colors
 * 
 */
__attribute__((hot))
void interpolate_rgb(RGB* result, const RGB start, const RGB end, const Normal ratio) {
    result->r = (uint8_t)(start.r + (end.r - start.r) * ratio);
    result->g = (uint8_t)(start.g + (end.g - start.g) * ratio);
    result->b = (uint8_t)(start.b + (end.b - start.b) * ratio);
}




/**
 * @brief normalize an 8 bit value (0-255) to a Normalized float 0-1
 */
__attribute__((pure))
Normal normalize_8(const uint8_t in) {
	return (Normal)(float)in / 255.0f;
}
__attribute__((pure))

/**
 * @brief normalize an 8 bit value (0-max_value) to a Normalized float 0-1
 * 
 * @param in - 8 bit input value
 * @param max_value  - 8 bit maximum value
 * @return Normal - a float in the range 0.0-1.0
 */
Normal normalize_any(const uint8_t in, const uint8_t max_value) {
	return (Normal)(float)in / (float)max_value;
}


// calculate the luminance of a color, return as a normal 0-1
__attribute__((pure))
Normal luminance(const RGBF *__restrict__ in) {
    // https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
    // over emphasized green on my displays, YMMV
    //Normal result = 0.2126 * in->r + 0.7152 * in->g + 0.0722 * in->b;
    Normal result = (0.299f * in->r) + (0.587f * in->g) + (0.114f * in->b);
    ASSERT(result >= 0.0f && result <= 1.0f);
    return result;
}



/**
 * @brief adjust the contrast and saturation of an RGBF pixel
 * 
 * @param in this RGBF value will be adjusted in place. no new RGBF value is returned
 * @param contrast - contrast value 0-1
 * @param saturation - saturation value 0-1
 */
void adjust_contrast_saturation(RGBF *__restrict__ in, const float contrast, const float saturation) {
	Normal lum  = luminance(in);

    // Adjust saturation: move the color towards or away from the grayscale value
    Normal red   = mixf(lum, in->r, saturation);
    Normal green = mixf(lum, in->g, saturation);
    Normal blue  = mixf(lum, in->b, saturation);

    // Adjust contrast: scale values around 0.5
    red   = (red   - 0.5f) * contrast + 0.5f;
    green = (green - 0.5f) * contrast + 0.5f;
    blue  = (blue  - 0.5f) * contrast + 0.5f;

    // Clamp values between 0 and 1 (maybe optimize with simple maxf?)
    in->r = clampf(red, 0.0f, 1.0f);
    in->g = clampf(green, 0.0f, 1.0f);
    in->b = clampf(blue, 0.0f, 1.0f);
}

/**
 * @brief  perform gamma correction on a single byte value (0-255)
 * 
 * @param x - value to gamma correct
 * @param gamma  - gamma correction factor, 1.0 - 2.4 is typical
 * @return uint8_t  - the gamma corrected value
 */
__attribute__((pure))
inline uint8_t byte_gamma_correct(const uint8_t x, const float gamma) {
    Normal normal = normalize_8(x);
    Normal correct = normal_gamma_correct(normal, gamma);
    return (uint8_t)MAX(0, MIN((correct * 255.0f), 255));
}


__attribute__((pure))
inline Normal normal_gamma_correct(const Normal x, const float gamma) {
    //return powf(x, 1.0f / gamma);
    ASSERT(x >= 0.0f && x <= 1.0f);
    return powf(x, gamma);
}

// Tone map function using ACES
__attribute__((pure))
inline Normal aces_tone_map(const Normal color) {
    return (color * (ACES_A * color + ACES_B)) / (color * (ACES_C * color + ACES_D) + ACES_E);
}

// Tone map function using reinhard, level should be 1.0
__attribute__((pure))
inline Normal reinhard_tone_map(const Normal color, const float level) {
    return color / (level + color);
}

// Hable's Uncharted 2 Tone Mapping function
__attribute__((pure))
inline Normal hable_tone_map(const Normal color) {
    float mapped_color = ((color * (UNCHART_A * color + UNCHART_C * UNCHART_B) + UNCHART_D * UNCHART_E) / (color * (UNCHART_A * color + UNCHART_B) + UNCHART_D * UNCHART_F)) - UNCHART_E / UNCHART_F;
    return mapped_color;
}

/**
 * @brief perform ACES tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
inline void aces_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) {
    out->r = aces_tone_map(in->r);
    out->g = aces_tone_map(in->g);
    out->b = aces_tone_map(in->b);
}

inline void sigmoid_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) {
    out->r = 1.0f / (1.0f + expf(-5.0f * (in->r - 0.5f)));
    out->g = 1.0f / (1.0f + expf(-5.0f * (in->g - 0.5f)));
    out->b = 1.0f / (1.0f + expf(-5.0f * (in->b - 0.5f)));
}

inline void saturation_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) {
    Normal lum = luminance(in);
    Normal gamma_lum = powf(lum, 1.0f / 2.2f);

    out->r = clampf((lum + level) * (in->r - lum), 0.0f, 1.0f);
    out->g = clampf((lum + level) * (in->g - lum), 0.0f, 1.0f);
    out->b = clampf((lum + level) * (in->b - lum), 0.0f, 1.0f);


    float max = fmaxf(fmaxf(out->r, out->g), out->b);
    if (max > 1.0f) {
        out->r /= max;
        out->g /= max;
        out->b /= max;
    }

    float ratio = lum / gamma_lum;
    out->r = clampf(out->r * ratio, 0.0f, 1.0f);
    out->g = clampf(out->g * ratio, 0.0f, 1.0f);
    out->b = clampf(out->b * ratio, 0.0f, 1.0f);
}


/**
 * @brief perform HABLE Uncharted 2 tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
inline void hable_tone_mapper(const RGB *__restrict__ in, RGB *__restrict__ out) {
    out->r = (uint8_t)(hable_tone_map(normalize_8(in->r)) * 255);
    out->g = (uint8_t)(hable_tone_map(normalize_8(in->g)) * 255);
    out->b = (uint8_t)(hable_tone_map(normalize_8(in->b)) * 255);
}

/**
 * @brief perform HABLE Uncharted 2 tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
inline void hable_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) {
    out->r = hable_tone_map((in->r));
    out->g = hable_tone_map((in->g));
    out->b = hable_tone_map((in->b));
}


/**
 * @brief perform reinhard tone mapping for a single pixel
 * 
 * @param in pointer to the input RGB 
 * @param out pointer to the output RGB 
 */
inline void reinhard_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) {
    out->r = reinhard_tone_map(in->r, level);
    out->g = reinhard_tone_map(in->g, level);
    out->b = reinhard_tone_map(in->b, level);
}



/**
 * @brief an empty tone mapper that does nothing
 * 
 * @param in 
 * @param out 
 */
void copy_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out, const float level) {
    out->r = in->r;
    out->g = in->g;
    out->b = in->b;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief map an input byte to a 32 bit pwm signal
 * 
 */
__attribute__((cold, pure))
uint32_t byte_to_bcm32(const uint8_t input, const uint8_t num_bits) {
    ASSERT((num_bits <= 32));

    // Calculate the number of '1's in the 11-bit result based on the 8-bit input
    uint32_t num_ones = (input * num_bits) / 255;  // Map 0-255 input to 0-num_bits ones
    //uint8_t  num_ones = (uint8_t)floorf(roundf((float)(input * num_bits) / 255.0f));  // Map 0-255 input to 0-num_bits ones
    uint32_t bcm_signal = 0;
    // bit mask that matches the number of bits we want to output
    // uint32_t result_mask = (1U << num_bits) - 1;


    // quant error for dithering is (input / 255) - (num_ones/num_bits);
    // TODO: keep this in floating point space for more precision!

    // dont divide by 0!
    if (num_ones == 0) {
        return bcm_signal;
    }
    //num_ones++;

    float step = (float)num_bits / ((float)num_ones);  // Step for evenly distributing 1's
    for (uint16_t i = 0; i < num_ones && i < 32; i++) {
        int shift = (int)((i + 0.0f) * step);
        bcm_signal |= (1 << (shift));
    }

    //printf("  BCM> @%d G:%d ONES:%d ", index, input, num_ones);
    //binary32(stdout, bcm_signal);
    //printf("\n");

    return bcm_signal;// & result_mask;
}

/**
 * @brief calculate the dither error for a given input byte.
 * reduces input to a bcm value (0-num_bits) and returns the quantization error.
 * 
 * @param input - normalized tone mapped, gamma corrected input value 0.0-1.0
 * @param num_bits - number of bits of BCM output (8-64)
 * @param index - the linear index of the pixel we are calculating
 * @return float - the quantization error (input - output) 0.0-1.0
 */
float byte_to_dither(const Normal input, const uint8_t num_bits, int index) {
    ASSERT((num_bits <= 64));
    ASSERT(input >= 0.0f && input <= 1.0f);

    uint8_t value = (uint8_t)(input * 255.0f);

    // Calculate the number of '1's in the 11-bit result based on the 8-bit input
    //uint8_t num_ones = (uint8_t)floorf(roundf((float)(value * num_bits) / 255.0f));  // Map 0-255 input to 0-num_bits ones
    uint32_t num_ones = (value * num_bits) / 255;  // Map 0-255 input to 0-num_bits ones
    if (num_ones == 0) {
        if (index < 1) {
            return 0.0f;
        }
    }
    //num_ones++;

    float quant_error = input - normalize_any(num_ones, num_bits);  // divide num_ones by num_bits to get normalized value
    // printf("   input -- (%f):%d:(%f)  QUANT:%f\n", input, num_ones, normalize_any(num_ones, num_bits), quant_error);
    return quant_error;
}

uint16_t bcm_to_quant(const uint64_t bcm_value, const uint8_t num_bits, uint8_t tone_val, uint8_t brightness) {
    ASSERT((num_bits <= 64));

    // count the number of bits set in bcm_value
    int num_ones = bit_count(bcm_value);
    float val_quant = (float)num_ones / (float)num_bits;
    float val_real = (float)(tone_val * brightness) / 255.0f;
    
    float quant_dist = val_real - val_quant;
    uint16_t err = (uint16_t)((quant_dist * 65535u) / num_bits); // return a 16 bit normalized value
    return err;
}


/**
 * @brief map an input byte to a 64 bit bcm signal
 * 
 */
__attribute__((cold, pure))
uint64_t byte_to_bcm64(const uint8_t input, const uint8_t bit_depth) {

    // constraints
    if (bit_depth == 0 || bit_depth > 64) {
        return 0ULL;
    }

    // map 0..255 to 0..bit_depth using round-to-nearest
    // this avoids systematic bias near midpoints
    uint32_t num_ones = (uint32_t)((input * (uint32_t)bit_depth + 127u) / 255u);

    // if you WANT to force a visible spark for any nonzero input, uncomment next two lines
    // if (input != 0 && num_ones == 0) num_ones = 1;

    if (num_ones == 0) {
        return 0ULL;
    }
    if (num_ones >= bit_depth) {
        // all on when fully saturated
        if (bit_depth == 64) return ~0ULL;
        return (1ULL << bit_depth) - 1ULL;
    }

    // Even distribution using Bresenham-style accumulator.
    // This sets exactly num_ones bits among bit_depth slots.
    uint64_t bcm_signal = 0ULL;
    uint32_t acc = 0;
    for (uint32_t j = 0; j < bit_depth; j++) {
        acc += num_ones;
        if (acc >= bit_depth) {
            acc -= bit_depth;
            bcm_signal |= (1ULL << j);   // j is always < bit_depth, so shift is defined
        }
    }
    return bcm_signal;
}



// helper: add and wrap an index 0..bit_depth-1
static inline uint8_t wrap_add_u8(uint8_t idx, uint8_t add, uint8_t mod) {
    uint8_t s = (uint8_t)(idx + add);
    return (s >= mod) ? (uint8_t)(s - mod) : s;  // mod is small (<=32), predictable
}


// build at init
static int32_t mid_dn_tbl[258], mid_up_tbl[258];  // +2  to handle the +1W access, +1 more for good measure

/* optional: keep a sanitized monotonic copy if you also use W elsewhere */
static inline void sd_build_mid_tables(const uint16_t *W_in) {
    /* 1) sanitize W to monotonic nondecreasing in 0..65535 */
    uint32_t W[258];
    uint32_t prev = 0;
    for (int i = 0; i < 257; ++i) {
        uint32_t wi = W_in[i];
        if (wi > 0xFFFFu) wi = 0xFFFFu;
        if (wi < prev)    wi = prev;        /* enforce monotonicity */
        W[i] = wi;
        prev = wi;
    }

    /* 2) edge sentinels so 0 never goes down, 255 never goes up */
    mid_dn_tbl[0]   = INT32_MIN / 2;        /* any e >= sentinel, so never lt_dn at v==0 */
    mid_up_tbl[255] = INT32_MAX / 2;        /* any e <= sentinel, so never gt_up at v==255 */
    mid_up_tbl[256] = INT32_MAX / 2;        /* any e <= sentinel, so never gt_up at v==255 */
    mid_up_tbl[257] = INT32_MAX / 2;        /* any e <= sentinel, so never gt_up at v==255 */

    /* 3) midpoints between neighbors with strict separation to avoid oscillation */
    for (int i = 1; i < 257; ++i) {
        uint32_t a = W[i - 1], b = W[i];
        uint32_t m = (a + b) >> 1;          /* floor((a+b)/2) */
        if (m >= b) m = (b > 0) ? b - 1 : 0;/* ensure mdn < b */
        mid_dn_tbl[i] = (int32_t)m;
    }
    for (int i = 0; i < 257; ++i) {
        uint32_t a = W[i], b = W[i + 1];
        uint32_t m = (a + b + 1) >> 1;      /* ceil((a+b)/2) */
        if (m <= a) m = (a < 0xFFFFu) ? a + 1 : 0xFFFFu; /* ensure mup > a */
        mid_up_tbl[i] = (int32_t)m;
    }
}

/**
 * @brief apply temporal dithering to a single 8-bit channel value. 
 * Purpose:
 *   Quantize an 8-bit value v to v, v+1, or v-1 using a running error accumulator
 *   and a precomputed weight table W[256]. This is useful for error-diffusion or
 *   temporal dithering where decisions depend on accumulated residual error.
 *
 * @param v the input value to be quantized, 0..255
 *        For v == 0, only stay or go up. For v == 255, only stay or go down.
 * @param acc an accumulator holding the residual error from prior steps. must be int32 width*height*3 wide to avoid overflow.
 * @param W a precomputed weight table, typically gamma-corrected. see: sd_build_weight_table()
 * @return the quantized output value, one of {v-1, v, v+1}
 */
__attribute__((hot))
static inline uint8_t sd_weight_step_fast(const uint8_t v,
                                          int32_t *acc,
                                          const uint16_t *W)
{
    // load once
    const int32_t wv  = (int32_t)W[v];
    // decision variable in signed domain
    const int32_t e   = *acc + wv;
    // midpoints to neighbors in signed domain, with safe sentinels at edges see: sd_build_mid_tables()
    const int32_t mdn = (int32_t)mid_dn_tbl[v];
    const int32_t mup = (int32_t)mid_up_tbl[v];

    // branch-lean decisions (compile to csel on aarch64, only one is set)
    int32_t gt_up = (int32_t)(e > mup);
    int32_t lt_dn = (int32_t)(e < mdn);

    /* hard edge guards */
    gt_up &= (int32_t)(v != 255u);
    lt_dn &= (int32_t)(v != 0u);

    // new code in {v-1, v, v+1}
    const uint8_t out = (uint8_t) (v + gt_up - lt_dn);

    // update residual for next step
    *acc = e - (int32_t)W[out];

    // return result
    return out;
}


// build once per process
static uint32_t PORT0_LUT[64], PORT1_LUT[64], PORT2_LUT[64];
static int      port_lut_built = 0;

// Cache all remaps once if you like, 6 * 64 bytes total.
static uint8_t IDX_REMAP[6][64];
static int idx_remap_built_mask = 0;


// Build a 6-bit index remap for the given order.
// Canonical idx layout is [R1,G1,B1,R2,G2,B2] with bit 0 = R1, 5 = B2.
static inline void build_idx_remap(panel_order_t order, uint8_t remap[64]) {
    // src_pos[wire] = which logical bit position supplies that wire for pixel1
    // wire: 0=Rwire, 1=Gwire, 2=Bwire
    uint8_t src_pos[3] = {0,0,0};
    switch (order) {
        case PANEL_RGB: src_pos[0]=0; src_pos[1]=1; src_pos[2]=2; break;
        case PANEL_RBG: src_pos[0]=0; src_pos[1]=2; src_pos[2]=1; break;
        case PANEL_GRB: src_pos[0]=1; src_pos[1]=0; src_pos[2]=2; break;
        case PANEL_GBR: src_pos[0]=1; src_pos[1]=2; src_pos[2]=0; break;
        case PANEL_BRG: src_pos[0]=2; src_pos[1]=0; src_pos[2]=1; break;
        case PANEL_BGR: src_pos[0]=2; src_pos[1]=1; src_pos[2]=0; break;
    }

    for (uint32_t idx = 0; idx < 64; ++idx) {
        // gather bits for pixel1
        uint32_t r1 = (idx >> 0) & 1u;
        uint32_t g1 = (idx >> 1) & 1u;
        uint32_t b1 = (idx >> 2) & 1u;
        // gather bits for pixel2
        uint32_t r2 = (idx >> 3) & 1u;
        uint32_t g2 = (idx >> 4) & 1u;
        uint32_t b2 = (idx >> 5) & 1u;

        const uint32_t pix1[3] = { r1, g1, b1 };
        const uint32_t pix2[3] = { r2, g2, b2 };

        // new_idx feeds the canonical RGB port LUTs
        uint32_t new_idx = 0;
        // wire R gets pix? [ src_pos[0] ], wire G gets pix? [ src_pos[1] ], wire B gets pix? [ src_pos[2] ]
        new_idx |= (pix1[src_pos[0]] << 0);
        new_idx |= (pix1[src_pos[1]] << 1);
        new_idx |= (pix1[src_pos[2]] << 2);
        new_idx |= (pix2[src_pos[0]] << 3);
        new_idx |= (pix2[src_pos[1]] << 4);
        new_idx |= (pix2[src_pos[2]] << 5);

        remap[idx] = (uint8_t)new_idx;
    }
}



static inline void build_port_luts(void) {
    if (port_lut_built) return;
    for (uint32_t i = 0; i < 64; ++i) {
        // bit layout in the LUT index: [R1,G1,B1,R2,G2,B2] per port, LSB first
        uint32_t v0 = 0;
        v0 |= ((i >> 0) & 1u) ? (1u << ADDRESS_P0_R1) : 0u;
        v0 |= ((i >> 1) & 1u) ? (1u << ADDRESS_P0_G1) : 0u;
        v0 |= ((i >> 2) & 1u) ? (1u << ADDRESS_P0_B1) : 0u;
        v0 |= ((i >> 3) & 1u) ? (1u << ADDRESS_P0_R2) : 0u;
        v0 |= ((i >> 4) & 1u) ? (1u << ADDRESS_P0_G2) : 0u;
        v0 |= ((i >> 5) & 1u) ? (1u << ADDRESS_P0_B2) : 0u;
        PORT0_LUT[i] = v0;

        uint32_t v1 = 0;
        v1 |= ((i >> 0) & 1u) ? (1u << ADDRESS_P1_R1) : 0u;
        v1 |= ((i >> 1) & 1u) ? (1u << ADDRESS_P1_G1) : 0u;
        v1 |= ((i >> 2) & 1u) ? (1u << ADDRESS_P1_B1) : 0u;
        v1 |= ((i >> 3) & 1u) ? (1u << ADDRESS_P1_R2) : 0u;
        v1 |= ((i >> 4) & 1u) ? (1u << ADDRESS_P1_G2) : 0u;
        v1 |= ((i >> 5) & 1u) ? (1u << ADDRESS_P1_B2) : 0u;
        PORT1_LUT[i] = v1;

        uint32_t v2 = 0;
        v2 |= ((i >> 0) & 1u) ? (1u << ADDRESS_P2_R1) : 0u;
        v2 |= ((i >> 1) & 1u) ? (1u << ADDRESS_P2_G1) : 0u;
        v2 |= ((i >> 2) & 1u) ? (1u << ADDRESS_P2_B1) : 0u;
        v2 |= ((i >> 3) & 1u) ? (1u << ADDRESS_P2_R2) : 0u;
        v2 |= ((i >> 4) & 1u) ? (1u << ADDRESS_P2_G2) : 0u;
        v2 |= ((i >> 5) & 1u) ? (1u << ADDRESS_P2_B2) : 0u;
        PORT2_LUT[i] = v2;
    }
    port_lut_built = 1;
}

/**
 * @brief ensure that the index remap for the given panel order is built, and return it.
 * 
 * @param order 
 * @return const uint8_t* 
 */
static inline const uint8_t* get_idx_remap(panel_order_t order) {
    int bit = 1 << (int)order;
    if (!(idx_remap_built_mask & bit)) {
        build_idx_remap(order, IDX_REMAP[order]);
        idx_remap_built_mask |= bit;
    }
    return IDX_REMAP[order];
}




static inline uint64_t init_mask(uint8_t phase, uint8_t bit_depth) {
    return 1ull << ((bit_depth == 64) ? (phase & 63) : (phase % bit_depth));
}

static inline uint64_t rotl1_mask(uint64_t m, uint8_t bit_depth) {
    if (bit_depth == 64) {
        return (m << 1) | (m >> 63);
    } else {
        const uint64_t wrap = (1ull << bit_depth) - 1ull;
        return ((m << 1) & wrap) | (m >> (bit_depth - 1));
    }
}

static inline uint64_t rotl2_mask(uint64_t m, uint8_t bit_depth) {
    if (bit_depth == 64) {
        return (m << 2) | (m >> 62);
    } else {
        const uint64_t wrap = (1ull << bit_depth) - 1ull;
        return ((m << 2) & wrap) | (m >> (bit_depth - 2));
    }
}



static const uint8_t bayer8x8_u0_63[64] = {
     0,48,12,60, 3,51,15,63,
    32,16,44,28,35,19,47,31,
     8,56, 4,52,11,59, 7,55,
    40,24,36,20,43,27,39,23,
     2,50,14,62, 1,49,13,61,
    34,18,46,30,33,17,45,29,
    10,58, 6,54, 9,57, 5,53,
    42,26,38,22,41,25,37,21
};

static inline uint8_t clamp_u8_int(int v) {
    if (v < 0) return 1;
    if (v > 250) return 250;
    return (uint8_t)v;
}

/**
 * Spatial Bayer dithering on dark values only.
 * img: interleaved RGB8 buffer
 * stride_bytes: bytes per row
 * cutoff: apply only when channel < cutoff, suggest 100
 * max_amp: maximum +/- offset in u8 units, suggest 1..3 (start with 2)
 */
__attribute__((hot))
void dither_spatial_bayer8_low(uint8_t *img, const int width, const int height,
                               const int image_stride, const int cutoff, const int max_amp)
{
    uint32_t offset = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, offset += image_stride) {
            uint8_t t_base = bayer8x8_u0_63[(y & 7) * 8 + (x & 7)];
            /* map 0..63 to roughly [-1, +1] then scale to [-max_amp, +max_amp], zero mean */

            /* per-channel decorrelated dither by phase shifting the tile */
            uint8_t t_r = t_base;
            uint8_t t_g = bayer8x8_u0_63[(y & 7) * 8 + ((x + 3) & 7)];
            uint8_t t_b = bayer8x8_u0_63[(y & 7) * 8 + ((x + 5) & 7)];
            int off_r = (int)((((int)t_r - 31) * (max_amp)) / 31.0f + 0.5f);
            int off_g = (int)((((int)t_g - 31) * (max_amp)) / 31.0f + 0.5f);
            int off_b = (int)((((int)t_b - 31) * (max_amp)) / 31.0f + 0.5f);

            if (img[offset] < cutoff) img[offset] = MAX(0, MIN(img[offset] + off_r, 254));
            if (img[offset+1] < cutoff) img[offset+1] = MAX(0, MIN(img[offset+1] + off_g, 254));
            if (img[offset+2] < cutoff) img[offset+2] = MAX(0, MIN(img[offset+2] + off_b, 254));
        }
    }
}

static inline uint32_t u32_hash(uint32_t x, uint32_t y) {
    uint32_t h = x * 0x9E3779B1u ^ (y + 0x7F4A7C15u);
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

void dither_spatial_hash_low(uint8_t *img, int width, int height,
                             int stride_bytes, int cutoff, int max_amp)
{
    uint32_t offset = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, offset += stride_bytes) {
            uint32_t h = u32_hash((uint32_t)x, (uint32_t)y);
            /* map to signed offset in [-max_amp, +max_amp], zero mean */
            int off_r = (int)((int)((h >>  0) & 0x3Fu) - 31);
            int off_g = (int)((int)((h >>  6) & 0x3Fu) - 31);
            int off_b = (int)((int)((h >> 12) & 0x3Fu) - 31);
            off_r = (int)(off_r * (max_amp / 31.0f) + 0.5f);
            off_g = (int)(off_g * (max_amp / 31.0f) + 0.5f);
            off_b = (int)(off_b * (max_amp / 31.0f) + 0.5f);

            if (img[offset] < cutoff) img[offset] = MAX(0, MIN(img[offset] + off_r, 254));
            if (img[offset+1] < cutoff) img[offset+1] = MAX(0, MIN(img[offset+1] + off_g, 254));
            if (img[offset+2] < cutoff) img[offset+2] = MAX(0, MIN(img[offset+2] + off_b, 254));
            //int v0 = px[0]; if (v0 < cutoff) px[0] = clamp_u8_int(v0 + off_r);
            //int v1 = px[1]; if (v1 < cutoff) px[1] = clamp_u8_int(v1 + off_g);
            //int v2 = px[2]; if (v2 < cutoff) px[2] = clamp_u8_int(v2 + off_b);
        }
    }
}





__attribute__((hot))
void update_bcm_signal_64_rgb(
    const scene_info *scene,
    const void *__restrict__ void_bits,
    uint32_t *__restrict__ bcm_signal,
    const uint8_t *__restrict__ image,
    uint16_t *__restrict__ quant_err_lut,
    uint8_t phase
) {

    static int32_t *accum = NULL;
    static int32_t cached_w = -1, cached_h = -1, cached_stride = -1;

    if (UNLIKELY(scene->width != cached_w || scene->panel_height != cached_h || scene->stride != cached_stride)) {
        cached_w = scene->width;
        cached_h = scene->panel_height;
        cached_stride = scene->stride;
        if (accum != NULL) {
            free(accum);
        }
        accum = (int32_t*)calloc(scene->width * scene->height * scene->stride, sizeof(int32_t));
        build_port_luts(); // once
    }

    const uint8_t bit_depth = scene->bit_depth;
    ASSERT(bit_depth % BIT_DEPTH_ALIGNMENT == 0);
    ASSERT(bit_depth >= 32);



    /* channel LUT planes for quant error, element offsets not bytes */
    const uint16_t *Wr = quant_err_lut + 0;
    const uint16_t *Wg = quant_err_lut + 256;
    const uint16_t *Wb = quant_err_lut + 512;

    // 3) helper macros for pointer/index math
    #define PIX_PTR(px_index)   (image + (size_t)(px_index) * (size_t)stride_bytes)
    #define ACC_IDX(px_index,c) ((px_index) * 3 + (c))   /* c: 0=R,1=G,2=B */


    // 2) derive pixel-stride geometry
    const int stride_bytes = scene->stride;                         // 3 or 4
    const int panel_stride_px = scene->width * (scene->panel_height / 2);

    // p*_px are pixel indices, not byte offsets
    const int p0t_px = 0;
    const int p0b_px = p0t_px + panel_stride_px;
    const int p1t_px = p0b_px + panel_stride_px;
    const int p1b_px = p1t_px + panel_stride_px;
    const int p2t_px = p1b_px + panel_stride_px;
    const int p2b_px = p2t_px + panel_stride_px;


    // 5) locate the six pixel base pointers once
    const uint8_t *p0t_ptr = PIX_PTR(p0t_px);
    const uint8_t *p0b_ptr = PIX_PTR(p0b_px);
    const uint8_t *p1t_ptr = PIX_PTR(p1t_px);
    const uint8_t *p1b_ptr = PIX_PTR(p1b_px);
    const uint8_t *p2t_ptr = PIX_PTR(p2t_px);
    const uint8_t *p2b_ptr = PIX_PTR(p2b_px);


    // 6) fetch with correct accum indexing; ternary evaluates only one side
    const uint8_t r0  = scene->quant_dither ? sd_weight_step_fast(p0t_ptr[0], &accum[ACC_IDX(p0t_px,0)], Wr) : p0t_ptr[0];
    const uint8_t g0  = scene->quant_dither ? sd_weight_step_fast(p0t_ptr[1], &accum[ACC_IDX(p0t_px,1)], Wg) : p0t_ptr[1];
    const uint8_t b0  = scene->quant_dither ? sd_weight_step_fast(p0t_ptr[2], &accum[ACC_IDX(p0t_px,2)], Wb) : p0t_ptr[2];

    const uint8_t r0b = scene->quant_dither ? sd_weight_step_fast(p0b_ptr[0], &accum[ACC_IDX(p0b_px,0)], Wr) : p0b_ptr[0];
    const uint8_t g0b = scene->quant_dither ? sd_weight_step_fast(p0b_ptr[1], &accum[ACC_IDX(p0b_px,1)], Wg) : p0b_ptr[1];
    const uint8_t b0b = scene->quant_dither ? sd_weight_step_fast(p0b_ptr[2], &accum[ACC_IDX(p0b_px,2)], Wb) : p0b_ptr[2];

    const uint8_t r1t = scene->quant_dither ? sd_weight_step_fast(p1t_ptr[0], &accum[ACC_IDX(p1t_px,0)], Wr) : p1t_ptr[0];
    const uint8_t g1t = scene->quant_dither ? sd_weight_step_fast(p1t_ptr[1], &accum[ACC_IDX(p1t_px,1)], Wg) : p1t_ptr[1];
    const uint8_t b1t = scene->quant_dither ? sd_weight_step_fast(p1t_ptr[2], &accum[ACC_IDX(p1t_px,2)], Wb) : p1t_ptr[2];

    const uint8_t r1b = scene->quant_dither ? sd_weight_step_fast(p1b_ptr[0], &accum[ACC_IDX(p1b_px,0)], Wr) : p1b_ptr[0];
    const uint8_t g1b = scene->quant_dither ? sd_weight_step_fast(p1b_ptr[1], &accum[ACC_IDX(p1b_px,1)], Wg) : p1b_ptr[1];
    const uint8_t b1b = scene->quant_dither ? sd_weight_step_fast(p1b_ptr[2], &accum[ACC_IDX(p1b_px,2)], Wb) : p1b_ptr[2];

    const uint8_t r2t = scene->quant_dither ? sd_weight_step_fast(p2t_ptr[0], &accum[ACC_IDX(p2t_px,0)], Wr) : p2t_ptr[0];
    const uint8_t g2t = scene->quant_dither ? sd_weight_step_fast(p2t_ptr[1], &accum[ACC_IDX(p2t_px,1)], Wg) : p2t_ptr[1];
    const uint8_t b2t = scene->quant_dither ? sd_weight_step_fast(p2t_ptr[2], &accum[ACC_IDX(p2t_px,2)], Wb) : p2t_ptr[2];

    const uint8_t r2b = scene->quant_dither ? sd_weight_step_fast(p2b_ptr[0], &accum[ACC_IDX(p2b_px,0)], Wr) : p2b_ptr[0];
    const uint8_t g2b = scene->quant_dither ? sd_weight_step_fast(p2b_ptr[1], &accum[ACC_IDX(p2b_px,1)], Wg) : p2b_ptr[1];
    const uint8_t b2b = scene->quant_dither ? sd_weight_step_fast(p2b_ptr[2], &accum[ACC_IDX(p2b_px,2)], Wb) : p2b_ptr[2];

    const uint64_t *bits_r = (const uint64_t*)void_bits + 0;
    const uint64_t *bits_g = (const uint64_t*)void_bits + 256;
    const uint64_t *bits_b = (const uint64_t*)void_bits + 512;

    const uint64_t R0  = bits_r[r0],  G0  = bits_g[g0],  B0  = bits_b[b0];
    const uint64_t R0B = bits_r[r0b], G0B = bits_g[g0b], B0B = bits_b[b0b];
    const uint64_t R1T = bits_r[r1t], G1T = bits_g[g1t], B1T = bits_b[b1t];
    const uint64_t R1B = bits_r[r1b], G1B = bits_g[g1b], B1B = bits_b[b1b];
    const uint64_t R2T = bits_r[r2t], G2T = bits_g[g2t], B2T = bits_b[b2t];
    const uint64_t R2B = bits_r[r2b], G2B = bits_g[g2b], B2B = bits_b[b2b];

    uint8_t  bcm_offset = 0;
    // mask for just this current BCM bit postion
    uint64_t m = init_mask(phase, bit_depth);

    const uint8_t *restrict remap = get_idx_remap(scene->panel_order);

    //memset(bcm_signal, 0, sizeof(uint32_t) * bit_depth);

    // unroll by 2 to cut loop overhead, requires bit_depth even, which it is
    #pragma GCC ivdep
    for (uint8_t j = 0; j < bit_depth; j += 2) {
        // slot j
        {
            // build 6-bit indices using branchless tests
            uint32_t idx0 =
                ((uint32_t)((R0  & m) != 0ull) << 0) |
                ((uint32_t)((G0  & m) != 0ull) << 1) |
                ((uint32_t)((B0  & m) != 0ull) << 2) |
                ((uint32_t)((R0B & m) != 0ull) << 3) |
                ((uint32_t)((G0B & m) != 0ull) << 4) |
                ((uint32_t)((B0B & m) != 0ull) << 5);

            uint32_t idx1 =
                ((uint32_t)((R1T & m) != 0ull) << 0) |
                ((uint32_t)((G1T & m) != 0ull) << 1) |
                ((uint32_t)((B1T & m) != 0ull) << 2) |
                ((uint32_t)((R1B & m) != 0ull) << 3) |
                ((uint32_t)((G1B & m) != 0ull) << 4) |
                ((uint32_t)((B1B & m) != 0ull) << 5);

            uint32_t idx2 =
                ((uint32_t)((R2T & m) != 0ull) << 0) |
                ((uint32_t)((G2T & m) != 0ull) << 1) |
                ((uint32_t)((B2T & m) != 0ull) << 2) |
                ((uint32_t)((R2B & m) != 0ull) << 3) |
                ((uint32_t)((G2B & m) != 0ull) << 4) |
                ((uint32_t)((B2B & m) != 0ull) << 5);


            // perform pixel order remap, remap LUT maps RGB order to panel order
            // PORTx_LUT maps the 6 bit linear RGB index to the actual GPIO bits to set
            uint32_t lut_word = PORT0_LUT[ remap[idx0] ] | PORT1_LUT[ remap[idx1] ] | PORT2_LUT[ remap[idx2] ];
            // store the result in the bcm_signal array
            bcm_signal[bcm_offset++] = lut_word;
        }

        // slot j+1
        {
            // mask off just this current BCM bit postion
            const uint64_t m2 = rotl1_mask(m, bit_depth);

            uint32_t idx0 =
                ((uint32_t)((R0  & m2) != 0ull) << 0) |
                ((uint32_t)((G0  & m2) != 0ull) << 1) |
                ((uint32_t)((B0  & m2) != 0ull) << 2) |
                ((uint32_t)((R0B & m2) != 0ull) << 3) |
                ((uint32_t)((G0B & m2) != 0ull) << 4) |
                ((uint32_t)((B0B & m2) != 0ull) << 5);

            uint32_t idx1 =
                ((uint32_t)((R1T & m2) != 0ull) << 0) |
                ((uint32_t)((G1T & m2) != 0ull) << 1) |
                ((uint32_t)((B1T & m2) != 0ull) << 2) |
                ((uint32_t)((R1B & m2) != 0ull) << 3) |
                ((uint32_t)((G1B & m2) != 0ull) << 4) |
                ((uint32_t)((B1B & m2) != 0ull) << 5);

            uint32_t idx2 =
                ((uint32_t)((R2T & m2) != 0ull) << 0) |
                ((uint32_t)((G2T & m2) != 0ull) << 1) |
                ((uint32_t)((B2T & m2) != 0ull) << 2) |
                ((uint32_t)((R2B & m2) != 0ull) << 3) |
                ((uint32_t)((G2B & m2) != 0ull) << 4) |
                ((uint32_t)((B2B & m2) != 0ull) << 5);

            // perform pixel order remap, remap LUT maps RGB order to panel order
            // PORTx_LUT maps the 6 bit linear RGB index to the actual GPIO bits to set
            uint32_t lut_word = PORT0_LUT[ remap[idx0] ] | PORT1_LUT[ remap[idx1] ] | PORT2_LUT[ remap[idx2] ];

            bcm_signal[bcm_offset++] = lut_word;

            // advance the rolling mask by 2
            m = rotl2_mask(m, bit_depth);
        }
    }
}




/**
 * @brief helper posix_memalign avoids "aligned_alloc size must be multiple of alignment"
 */
static inline void *aligned_alloc64(size_t size) {
    void *p = NULL;
    size_t rounded = (size + 63u) & ~((size_t)63u);
    if (posix_memalign(&p, 64, rounded) != 0) return NULL;
    return p;
}
 
/**
 * @brief create a bcm signal map from linear sRGB space to the bcm(pwm) signal.
 * the returned pointer will be either uint32_t* or uint64_t* depending on the size
 * of bit_depth.
 * 
 * each bcm entry will be a right aligned bit mask of length bit_depth.
 * 
 * looking up any linear 8 bit value in the map will return a BCM bit mask of length bit_depth
 * 
 * @param scene contains reference to jitter_brightness, gamma, 
 * brightness, red_linear, green_linear, blue_linear, red_gamma, green_gamma, blue_gamma, 
 * bit_depth, tone_mapper.
 * @param bit_depth  number of bits of BCM data (good values from 8-64) try to make them multiples of 4 or 8
 * @param quant_errors pointer to an array of 3*256 floats to store the quantization errors for dithering.
 * @return void* pointer to the bcm signal map. 0-255 red, 256-511 green, 512-767 blue. caller must free() 
 */
void *tone_map_rgb_bits(const scene_info *scene, const int bit_depth, uint16_t *quant_errors) {
    if (bit_depth > 64 || bit_depth < 8) {
        die("bit depth must be between 8 and 64\n");
    }

    // one unified buffer, 64B aligned
    const size_t entries = 3u * 257u;                     // keep your 257 convention
    const size_t bytes   = entries * sizeof(uint64_t);
    uint64_t *bits = (uint64_t *)aligned_alloc64(bytes);
    if (UNLIKELY(!bits)) die("tone_map_rgb_bits: out of memory\n");
    memset(bits, 0, bytes);

    const uint8_t brightness = (scene->jitter_brightness) ? 255 : scene->brightness;
    for (uint16_t i=0; i<=255; i++) {
        RGBF tone_pixel = {0, 0, 0};
        RGBF gamma_pixel = {
            normal_gamma_correct(normalize_8(i), scene->gamma),
            normal_gamma_correct(normalize_8(i), scene->gamma),
            normal_gamma_correct(normalize_8(i), scene->gamma)
        };

        // tone map the value ...
        if (scene->tone_mapper != NULL) {
            scene->tone_mapper(&gamma_pixel, &tone_pixel, scene->tone_level);
        }
        // jsut gamma correct the value ...
        else {
            tone_pixel.r = gamma_pixel.r;
            tone_pixel.g = gamma_pixel.g;
            tone_pixel.b = gamma_pixel.b;
        }

        // calculate quant errors from gamma correction for dithering
        // quant errors need to calculate difference between the BCM value (0-32) and the original value (255)
        // ideally this happens as a normalized float, not a byte.
        
        uint8_t r = MIN(tone_pixel.r * brightness, 255);
        uint8_t g = MIN(tone_pixel.g * brightness, 255);
        uint8_t b = MIN(tone_pixel.b * brightness, 255);

        bits[i]     = byte_to_bcm64(r, bit_depth);
        bits[i+256] = byte_to_bcm64(g, bit_depth);
        bits[i+512] = byte_to_bcm64(b, bit_depth);

        quant_errors[i]     = bcm_to_quant(bits[i], bit_depth, r, brightness);
        quant_errors[i+256] = bcm_to_quant(bits[i+256], bit_depth, g, brightness);
        quant_errors[i+512] = bcm_to_quant(bits[i+512], bit_depth, b, brightness);
    }

    // build the mid tables so we can remove all branching from temporal dithering
    sd_build_mid_tables(quant_errors);

    return bits;
}




// scale then offset per channel:
// out = clamp_u8( ((in * q8 + 128) >> 8) + off )
// pixels layout: [R,G,B,A] per pixel, A copied unchanged.
// image_stride is bytes per pixel, expected 4 for RGBA8.
static inline void scale_rect_rgb_q8_offset(uint8_t *pixels, uint8_t *mapped_pixels,
                                            int width, int height, uint8_t image_stride,
                                            int x0, int y0, int w, int h,
                                            uint16_t red_q8, uint16_t green_q8, uint16_t blue_q8,
                                            int16_t red_off, int16_t green_off, int16_t blue_off)
{
    if (x0 < 0) { w += x0; x0 = 0; }
    if (y0 < 0) { h += y0; y0 = 0; }
    if (x0 + w > width)  w = width  - x0;
    if (y0 + h > height) h = height - y0;
    if (w <= 0 || h <= 0) return;

    const size_t row_stride = (size_t)width * (size_t)image_stride;
    uint8_t *src_row = pixels        + (size_t)y0 * row_stride + (size_t)x0 * image_stride;
    uint8_t *dst_row = mapped_pixels + (size_t)y0 * row_stride + (size_t)x0 * image_stride;

#if defined(__ARM_NEON) || defined(__aarch64__)
    const uint16x8_t r16v = vdupq_n_u16((uint16_t)red_q8);
    const uint16x8_t g16v = vdupq_n_u16((uint16_t)green_q8);
    const uint16x8_t b16v = vdupq_n_u16((uint16_t)blue_q8);
    const uint16x8_t round_bias = vdupq_n_u16(128); // for rounding before >> 8
    const int16x8_t  r_off_v = vdupq_n_s16(red_off);
    const int16x8_t  g_off_v = vdupq_n_s16(green_off);
    const int16x8_t  b_off_v = vdupq_n_s16(blue_off);
#endif

    for (int y = 0; y < h; ++y) {
        uint8_t *p_src = src_row;
        uint8_t *p_dst = dst_row;

#if defined(__ARM_NEON) || defined(__aarch64__)
        if ((((uintptr_t)p_src | (uintptr_t)p_dst) & 0xF) == 0 && image_stride == 4) {
            int x = 0;
            const int vec_pixels = w & ~15; // multiple of 16
            for (; x < vec_pixels; x += 16, p_src += 16 * 4, p_dst += 16 * 4) {
                __builtin_prefetch(p_src + 128, 0, 3);

                // load and deinterleave 16 RGBA pixels
                uint8x16x4_t rgba = vld4q_u8(p_src);

                // widen to u16
                uint16x8_t r_lo = vmovl_u8(vget_low_u8 (rgba.val[0]));
                uint16x8_t r_hi = vmovl_u8(vget_high_u8(rgba.val[0]));
                uint16x8_t g_lo = vmovl_u8(vget_low_u8 (rgba.val[1]));
                uint16x8_t g_hi = vmovl_u8(vget_high_u8(rgba.val[1]));
                uint16x8_t b_lo = vmovl_u8(vget_low_u8 (rgba.val[2]));
                uint16x8_t b_hi = vmovl_u8(vget_high_u8(rgba.val[2]));

                // rounded scale to u16 in 0..255
                r_lo = vshrq_n_u16(vaddq_u16(vmulq_u16(r_lo, r16v), round_bias), 8);
                r_hi = vshrq_n_u16(vaddq_u16(vmulq_u16(r_hi, r16v), round_bias), 8);
                g_lo = vshrq_n_u16(vaddq_u16(vmulq_u16(g_lo, g16v), round_bias), 8);
                g_hi = vshrq_n_u16(vaddq_u16(vmulq_u16(g_hi, g16v), round_bias), 8);
                b_lo = vshrq_n_u16(vaddq_u16(vmulq_u16(b_lo, b16v), round_bias), 8);
                b_hi = vshrq_n_u16(vaddq_u16(vmulq_u16(b_hi, b16v), round_bias), 8);

                // add signed offsets in s16, then saturating narrow to u8
                int16x8_t r_lo_s = vreinterpretq_s16_u16(r_lo);
                int16x8_t r_hi_s = vreinterpretq_s16_u16(r_hi);
                int16x8_t g_lo_s = vreinterpretq_s16_u16(g_lo);
                int16x8_t g_hi_s = vreinterpretq_s16_u16(g_hi);
                int16x8_t b_lo_s = vreinterpretq_s16_u16(b_lo);
                int16x8_t b_hi_s = vreinterpretq_s16_u16(b_hi);

                r_lo_s = vaddq_s16(r_lo_s, r_off_v);
                r_hi_s = vaddq_s16(r_hi_s, r_off_v);
                g_lo_s = vaddq_s16(g_lo_s, g_off_v);
                g_hi_s = vaddq_s16(g_hi_s, g_off_v);
                b_lo_s = vaddq_s16(b_lo_s, b_off_v);
                b_hi_s = vaddq_s16(b_hi_s, b_off_v);

                uint8x8_t r_lo8 = vqmovun_s16(r_lo_s);
                uint8x8_t r_hi8 = vqmovun_s16(r_hi_s);
                uint8x8_t g_lo8 = vqmovun_s16(g_lo_s);
                uint8x8_t g_hi8 = vqmovun_s16(g_hi_s);
                uint8x8_t b_lo8 = vqmovun_s16(b_lo_s);
                uint8x8_t b_hi8 = vqmovun_s16(b_hi_s);

                rgba.val[0] = vcombine_u8(r_lo8, r_hi8);
                rgba.val[1] = vcombine_u8(g_lo8, g_hi8);
                rgba.val[2] = vcombine_u8(b_lo8, b_hi8);
                // alpha unchanged from source

                vst4q_u8(p_dst, rgba);
            }

            // scalar tail
            for (; x < w; ++x, p_src += 4, p_dst += 4) {
                int r = (p_src[0] * red_q8   + 128) >> 8; r += red_off;   if (r < 0) r = 0; if (r > 255) r = 255;
                int g = (p_src[1] * green_q8 + 128) >> 8; g += green_off; if (g < 0) g = 0; if (g > 255) g = 255;
                int b = (p_src[2] * blue_q8  + 128) >> 8; b += blue_off;  if (b < 0) b = 0; if (b > 255) b = 255;
                p_dst[0] = (uint8_t)r;
                p_dst[1] = (uint8_t)g;
                p_dst[2] = (uint8_t)b;
                p_dst[3] = p_src[3];
            }
        } else
#endif
        {
            // scalar path, unaligned or no NEON, or non 4-byte stride
            for (int x = 0; x < w; ++x, p_src += image_stride, p_dst += image_stride) {
                int r = (p_src[0] * red_q8   + 128) >> 8; r += red_off;   if (r < 0) r = 0; if (r > 255) r = 255;
                int g = (p_src[1] * green_q8 + 128) >> 8; g += green_off; if (g < 0) g = 0; if (g > 255) g = 255;
                int b = (p_src[2] * blue_q8  + 128) >> 8; b += blue_off;  if (b < 0) b = 0; if (b > 255) b = 255;
                p_dst[0] = (uint8_t)r;
                p_dst[1] = (uint8_t)g;
                p_dst[2] = (uint8_t)b;
                p_dst[3] = p_src[3];
            }
        }

        src_row += row_stride;
        dst_row += row_stride;
    }
}


/* Copy an axis-aligned rectangle from pixels -> mapped_pixels using memcpy only.
 * pixels layout: interleaved, image_stride bytes per pixel (typically 4 for RGBA8).
 */
static inline void copy_rect_rgb(const uint8_t *pixels, uint8_t *mapped_pixels,
                                    int image_stride, int width, int height,
                                    int x0, int y0, int w, int h)
{
    /* clip rectangle to image bounds */
    if (x0 < 0) { w += x0; x0 = 0; }
    if (y0 < 0) { h += y0; y0 = 0; }
    if (x0 + w > width)  w = width  - x0;
    if (y0 + h > height) h = height - y0;
    if (w <= 0 || h <= 0) return;

    const size_t row_stride = (size_t)width * (size_t)image_stride;
    const size_t row_bytes  = (size_t)w     * (size_t)image_stride;

    const uint8_t *src_row = pixels        + (size_t)y0 * row_stride + (size_t)x0 * image_stride;
    uint8_t       *dst_row = mapped_pixels + (size_t)y0 * row_stride + (size_t)x0 * image_stride;

    for (int y = 0; y < h; ++y) {
        memcpy(dst_row, src_row, row_bytes);
        src_row += row_stride;
        dst_row += row_stride;
    }
}


static inline void apply_panel_brightness_q8(uint8_t * pixels, uint8_t *mapped_pixels, const scene_info *scene) {
    for (int py = 0; py < scene->num_ports; ++py) {
        for (int px = 0; px < scene->num_chains; ++px) {
            const int idx = py * scene->num_chains + px;
            const int panel_type = scene->panel_types[idx] - 1;
            if ((unsigned)panel_type >= (unsigned)scene->num_panel_types) continue;

            const uint16_t rq = scene->panel_scale[panel_type].red_q8;
            const uint16_t gq = scene->panel_scale[panel_type].green_q8;
            const uint16_t bq = scene->panel_scale[panel_type].blue_q8;

            const uint16_t ro = scene->panel_offset[panel_type].red_q8;
            const uint16_t go = scene->panel_offset[panel_type].green_q8;
            const uint16_t bo = scene->panel_offset[panel_type].blue_q8;

            // nothing to do for this panel
            if (rq > 254 && gq > 254 && bq > 254) {
                copy_rect_rgb(pixels, mapped_pixels, scene->stride,
                              scene->width, scene->height,
                              px * scene->panel_width,
                              py * scene->panel_height,
                              scene->panel_width,
                              scene->panel_height);
            } else {
                scale_rect_rgb_q8_offset(pixels, mapped_pixels,
                              scene->width, scene->height,
                              scene->stride,
                              px * scene->panel_width,
                              py * scene->panel_height,
                              scene->panel_width,
                              scene->panel_height,
                              rq, gq, bq, ro, go, bo);
            }
        }
    }
}


void remap_interleaved_to_plane_major(const scene_info *scene,
                                      const uint32_t *restrict src,
                                      uint32_t *restrict dst,
                                    const size_t dst_bytes)
{
    const unsigned bit_planes   = scene->bit_depth;
    const size_t   half_height  = (size_t)scene->panel_height / 2;   /* or scene->half_height */
    const size_t   pixels       = (size_t)scene->width * half_height;
    const size_t   plane_stride = pixels;

    /* if the interleaved blocks include padding, keep +1, else set to bit_planes */
    const size_t   stride_words = (size_t)bit_planes + 1;

    assert(src != NULL && dst != NULL);

    for (size_t p = 0; p < pixels; ++p) {
        const uint32_t *block = src + p * stride_words;   /* block[0..bit_planes-1] are planes */
        for (unsigned b = 0; b < bit_planes; ++b) {
            const uint32_t sval = block[b];
            const uint32_t dst_off = b * plane_stride + p;
            dst[dst_off] = sval;
        }
    }
}

/**
 * @brief this function takes the image data and maps it to the bcm signal.
 * 
 * if scene->tone_mapper is updated, new bcm bit masks will be created.
 * 
 * @param scene the scene information
 * @param image the image to map to the scene bcm data. if NULL scene->image will be used
 */
__attribute__((hot))
void map_byte_image_to_bcm(const scene_info *scene, uint8_t *image) {

    // tone map the bits for the current scene, update if the lookup table if scene tone mapping changes....
    // TODO: create per panel tone mapping tables if panels have different characteristics
    static void     *bits = NULL;
    static uint16_t *quant_errors = NULL;
    static uint8_t  *mapped_image = NULL;
    static uint8_t  *mapped_image2 = NULL;
    static uint32_t *tmp_bcm = NULL;
    static uint8_t  phase = 1;
    phase = phase + 1 % 64;

    if (UNLIKELY(bits == NULL)) {
        if (mapped_image == NULL) {
            mapped_image = (uint8_t*)calloc(scene->width * scene->height * scene->stride, sizeof(uint8_t));
        }
        if (mapped_image2 == NULL) {
            mapped_image2 = (uint8_t*)calloc(scene->width * (scene->height +1)* scene->stride, sizeof(uint8_t));
        }
        if (tmp_bcm == NULL) {
            tmp_bcm = (uint32_t*)calloc(scene->width * scene->height * scene->stride * scene->bit_depth, sizeof(uint8_t));
        }
        if (quant_errors == NULL) {
            quant_errors = (uint16_t*)calloc(768*2, sizeof(uint16_t));
        }
        if (bits != NULL) { // don't leak memory!
            free(bits);
        }   
        bits = (uint64_t*)tone_map_rgb_bits(scene, scene->bit_depth, quant_errors);
        debug("new tone mapped bits created\n");
    }

    // select our image source
    uint8_t *image_ptr = (image == NULL) ? scene->image : image;

    // map the image to handle weird panel chain configurations
    // the image mapper should take a normal image and map it to match the chain configuration
    // the image mapper should operate on the image in place
    if (scene->image_mapper != NULL) {
        scene->image_mapper(image_ptr, mapped_image, scene);
        image_ptr = mapped_image;
    }


    if (scene->num_panel_types > 1) {
        apply_panel_brightness_q8(image_ptr, mapped_image2, scene);
        image_ptr = mapped_image2;
    }

    if (scene->dither) {
        //dither_spatial_bayer8_low(image_ptr, scene->width, scene->height, scene->stride, 80, (uint8_t)scene->dither);
        dither_spatial_hash_low(image_ptr, scene->width, scene->height, scene->stride, 80, (uint8_t)scene->dither);
    }


   
    //update_bcm_signal_fn update_bcm_signal = NULL;
    //update_bcm_signal = (update_bcm_signal_fn)update_bcm_signal_64_rgb;

    ASSERT(scene->panel_height % 16 == 0);
    ASSERT(scene->panel_width % 16 == 0);
    const uint8_t  half_height __attribute__((aligned(16))) = scene->panel_height / 2;
    // ensure 16 bit alignment for width
    const uint16_t width __attribute__((aligned(32))) = scene->width;

    // ensure alignment for the compiler to optimize these loops
    ASSERT(scene->bit_depth % BIT_DEPTH_ALIGNMENT == 0);
    ASSERT(half_height % 16 == 0);
    ASSERT(width % 32 == 0);                        // Ensure length is a multiple of 32


    // try and acquire a buffer to render to
    scene_info *mutable_scene = (scene_info *)scene;
    void *frame_pointer = NULL;
    while(!spsc_frame_try_acquire(&mutable_scene->dst_ctx, &frame_pointer)) {
        // wait until we get one...
        sched_yield();
    }
    uint32_t *bcm_signal = (uint32_t *)frame_pointer;

    // convenience variables
    const uint16_t stride     = scene->stride;
    const uint8_t  bit_depth  = scene->bit_depth;

    // we only need to process half the height of the first panel, since we are clocking in
    // 2 rows at a time (upper and lower) aand 3 ports at a time
    uint32_t *tmp_dst_ptr = tmp_bcm;
    for (uint16_t y=0; y < half_height; y ++) {
        for (uint16_t x=0; x < width; x++) {

            // create the bcm signal for the current pixel, 
            // writes bit_depth *(sizeof(uint32_t)) bytes to bcm_signal
            //update_bcm_signal_64_rgb(scene, bits, bcm_signal, image_ptr, quant_errors, phase);
            update_bcm_signal_64_rgb(scene, bits, tmp_dst_ptr, image_ptr, quant_errors, phase);

            //bcm_signal += bit_depth + 1;
            tmp_dst_ptr += bit_depth + 1;
            image_ptr += stride;
        }
    }

    const size_t max_size = scene->width * half_height * bit_depth;

    remap_interleaved_to_plane_major(scene, (const uint32_t *)tmp_bcm, bcm_signal, max_size);
    //spsc_frame_ring *dst = &mutable_scene->dst_ctx;
    // publish the frame we just rendered to the display thread
    spsc_frame_produce(&mutable_scene->dst_ctx);
}



float gradient_horiz(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1) {
    return r0;//(p1 - p3) / (p2 - p4);
}
float gradient_vert(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1) {
    return r1;//(p1 - p2) / (p3 - p4);
}
float gradient_max(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1) {
    return MAX(r0, r1);//(p1 - p2) / (p3 - p4);
}
float gradient_min(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1) {
    return MIN(r0, r1);//(p1 - p2) / (p3 - p4);
}
float gradient_quad(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r0, float r1) {
    return (r0 < r1) ? r0 / r1 : r1 / r0;
}


/**
 * @brief helper method to set a pixel in a 24 bpp RGB image buffer
 * 
 * @param scene the scene to draw the pixel at
 * @param x horizontal position (starting at 0) clamped to scene->width
 * @param y vertical position (starting at 0) clamped to scene->height
 * @param pixel RGB value to set at pixel x,y
 */
inline void hub_pixel(scene_info *scene, const int x, const int y, const RGB pixel) {
    const uint16_t fx = MIN(x, scene->width-1);
    const uint16_t fy = MIN(y, scene->height-1);
    const int offset = (fy * scene->width + fx) * scene->stride;
    ASSERT(offset < scene->width * scene->height * scene->stride);

    scene->image[offset] = pixel.r;
    scene->image[offset + 1] = pixel.g;
    scene->image[offset + 2] = pixel.b;
}

/**
 * @brief helper method to set a pixel in a 24 bpp RGB image buffer, each
 * rgb channel is scaled by factor. if scaling exceeds byte storage (255)
 * the value will wrap. saturated artithmetic is still not portable....
 * 
 * @param scene the scene to draw the pixel at
 * @param x horizontal position (starting at 0)
 * @param y vertical position (starting at 0)
 * @param pixel RGB value to set at pixel x,y
 */
inline void hub_pixel_factor(scene_info *scene, const int x, const int y, const RGB pixel, const float factor) {
    const uint16_t fx = MIN(x, scene->width-1);
    const uint16_t fy = MIN(y, scene->height-1);
    const int offset = (fy * scene->width + fx) * scene->stride;
    ASSERT(offset < scene->width * scene->height * scene->stride);

    scene->image[offset] = pixel.r * factor;
    scene->image[offset + 1] = pixel.g * factor;
    scene->image[offset + 2] = pixel.b * factor;
}



/**
 * @brief helper method to set a pixel in a 32 bit RGBA image buffer
 * NOTE: You probably want hub_pixel_factor for most cases
 * 
 * @param scene the scene to draw the pixel at
 * @param x horizontal position (starting at 0)
 * @param y vertical position (starting at 0)
 * @param pixel RGB value to set at pixel x,y
 */
inline void hub_pixel_alpha(scene_info *scene, const int x, const int y, const RGBA pixel) {
    const uint16_t fx = MIN(x, scene->width-1);
    const uint16_t fy = MIN(y, scene->height-1);
    const int offset = (fy * scene->width + fx) * scene->stride;
    ASSERT(scene->stride == 4);
    ASSERT(offset < scene->width * scene->height * scene->stride);

    Normal alpha = normalize_8(pixel.a);

    scene->image[offset] += (pixel.r * alpha);
    scene->image[offset + 1] += (pixel.g * alpha);
    scene->image[offset + 2] += (pixel.b * alpha);
    scene->image[offset + 3] = pixel.a;
}


/**
 * @brief fill in a rectangle from x1,y1 to x2,y2. x2,y2 do not need to be > x1,y1
 * 
 * @param scene 
 * @param x1
 * @param y1
 * @param x2 inclusive
 * @param y2 inclusive
 * @param color 
 */
void hub_fill(scene_info *scene, const uint16_t x1, const uint16_t y1, const uint16_t x2, const uint16_t y2, const RGB color) {
    uint16_t fx1 = x1 % scene->width;
    uint16_t fx2 = x2 % scene->width;
    uint16_t fy1 = y1 % scene->height;
    uint16_t fy2 = y2 % scene->height;

    if (fx2 < fx1) {
        uint16_t temp = fx1;
        fx1 = fx2;
        fx2 = temp;
    }
    if (fy2 < fy1) {
        uint16_t temp = fy1;
        fy1 = fy2;
        fy2 = temp;
    }
    for (int y = fy1; y <= fy2; y++) {
        for (int x = fx1; x <= fx2; x++) {
            hub_pixel(scene, x, y, color);
        }
    }
}

/**
 * @brief fill in a rectangle of width,height at x,y with the specified color
 * 
 * @param scene 
 * @param x 
 * @param y 
 * @param width 
 * @param height 
 * @param color 
 */
void hub_fill_grad(scene_info *scene, uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, Gradient gradient) {
    if (x1 < x0) {
        uint16_t temp = x0;
        x0 = x1;
        x1 = temp;
    }
    if (y1 < y0) {
        uint16_t temp = y0;
        y0 = y1;
        y1 = temp;
    }
    ASSERT(y1 < scene->height);
    ASSERT(x1 < scene->width);
    if (CONSOLE_DEBUG) {
        printf("%dx%d, %dx%d\n", x0, y0, x1, y1);
    }

    RGB left, right, final;
    float h_ratio, v_ratio = 0.0f;
    for (int y = y0; y < y1; y++) {
        v_ratio = (float)(y - y0) / (y1 - y0);

        //float vertical = gradient.type(y0, y1, x0, x1, v_ratio, 0);
        float vertical = gradient.type(x0, y0, x1, y1, v_ratio, 0);
        interpolate_rgb(&left, gradient.colorA1, gradient.colorA2, vertical);
        interpolate_rgb(&right, gradient.colorB1, gradient.colorB2, vertical);

        for (int x = x0; x < x1; x++) {
            h_ratio = (float)(x - x0) / (x1 - x0);

            //float horizontal = gradient.type(y0, y1, x0, x1, v_ratio, h_ratio);
            float horizontal = gradient.type(y0, y1, x0, x1, v_ratio, h_ratio);
            if (CONSOLE_DEBUG) {
                printf("v: %f, h: %f\n", (double)vertical, (double)horizontal);
            }
            interpolate_rgb(&final, left, right, horizontal);

            hub_pixel(scene, x, y, final);
        }
    }
}


// Draw an unfilled circle using Bresenham's algorithm
void hub_circle(scene_info *scene, const uint16_t centerX, const uint16_t centerY, const uint16_t radius, const RGB color) {
    int x = radius;
    int y = 0;
    int decisionOver2 = 1 - x; // Decision variable


    while (x >= y) {
        hub_pixel(scene, centerX + x, centerY + y, color);
        hub_pixel(scene, centerX + y, centerY + x, color);
        hub_pixel(scene, centerX - y, centerY + x, color);
        hub_pixel(scene, centerX - x, centerY + y, color);

        hub_pixel(scene, centerX - x, centerY - y, color);
        hub_pixel(scene, centerX - y, centerY - x, color);
        hub_pixel(scene, centerX + y, centerY - x, color);
        hub_pixel(scene, centerX + x, centerY - y, color);
        
        y++;

        // Update decision variable
        if (decisionOver2 <= 0) {
            decisionOver2 += 2 * y + 1; // East
        } else {
            x--;
            decisionOver2 += 2 * (y - x) + 1; // Southeast
        }
    }
}


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
void hub_line(scene_info *scene, int x0, int y0, int x1, int y1, RGB color) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1; // Step in the x direction
    int sy = (y0 < y1) ? 1 : -1; // Step in the y direction
    int err = dx - dy;           // Error value

    while (1) {
        hub_pixel(scene, x0, y0, color); // Set pixel

        // Check if we've reached the end point
        if (x0 == x1 && y0 == y1) break;

        int err2 = err * 2;
        if (err2 > -dy) { // Error term for the x direction
            err -= dy;
            x0 += sx;
        }
        if (err2 < dx) { // Error term for the y direction
            err += dx;
            y0 += sy;
        }
    }
}


/**
 * @brief draw an anti-aliased line using Xiolin Wu's anti-aliased line drawing algorithm
 * 
 * @param scene 
 * @param x0 start pixel x location
 * @param y0 start pixel y location
 * @param x1 end pixel x
 * @param y1 end pixel y
 * @param color color to draw the line
 */
void hub_line_aa(scene_info *scene, const int x0, const int y0, const int x1, const int y1, const RGB color) {

    int fx0 = MIN(x0, scene->width-1);
    int fx1 = MIN(x1, scene->width-1);
    int fy0 = MIN(y0, scene->height-1);
    int fy1 = MIN(y1, scene->height-1);

    float dx = (float)(fx1 - fx0);
    float dy = (float)(fy1 - fy0);
    
    int steep = fabs(dy) > fabs(dx);
    
    if (steep) {
        // Swap x and y
        int tmp;
        tmp = fx0; fx0 = fy0; fy0 = tmp;
        tmp = fx1; fx1 = fy1; fy1 = tmp;
        dx = (float)(fx1 - fx0);
        dy = (float)(fy1 - fy0);
    }
    
    if (fx0 > fx1) {
        // Swap (fx0, fy0) with (fx1, fy1)
        int tmp;
        tmp = fx0; fx0 = fx1; fx1 = tmp;
        tmp = fy0; fy0 = fy1; fy1 = tmp;
        dx = (float)(fx1 - fx0);
        dy = (float)(fy1 - fy0);
    }

    float gradient = (dx == 0.0f) ? 1.0f : dy / dx;

    // Handle the first endpoint
    float xend = roundf(fx0);
    float yend = fy0 + gradient * (xend - fx0);
    float xgap = rfpart(fx0 + 0.5f);
    int xpxl1 = (int)xend;
    int ypxl1 = ipart(yend);
    if (steep) {
        hub_pixel_factor(scene, ypxl1, xpxl1, color, rfpart(yend) * xgap);
        hub_pixel_factor(scene, ypxl1 + 1, xpxl1, color, fpart(yend) * xgap);
    } else {
        hub_pixel_factor(scene, xpxl1, ypxl1, color, rfpart(yend) * xgap);
        hub_pixel_factor(scene, xpxl1, ypxl1 + 1, color, fpart(yend) * xgap);
    }
    float intery = yend + gradient;  // First y-intersection for the main loop

    // Handle the second endpoint
    xend = roundf(fx1);
    yend = fy1 + gradient * (xend - fx1);
    xgap = fpart(fx1 + 0.5f);
    int xpxl2 = (int)xend;
    int ypxl2 = ipart(yend);
    if (steep) {
        hub_pixel_factor(scene, ypxl2, xpxl2, color, rfpart(yend) * xgap);
        hub_pixel_factor(scene, ypxl2 + 1, xpxl2, color, fpart(yend) * xgap);
    } else {
        hub_pixel_factor(scene, xpxl2, ypxl2, color, rfpart(yend) * xgap);
        hub_pixel_factor(scene, xpxl2, ypxl2 + 1, color, fpart(yend) * xgap);
    }

    // Main loop
    if (steep) {
        for (int x = xpxl1 + 1; x < xpxl2; x++) {
            hub_pixel_factor(scene, ipart(intery), x, color, rfpart(intery));
            hub_pixel_factor(scene, ipart(intery) + 1, x, color, fpart(intery));
            intery += gradient;
        }
    } else {
        for (int x = xpxl1 + 1; x < xpxl2; x++) {
            hub_pixel_factor(scene, x, ipart(intery), color, rfpart(intery));
            hub_pixel_factor(scene, x, ipart(intery) + 1, color, fpart(intery));
            intery += gradient;
        }
    }
}


/**
 * @brief  draw an un-anti aliased triangle
 * 
 * @param scene 
 * @param x0 
 * @param y0 
 * @param x1 
 * @param y1 
 * @param x2 
 * @param y2 
 * @param color 
 */
void hub_triangle(scene_info *scene, int x0, int y0, int x1, int y1, int x2, int y2, RGB color) {
    hub_line(scene, x0, y0, x1, y1, color);
    hub_line(scene, x1, y1, x2, y2, color);
    hub_line(scene, x2, y2, x0, y0, color);
}

void hub_triangle_aa(scene_info *scene, int x0, int y0, int x1, int y1, int x2, int y2, RGB color) {
    hub_line_aa(scene, x0, y0, x1, y1, color);
    hub_line_aa(scene, x1, y1, x2, y2, color);
    hub_line_aa(scene, x2, y2, x0, y0, color);
}


