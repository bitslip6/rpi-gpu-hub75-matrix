#include <stdint.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <smmintrin.h>
#endif

#include <math.h>
#include "hub75gpu.h"

extern int32_t __builtin_arm_qadd (int32_t, int32_t);  // ARM intrinsic for saturated addition



/**
 * @brief clamp a float to the range 0-1
 */
static inline float clamp1f(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

/**
 * @brief one step of Newton refinement helps ARM accuracy if you enable -ffast-math
 * @return float 
 */
static inline float safe_rcp(float x) {
    float r = 1.0f / x;
    // r = r * (2 - x*r); // uncomment if you use a fast approx to start
    return r;
}


/**
 * @brief return just the fractional part of x
 * @return float - x - floorf(x)
 */
static inline float fract(float x) {
    return x - floorf(x);
}

/**
 * @brief linear interpolate between two floats
 * 
 * @param x value A
 * @param y value B
 * @param a Normal 0-1 interpolation amount
 * @return float 
 */
__attribute__((pure))
static inline float mixf(const float x, const float y, const Normal a) {
    return x * (1.0f - a) + y * a;
}

/**
 * @brief  clamp a value between >= lower and <= upper
 * 
 * @param x value to clamp
 * @param lower  lower bound inclusive
 * @param upper  upper bound inclusive
 * @return float 
 */
__attribute__((pure))
static inline float clampf(const float x, const float lower, const float upper) {
	return fmaxf(lower, fminf(x, upper));
}

/**
 * @brief  clamp an integer between >= lo and <= hi
 */
__attribute__((pure))
static inline int clamp_int(const int v, const int lo, const int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}


/**
 * @brief Convert normalized coordinate to pixel coordinate
 * 
 * @param nx Normalized coordinate (0.0 to 1.0)
 * @param width Width of the display in pixels
 * @return int Pixel coordinate (0 to width-1)
 * 
 * Converts a normalized coordinate (0.0 = left edge, 1.0 = right edge)
 * to a pixel coordinate. Includes rounding and clamping to valid range.
 */
__attribute__((pure))
static inline int norm_to_px(const Normal nx, const int width) {
    float fx = nx * (float)(width  - 1);
    int   ix = (int)(fx + 0.5f);
    return clamp_int(ix, 0, width - 1);
}


/**
 * @brief hardware saturated addition of two int32_t values
 * @param a 
 * @param b 
 * @return int32_t 
 */
static inline int32_t saturating_add(int32_t a, int32_t b) {
#if defined(__arm__) || defined(__aarch64__)
    return __builtin_arm_qadd(a, b);  // GCC/Clang built-in for ARM saturated add


#elif defined(__SSE2__)  // x86 with SSE2
    __m128i va = _mm_set1_epi32(a);
    __m128i vb = _mm_set1_epi32(b);
    __m128i result = _mm_add_epi32(va, vb);  // SSE2 saturated add
    return _mm_cvtsi128_si32(result);

#else  // Portable software-based saturated arithmetic
    int32_t result = a + b;
    if (((b > 0) && (result < a)) || ((b < 0) && (result > a))) {
        result = (b > 0) ? INT32_MAX : INT32_MIN;  // Saturate on overflow or underflow
    }
    return result;
#endif
}

/**
 * helper: lerp between two uint8 channels
 */
static inline uint8_t lerp_u8(uint8_t a, uint8_t b, Normal t) {
    float v = (1.0f - t) * (float)a + t * (float)b;
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return (uint8_t)(v + 0.5f);
}

#define bit_count(x) __builtin_popcount(x)  // GCC/Clang built-in for counting set bits
