#include <math.h>
#include <float.h>
#include "rpihub75.h"


#ifndef BF_EPS
#define BF_EPS 1e-7f
#endif


/**
 * @brief  Branch-lean RGB→HSL. h ∈ [0,1), s ∈ [0,1], l ∈ [0,1].
 * convert RGB to HSL color space
 * 
 * @param in 
 * @param out 
 */
static inline void rgb_to_hsl(const RGBF *in, HSLF *out) {
    float r = clamp1f(in->r);
    float g = clamp1f(in->g);
    float b = clamp1f(in->b);

    // pairwise min/max are cheap and compile well on ARM
    float max_rg = fmaxf(r, g);
    float min_rg = fminf(r, g);
    float max_v  = fmaxf(max_rg, b);
    float min_v  = fminf(min_rg, b);

    float chroma = max_v - min_v;
    float l = 0.5f * (max_v + min_v);
    out->l = l;

    // Fast path for gray, still a single predictable branch
    if (chroma <= BF_EPS) {
        out->h = 0.0f;
        out->s = 0.0f;
        return;
    }

    // Branchless saturation: s = chroma / (1 - |2l - 1|)
    float denom_s = 1.0f - fabsf(2.0f * l - 1.0f);
    denom_s = fmaxf(denom_s, BF_EPS);
    out->s = chroma * safe_rcp(denom_s);

    // Choose hue sector without cascaded if/else.
    // Break ties deterministically to avoid double matches.
    int r_is_max = (r >= g) & (r >= b);
    int g_is_max = (!r_is_max) & (g >= b);
    // b_is_max is implied when both above are zero.

    float inv_chroma = safe_rcp(chroma);

    // Compute each candidate hue base, then blend using masks.
    float h_r = (g - b) * inv_chroma;                 // sector 0
    float h_g = (b - r) * inv_chroma + 2.0f;          // sector 2
    float h_b = (r - g) * inv_chroma + 4.0f;          // sector 4

    // Convert masks to floats, then blend without branches.
    float fr = (float)r_is_max;
    float fg = (float)g_is_max;
    float fb = 1.0f - fr - fg;

    float h = (fr * h_r + fg * h_g + fb * h_b) * (1.0f / 6.0f);

    // Normalize to [0,1) with branchless corrections
    h += (h < 0.0f);
    h -= (h >= 1.0f);

    out->h = h;
}



/**
 * @brief triangular wave that maps hue to channel weight in [0,1]
 * @param h 
 * @return float 
 */
static inline float hue_tri_unit(float h) {
    // assume h is any real, we wrap using fract
    float t = fabsf(fract(h) * 6.0f - 3.0f) - 1.0f; // in [-1,1]
    // branchless clamp to [0,1]
    return fminf(fmaxf(t, 0.0f), 1.0f);
}

/**
 * @brief  Branch-lean HSL -> RGB. h in [0,1) preferred, s,l in [0,1].
 */
void hsl_to_rgb_fast(HSLF *in, RGBF *out) {

    // optional input clamp if upstream may overshoot
    in->h = in->h - floorf(in->h);           // wrap to [0,1)
    in->s = clamp1f(in->s);
    in->l = clamp1f(in->l);

    // chroma and match term
    // c = (1 - |2l - 1|) * s
    float c = (1.0f - fabsf(2.0f * in->l - 1.0f)) * in->s;
    float m = in->l - 0.5f * c;

    // channel weights via shifted hue triangular waves
    float r1 = hue_tri_unit(in->h + 1.0f / 3.0f);
    float g1 = hue_tri_unit(in->h);
    float b1 = hue_tri_unit(in->h - 1.0f / 3.0f);

    // scale and add match
    out->r = m + c * r1;
    out->g = m + c * g1;
    out->b = m + c * b1;
}