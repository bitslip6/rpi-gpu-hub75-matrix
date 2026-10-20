#include "hub75gpu.h"
#include <math.h>

/**
 * @brief  the classic smoothstep function
 * @param 
 */
Normal step_smooth(const Normal left_edge, const Normal right_edge,
                   const Normal value) {
  Normal t = Normal_clamp((value - left_edge) / (right_edge - left_edge));
  return (const Normal)(t * t * (3.0f - 2.0f * t));
}

/**
 * @brief cubic smooth step that fades more
 *
 */
Normal step_smoother(const Normal left_edge, const Normal right_edge,
                     const Normal value) {
  float t = Normal_clamp((value - left_edge) / (right_edge - left_edge));
  float t2 = t * t, t3 = t2 * t;
  // 6t^5 - 15t^4 + 10t^3
  return (const Normal)(t3 * (6.0f * t2 - 15.0f * t + 10.0f));
}

/**
 * @brief logistic step function
 * @param edge_center - center of the step
 * @param stepness - steepness of the step 1-50 (8-20 typical)
 * @param value - input value
 */
Normal step_logic(const Normal edge_center, const Normal stepness,
                  const Normal value) {
  float dist = (value - edge_center);
  float result = 1.0f / (1.0f + expf(-(float)stepness * dist));
  return Normal_clamp(result);
}

/**
 * @brief Exponential-soft edge (gamma-like control; g>=1 sharper)
 * e0 (start of band): any finite float. Typically e0 ∈ [0,1].
 * e1 (end of band): must be different from e0. For a forward ramp, require e1 > e0.
 * x (sample): expected x ∈ [0,1] if normalized.
 * g (exponent): g ≥ 1 for a conventional soft edge. g = 1 is linear. g > 1
 * sharpens. 0 < g < 1 would soften further but is outside the intended range.
 */
Normal step_exp(const Normal e0, const Normal e1, const Normal x,
                const Normal g) {
  float t = Normal_clamp((x - e0) / (e1 - e0));
  return 1.0f - powf(1.0f - t, g);
}

/**
 * @brief Saturating difference using an inverted subtrahend.
 *
 * Computes clamp(minuend - (1.0f - subtrahend)), equivalent to
 * clamp(minuend + subtrahend - 1.0f). Inputs are expected in [0,1], and the
 * result is saturated to [0,1].
 *
 * @param minuend     Normal in [0,1].
 * @param subtrahend  Normal in [0,1]; inverted before subtraction.
 * @return Normal     Saturated result in [0,1].
 */
Normal step_difference(const Normal minuend, const Normal subtrahend) {
    return Normal_clamp(minuend - subtrahend);
}

Normal step_cutout(const Normal minuend, const Normal subtrahend) {
    return Normal_clamp(minuend * (1.0f - subtrahend));
}



/**
 * @brief Purpose: Gaussian weight that decays with distance from the edge.
 * Parameters and valid ranges
 * d (distance): any finite float. Convention here, d >= 0 is outside, d < 0
 * inside, but the function is even in d. sigma (spread): sigma > 0. Typical
 * sigma ∈ (0, 32] depending on pixel scale. Output range: (0,1]. At d = 0,
 * returns 1. As |d| → ∞, tends to 0. Behavior notes Extremely small sigma makes
 * the falloff very steep. Very large d/sigma can underflow toward zero, which
 * is expected.
 */
float step_gauss(const NormalSigned d, float sigma) {
  // d>=0 outside edge, d<0 inside; sigma in pixels/edge-units
  return expf(-(d * d) / (2.0f * sigma * sigma));
}

/**
 * @brief Purpose: Smoothstep-based coverage around edge center e with half-band
 * k. Parameters and valid ranges
 * @param (sample): expected s ∈ [0,1].
 * @param (edge_center): typically e ∈ [0,1], often ≈ 0.5.
 * @param (smoothness): smoothness > 0. Typical smoothness ∈ (0, 0.5]. The
 * transition spans [edge - smooth, edge + smooth]. Output range Normal: [0,1].
 * Behavior notes
 * Equivalent to smoothstepf(e - k, e + k, s). Wider k gives softer transition.
 */
Normal sdf_coverage_smooth(const Normal sample,
                           const Normal edge_center,
                           const Normal smoothness) {
  // Classic AA stroke coverage
	//float a_stroke = smoothstepf(0.5f - k + weight_bias, 0.5f + k + weight_bias, sample);
  return step_smooth(edge_center - smoothness, edge_center + smoothness,
                     sample);
}

/**
 * @see sdf_coverage_smooth
 */
Normal sdf_coverage_smoother(const Normal sample,
                             const Normal edge_center,
                             const Normal smoothness) {
  return step_smoother(edge_center - smoothness, edge_center + smoothness,
                       sample);
}

Normal sdf_coverage_logistic(const float s, const float e,
                            const float k, const float steep) {
  // Map to a symmetric band with logistic curve
  float t0 = step_logic(e - k, steep, s);
  float t1 = step_logic(e + k, steep, s);
  return Normal_clamp(
      (t0 + (1.0f - t1))); // approximate band; 1 inside band, 0 outside
}

/**
 * Weight as threshold bias (weight >1 bold, weight <1 thin)
 * Purpose: Compute a bias offset to widen or thin a stroke band.
 * @param weight: practical positive range. weight = 1 yields zero bias. weight
 * > 1 thickens, 0 < weight < 1 thins. Reasonable range weight ∈ (0, 4].
 * @param base_band: a positive half-band scale factor, typically the same
 * smoothness used in coverage. Practical base_band ∈ (0, 0.5]. Output range:
 * Unbounded real, but intended as a small signed offset. For typical weight and
 * base_band, the bias is in [-base_band, +∞) if weight ≥ 0.
 */
float sdf_weight_bias(const float weight,
                      const Normal base_band) {
  return (1.0f - weight) * base_band;
}

/**
 * Crisp outline slab with small AA at both edges
 * Purpose: Produce a flat unity band between two edges with small antialiased
 *shoulders.
 * @param sample - the glyph texture sample
 * @param inner_edge (e1): edge position where the inner shoulder ends. Typical
 *∈ [0,1].
 * @param outer_edge (o0): the opposite edge. For a noninverted band, require
 *outer_edge < inner_edge.
 * @param aa: antialias width, aa > 0. Practical aa ∈ (0, 0.1] relative to
 *normalized SDF.
 *
 * Behavior notes
 * outer = smoothstepf(outer_edge, outer_edge + aa, s) ramps from 0 to 1 near
 * outer_edge. inner = 1 - smoothstepf(inner_edge - aa, inner_edge, s) ramps from
 * 1 to 0 near inner_edge. The product gives a flat top between edges and soft
 * edges at both sides. If outer_edge >= inner_edge, the band collapses. Ensure
 * ordering.
 */
Normal
sdf_outline_slab(const Normal sample,
                 const Normal inner_edge, // stroke outer edge (e1)
                 const Normal outer_edge, // outer limit (o0)
                 const Normal aa)         // small AA width
{
  // Outer edge ramp up (0→1 from outer_edge to outer_edge+aa)
  Normal outer = step_smooth(outer_edge, outer_edge + aa, sample);
  // Inner edge ramp down (1→0 from inner_edge-aa to inner_edge)
  Normal inner = 1.0f - step_smooth(inner_edge - aa, inner_edge, sample);
  return Normal_clamp(outer * inner); // flat 1.0 in the middle band
}

/**
* Glow outside the stroke: unity at edge, 0 at radius
* Purpose: Smooth falloff starting at the edge and going outward a fixed width.
* Parameters and valid ranges
* s: expected ∈ [0,1].
* inner_edge (e1): the stroke edge location, typical ∈ [0,1].
* band_width: positive radius in normalized units, band_width > 0. Practical ∈ (0, 1].
* Output range: [0,1].
* Behavior notes
* Equivalent to smoothstepf(inner_edge - band_width, inner_edge, s).
* Returns 1 at the edge, fades to 0 at band_edge = inner_edge - band_width.
*/
Normal sdf_glow_smooth(const Normal sample,
                const Normal inner_edge, // e1
                const Normal band_width) // mapped from px radius
{
  Normal diff = Normal_clamp(inner_edge - band_width);
  return step_smooth(diff, inner_edge, sample);
}

/**
* Gaussian variant (softer tail). d is measured from edge: d = inner_edge - s (>=0 outside)
* Purpose: Gaussian glow outside the edge with standard deviation sigma.
* Parameters and valid ranges
* s: expected ∈ [0,1].
* inner_edge: typical ∈ [0,1].
* sigma: sigma > 0. Practical ∈ (0, 0.5] in normalized units.
* Output range: (0,1]. Exactly 1 at the edge (s == inner_edge), decreasing toward 0 with distance.
* Behavior notes
* Computes d = max(0, inner_edge - s), so only outside-of-edge contributes.
* For very small sigma, falloff is almost a step. For larger sigma, tail is wide.
*/
float sdf_glow_gauss_from_sample(const Normal sample,
                                 const Normal inner_edge,
                                 const Normal sigma) {
  float d = fmaxf(0.0f, inner_edge - sample);
  return step_gauss(d, sigma);
}
