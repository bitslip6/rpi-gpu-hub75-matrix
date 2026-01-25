#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "hub75gpu.h"

/* Declarations for math helpers used across translation units. Implementations
 * live in src/functions.c (external linkage). */

Normal step_smooth(const Normal left_edge, const Normal right_edge, const Normal value);
Normal step_smoother(const Normal left_edge, const Normal right_edge, const Normal value);
Normal step_logic(const Normal edge_center, const Normal stepness, const Normal value);
Normal step_exp(const Normal e0, const Normal e1, const Normal x, const Normal g);
float  step_gauss(const NormalSigned d, float sigma);
Normal step_difference(const Normal minuend, const Normal subtrahend);

Normal sdf_coverage_smooth(const Normal sample, const Normal edge_center, const Normal smoothness);
Normal sdf_coverage_smoother(const Normal sample, const Normal edge_center, const Normal smoothness);
Normal  sdf_coverage_logistic(const float s, const float e, const float k, const float steep);
float  sdf_weight_bias(const float weight, const Normal base_band);

Normal sdf_outline_slab(const float sample, const Normal inner_edge, const Normal outer_edge, const Normal aa);
float  sdf_glow_smooth(const Normal sample, const Normal inner_edge, const Normal band_width);
float  sdf_glow_gauss_from_sample(const Normal sample, const Normal inner_edge, const Normal sigma);

#endif // FUNCTIONS_H

