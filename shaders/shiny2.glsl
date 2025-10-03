// https://www.shadertoy.com/view/w3SSWd
/*
    Inspired by Xor's recent raymarchers with comments!
    https://www.shadertoy.com/view/tXlXDX
*/

#define OPT_FAST_TONEMAP 0

// explicit, readable version of your shader
void mainImage(out vec4 frag_color, in vec2 frag_coord)
{
    // constants
    const int   OUTER_STEPS            = 70;
    const float INNER_S_START          = 0.10;
    const float INNER_S_LIMIT          = 2.00;
    const float INNER_S_SCALE          = 1.42;
    const float DEFORM_MULTIPLIER      = 16.0;
    const float DEFORMATION_MIX        = 0.30;
    const float DIST_BASE_INCREMENT    = 0.02;
    const float DIST_SCALE             = 0.10;
    const float DIST_RADIUS            = 3.00;
    const vec4  COSINE_PHASE           = vec4(4.0, 2.0, 1.0, 0.0);
    const float TONE_DIVISOR           = 2000.0;

    // uniforms
    float time_seconds = iTime;

    // accumulators
    float distance_along_ray = 0.0;
    vec4  accum_color        = vec4(0.0);

    // replicate original camera mapping: normalize(vec3(2*u,0) - iResolution.xyy)
    // requires iResolution to be a vec3
    vec3 resolution_xyy = vec3(iResolution.x, iResolution.y, iResolution.y);

    for (int step = 0; step < OUTER_STEPS; ++step)
    {
        vec3 ray_dir = normalize(vec3(2.0 * frag_coord, 0.0) - resolution_xyy);
        vec3 p       = distance_along_ray * ray_dir;
        p.z         -= time_seconds;

        // inner deformation loop
        float s = INNER_S_START;
        while (s < INNER_S_LIMIT)
        {
            vec3 cos_term = cos(time_seconds + p * (s * DEFORM_MULTIPLIER));
            float dot_term = dot(cos_term, vec3(0.01));
            p -= (dot_term / s);
            p += sin(p.yzx * 0.90) * DEFORMATION_MIX;
            s *= INNER_S_SCALE;
        }

        float step_size = DIST_BASE_INCREMENT
                        + abs(DIST_RADIUS - length(p.yx)) * DIST_SCALE;

        distance_along_ray += step_size;

        vec4 step_color = (1.0 + cos(distance_along_ray + COSINE_PHASE)) / step_size;
        accum_color += step_color;
    }

//    frag_color = tanh(accum_color / TONE_DIVISOR);

#if OPT_FAST_TONEMAP
    // micro-opt 3: cheaper tanh approximation
    vec4 x = accum_color / TONE_DIVISOR;
    frag_color = x / (1.0 + abs(x));
#else
    frag_color = tanh(accum_color / TONE_DIVISOR);
#endif
}
