#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "hub75gpu.h"
#include "mymath.h"

/**
 * @brief Apply easing function to a normalized value
 * 
 * @param t Normalized input value (0.0 to 1.0)
 * @param easing Easing function type to apply
 * @return float Eased output value (0.0 to 1.0)
 * 
 * Applies the specified easing function to smooth gradient transitions.
 * Input and output values should be in the range [0.0, 1.0].
 */
float apply_easing(float t, easing_function_t easing) {
    // Clamp input to valid range
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    
    switch (easing) {
        case EASE_LINEAR:
            return t;
            
        case EASE_IN_QUAD:
            return t * t;
            
        case EASE_OUT_QUAD:
            return 1.0f - (1.0f - t) * (1.0f - t);
            
        case EASE_IN_OUT_QUAD:
            if (t < 0.5f) {
                return 2.0f * t * t;
            } else {
                float f = 1.0f - t;
                return 1.0f - 2.0f * f * f;
            }
            
        default:
            return t; // Fallback to linear
    }
}

/**
 * @brief Calculate gradient factor for a pixel based on direction and position
 * 
 * @param x Pixel X coordinate
 * @param y Pixel Y coordinate  
 * @param minx Polygon bounding box minimum X
 * @param miny Polygon bounding box minimum Y
 * @param maxx Polygon bounding box maximum X
 * @param maxy Polygon bounding box maximum Y
 * @param direction Gradient direction
 * @return float Normalized gradient factor (0.0 to 1.0)
 * 
 * Calculates how far along the gradient this pixel should be based on
 * its position within the polygon's bounding box and the gradient direction.
 */
static float calculate_gradient_factor(int x, int y, int minx, int miny, int maxx, int maxy, gradient_direction_t direction) {
    if (maxx <= minx || maxy <= miny) return 0.0f; // Degenerate case
    
    float width = (float)(maxx - minx);
    float height = (float)(maxy - miny);
    float norm_x = (float)(x - minx) / width;
    float norm_y = (float)(y - miny) / height;
    
    switch (direction) {
        case GRADIENT_HORIZONTAL:
            return norm_x;
            
        case GRADIENT_VERTICAL:
            return norm_y;
            
        case GRADIENT_DIAGONAL:
            return (norm_x + norm_y) * 0.5f;
            
        case GRADIENT_RADIAL: {
            // Distance from center, normalized to bounding box
            float center_x = 0.5f;
            float center_y = 0.5f;
            float dx = norm_x - center_x;
            float dy = norm_y - center_y;
            float dist = sqrtf(dx * dx + dy * dy);
            // Normalize to max possible distance (corner to center)
            float max_dist = sqrtf(0.5f * 0.5f + 0.5f * 0.5f);
            return fminf(dist / max_dist, 1.0f);
        }
        
        default:
            return norm_x; // Fallback to horizontal
    }
}

/**
 * @brief Fill a horizontal span with a simple gradient 
 * 
 * @param scene Scene to draw into
 * @param y Row to fill
 * @param x0 Starting X coordinate (inclusive)
 * @param x1 Ending X coordinate (inclusive)
 * @param gradient Simple gradient configuration
 * @param minx Polygon bounding box minimum X
 * @param miny Polygon bounding box minimum Y
 * @param maxx Polygon bounding box maximum X
 * @param maxy Polygon bounding box maximum Y
 * 
 * Fills pixels from x0 to x1 (inclusive) on row y with gradient colors.
 * Uses the simple gradient system with two colors, direction, and easing.
 */
void gradient_fill(scene_info *scene, int y, int x0, int x1, 
                                             const SimpleGradient *gradient, 
                                             int minx, int miny, int maxx, int maxy) {
    if ((unsigned)y >= (unsigned)scene->height) return;
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (x1 < 0 || x0 >= scene->width) return;

    x0 = clamp_int(x0, 0, scene->width - 1);
    x1 = clamp_int(x1, 0, scene->width - 1);

    uint8_t *row = scene->image + (y * scene->width * scene->stride);
    size_t off = (size_t)x0 * 3u; /* RGB888 write */
    
    for (int x = x0; x <= x1; ++x) {
        // Calculate gradient factor for this pixel
        float grad_factor = calculate_gradient_factor(x, y, minx, miny, maxx, maxy, gradient->direction);
        
        // Apply easing function
        float eased_factor = apply_easing(grad_factor, gradient->easing);
        
        // Interpolate between start and end colors
        RGB color;
        color.r = (uint8_t)(gradient->start_color.r + (gradient->end_color.r - gradient->start_color.r) * eased_factor);
        color.g = (uint8_t)(gradient->start_color.g + (gradient->end_color.g - gradient->start_color.g) * eased_factor);
        color.b = (uint8_t)(gradient->start_color.b + (gradient->end_color.b - gradient->start_color.b) * eased_factor);
        
        row[off + 0] = color.r;
        row[off + 1] = color.g;
        row[off + 2] = color.b;
        off += scene->stride;
    }
}

/**
 * @brief Draw a filled polygon with a simple gradient
 * 
 * @param scene Scene containing the image buffer to draw into
 * @param poly Polygon with normalized coordinates (0.0 to 1.0)
 * @param gradient Simple gradient definition with two colors, direction, and easing
 * 
 * Renders a filled polygon using scanline rasterization with a simple two-color gradient.
 * The gradient direction and easing function determine how colors blend across the polygon.
 * 
 * Algorithm:
 * 1. Convert normalized coordinates to pixel coordinates
 * 2. Calculate polygon's bounding box for gradient calculations
 * 3. For each scanline, find edge intersections
 * 4. Fill between intersection pairs using gradient colors
 */
void gradient_polygon(scene_info *scene, Polygonf_t *poly, SimpleGradient gradient)
{
    if (!scene || !scene->image || !poly || poly->num_points < 3) {
        debug("gradient_polygon: bad args\n");
        return;
    }

    size_t n = poly->num_points;
    if (n > MAX_POLY_POINTS) n = MAX_POLY_POINTS; /* truncate safely */

    int vx[MAX_POLY_POINTS];
    int vy[MAX_POLY_POINTS];
    int miny = scene->height - 1;
    int maxy = 0;
    int minx = scene->width - 1;
    int maxx = 0;

    // Convert coordinates and find bounding box
    for (size_t i = 0; i < n; ++i) {
        vx[i] = norm_to_px(poly->points[i].x, scene->width);
        vy[i] = norm_to_px(poly->points[i].y, scene->height);
        if (vy[i] < miny) miny = vy[i];
        if (vy[i] > maxy) maxy = vy[i];
        if (vx[i] < minx) minx = vx[i];
        if (vx[i] > maxx) maxx = vx[i];
    }
    
    if (miny > maxy || minx > maxx) {
        debug("gradient_polygon: degenerate polygon\n");
        return;
    }

    miny = clamp_int(miny, 0, scene->height - 1);
    maxy = clamp_int(maxy, 0, scene->height - 1);
    minx = clamp_int(minx, 0, scene->width - 1);
    maxx = clamp_int(maxx, 0, scene->width - 1);

    int xints[MAX_POLY_POINTS];

    for (int y = miny; y <= maxy; ++y) {
        size_t cnt = 0;

        /* build intersections for this scanline, using upper-exclusive rule */
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            int x0 = vx[j], y0 = vy[j];
            int x1 = vx[i], y1 = vy[i];

            /* edge crosses the scanline if one end is above and the other strictly below-or-equal */
            if ( ((y0 <= y) && (y1 > y)) || ((y1 <= y) && (y0 > y)) ) {
                int dy = y1 - y0; /* nonzero due to predicate */
                int dx = x1 - x0;
                float t = (float)(y - y0) / (float)dy;
                int xi = (int)((float)x0 + t * (float)dx + 0.5f); /* round to nearest */
                if (cnt < MAX_POLY_POINTS) xints[cnt++] = xi;
            }
        }
        if (cnt == 0) {
            continue;
        }

        /* insertion sort, n ≤ 32 is tiny */
        for (size_t i = 1; i < cnt; ++i) {
            int v = xints[i];
            size_t k = i;
            while (k > 0 && xints[k - 1] > v) { xints[k] = xints[k - 1]; --k; }
            xints[k] = v;
        }

        /* fill pairs with gradient: [0,1], [2,3], ... */
        for (size_t i = 0; i + 1 < cnt; i += 2) {
            int x_start = xints[i];
            int x_end   = xints[i + 1] - 1; /* half-open to avoid overdraw at vertical edges */
            gradient_fill(scene, y, x_start, x_end, &gradient, minx, miny, maxx, maxy);
        }
    }
}
