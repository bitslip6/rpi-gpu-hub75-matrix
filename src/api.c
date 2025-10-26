#include <threads.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "hub75gpu.h"
#include "pixels.h"
#include "gradient.h"

#include "debug.h"


/* per-thread current scene */
#if defined(__STDC_NO_THREADS__)
#  error "need thread local storage support (_Thread_local)"
#endif

/* Thread-local storage for the current scene being processed by this thread */
static _Thread_local scene_info *tls_scene = NULL;

/**
 * @brief Clear the current scene's image buffer by setting all pixels to black
 * 
 * Sets all pixels in the scene's image buffer to 0 (black). Checks that a scene
 * is set before attempting to clear.
 */
static void api_clear() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    memset(tls_scene->image, 0, (size_t)(tls_scene->width * tls_scene->height * tls_scene->stride));
}

/**
 * @brief Set a single pixel to the specified color
 * 
 * @param x X coordinate of the pixel
 * @param y Y coordinate of the pixel 
 * @param color RGB color value to set
 * 
 * Sets a pixel at the given coordinates to the specified color. Checks that
 * a scene is set before attempting to draw.
 */
static void api_pixel(uint16_t x, uint16_t y, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_pixel(tls_scene, x, y, color);
}

/**
 * @brief Draw a line between two points using basic line algorithm
 * 
 * @param x1 Starting X coordinate
 * @param y1 Starting Y coordinate
 * @param x2 Ending X coordinate
 * @param y2 Ending Y coordinate
 * @param color RGB color for the line
 * 
 * Draws a line from (x1,y1) to (x2,y2) using a basic line drawing algorithm.
 * Checks that a scene is set before attempting to draw.
 */
static void api_line(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_line(tls_scene, x1, y1, x2, y2, color);
}

/**
 * @brief Draw an anti-aliased line between two points
 * 
 * @param x1 Starting X coordinate
 * @param y1 Starting Y coordinate
 * @param x2 Ending X coordinate
 * @param y2 Ending Y coordinate
 * @param color RGB color for the line
 * 
 * Draws a smooth anti-aliased line from (x1,y1) to (x2,y2) for better visual quality.
 * Checks that a scene is set before attempting to draw.
 */
static void api_line_aa(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_line_aa(tls_scene, x1, y1, x2, y2, color);
}


/**
 * @brief Begin a new frame for drawing operations
 * 
 * Acquires a new frame buffer from the ring buffer for drawing operations.
 * This must be called before any drawing operations and paired with api_end_frame().
 * Waits up to 10ms for a buffer to become available.
 * 
 * Sets tls_scene->frame_ready to false and updates tls_scene->image pointer.
 */
static void api_begin_frame() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    if (!tls_scene->frame_ready) {
        debug("previous frame not released\n");
        return;
    }
    uint8_t *image = NULL;
    image = spsc_push_ptr_begin(tls_scene->ring_buf_mapper, 200);
    for(int i = 0; i < 10 && image == NULL; i++) {
        usleep(1000);
        image = spsc_push_ptr_begin(tls_scene->ring_buf_mapper, 200);
    }
    if (image == NULL) {
        debug("timed out waiting for frame buffer\n");
        return;
    }
    tls_scene->frame_ready = false;
    
    tls_scene->image = image;
}

/**
 * @brief Complete the current frame and submit it for display
 * 
 * Finalizes the current frame and commits it to the ring buffer for display.
 * This must be called after api_begin_frame() and all drawing operations are complete.
 * 
 * Sets tls_scene->frame_ready to true and commits the buffer to the mapper.
 */
static void api_end_frame() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }

    if (tls_scene->frame_ready) {
        debug("begin frame not called\n");
        return;
    }

    tls_scene->frame_ready = true;
    spsc_push_ptr_commit(tls_scene->ring_buf_mapper);
}


/* helpers */
/**
 * @brief Clamp an integer value to a specified range
 * 
 * @param v Value to clamp
 * @param lo Lower bound (inclusive)
 * @param hi Upper bound (inclusive)
 * @return int Clamped value between lo and hi
 * 
 * Ensures the value v is within the range [lo, hi]. If v is less than lo,
 * returns lo. If v is greater than hi, returns hi. Otherwise returns v.
 */
static inline int clamp_int(int v, int lo, int hi) {
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
static inline int norm_to_px(Normal nx, int width) {
    float fx = nx * (float)(width  - 1);
    int   ix = (int)(fx + 0.5f);
    return clamp_int(ix, 0, width - 1);
}


/**
 * @brief Fill a horizontal span of pixels with the specified color
 * 
 * @param scene Scene containing the image buffer
 * @param y Y coordinate (row) to fill
 * @param x0 Starting X coordinate (inclusive)
 * @param x1 Ending X coordinate (inclusive)
 * @param c RGB color to fill with
 * 
 * Fills pixels from x0 to x1 (inclusive) on row y with the specified color.
 * Handles coordinate swapping if x0 > x1 and clamps coordinates to valid ranges.
 * Uses RGB888 format (3 bytes per pixel).
 */
static inline void fill_span_rgb(scene_info *scene, int y, int x0, int x1, RGB c) {
    if ((unsigned)y >= (unsigned)scene->height) return;
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (x1 < 0 || x0 >= scene->width) return;

    x0 = clamp_int(x0, 0, scene->width - 1);
    x1 = clamp_int(x1, 0, scene->width - 1);

    uint8_t *row = scene->image + (y * scene->width * scene->stride);
    size_t off = (size_t)x0 * 3u;                 /* RGB888 write */
    for (int x = x0; x <= x1; ++x) {
        row[off + 0] = c.r;
        row[off + 1] = c.g;
        row[off + 2] = c.b;
        off += scene->stride;
    }
}

/**
 * @brief Draw a filled polygon with specified colors based on winding order
 * 
 * @param scene Scene containing the image buffer to draw into
 * @param poly Polygon with normalized coordinates (0.0 to 1.0)
 * @param color1 Color for counter-clockwise winding
 * @param color2 Color for clockwise winding  
 * 
 * Renders a filled polygon using scanline rasterization. The polygon vertices
 * should be in normalized coordinates. Uses different colors based on the
 * polygon's winding order (determined by signed area calculation).
 * 
 * Algorithm:
 * 1. Convert normalized coordinates to pixel coordinates
 * 2. Calculate polygon's signed area to determine winding
 * 3. For each scanline, find edge intersections
 * 4. Fill between intersection pairs
 */
void draw_polygon_fill(scene_info *scene, Polygonf_t *poly, RGB color)
{
    if (!scene || !scene->image || !poly || poly->num_points < 3) {
        debug("draw_polygon: bad args\n");
        return;
    }

    size_t n = poly->num_points;
    if (n > MAX_POLY_POINTS) n = MAX_POLY_POINTS;   /* truncate safely */

    int vx[MAX_POLY_POINTS];
    int vy[MAX_POLY_POINTS];
    int miny = scene->height - 1;
    int maxy = 0;

    for (size_t i = 0; i < n; ++i) {
        vx[i] = norm_to_px(poly->points[i].x, scene->width);
        vy[i] = norm_to_px(poly->points[i].y, scene->height);
        if (vy[i] < miny) miny = vy[i];
        if (vy[i] > maxy) maxy = vy[i];
    }
    if (miny > maxy) {
        debug("draw_polygon: degenerate polygon\n");
        return;
    }

    miny = clamp_int(miny, 0, scene->height - 1);
    maxy = clamp_int(maxy, 0, scene->height - 1);

    int a = 0; // is the polygon rolled forward or backward?
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        a += vx[j] * vy[i] - vx[i] * vy[j];
    }

    int xints[MAX_POLY_POINTS];

    for (int y = miny; y <= maxy; ++y) {
        size_t cnt = 0;

        /* build intersections for this scanline, using upper-exclusive rule */
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            int x0 = vx[j], y0 = vy[j];
            int x1 = vx[i], y1 = vy[i];

            /* edge crosses the scanline if one end is above and the other strictly below-or-equal */
            if ( ((y0 <= y) && (y1 > y)) || ((y1 <= y) && (y0 > y)) ) {
                int dy = y1 - y0;                  /* nonzero due to predicate */
                int dx = x1 - x0;
                float t = (float)(y - y0) / (float)dy;
                int xi = (int)((float)x0 + t * (float)dx + 0.5f);   /* round to nearest */
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


        /* fill pairs: [0,1], [2,3], ... */
        for (size_t i = 0; i + 1 < cnt; i += 2) {
            int x_start = xints[i];
            int x_end   = xints[i + 1] - 1;   /* half-open to avoid overdraw at vertical edges */
            fill_span_rgb(scene, y, x_start, x_end, color);
        }
    }
}

/**
 * @brief Determine the winding order of a polygon
 * 
 * @param poly Polygon with normalized coordinates to analyze
 * @return poly_winding_t Winding order: POLY_CCW, POLY_CW, or POLY_DEGENERATE
 * 
 * Calculates the signed area of the polygon to determine its winding order:
 * - Positive area = counter-clockwise (CCW)
 * - Negative area = clockwise (CW) 
 * - Near-zero area = degenerate (collinear points)
 * 
 * Uses the shoelace formula with double precision for numerical robustness.
 */
poly_winding_t polygon_winding(const Polygonf_t *poly)
{
    if (!poly || poly->num_points < 3) return POLY_DEGENERATE;

    double a = 0.0; /* use double for robustness */
    size_t n = poly->num_points;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly->points[i].x;
        double yi = poly->points[i].y;
        double xj = poly->points[j].x;
        double yj = poly->points[j].y;
        a += xj * yi - xi * yj;
    }

    /* treat tiny areas as degenerate to avoid jitter on almost-collinear input */
    const double eps = 1e-12;
    if (a > eps)  return POLY_CCW;
    if (a < -eps) return POLY_CW;
    return POLY_DEGENERATE;
}

/**
 * @brief Draw a polygon with simple gradient using the current thread-local scene
 * 
 * @param poly Polygon with normalized coordinates to draw
 * @param gradient Simple gradient definition
 * 
 * Public API function that draws a gradient polygon using the current thread's scene.
 * This is a wrapper around gradient_polygon() that uses the thread-local scene.
 * Checks that a scene is set before attempting to draw.
 */
void api_poly_gradient(Polygonf_t *poly, SimpleGradient gradient) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }

    gradient_polygon(tls_scene, poly, gradient);
}


/**
 * @brief Fill a horizontal span with a simple gradient using the current thread-local scene
 */
void api_fill_gradient(int y, int x0, int x1, 
                                             const SimpleGradient *gradient, 
                                             int minx, int miny, int maxx, int maxy) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }

    gradient_fill(tls_scene, y, x0, x1, gradient, minx, miny, maxx, maxy);
}

/**
 * @brief Draw a polygon using the current thread-local scene
 * 
 * @param poly Polygon with normalized coordinates to draw
 * @param color1 Color for counter-clockwise winding
 * @param color2 Color for clockwise winding
 * 
 * Public API function that draws a polygon using the current thread's scene.
 * This is a wrapper around draw_polygon_fill() that uses the thread-local scene.
 * Checks that a scene is set before attempting to draw.
 */
void api_poly(Polygonf_t *poly, RGB color1) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }

    draw_polygon_fill(tls_scene, poly, color1);
}

/**
 * @brief Request a graceful shutdown of the current scene's rendering
 */
void api_shutdown() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub75_request_shutdown(tls_scene);
}

void api_pixel_factor (int x, int y, RGB pixel, float factor) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_pixel_factor(tls_scene, x, y, pixel, factor);
}

void api_pixel_alpha (int x, int y, RGBA pixel) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_pixel_alpha(tls_scene, x, y, pixel);
}   

camera_t *api_new_camera(void) {
    camera_t *cam = calloc(1, sizeof(camera_t));

    cam->position.z = 5.0f;
    cam->up.y = 1.0f;
    cam->fov_y = (float)M_PI / 3.0f;
    cam->aspect = 16.0f / 9.0f;
    cam->z_near = 0.1f;
    cam->z_far = 100.0f;

    return cam;
}

transform_t *api_new_transform(void) {
    transform_t *xform = calloc(1, sizeof(transform_t));

    xform->scale.x = 1.0f;
    xform->scale.y = 1.0f;
    xform->scale.z = 1.0f;

    return xform;
}


/**
 * @brief Project an object's transform using the specified camera
 */
mat4 api_geo_project(camera_t *cam, transform_t *obj_xform) {
    return camera_project(cam, obj_xform);
}

void api_geo_render(const vec3 *in_vertices, const size_t n, const mat4 mvp, vec3 *out_ndc) {
    transform_mesh_to_ndc(in_vertices, n, mvp, out_ndc);
}

object_t *api_new_object(uint16_t num_vertices, uint16_t num_edges, uint16_t num_faces) {
    return object_new((uint16_t)num_vertices, (uint16_t)num_edges, (uint16_t)num_faces);
}

object_t *api_new_cube(void) {
    return object_cube();
}

object_t *api_new_tetrahedron(void) {
    return object_tetrahedron();
}

object_t *api_new_octahedron(void) {
    return object_octahedron();
}

object_t *api_new_pyramid(void) {
    return object_pyramid();
}

object_t *api_new_cylinder(uint16_t segments) {
    return object_cylinder(segments);
}

object_t *api_new_sphere(uint16_t subs) {
    return object_sphere(subs);
}

object_t *api_new_plane(uint16_t width_segments, uint16_t height_segments) {
    return object_plane(width_segments, height_segments);
}


object_t* api_new_torus(uint16_t major_segments, uint16_t minor_segments) {
    return object_torus(major_segments, minor_segments);
}

void api_geo_render_wire(const camera_t *cam, object_t *obj, const transform_t *obj_xform) {

    mat4 mvp = camera_project(cam, obj_xform);

    //vec3 ndc[8];
    //api.geo_render(obj->vertex_list->vertices, obj->vertex_list->num_vertices, mvp, ndc);
    transform_mesh_to_ndc(obj->verticies->list, obj->verticies->length, mvp, obj->rendered_vertices);


    const uint16_t w = (uint16_t)(tls_scene->width * 0.5f);
    const uint16_t h = (uint16_t)(tls_scene->height * 0.5f);
    for (size_t i = 0; i < obj->edges->length; i++) {
        //debug("Vertex %zu: NDC (%.3f, %.3f, %.3f)\n", i, obj->rendered_vertices[i].x, obj->rendered_vertices[i].y, obj->rendered_vertices[i].z);
        vec2 edge = obj->edges->list[i];
        vec3 v1 = obj->rendered_vertices[(size_t)edge.x];
        vec3 v2 = obj->rendered_vertices[(size_t)edge.y];

        const uint16_t x1 = (uint16_t)(w * (v1.x + 1.0f));
        const uint16_t y1 = (uint16_t)(h * (v1.y + 1.0f));
        const uint16_t x2 = (uint16_t)(w * (v2.x + 1.0f));
        const uint16_t y2 = (uint16_t)(w * (v2.x + 1.0f));
        
        hub_line(tls_scene, x1, y1, x2, y2, obj->edge_colors->list[i]);
    }
}

void api_geo_render_filled(const camera_t *cam, object_t *obj, const transform_t *obj_xform) {
    if (!obj || !obj->faces || !obj->verticies) return;

    mat4 mvp = camera_project(cam, obj_xform);
    
    /* Transform vertices to NDC space */
    transform_mesh_to_ndc(obj->verticies->list, obj->verticies->length, mvp, obj->rendered_vertices);

    const uint16_t w = (uint16_t)(tls_scene->width * 0.5f);
    const uint16_t h = (uint16_t)(tls_scene->height * 0.5f);
    
    /* Render each triangle face */
    for (size_t i = 0; i < obj->faces->length; i++) {
        vec3 face = obj->faces->list[i];
        
        /* Get the three vertices of the triangle */
        vec3 v1 = obj->rendered_vertices[(size_t)face.x];
        vec3 v2 = obj->rendered_vertices[(size_t)face.y];
        vec3 v3 = obj->rendered_vertices[(size_t)face.z];
        
        /* Convert NDC to screen coordinates */
        uint16_t x1 = (uint16_t)(w * (v1.x + 1.0f));
        uint16_t y1 = (uint16_t)(h * (v1.y + 1.0f));
        uint16_t x2 = (uint16_t)(w * (v2.x + 1.0f));
        uint16_t y2 = (uint16_t)(h * (v2.y + 1.0f));
        uint16_t x3 = (uint16_t)(w * (v3.x + 1.0f));
        uint16_t y3 = (uint16_t)(h * (v3.y + 1.0f));
        
        /* Backface culling - check if triangle is facing away */
        if (obj->normals && obj->normals->list) {
            vec3 normal = obj->normals->list[i];
            /* Simple backface culling: if normal.z < 0, triangle faces away */
            if (normal.z < 0.0f) continue;
        }
        
        /* Create a triangle polygon for rendering */
        Polygonf_t triangle;
        triangle.num_points = 3;
        triangle.points[0] = (Pointf_t){(float)x1, (float)y1};
        triangle.points[1] = (Pointf_t){(float)x2, (float)y2};
        triangle.points[2] = (Pointf_t){(float)x3, (float)y3};
        
        /* Use edge color for face color (index by face) */
        RGB face_color = (i < obj->edge_colors->length) ? 
                        obj->edge_colors->list[i] : 
                        (RGB){255, 255, 255}; /* default white */
        
        /* Render the filled triangle */
        draw_polygon_fill(tls_scene, &triangle, face_color);
    }
}




/**
 * @brief Function pointer table containing all drawing API functions
 * 
 * Static constant structure that maps function pointers to the implementation
 * functions. This table is returned by hub75gpu() to provide the drawing API.
 */
static const hub75gpu_t api_table = {
    .clear = api_clear,
    .pixel = api_pixel,
    .pixel_factor = api_pixel_factor,
    .pixel_alpha = api_pixel_alpha,
    .line = api_line,
    .line_aa = api_line_aa,
    .poly = api_poly,
    .poly_gradient = api_poly_gradient,
    .fill_gradient = api_fill_gradient,
    .begin_frame = api_begin_frame,

    .geo_object = api_new_object,
    .geo_cube = api_new_cube,
    .geo_tetrahedron = api_new_tetrahedron,
    .geo_octahedron = api_new_octahedron,
    .geo_pyramid = api_new_pyramid,
    .geo_cylinder = api_new_cylinder,
    .geo_sphere = api_new_sphere,
    .geo_torus = api_new_torus,
    .geo_plane = api_new_plane,


    .geo_camera = api_new_camera,
    .geo_transform = api_new_transform,
    .geo_project = api_geo_project,
    .geo_render_wire = api_geo_render_wire,
    .geo_render_filled = api_geo_render_filled,

    .end_frame = api_end_frame,
    .shutdown = api_shutdown,
};


/**
 * @brief Initialize the drawing API for a specific scene
 * 
 * @param scene Scene to associate with the current thread for drawing operations
 * @return hub75gpu_t Function pointer table for drawing operations
 * 
 * Sets up the thread-local scene and returns the API function table.
 * This allows each thread to have its own scene context while using the
 * same drawing functions. The scene parameter is stored in thread-local
 * storage and used by all subsequent drawing operations in this thread.
 */
hub75gpu_t hub75gpu(scene_info *scene) {
    tls_scene = scene;
    return api_table;
}

