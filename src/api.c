#include <threads.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "hub75gpu.h"
#include "pixels.h"
#include "gradient.h"
#include "mymath.h"

#include "debug.h"


/* per-thread current scene */
#if defined(__STDC_NO_THREADS__)
#  error "need thread local storage support (_Thread_local)"
#endif

/* Thread-local storage for the current scene being processed by this thread */
static _Thread_local hub75_display_t *tls_scene = NULL;
static _Thread_local scene3d_t *tls_current_os = NULL;

/* Minimal 3D helpers for normal-based culling */
static inline vec3 v3_add(vec3 a, vec3 b){ return (vec3){a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline vec3 v3_sub(vec3 a, vec3 b){ return (vec3){a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline float v3_dot(vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b){
    return (vec3){ a.y*b.z - a.z*b.y,
                   a.z*b.x - a.x*b.z,
                   a.x*b.y - a.y*b.x };
}
static inline vec3 v3_norm(vec3 v){ float d = sqrtf(v3_dot(v,v)); return d>0? (vec3){v.x/d,v.y/d,v.z/d} : (vec3){0,0,0}; }

/* Build model rotation 3x3 (column-major) from Euler angles, order Rz*Ry*Rx */
static inline void mat3_model_rotation(const vec3 euler, float Rm[9]){
    float cx = cosf(euler.x), sx = sinf(euler.x);
    float cy = cosf(euler.y), sy = sinf(euler.y);
    float cz = cosf(euler.z), sz = sinf(euler.z);

    /* Column-major rotation matrices matching 3d.c (column vectors) */
    /* Rx */
    float Rx[9] = { 1, 0, 0,
                    0, cx, -sx,
                    0, sx,  cx };
    /* Ry */
    float Ry[9] = {  cy, 0, sy,
                     0,  1, 0,
                    -sy, 0, cy };
    /* Rz */
    float Rz[9] = {  cz, -sz, 0,
                     sz,  cz, 0,
                     0,   0,  1 };

    /* temp = Ry*Rx, then Rm = Rz*temp (column-major multiply) */
    float T[9];
    for(int col=0; col<3; ++col){
        for(int row=0; row<3; ++row){
            T[col*3+row] = Ry[0*3+row]*Rx[col*3+0]
                         + Ry[1*3+row]*Rx[col*3+1]
                         + Ry[2*3+row]*Rx[col*3+2];
        }
    }
    for(int col=0; col<3; ++col){
        for(int row=0; row<3; ++row){
            Rm[col*3+row] = Rz[0*3+row]*T[col*3+0]
                          + Rz[1*3+row]*T[col*3+1]
                          + Rz[2*3+row]*T[col*3+2];
        }
    }
}

/* Build view rotation 3x3 (column-major) using look-at axes: columns = s, u, -f */
static inline void mat3_view_rotation(const camera_t *cam, float Rv[9]){
    vec3 f = v3_norm(v3_sub(cam->target, cam->position));
    vec3 up = cam->up.x==0 && cam->up.y==0 && cam->up.z==0 ? (vec3){0,1,0} : cam->up;
    vec3 s = v3_norm(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    vec3 mf = (vec3){-f.x, -f.y, -f.z};
    /* columns: s, u, -f */
    Rv[0]=s.x; Rv[3]=s.y; Rv[6]=s.z;
    Rv[1]=u.x; Rv[4]=u.y; Rv[7]=u.z;
    Rv[2]=mf.x;Rv[5]=mf.y;Rv[8]=mf.z;
}

static inline vec3 mat3_mul_v3(const float M[9], vec3 v){
    return (vec3){ M[0]*v.x + M[3]*v.y + M[6]*v.z,
                   M[1]*v.x + M[4]*v.y + M[7]*v.z,
                   M[2]*v.x + M[5]*v.y + M[8]*v.z };
}

/* Minimal 4x4 helpers (row-major) to compute view-space vertices for culling */
static inline mat4 m4_identity(void){
    mat4 r = { .m = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    }}; return r;
}
static inline mat4 m4_mul(mat4 a, mat4 b){
    mat4 r = {0};
    for(int row=0; row<4; ++row){
        for(int col=0; col<4; ++col){
            r.m[row*4+col] = a.m[row*4+0]*b.m[0*4+col]
                           + a.m[row*4+1]*b.m[1*4+col]
                           + a.m[row*4+2]*b.m[2*4+col]
                           + a.m[row*4+3]*b.m[3*4+col];
        }
    }
    return r;
}
static inline vec3 m4_mul_point(mat4 m, vec3 p){
    float x = m.m[0]*p.x + m.m[1]*p.y + m.m[2]*p.z + m.m[3];
    float y = m.m[4]*p.x + m.m[5]*p.y + m.m[6]*p.z + m.m[7];
    float z = m.m[8]*p.x + m.m[9]*p.y + m.m[10]*p.z + m.m[11];
    float w = m.m[12]*p.x + m.m[13]*p.y + m.m[14]*p.z + m.m[15];
    if (w != 0.0f) { x /= w; y /= w; z /= w; }
    return (vec3){x,y,z};
}
static inline vec3 m4_mul_point_affine(mat4 m, vec3 p){
    float x = m.m[0]*p.x + m.m[1]*p.y + m.m[2]*p.z + m.m[3];
    float y = m.m[4]*p.x + m.m[5]*p.y + m.m[6]*p.z + m.m[7];
    float z = m.m[8]*p.x + m.m[9]*p.y + m.m[10]*p.z + m.m[11];
    return (vec3){x,y,z};
}
static inline mat4 m4_translate(vec3 t){ mat4 r = m4_identity(); r.m[3]=t.x; r.m[7]=t.y; r.m[11]=t.z; return r; }
static inline mat4 m4_rotate_x(float a){ float c=cosf(a), s=sinf(a); mat4 r=m4_identity(); r.m[5]=c; r.m[6]=-s; r.m[9]=s; r.m[10]=c; return r; }
static inline mat4 m4_rotate_y(float a){ float c=cosf(a), s=sinf(a); mat4 r=m4_identity(); r.m[0]=c; r.m[2]=s; r.m[8]=-s; r.m[10]=c; return r; }
static inline mat4 m4_rotate_z(float a){ float c=cosf(a), s=sinf(a); mat4 r=m4_identity(); r.m[0]=c; r.m[1]=-s; r.m[4]=s; r.m[5]=c; return r; }
static inline mat4 m4_scale(vec3 s){ mat4 r=m4_identity(); r.m[0]=s.x? s.x:1.0f; r.m[5]=s.y? s.y:1.0f; r.m[10]=s.z? s.z:1.0f; return r; }
static inline mat4 build_model_matrix(const transform_t *t){
    mat4 S = m4_scale(t->scale.x==0&&t->scale.y==0&&t->scale.z==0?(vec3){1,1,1}:t->scale);
    mat4 Rx = m4_rotate_x(t->rotation.x);
    mat4 Ry = m4_rotate_y(t->rotation.y);
    mat4 Rz = m4_rotate_z(t->rotation.z);
    mat4 Tr = m4_translate(t->position);
    mat4 R = m4_mul(Rz, m4_mul(Ry, Rx));
    return m4_mul(Tr, m4_mul(R, S));
}
static inline mat4 build_view_matrix_old(const camera_t *c){
    vec3 f = v3_norm(v3_sub(c->target, c->position));
    vec3 up = (c->up.x==0&&c->up.y==0&&c->up.z==0)? (vec3){0,1,0} : c->up;
    vec3 s = v3_norm(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    mat4 r = { .m = {
        s.x, u.x, -f.x, 0.0f,
        s.y, u.y, -f.y, 0.0f,
        s.z, u.z, -f.z, 0.0f,
        -v3_dot(s, c->position), -v3_dot(u, c->position), v3_dot(f, c->position), 1.0f
    }};
    return r;
}

static inline mat4 build_view_matrix(const camera_t *c){
    vec3 f  = v3_norm(v3_sub(c->target, c->position));
    vec3 up = (c->up.x==0&&c->up.y==0&&c->up.z==0)? (vec3){0,1,0} : c->up;
    vec3 s  = v3_norm(v3_cross(f, up));
    vec3 u  = v3_cross(s, f);

    // row-major, last column is translation
    mat4 r = { .m = {
        s.x,  u.x,  -f.x, -v3_dot(s, c->position),
        s.y,  u.y,  -f.y, -v3_dot(u, c->position),
        s.z,  u.z,  -f.z,  v3_dot(f, c->position),
        0.0f, 0.0f,  0.0f, 1.0f
    }};
    return r;
}

static inline float tri_orientation_clip_xyw(vec4 v0, vec4 v1, vec4 v2)
{
    float t0 = v1.y * v2.w - v2.y * v1.w;
    float t1 = v2.y * v0.w - v0.y * v2.w;
    float t2 = v0.y * v1.w - v1.y * v0.w;
    return v0.x * t0 + v1.x * t1 + v2.x * t2; // >0 = CCW, <0 = CW
}

static inline bool is_backface_ccw_clip(vec4 c0, vec4 c1, vec4 c2)
{
    const float s = tri_orientation_clip_xyw(c0, c1, c2);
    return s <= 1e-12f;
}

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
 * This must be called before any drawing operations and paired with api_frame_end().
 * Waits up to 10ms for a buffer to become available.
 * 
 * Sets tls_scene->frame_ready to false and updates tls_scene->image pointer.
 */
static void api_frame_begin() {
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
 * This must be called after api_frame_begin() and all drawing operations are complete.
 * 
 * Sets tls_scene->frame_ready to true and commits the buffer to the mapper.
 */
static void api_frame_end() {
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
static inline void fill_span_rgb(hub75_display_t *scene, int y, int x0, int x1, RGB c) {
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
void draw_polygon_fill(hub75_display_t *scene, Polygonf_t *poly, RGB color)
{
    if (!scene || !scene->image || !poly || poly->num_points < 3) {
        debug("draw_polygon: bad args\n");
        return;
    }

    /* Debug: verify coordinates look normalized (0..1). If they don't, log once per call. */
    /*
    {
        float minx =  1e9f, maxx = -1e9f, miny =  1e9f, maxy = -1e9f;
        for (size_t i = 0; i < poly->num_points; ++i) {
            if (poly->points[i].x < minx) minx = poly->points[i].x;
            if (poly->points[i].x > maxx) maxx = poly->points[i].x;
            if (poly->points[i].y < miny) miny = poly->points[i].y;
            if (poly->points[i].y > maxy) maxy = poly->points[i].y;
        }
        if (minx < -0.01f || maxx > 1.01f || miny < -0.01f || maxy > 1.01f) {
            printf("[draw_polygon_fill] Warning: coordinates not normalized. min(%.3f,%.3f) max(%.3f,%.3f)\n",
                   (double)minx, (double)miny, (double)maxx, (double)maxy);
        }
    }
    */

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
    hub75_display_request_shutdown(tls_scene);
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

/**
 * @brief Create a new camera with default parameters; use current tls_scene if available
 */
camera_t *api_new_camera(void) {
    camera_t *cam = calloc(1, sizeof(camera_t));
    // Use actual image aspect ratio if a scene is set
    if (tls_scene) {
        cam->aspect = (float)tls_scene->width / (float)tls_scene->height;
    } else {
        cam->aspect = 1.0f;
    }
    // 45 degree FOV
    cam->fov_y = (float)M_PI / 4.0f;
    // move camera back and up to see the default unit sized scene 
    cam->position = (vec3){0, -2.0f, 5.0f};
    // look at origin
    cam->target = (vec3){0, 0, 0};

    // default the camera up vector to +Y
    cam->up.y = 1.0f;

    // clipping regions
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

object_t *api_new_cube(const object_draw_mode_t draw_mode, const bool cull_backface) {
    return object_cube(draw_mode, cull_backface);
}

object_t *api_new_tetrahedron(const object_draw_mode_t draw_mode, const bool cull_backface) {
    return object_tetrahedron(draw_mode, cull_backface);
}

object_t *api_new_octahedron(const object_draw_mode_t draw_mode, const bool cull_backface) {
    return object_octahedron(draw_mode, cull_backface);
}

object_t *api_new_pyramid(void) {
    return object_pyramid();
}

object_t *api_new_cylinder(const uint16_t segments, const object_draw_mode_t mode, const bool cull_backface) {
    return object_cylinder(segments, mode, cull_backface);
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

void debug_object(object_t *obj) {
    if (!obj) {
        printf("printf_object: null object\n");
        return;
    }

    printf("Object printfg Info:\n");
    printf("Vertices (%u):\n", (unsigned)obj->verticies->length);
    for (size_t i = 0; i < obj->verticies->length; i++) {
        vec3 v = obj->verticies->list[i];
        printf("  Vertex %zu: (%.3f, %.3f, %.3f)\n", i, (double)v.x, (double)v.y, (double)v.z);
    }

    printf("Edges (%u):\n", (unsigned)obj->edges->length);
    for (size_t i = 0; i < obj->edges->length; i++) {
        vec2 e = obj->edges->list[i];
        printf("  Edge %zu: Vertex %u to Vertex %u\n", i, (uint16_t)e.x, (uint16_t)e.y);
    }

    printf("Faces (%u):\n", (unsigned)obj->faces->length);
    for (size_t i = 0; i < obj->faces->length; i++) {
        vec3 f = obj->faces->list[i];
        printf("  Face %zu: Vertex %u, Vertex %u, Vertex %u\n", i, (uint16_t)f.x, (uint16_t)f.y, (uint16_t)f.z);
    }
}

void api_geo_render_wire(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene3d_lighting_t *lighting) {
    (void)lighting; /* allow NULL; lighting not used in wireframe */

    // Debug camera and transform (gated behind enhanced_debug)
    if (tls_scene && tls_scene->enhanced_debug) {
     printf("Camera: pos(%.3f, %.3f, %.3f), target(%.3f, %.3f, %.3f), fov=%.3f, aspect=%.3f, near=%.3f, far=%.3f\n", 
         (double)cam->position.x, (double)cam->position.y, (double)cam->position.z, 
         (double)cam->target.x, (double)cam->target.y, (double)cam->target.z,
         (double)cam->fov_y, (double)cam->aspect, (double)cam->z_near, (double)cam->z_far);
     printf("Transform: pos(%.3f, %.3f, %.3f), rot(%.3f, %.3f, %.3f), scale(%.3f, %.3f, %.3f)\n",
         (double)obj_xform->position.x, (double)obj_xform->position.y, (double)obj_xform->position.z,
         (double)obj_xform->rotation.x, (double)obj_xform->rotation.y, (double)obj_xform->rotation.z,
         (double)obj_xform->scale.x, (double)obj_xform->scale.y, (double)obj_xform->scale.z);

        debug_object(obj);
    }

    mat4 mvp = camera_project(cam, obj_xform);

    transform_mesh_to_ndc(obj->verticies->list, obj->verticies->length, mvp, obj->rendered_vertices);


    const uint16_t w = (uint16_t)(tls_scene->width * 0.5f);
    const uint16_t h = (uint16_t)(tls_scene->height * 0.5f);

    /* If enabled, compute front-facing faces in VIEW space using geometry (more robust).
       We'll still render as edges, filtering edges that do not belong to any front-facing face. */
    bool do_cull_edges = false;
    bool *front_face = NULL;
    if (obj->cull_backface && obj->faces && obj->verticies && obj->faces->length > 0) {
        /* Build view * model (row-major) to transform object vertices into view space */
        mat4 V = build_view_matrix(cam);
        mat4 M = build_model_matrix(obj_xform);
        mat4 MV = m4_mul(V, M);

        size_t nf = obj->faces->length;
        front_face = (bool*)calloc(nf, sizeof(bool));
        if (front_face) {
            for (size_t i = 0; i < nf; ++i) {
                vec3 f = obj->faces->list[i];
                vec3 p0 = obj->verticies->list[(size_t)f.x];
                vec3 p1 = obj->verticies->list[(size_t)f.y];
                vec3 p2 = obj->verticies->list[(size_t)f.z];
                /* transform to view space */
                vec3 v0 = m4_mul_point(MV, p0);
                vec3 v1 = m4_mul_point(MV, p1);
                vec3 v2 = m4_mul_point(MV, p2);
                /* compute face normal in view space */
                vec3 e1 = v3_sub(v1, v0);
                vec3 e2 = v3_sub(v2, v0);
                vec3 n  = v3_cross(e1, e2);
                /* In our view space, camera looks down -Z; front faces have n.z > 0 */
                front_face[i] = (n.z > 1e-6f);
            }
            do_cull_edges = true;
        }
    }

    for (size_t i = 0; i < obj->edges->length; i++) {
        //debug("Vertex %zu: NDC (%.3f, %.3f, %.3f)\n", i, obj->rendered_vertices[i].x, obj->rendered_vertices[i].y, obj->rendered_vertices[i].z);
        vec2 edge = obj->edges->list[i];
        vec3 v1 = obj->rendered_vertices[(size_t)edge.x];
        vec3 v2 = obj->rendered_vertices[(size_t)edge.y];

        /* If culling, only draw this edge if it belongs to any front-facing face */
        if (do_cull_edges) {
            uint16_t a = (uint16_t)edge.x;
            uint16_t b = (uint16_t)edge.y;
            bool draw_edge = false;
            for (size_t fi = 0; fi < obj->faces->length; ++fi) {
                if (!front_face[fi]) continue;
                vec3 f = obj->faces->list[fi];
                uint16_t i0 = (uint16_t)f.x, i1 = (uint16_t)f.y, i2 = (uint16_t)f.z;
                /* unordered edge match against triangle edges */
                if ((a==i0 && b==i1) || (a==i1 && b==i0) ||
                    (a==i1 && b==i2) || (a==i2 && b==i1) ||
                    (a==i2 && b==i0) || (a==i0 && b==i2)) {
                    draw_edge = true;
                    break;
                }
            }
            if (!draw_edge) continue;
        }

        // Skip edges where vertices are outside the view frustum (simple z clipping)
        if (v1.z < -1.0f || v1.z > 1.0f || v2.z < -1.0f || v2.z > 1.0f) {
            if (tls_scene && tls_scene->enhanced_debug) {
                printf("Skipping edge %zu due to z clipping: V1.z=%.3f, V2.z=%.3f\n", i, (double)v1.z, (double)v2.z);
            }
            continue;
        }

        // print out debugging info for each point
        if (tls_scene && tls_scene->enhanced_debug) {
            printf("Edge %zu: V1 NDC (%.3f, %.3f, %.3f), V2 NDC (%.3f, %.3f, %.3f)\n", i,
                   (double)v1.x, (double)v1.y, (double)v1.z, (double)v2.x, (double)v2.y, (double)v2.z);
        }
        
        // Convert NDC to screen coordinates with clamping
        int x1 = (int)(w * (v1.x + 1.0f));
        int y1 = (int)(h * (v1.y + 1.0f));
        int x2 = (int)(w * (v2.x + 1.0f));
        int y2 = (int)(h * (v2.y + 1.0f));
        
        // Clamp to screen bounds
        x1 = (x1 < 0) ? 0 : (x1 >= tls_scene->width) ? tls_scene->width - 1 : x1;
        y1 = (y1 < 0) ? 0 : (y1 >= tls_scene->height) ? tls_scene->height - 1 : y1;
        x2 = (x2 < 0) ? 0 : (x2 >= tls_scene->width) ? tls_scene->width - 1 : x2;
        y2 = (y2 < 0) ? 0 : (y2 >= tls_scene->height) ? tls_scene->height - 1 : y2;
        
        if (tls_scene && tls_scene->enhanced_debug) {
            printf("line: (%d, %d) to (%d, %d)\n", x1, y1, x2, y2);
        }
        
        hub_line(tls_scene, (uint16_t)x1, (uint16_t)y1, (uint16_t)x2, (uint16_t)y2, obj->edge_colors->list[i]);
    }

    if (front_face) free(front_face);
}

typedef struct {
    float depth;      /* average NDC z */
    Polygonf_t poly;  /* 3 points normalized to [0,1] */
    RGB color;        /* shaded color */
} _TriFill;

/* qsort comparator: sort by depth descending (far to near) */
static int _cmp_trifill_desc(const void *a, const void *b) {
    const _TriFill *ta = (const _TriFill*)a;
    const _TriFill *tb = (const _TriFill*)b;
    if (ta->depth < tb->depth) return 1;   /* b before a */
    if (ta->depth > tb->depth) return -1;  /* a before b */
    return 0;
}

void api_geo_render_filled(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene3d_lighting_t *lighting) {
    if (!obj || !obj->faces || !obj->verticies) return;
    mat4 mvp = camera_project(cam, obj_xform);

    /* Transform all vertices to NDC */
    transform_mesh_to_ndc(obj->verticies->list, obj->verticies->length, mvp, obj->rendered_vertices);

    /* Precompute world normal matrix only if lighting is provided */
    float N3[9];
    if (lighting) {
        mat4 M_model = model_matrix(obj_xform);
        normal_matrix_from_model(M_model, N3);
    }

    size_t nf = obj->faces->length;
    _TriFill *tri = (_TriFill*)malloc(sizeof(_TriFill) * nf);
    if (!tri) return;
    size_t tcount = 0;

    for (size_t i = 0; i < nf; i++) {
        vec3 face = obj->faces->list[i];
        vec3 v1 = obj->rendered_vertices[(size_t)face.x];
        vec3 v2 = obj->rendered_vertices[(size_t)face.y];
        vec3 v3 = obj->rendered_vertices[(size_t)face.z];

        /* Backface culling in NDC (CCW = front) */
        if (obj->cull_backface) {
            float ax = v2.x - v1.x, ay = v2.y - v1.y;
            float bx = v3.x - v1.x, by = v3.y - v1.y;
            float area = ax * by - ay * bx;
            if (area < 0.0f) continue;
        }

        float nx1 = 0.5f * (v1.x + 1.0f);
        float ny1 = 0.5f * (v1.y + 1.0f);
        float nx2 = 0.5f * (v2.x + 1.0f);
        float ny2 = 0.5f * (v2.y + 1.0f);
        float nx3 = 0.5f * (v3.x + 1.0f);
        float ny3 = 0.5f * (v3.y + 1.0f);

        /* Shade */
        RGB base_rgb = (i < obj->edge_colors->length) ? obj->edge_colors->list[i] : (RGB){255,255,255};
        RGB shaded_rgb = base_rgb;
        if (lighting) {
            vec3 n_obj = (obj->normals && i < obj->normals->length) ? obj->normals->list[i] : (vec3){0,0,1};
            vec3 n_world = mat3_mul_vec3(N3, n_obj);
            float n_len = sqrtf(n_world.x*n_world.x + n_world.y*n_world.y + n_world.z*n_world.z);
            if (n_len > 1e-6f) { n_world.x/=n_len; n_world.y/=n_len; n_world.z/=n_len; }

            float br = base_rgb.r/255.0f, bg = base_rgb.g/255.0f, bb = base_rgb.b/255.0f;
            float lr = lighting->ambient.r, lg = lighting->ambient.g, lb = lighting->ambient.b;
            for (uint16_t li = 0; li < lighting->num_lights; ++li) {
                const light_t *L = &lighting->lights[li];
                if (L->intensity <= 0.0f) continue;
                if (L->type == LIGHT_DIRECTIONAL) {
                    float Lx = -L->direction.x, Ly = -L->direction.y, Lz = -L->direction.z;
                    float Llen = sqrtf(Lx*Lx + Ly*Ly + Lz*Lz);
                    if (Llen > 1e-6f) { Lx/=Llen; Ly/=Llen; Lz/=Llen; }
                    float ndotl = n_world.x*Lx + n_world.y*Ly + n_world.z*Lz;
                    if (ndotl > 0.0f) {
                        lr += L->color.r * L->intensity * ndotl;
                        lg += L->color.g * L->intensity * ndotl;
                        lb += L->color.b * L->intensity * ndotl;
                    }
                }
            }
            float cr = fminf(fmaxf(br * lr, 0.0f), 1.0f);
            float cg = fminf(fmaxf(bg * lg, 0.0f), 1.0f);
            float cb = fminf(fmaxf(bb * lb, 0.0f), 1.0f);
            shaded_rgb.r = (uint8_t)(cr * 255.0f);
            shaded_rgb.g = (uint8_t)(cg * 255.0f);
            shaded_rgb.b = (uint8_t)(cb * 255.0f);
        }

        _TriFill t;
        t.depth = (v1.z + v2.z + v3.z) / 3.0f; /* NDC z: -1 near, +1 far */
        t.poly.num_points = 3;
        t.poly.points[0] = (Pointf_t){nx1, ny1};
        t.poly.points[1] = (Pointf_t){nx2, ny2};
        t.poly.points[2] = (Pointf_t){nx3, ny3};
        t.color = shaded_rgb;
        tri[tcount++] = t;
    }

    /* Painter's algorithm: draw far to near => sort by depth descending (far ~ +1) */
    if (tcount > 1) {
        qsort(tri, tcount, sizeof(_TriFill), _cmp_trifill_desc);
    }

    for (size_t i = 0; i < tcount; ++i) {
        if (tls_scene && tls_scene->enhanced_debug) {
            printf("Filling triangle p1 (%.3f, %.3f), p2 (%.3f, %.3f), p3 (%.3f, %.3f) depth=%.3f\n",
                (double)tri[i].poly.points[0].x, (double)tri[i].poly.points[0].y,
                (double)tri[i].poly.points[1].x, (double)tri[i].poly.points[1].y,
                (double)tri[i].poly.points[2].x, (double)tri[i].poly.points[2].y,
                (double)tri[i].depth);
        }
        draw_polygon_fill(tls_scene, &tri[i].poly, tri[i].color);
    }

    free(tri);
}

/* --- Clip-space XYW orientation helpers for robust backface culling --- */
static inline vec4 mat4_mul_point_clip_xyw(const mat4 m, const vec3 p) {
    /* Column-major multiply with column vector: clip = M * [p.x, p.y, p.z, 1]^T */
    float x = m.m[0]*p.x + m.m[4]*p.y + m.m[8]*p.z + m.m[12];
    float y = m.m[1]*p.x + m.m[5]*p.y + m.m[9]*p.z + m.m[13];
    float z = m.m[2]*p.x + m.m[6]*p.y + m.m[10]*p.z + m.m[14];
    float w = m.m[3]*p.x + m.m[7]*p.y + m.m[11]*p.z + m.m[15];
    (void)z; /* z not needed for orientation, but kept for completeness */
    return (vec4){x, y, z, w};
}

/* Wireframe renderer that culls backfaces using clip-space XYW orientation */
void api_geo_render_wire_clip_cull(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene3d_lighting_t *lighting) {
    (void)lighting; /* not used for wireframe */
    if (!obj || !obj->verticies || !obj->edges) return;

    /* Build MVP and compute both: clip-space XYW (for culling) and NDC (for raster) */
    mat4 mvp = camera_project(cam, obj_xform);

    /* Compute clip-space for all vertices (keep x,y,w) */
    size_t nv = obj->verticies->length;
    vec4 *clip_xyw = (vec4*)malloc(sizeof(vec4) * nv);
    if (!clip_xyw) return;
    for (size_t i = 0; i < nv; ++i) {
        vec3 p = obj->verticies->list[i];
        clip_xyw[i] = mat4_mul_point_clip_xyw(mvp, p);
    }

    /* Also compute NDC positions used for drawing */
    transform_mesh_to_ndc(obj->verticies->list, obj->verticies->length, mvp, obj->rendered_vertices);

    const uint16_t w = (uint16_t)(tls_scene->width * 0.5f);
    const uint16_t h = (uint16_t)(tls_scene->height * 0.5f);

    bool do_cull_edges = obj->cull_backface && obj->faces && obj->faces->length > 0;
    bool *front_face = NULL;
    if (do_cull_edges) {
        size_t nf = obj->faces->length;
        front_face = (bool*)calloc(nf, sizeof(bool));
        if (front_face) {
            /* Allow compile-time flip of front-face convention if needed */
            #ifndef FRONT_FACE_CCW
            #define FRONT_FACE_CCW 1
            #endif
            size_t front_count = 0;
            for (size_t i = 0; i < nf; ++i) {
                vec3 f = obj->faces->list[i];
                vec4 a = clip_xyw[(size_t)f.x];
                vec4 b = clip_xyw[(size_t)f.y];
                vec4 c = clip_xyw[(size_t)f.z];
                float orient = tri_orientation_clip_xyw(a, b, c);
                /* CCW => front by default. Flip with FRONT_FACE_CCW==0 if needed. */
                bool is_front = FRONT_FACE_CCW ? (orient > 0.0f) : (orient < 0.0f);
                front_face[i] = is_front;
                if (is_front) front_count++;
            }
            /* Safety: if nothing classified as front, disable culling to avoid blank output */
            if (front_count == 0) {
                if (tls_scene && tls_scene->enhanced_debug) {
                    printf("Clip-cull: 0 front faces detected; disabling cull for this object.\n");
                }
                do_cull_edges = false;
                free(front_face); front_face = NULL;
            }
        } else {
            do_cull_edges = false;
        }
    }

    for (size_t i = 0; i < obj->edges->length; i++) {
        vec2 edge = obj->edges->list[i];
        vec3 v1 = obj->rendered_vertices[(size_t)edge.x];
        vec3 v2 = obj->rendered_vertices[(size_t)edge.y];

        /* If culling, only draw this edge if it belongs to any front-facing face */
        if (do_cull_edges) {
            uint16_t a = (uint16_t)edge.x;
            uint16_t b = (uint16_t)edge.y;
            bool draw_edge = false;
            for (size_t fi = 0; fi < obj->faces->length; ++fi) {
                if (!front_face[fi]) continue;
                vec3 f = obj->faces->list[fi];
                uint16_t i0 = (uint16_t)f.x, i1 = (uint16_t)f.y, i2 = (uint16_t)f.z;
                /* unordered edge match against triangle edges */
                if ((a==i0 && b==i1) || (a==i1 && b==i0) ||
                    (a==i1 && b==i2) || (a==i2 && b==i1) ||
                    (a==i2 && b==i0) || (a==i0 && b==i2)) {
                    draw_edge = true;
                    break;
                }
            }
            if (!draw_edge) continue;
        }

        /* Simple z clipping in NDC space */
        if (v1.z < -1.0f || v1.z > 1.0f || v2.z < -1.0f || v2.z > 1.0f) {
            if (tls_scene && tls_scene->enhanced_debug) {
                printf("Skipping edge %zu due to z clipping: V1.z=%.3f, V2.z=%.3f\n", i, (double)v1.z, (double)v2.z);
            }
            continue;
        }

        int x1 = (int)(w * (v1.x + 1.0f));
        int y1 = (int)(h * (v1.y + 1.0f));
        int x2 = (int)(w * (v2.x + 1.0f));
        int y2 = (int)(h * (v2.y + 1.0f));

        /* Clamp to screen bounds */
        x1 = (x1 < 0) ? 0 : (x1 >= tls_scene->width) ? tls_scene->width - 1 : x1;
        y1 = (y1 < 0) ? 0 : (y1 >= tls_scene->height) ? tls_scene->height - 1 : y1;
        x2 = (x2 < 0) ? 0 : (x2 >= tls_scene->width) ? tls_scene->width - 1 : x2;
        y2 = (y2 < 0) ? 0 : (y2 >= tls_scene->height) ? tls_scene->height - 1 : y2;

        hub_line_aa(tls_scene, (uint16_t)x1, (uint16_t)y1, (uint16_t)x2, (uint16_t)y2, obj->edge_colors->list[i]);
    }

    if (front_face) free(front_face);
    free(clip_xyw);
}

/* Render a list of object instances with per-object draw mode */
static void api_render_scene3d(const camera_t *cam, const scene3d_t *os, const scene3d_lighting_t *lighting) {
    if (!os || !os->instances || os->count == 0) return;
    /* Prefer explicitly provided lighting; else fall back to scene-owned lighting */
    const scene3d_lighting_t *L = lighting ? lighting : &os->lighting;
    for (uint16_t i = 0; i < os->count; ++i) {
        const object_instance_t *inst = &os->instances[i];
        object_t *obj = inst->object;
        const transform_t *xf = inst->xform;
        if (!obj || !xf) continue;
        switch (obj->draw_mode) {
            case DRAW_FILLED:
                api_geo_render_filled(cam, obj, xf, L);
                break;
            case DRAW_WIRE:
            default:
                /* Use clip-space culling variant for consistency with API table */
                api_geo_render_wire_clip_cull(cam, obj, xf, L);
                break;
        }
    }
}

/* Public wrappers for scene rendering (FFI-friendly) */
void api_render_geo(const camera_t *cam, const scene3d_t *os, const scene3d_lighting_t *lighting) {
    api_render_scene3d(cam, os, lighting);
}

/* -------- Lighting helpers (FFI-friendly) -------- */
scene3d_lighting_t *api_lighting_new(uint16_t num_lights, RGBF ambient) {
    scene3d_lighting_t *L = (scene3d_lighting_t*)calloc(1, sizeof(scene3d_lighting_t));
    if (!L) return NULL;
    L->num_lights = num_lights;
    if (num_lights > 0) {
        L->lights = (light_t*)calloc(num_lights, sizeof(light_t));
        if (!L->lights) { free(L); return NULL; }
    }
    L->ambient = ambient;
    return L;
}

void api_lighting_free(scene3d_lighting_t *l) {
    if (!l) return;
    if (l->lights) free(l->lights);
    free(l);
}

void api_lighting_set_directional(scene3d_lighting_t *l, uint16_t index,
                                  light_vec3 direction, RGBF color,
                                  float intensity, bool casts_shadows) {
    if (!l || !l->lights || index >= l->num_lights) return;
    light_t *L = &l->lights[index];
    L->type = LIGHT_DIRECTIONAL;
    L->direction = direction;
    L->color = color;
    L->intensity = intensity;
    L->casts_shadows = casts_shadows;
}

/* -------- Object scene helpers (FFI-friendly) -------- */
/* ---- object_scene OO-style lighting helpers ---- */
static void os_set_ambient(scene3d_t *os, RGBF ambient) {
    if (!os) return;
    os->lighting.ambient = ambient;
}

static uint16_t os_add_directional(scene3d_t *os,
                                   light_vec3 direction,
                                   RGBF color,
                                   float intensity,
                                   bool casts_shadows) {
    if (!os) return UINT16_MAX;
    uint16_t n = os->lighting.num_lights;
    light_t *newlights = (light_t*)realloc(os->lighting.lights, (size_t)(n + 1) * sizeof(light_t));
    if (!newlights) return UINT16_MAX;
    os->lighting.lights = newlights;
    light_t *L = &os->lighting.lights[n];
    L->type = LIGHT_DIRECTIONAL;
    L->direction = direction;
    L->color = color;
    L->intensity = intensity;
    L->casts_shadows = casts_shadows;
    L->position = (light_vec3){0,0,0};
    L->range = 0.0f; L->inner_cos = 1.0f; L->outer_cos = 1.0f;
    os->lighting.num_lights = (uint16_t)(n + 1);
    return n;
}

/* Set a directional light using a camera-like pose (position + look_at) */
static void os_set_directional_pose(scene3d_t *os, uint16_t id,
                                    light_vec3 position, light_vec3 look_at) {
    if (!os || id >= os->lighting.num_lights || !os->lighting.lights) return;
    light_t *L = &os->lighting.lights[id];
    /* Store position for completeness */
    L->position = position;
    /* Compute direction = normalize(look_at - position) */
    float dx = look_at.x - position.x;
    float dy = look_at.y - position.y;
    float dz = look_at.z - position.z;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len > 1e-6f) { dx/=len; dy/=len; dz/=len; }
    else { dx = 0.0f; dy = -1.0f; dz = 0.0f; }
    L->direction = (light_vec3){ dx, dy, dz };
}

static uint16_t os_add_object(scene3d_t *os, object_t *obj, transform_t *xform) {
    if (!os || !obj || !xform) return UINT16_MAX;
    /* Ensure capacity */
    if (os->count >= os->capacity) {
        uint16_t new_cap = (os->capacity == 0) ? 1u : (uint16_t)(os->capacity * 2u);
        object_instance_t *new_arr = (object_instance_t*)realloc(os->instances, (size_t)new_cap * sizeof(object_instance_t));
        if (!new_arr) return UINT16_MAX;
        /* Zero-init new tail */
        if (new_cap > os->capacity) {
            size_t old = os->capacity;
            memset(new_arr + old, 0, (size_t)(new_cap - old) * sizeof(object_instance_t));
        }
        os->instances = new_arr;
        os->capacity = new_cap;
    }
    uint16_t id = os->count;
    os->instances[id].object = obj;
    os->instances[id].xform = xform;
    os->count = (uint16_t)(id + 1);
    return id;
}

static object_t *os_get_object(scene3d_t *os, uint16_t id) {
    if (!os || id >= os->count) return NULL;
    return os->instances[id].object;
}

static light_t *os_get_directional(scene3d_t *os, uint16_t id) {
    if (!os || id >= os->lighting.num_lights) return NULL;
    return &os->lighting.lights[id];
}

static transform_t *os_get_transform(scene3d_t *os, uint16_t id) {
    if (!os || id >= os->count) return NULL;
    return os->instances[id].xform;
}

scene3d_t *api_object_scene_new(uint16_t object_count) {
    scene3d_t *os = (scene3d_t*)calloc(1, sizeof(scene3d_t));
    if (!os) return NULL;
    os->capacity = object_count;
    os->count = object_count;
    if (os->capacity > 0) {
        os->instances = (object_instance_t*)calloc(os->capacity, sizeof(object_instance_t));
        if (!os->instances) { free(os); return NULL; }
    }
    // Initialize embedded lighting to defaults (black ambient, no lights)
    os->lighting.ambient = (RGBF){0.0f, 0.0f, 0.0f};
    os->lighting.num_lights = 0;
    os->lighting.lights = NULL;
    // Wire OO-style helpers 
    os->set_ambient = os_set_ambient;
    os->add_directional = os_add_directional;
    os->set_directional_pose = os_set_directional_pose;
    os->get_directional = os_get_directional;
    os->add_object = os_add_object;
    os->get_object = os_get_object;
    os->get_transform = os_get_transform;
    return os;
}

void api_object_scene_set(scene3d_t *os, uint16_t index, object_t *obj, transform_t *xform) {
    if (!os || !os->instances || index >= os->count) return;
    os->instances[index].object = obj;
    os->instances[index].xform = xform;
}

void api_object_scene_free(scene3d_t *os) {
    if (!os) return;
    if (os->instances) free(os->instances);
    if (os->lighting.lights) free(os->lighting.lights);
    free(os);
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
    .frame_begin = api_frame_begin,

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
    .render_wire = api_geo_render_wire_clip_cull,
    .render_filled = api_geo_render_filled,
    .render_scene3d = api_render_scene3d,
    .scene3d_new = api_object_scene_new,

    .frame_end = api_frame_end,
    .shutdown = api_shutdown,

    /* convenience scene wrappers */
    .scene3d_set_current = api_scene3d_set_current,
    .scene3d_clear_current = api_scene3d_clear_current,
    .scene3d_set_ambient = api_scene3d_set_ambient,
    .scene3d_add_directional = api_scene3d_add_directional,
    .scene3d_set_directional_pose = api_scene3d_set_directional_pose,
    .scene3d_get_directional = api_scene3d_get_directional,
    .scene3d_add_object = api_scene3d_add_object,
    .scene3d_get_object = api_scene3d_get_object,
    .scene3d_get_transform = api_scene3d_get_transform,
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
hub75gpu_t hub75_api(hub75_display_t *scene) {
    tls_scene = scene;
    return api_table;
}

/* -------- Convenience scene wrappers (thread-local current scene3d) -------- */
void api_scene3d_set_current(scene3d_t *os) { tls_current_os = os; }
void api_scene3d_clear_current(void) { tls_current_os = NULL; }
void api_scene3d_set_ambient(RGBF ambient) {
    if (tls_current_os && tls_current_os->set_ambient) tls_current_os->set_ambient(tls_current_os, ambient);
}
uint16_t api_scene3d_add_directional(light_vec3 direction, RGBF color, float intensity, bool casts_shadows) {
    if (!tls_current_os || !tls_current_os->add_directional) return UINT16_MAX;
    return tls_current_os->add_directional(tls_current_os, direction, color, intensity, casts_shadows);
}
uint16_t api_scene3d_add_object(object_t *obj, transform_t *xform) {
    if (!tls_current_os || !tls_current_os->add_object) return UINT16_MAX;
    return tls_current_os->add_object(tls_current_os, obj, xform);
}
object_t* api_scene3d_get_object(uint16_t id) {
    if (!tls_current_os || !tls_current_os->get_object) return NULL;
    return tls_current_os->get_object(tls_current_os, id);
}
light_t* api_scene3d_get_directional(uint16_t id) {
    if (!tls_current_os || !tls_current_os->get_directional) return NULL;
    return tls_current_os->get_directional(tls_current_os, id);
}

transform_t* api_scene3d_get_transform(uint16_t id) {
    if (!tls_current_os || !tls_current_os->get_transform) return NULL;
    return tls_current_os->get_transform(tls_current_os, id);
}

void api_scene3d_set_directional_pose(uint16_t id, light_vec3 position, light_vec3 look_at) {
    if (tls_current_os && tls_current_os->set_directional_pose) {
        tls_current_os->set_directional_pose(tls_current_os, id, position, look_at);
    }
}

