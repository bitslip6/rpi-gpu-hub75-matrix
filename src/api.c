#include <threads.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "hub75gpu.h"
#include "pixels.h"

#include "debug.h"


/* per-thread current scene */
#if defined(__STDC_NO_THREADS__)
#  error "need thread local storage support (_Thread_local)"
#endif

static _Thread_local scene_info *tls_scene = NULL;

static void api_clear() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    memset(tls_scene->image, 0, (size_t)(tls_scene->width * tls_scene->height * tls_scene->stride));
}

static void api_pixel(uint16_t x, uint16_t y, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_pixel(tls_scene, x, y, color);
}

static void api_line(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_line(tls_scene, x1, y1, x2, y2, color);
}

static void api_line_aa(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    hub_line_aa(tls_scene, x1, y1, x2, y2, color);
}


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
static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int norm_to_px(Normal nx, int width) {
    float fx = nx * (float)(width  - 1);
    int   ix = (int)(fx + 0.5f);
    return clamp_int(ix, 0, width - 1);
}


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

void draw_polygon(scene_info *scene, Polygonf_t *poly, RGB color1, RGB color2)
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
        printf("draw_polygon: degenerate polygon\n");
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
                int xi = (int)(x0 + t * (float)dx + 0.5f);   /* round to nearest */
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
            fill_span_rgb(scene, y, x_start, x_end, (a > 0) ? color1 : color2);
        }
    }
}

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
    constexpr double eps = 1e-12;
    if (a > eps)  return POLY_CCW;
    if (a < -eps) return POLY_CW;
    return POLY_DEGENERATE;
}

void api_poly(Polygonf_t *poly, RGB color1, RGB color2) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }

    draw_polygon(tls_scene, poly, color1, color2);
}

static const hub75gpu_t api_table = {
    .clear = api_clear,
    .pixel = api_pixel,
    .line = api_line,
    .line_aa = api_line_aa,
    .poly = api_poly,
    .begin_frame = api_begin_frame,
    .end_frame = api_end_frame,
};


hub75gpu_t hub75gpu(scene_info *scene) {
    tls_scene = scene;
    return api_table;
}

