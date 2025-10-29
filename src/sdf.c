/* sdf_text_render.h or .c
 *
 * Minimal SDF text renderer that:
 *  - loads CSV metrics and per-glyph SDF PNGs
 *  - renders ASCII strings to an RGB bitmap
 *  - uses bilinear sampling and smooth coverage
 *  - draws a fallback box glyph when missing
 *
 * CSV columns:
 *   char,advance_width,bearing_x,bearing_y,width,height,path
 * char can be a single printable character or a codepoint integer like 65
 *
 * Dependencies:
 *   stb_image.h        (stbi_load)
 *   stb_image_write.h  (stbi_write_png)  only if you use the example saver
 *
 * Compile:
 *   gcc -O2 -std=c17 your_app.c -o app
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "stb_image.h"
#include "stb_image_write.h" /* only used by the example save at the bottom */

/* -----------------------------
   Types
   ----------------------------- */

typedef struct {
    float advance_width;   /* pixels at this raster size */
    float bearing_x;       /* left side bearing in pixels */
    float bearing_y;       /* top bearing relative to baseline in pixels */
    int   width;           /* PNG width */
    int   height;          /* PNG height */
    uint8_t *sdf;          /* grayscale SDF, width*height bytes */
} glyph_sdf;

typedef struct {
    glyph_sdf *table[256]; /* ASCII cache for simplicity */
} glyph_cache;

typedef struct {
    int width;
    int height;
    uint8_t *rgb;          /* 3*width*height bytes, row-major */
} image_rgb;

/* -----------------------------
   Small helpers
   ----------------------------- */

static void* xmalloc(size_t n) { void *p = malloc(n); if (!p) { fprintf(stderr, "oom\n"); exit(1); } return p; }

static void image_rgb_init(image_rgb *img, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    img->width = w; img->height = h;
    img->rgb = (uint8_t*)xmalloc((size_t)w*(size_t)h*3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t i = ((size_t)y*(size_t)w + (size_t)x) * 3u;
            img->rgb[i+0] = r;
            img->rgb[i+1] = g;
            img->rgb[i+2] = b;
        }
    }
}

static inline void blend_rgb(uint8_t *dst, uint8_t sr, uint8_t sg, uint8_t sb, float a) {
    float ia = 1.0f - a;
    dst[0] = (uint8_t)(sr * a + dst[0] * ia);
    dst[1] = (uint8_t)(sg * a + dst[1] * ia);
    dst[2] = (uint8_t)(sb * a + dst[2] * ia);
}

static inline float smoothstep(float a, float b, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - a) / (b - a)));
    return t * t * (3.0f - 2.0f * t);
}

/* -----------------------------
   CSV parsing
   ----------------------------- */

typedef struct {
    unsigned ch;    /* ASCII codepoint or byte value */
    float    adv;   /* advance_width */
    float    bx;    /* bearing_x */
    float    by;    /* bearing_y (top from baseline) */
    int      w;     /* bitmap width  */
    int      h;     /* bitmap height */
    char     path[512]; /* glyph PNG path */
} csv_row;

static char* str_trim(char *s) {
    while (*s && (*s==' ' || *s=='\t' || *s=='\r' || *s=='\n')) ++s;
    if (!*s) return s;
    char *e = s + strlen(s) - 1;
    while (e > s && (*e==' ' || *e=='\t' || *e=='\r' || *e=='\n')) { *e = '\0'; --e; }
    return s;
}

static int csv_split(char *line, char *fields[], int max_fields) {
    int n = 0;
    char *p = line;
    while (*p && n < max_fields) {
        fields[n++] = p;
        char *c = strchr(p, ',');
        if (!c) break;
        *c = '\0';
        p = c + 1;
    }
    return n;
}

static unsigned parse_uint(const char *t) {
    char *end = NULL;
    unsigned v = (unsigned)strtoul(t, &end, 10);
    return v;
}

static float parse_float(const char *t) {
    char *end = NULL;
    float v = strtof(t, &end);
    return v;
}

static csv_row* load_metrics_csv(const char *csv_path, size_t *out_count) {
    *out_count = 0;
    FILE *f = fopen(csv_path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open csv: %s\n", csv_path);
        return NULL;
    }

    size_t cap = 128, n = 0;
    csv_row *rows = (csv_row*)malloc(cap * sizeof(csv_row));
    if (!rows) { fclose(f); return NULL; }

    char line[4096];
    int is_header = 1;
    while (fgets(line, sizeof(line), f)) {
        char *ln = str_trim(line);
        if (!*ln) continue;

        if (is_header) { is_header = 0; continue; }

        char *fields[8] = {0};
        int nf = csv_split(ln, fields, 8);
        if (nf < 7) {
            fprintf(stderr, "csv parse warning, expected 7 fields got %d\n", nf);
            continue;
        }

        csv_row row = {0};
        char *f0 = str_trim(fields[0]);
        if (strlen(f0) == 1) row.ch = (unsigned)(unsigned char)f0[0];
        else                 row.ch = parse_uint(f0);

        row.adv = parse_float(str_trim(fields[1]));
        row.bx  = parse_float(str_trim(fields[2]));
        row.by  = parse_float(str_trim(fields[3]));
        row.w   = (int)strtol(str_trim(fields[4]), NULL, 10);
        row.h   = (int)strtol(str_trim(fields[5]), NULL, 10);

        char *pstr = str_trim(fields[6]);
        strncpy(row.path, pstr, sizeof(row.path)-1);
        row.path[sizeof(row.path)-1] = '\0';

        if (n == cap) {
            cap *= 2;
            csv_row *tmp = (csv_row*)realloc(rows, cap * sizeof(csv_row));
            if (!tmp) { free(rows); fclose(f); return NULL; }
            rows = tmp;
        }
        rows[n++] = row;
    }

    fclose(f);
    *out_count = n;
    return rows;
}

static const csv_row* csv_find_row(const csv_row *rows, size_t nrows, unsigned ch) {
    for (size_t i = 0; i < nrows; ++i) {
        if (rows[i].ch == ch) return &rows[i];
    }
    return NULL;
}

/* -----------------------------
   Glyph cache and loading
   ----------------------------- */

static glyph_cache* glyph_cache_create(void) {
    glyph_cache *c = (glyph_cache*)xmalloc(sizeof(*c));
    memset(c, 0, sizeof(*c));
    return c;
}

static void glyph_cache_destroy(glyph_cache *c) {
    if (!c) return;
    for (int i = 0; i < 256; ++i) {
        glyph_sdf *g = c->table[i];
        if (g) {
            stbi_image_free(g->sdf);
            free(g);
        }
    }
    free(c);
}

static glyph_sdf* load_one_glyph_from_row(const csv_row *row) {
    int w=0, h=0, n=0;
    uint8_t *img = stbi_load(row->path, &w, &h, &n, 1); /* 1 channel */
    if (!img) {
        fprintf(stderr, "failed to load png: %s\n", row->path);
        return NULL;
    }
    glyph_sdf *g = (glyph_sdf*)malloc(sizeof(*g));
    if (!g) { stbi_image_free(img); return NULL; }
    g->sdf = img;
    g->width = w;
    g->height = h;
    g->advance_width = row->adv;
    g->bearing_x = row->bx;
    g->bearing_y = row->by;
    return g;
}

static int glyph_cache_load_from_csv(glyph_cache *cache,
                                     const char *csv_path,
                                     const char *charset)
{
    size_t nrows = 0;
    csv_row *rows = load_metrics_csv(csv_path, &nrows);
    if (!rows) return -1;

    int loaded = 0;
    for (const unsigned char *p = (const unsigned char*)charset; *p; ++p) {
        unsigned ch = *p;
        if (cache->table[ch]) continue;
        const csv_row *row = csv_find_row(rows, nrows, ch);
        if (!row) {
            fprintf(stderr, "warning: metrics missing for '%c' (%u)\n", (char)ch, ch);
            continue;
        }
        glyph_sdf *g = load_one_glyph_from_row(row);
        if (!g) continue;
        cache->table[ch] = g;
        ++loaded;
    }

    free(rows);
    return loaded;
}

/* -----------------------------
   SDF sampling and coverage
   ----------------------------- */

/* bilinear sample the 8-bit SDF as a normalized float in [0,1] */
static inline float glyph_sample_bilinear(const glyph_sdf *g, float fx, float fy) {
    if (fx < 0.0f || fy < 0.0f || fx > (float)(g->width-1) || fy > (float)(g->height-1)) {
        return 0.0f;
    }
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = x0 + 1; if (x1 >= g->width)  x1 = g->width - 1;
    int y1 = y0 + 1; if (y1 >= g->height) y1 = g->height - 1;

    float tx = fx - (float)x0;
    float ty = fy - (float)y0;

    uint8_t s00 = g->sdf[(size_t)y0 * (size_t)g->width + (size_t)x0];
    uint8_t s10 = g->sdf[(size_t)y0 * (size_t)g->width + (size_t)x1];
    uint8_t s01 = g->sdf[(size_t)y1 * (size_t)g->width + (size_t)x0];
    uint8_t s11 = g->sdf[(size_t)y1 * (size_t)g->width + (size_t)x1];

    float a = (float)s00 * (1.0f - tx) + (float)s10 * tx;
    float b = (float)s01 * (1.0f - tx) + (float)s11 * tx;
    float s = a * (1.0f - ty) + b * ty;

    return s * (1.0f / 255.0f);
}

/* Map normalized SDF to coverage in [0,1].
   s = 0.5 is the contour. edge band scales with 1/spread_px. */
static inline float sdf_to_coverage(float s_norm, float spread_px) {
    float edge_soft = 0.5f * (1.0f / fmaxf(spread_px, 1.0f));
    return smoothstep(0.5f - edge_soft, 0.5f + edge_soft, s_norm);
}

/* -----------------------------
   Fallback box glyph
   ----------------------------- */

/* draw a 1 px rectangle outline with an X inside, blended at full opacity */
static void draw_fallback_box(image_rgb *img, int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    if (w <= 0 || h <= 0) return;
    int x1 = x0 + w - 1;
    int y1 = y0 + h - 1;
    for (int x = x0; x <= x1; ++x) {
        if ((unsigned)y0 < (unsigned)img->height && (unsigned)x < (unsigned)img->width) {
            uint8_t *p = &img->rgb[((size_t)y0 * img->width + x) * 3u];
            blend_rgb(p, r, g, b, 1.0f);
        }
        if ((unsigned)y1 < (unsigned)img->height && (unsigned)x < (unsigned)img->width) {
            uint8_t *p = &img->rgb[((size_t)y1 * img->width + x) * 3u];
            blend_rgb(p, r, g, b, 1.0f);
        }
    }
    for (int y = y0; y <= y1; ++y) {
        if ((unsigned)y < (unsigned)img->height && (unsigned)x0 < (unsigned)img->width) {
            uint8_t *p = &img->rgb[((size_t)y * img->width + x0) * 3u];
            blend_rgb(p, r, g, b, 1.0f);
        }
        if ((unsigned)y < (unsigned)img->height && (unsigned)x1 < (unsigned)img->width) {
            uint8_t *p = &img->rgb[((size_t)y * img->width + x1) * 3u];
            blend_rgb(p, r, g, b, 1.0f);
        }
    }
    /* diagonals */
    int dx = x1 - x0, dy = y1 - y0;
    int n = dx > dy ? dx + 1 : dy + 1;
    for (int i = 0; i < n; ++i) {
        int x = x0 + (int)lroundf((float)i * dx / (float)(n - 1));
        int y = y0 + (int)lroundf((float)i * dy / (float)(n - 1));
        if ((unsigned)y < (unsigned)img->height && (unsigned)x < (unsigned)img->width) {
            uint8_t *p = &img->rgb[((size_t)y * img->width + x) * 3u];
            blend_rgb(p, r, g, b, 1.0f);
        }
        int yb = y1 - (int)lroundf((float)i * dy / (float)(n - 1));
        if ((unsigned)yb < (unsigned)img->height && (unsigned)x < (unsigned)img->width) {
            uint8_t *p2 = &img->rgb[((size_t)yb * img->width + x) * 3u];
            blend_rgb(p2, r, g, b, 1.0f);
        }
    }
}

/* -----------------------------
   Core SDF text rendering
   ----------------------------- */

/* Render an ASCII string using SDF glyphs.
   baseline_y_ratio controls baseline placement as fraction of image height, typical 0.7..0.8. */
static void render_text_sdf(image_rgb *img,
                            const glyph_cache *cache,
                            const char *text,
                            float spread_px,
                            float baseline_y_ratio,
                            uint8_t text_r, uint8_t text_g, uint8_t text_b)
{
    int pen_x = 8;
    int baseline_y = (int)lroundf(img->height * baseline_y_ratio);

    for (const unsigned char *p = (const unsigned char*)text; *p; ++p) {
        unsigned ch = *p;

        if (ch == '\n') {
            baseline_y += (int)lroundf(img->height * 0.5f);
            pen_x = 8;
            continue;
        }

        const glyph_sdf *g = cache->table[ch];
        if (!g) {
            /* draw fallback box roughly the size of a capital glyph */
            int box_w = (int)lroundf(img->height * 0.4f);
            int box_h = (int)lroundf(img->height * 0.8f);
            int x0 = pen_x;
            int y0 = baseline_y - box_h; /* top aligned to baseline region */
            draw_fallback_box(img, x0, y0, box_w, box_h, text_r, text_g, text_b);
            pen_x += box_w;
            continue;
        }

        /* top-left of glyph bitmap in destination space */
        int dst_x0 = pen_x + (int)lroundf(g->bearing_x);
        int dst_y0 = baseline_y - (int)lroundf(g->bearing_y);

        /* iterate destination pixels covered by the glyph rectangle */
        for (int gy = 0; gy < g->height; ++gy) {
            int y = dst_y0 + gy;
            if ((unsigned)y >= (unsigned)img->height) continue;

            for (int gx = 0; gx < g->width; ++gx) {
                int x = dst_x0 + gx;
                if ((unsigned)x >= (unsigned)img->width) continue;

                /* bilinear sample at glyph pixel center */
                float s_norm = glyph_sample_bilinear(g, (float)gx + 0.5f, (float)gy + 0.5f);
                if (s_norm <= 0.0f) continue;

                float a = sdf_to_coverage(s_norm, spread_px);
                if (a <= 0.001f) continue;

                uint8_t *dst = &img->rgb[((size_t)y * img->width + x) * 3u];
                blend_rgb(dst, text_r, text_g, text_b, a);
            }
        }

        pen_x += (int)lroundf(g->advance_width);
    }
}

/* -----------------------------
   Example usage (optional)
   ----------------------------- */
/* Define SDF_TEXT_DEMO to compile a small CLI demo. */
#ifdef SDF_TEXT_DEMO
int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <csv_path> <text> <canvas_h> <spread_px> <out_png>\n", argv[0]);
        return 1;
    }
    const char *csv_path  = argv[1];
    const char *text      = argv[2];
    int canvas_h          = atoi(argv[3]);
    float spread_px       = (float)atof(argv[4]);
    const char *out_png   = argv[5];

    int canvas_w = (int)lroundf(canvas_h * 2.2f);
    image_rgb img = {0};
    image_rgb_init(&img, canvas_w, canvas_h, 255, 255, 255);

    glyph_cache *cache = glyph_cache_create();
    int n_loaded = glyph_cache_load_from_csv(cache, csv_path, text);
    if (n_loaded <= 0) {
        fprintf(stderr, "warning: no glyphs loaded from csv, proceeding with fallbacks only\n");
    }

    render_text_sdf(&img, cache, text, spread_px, 0.72f, 10, 10, 10);

    if (!stbi_write_png(out_png, img.width, img.height, 3, img.rgb, img.width * 3)) {
        fprintf(stderr, "failed to write %s\n", out_png);
    } else {
        printf("wrote %s\n", out_png);
    }

    glyph_cache_destroy(cache);
    free(img.rgb);
    return 0;
}
#endif

