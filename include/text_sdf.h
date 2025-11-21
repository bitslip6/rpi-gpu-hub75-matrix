#ifndef TEXT_SDF_H
#define TEXT_SDF_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <errno.h>

#include "hub75gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 Thread-safety and ownership
 ---------------------------
 - Loading:
     * sdf_metrics_csv_load() and sdf_font_load() perform file I/O and allocations; call these during init.
     * They are not thread-safe if invoked concurrently on the same paths unless externally synchronized.
 - Font lifetime and sharing:
     * After sdf_font_load() returns, treat sdf_font_t as immutable (read-only). Do not mutate glyph buffers or metrics.
     * Multiple threads may concurrently read from the same sdf_font_t (rendering in parallel is OK) as long as the font
         is not being freed. Ensure sdf_font_free() is not called while in use.
     * The font must outlive all sdf_text_t instances that reference it.
 - Text objects:
     * sdf_text_t is not thread-safe for concurrent mutation. Restrict mutations (set_* calls) and rendering to a single
         thread per instance, or guard with a mutex if you need cross-thread updates.
     * It is safe to render different sdf_text_t instances in parallel as long as they reference read-only, live fonts.
 - Rendering:
     * sdf_text_render() is not re-entrant per sdf_text_t instance without external synchronization. It does not hold any
         global state and can be used from multiple threads on distinct instances.
 - Memory ownership:
     * All allocations use malloc/free. The caller owns freeing of sdf_text_t via sdf_text_destroy(), and sdf_font_t via
         sdf_font_free(). Do not free pixel buffers or internal pointers directly.
*/

/*
 * SDF glyph representation
 * - 8-bit grayscale SDF buffer, row-major, stride in bytes
 * - width/height in pixels
 * - bearings and advance in pixels at the glyph's generated size
 *   bearing_y is the TOP bearing (positive up) relative to the baseline
 * - pixels use the convention: 128 == contour, >128 inside, <128 outside
 */
typedef struct sdf_glyph_t {
    uint8_t width;              /* glyph bitmap width in pixels */
    uint8_t height;             /* glyph bitmap height in pixels */

    float bearing_x;        /* left-side bearing (pixels) */
    float bearing_y;        /* top bearing relative to baseline (pixels, +up) */
    float advance_width;    /* advance width in pixels */

    uint8_t *pixels;        /* pointer to 8-bit SDF buffer, size >= height*stride */
    /* Ink bounds (rows) detected with SDF threshold >= 128 at pixel centers */
    uint8_t ink_top;        /* first row index with any ink (0..height-1), 0 if none */
    uint8_t ink_bottom;     /* last row index with any ink (0..height-1), height-1 if none */
} sdf_glyph_t;

/*
 * SDF font container
 * - ASCII-only table (0..255) of glyph pointers for v1
 * - Fallback box glyph used when a character is missing
 * - Optional metadata for identification
 */
typedef struct sdf_font_t {
    sdf_glyph_t *table[256];   /* ASCII glyph pointers (NULL if missing) */

    /* Fallback box glyph (embedded); use box_valid to check presence */
    sdf_glyph_t box;
    bool box_valid;

    /* Optional identifiers (owned strings or external; loader defines policy) */
    const char *name;          /* font name or directory */
    const char *dir;           /* asset directory path containing metrics.csv and PNGs */

    uint32_t glyph_count;      /* number of loaded glyphs in table */

    /* Optional kerning pairs (ASCII v1). If kern_count==0 or pairs==NULL, kerning is unavailable */
    struct sdf_kern_pair_t *kern_pairs;
    size_t kern_count;

    /* Derived line metrics (at current glyph sizes) */
    float line_top;            /* max(bearing_y) across glyphs */
    float line_bottom;         /* max(height - bearing_y) across glyphs */
    float line_height;         /* line_top + line_bottom */
} sdf_font_t;

/*
 * Parsed CSV metrics row for a glyph.
 * width/height are the PNG dimensions (including padding); path is the glyph PNG path from CSV.
 */
typedef struct sdf_metrics_row_t {
    unsigned ch;         /* ASCII code (0..255) for v1 */
    float advance_width; /* pixels */
    float bearing_x;     /* pixels */
    float bearing_y;     /* pixels, top bearing (+up) */
    int   width;         /* PNG width in pixels */
    int   height;        /* PNG height in pixels */
    char *path;          /* heap-allocated string to PNG path */
} sdf_metrics_row_t;

/* CSV loader API */
/* Returns 0 on success; rows must be freed with sdf_metrics_rows_free. */
int sdf_metrics_csv_load(const char *csv_path, sdf_metrics_row_t **out_rows, size_t *out_count);
void sdf_metrics_rows_free(sdf_metrics_row_t *rows, size_t count);

/* Font loader: loads metrics.csv + glyph PNGs under font_dir; preloads ASCII 32..126 */
/* Returns NULL on failure. Caller should free with sdf_font_free (to be implemented in 1.7). */
sdf_font_t* sdf_font_load(const char *font_dir);

/* Load and immediately scale the font to a target line height (in pixels). */
sdf_font_t* sdf_font_load_scaled(const char *font_dir, float target_line_height_px);

/* Free APIs for font and glyphs */
void sdf_glyph_free(sdf_glyph_t *g);
void sdf_font_free(sdf_font_t *font);

/* Create a scaled copy of an already-loaded font to match target line height (in pixels). */
sdf_font_t* sdf_font_create_scaled(const sdf_font_t *base, float target_line_height_px);

/* Query the current line height of a font (pixels). */
static inline float sdf_font_get_line_height(const sdf_font_t *font) { return font ? font->line_height : 0.0f; }

typedef struct sdf_effect_t {
    /* Effect placeholders (V2) */
    RGBA  glow_color;     // the glow color
    Normal glow_radius;    // px 
    RGBA  outline_color;  // outline color
    Normal outline_smooth;  // outline color
    Normal outline_width;  // px 
    float weight;         // 1.0 = normal; >1.0 bold (grow), <1.0 thin (shrink) 
} sdf_effect_t;

/* Feature macro for conditional compilation in C sources */
#define SDF_HAS_OUTLINE_COLOR 1

/* Text orientation: horizontal (default) or vertical stack */
typedef enum sdf_orientation_t {
    SDF_ORIENT_HORIZONTAL = 0,
    SDF_ORIENT_VERTICAL   = 1
} sdf_orientation_t;

/* Vertical alignment options for text placement */
typedef enum sdf_valign_t {
    SDF_VALIGN_BASELINE = 0, /* place glyphs using typographic baseline */
    SDF_VALIGN_TOP      = 1, /* align top of visible ink to line top */
    SDF_VALIGN_BOTTOM   = 2  /* align bottom of visible ink to line bottom */
} sdf_valign_t;


/*
 * Retained-mode text object (per-instance state)
 * - Holds immutable/cached layout inputs plus dynamic animation state
 * - Rendering API will consume this to rasterize into a destination buffer
 */
typedef struct sdf_text_t {
    /* Content + font */
    sdf_font_t *font;     /* current font used for rendering (may be a scaled copy) */
    sdf_font_t *base_font;/* original font provided at create time (never mutated) */
    char       *text;     /* UTF-8 string (ASCII for v1); owned or external TBD in 2.2 */
    size_t      text_len; /* cached byte length (excludes NUL) */

    /* Transform and motion */
    float x, y;           /* current position (top-left or baseline origin, dependent on renderer) */
    float x0, y0;         /* base position at t=0 for absolute-time animation */
    float dir_x, dir_y;   /* normalized direction vector for scrolling */
    float speed;          /* pixels per second (applied along dir) */
    float wrap_mod;       /* time in seconds to wrap the text */
    float angle;          /* radians, rotation about (x,y) (optional v1) */
    float size_px;        /* desired line height in pixels; 0 => use current font size */

    float softness;       // softness of the glyph edges


    /* Appearance */
    RGBA color;      /* color */
    uint8_t alpha;        /* overall alpha (0..255) */

    /* Layout */
    float tracking;       /* additional spacing in pixels between glyphs */
    bool  kerning_enable; /* enable kerning when available (future) */
    bool  wrap;           /* enable wrapping (future) */

    /* Cached shaping (simple v1 cache) */
    uint16_t *_glyph_indices; /* ASCII codepoints or glyph ids (v1 uses ASCII) */
    float    *_advances;      /* per-glyph advances (pixels) */
    int      *_ofs_x;         /* per-glyph x offset from baseline origin (pixels, integer placement) */
    int      *_ofs_y;         /* per-glyph y offset from baseline origin (pixels, integer placement) */
    size_t    _glyph_count;   /* number of shaped glyphs */
    bool      shape_dirty;   /* recompute layout cache when true */

    sdf_effect_t effects;

    /* Alloc/ownership */
    bool owns_font;          /* if true, text owns and will free its current font on destroy/resize */

    /* Vertical alignment mode */
    sdf_valign_t valign;

    /* Orientation */
    sdf_orientation_t orient;

    vec2u dimensions;
} sdf_text_t;


/* Text object API */
sdf_text_t* sdf_text_create(sdf_font_t *font, const char *text);
void        sdf_text_destroy(sdf_text_t *t);

/* Setters: mark shape_dirty when layout-affecting changes occur */
void sdf_text_set_position(sdf_text_t *t, float x, float y);
void sdf_text_set_direction(sdf_text_t *t, float dx, float dy);
void sdf_text_set_speed(sdf_text_t *t, float speed);
void sdf_text_set_size_px(sdf_text_t *t, float size_px);
void sdf_text_set_color(sdf_text_t *t, RGBA color);
void sdf_text_set_alpha(sdf_text_t *t, uint8_t a);
void sdf_text_set_tracking(sdf_text_t *t, float tracking);
void sdf_text_set_kerning(sdf_text_t *t, bool enable);
void sdf_text_set_wrap(sdf_text_t *t, bool enable);
void sdf_text_set_angle(sdf_text_t *t, float angle);
void sdf_text_set_text(sdf_text_t *t, const char *text);
void sdf_text_set_valign(sdf_text_t *t, sdf_valign_t valign);
void sdf_text_set_orientation(sdf_text_t *t, sdf_orientation_t orient);

/* Force a layout/metrics refresh after batching attribute changes.
 * This recomputes any internal cached data (font scaling, shaping arrays)
 * so the text object is consistent and ready to render. */
void sdf_text_update(sdf_text_t *t, const int32_t display_width);

/* Render entry point. time_sec is an ABSOLUTE timestamp (seconds since animation start)
 * used to position scrolling text deterministically: pos(t) = (x0,y0) + dir*speed*t. */
void sdf_text_render(sdf_text_t *t, uint8_t *dst, int w, int h, int stride, float time_sec);

/* Layout helpers */
typedef struct sdf_line_extents_t {
    float top;      /* max(bearing_y) across used glyphs */
    float bottom;   /* max(height - bearing_y) across used glyphs */
    float height;   /* top + bottom */
} sdf_line_extents_t;

/* Compute line extents for the current text (ASCII v1). Empty/missing -> zeros. */
sdf_line_extents_t sdf_text_compute_line_extents(const sdf_text_t *t);


/* Measure the rendered size of the current text (in pixels).
 * - Horizontal: width = sum of advances; height = line extents height
 * - Vertical:   height = sum of advances; width = max glyph width
 * Ensures pending size/shape updates are applied before measuring. */
vec2u sdf_text_measure_px(sdf_text_t *t);

/* Baseline helpers (y coordinates in pixels) */
/*
 - bottom-aligned: baseline at B - line_bottom
 - top-aligned:    baseline at T + line_top
 - center:         baseline at T + (H + line_top - line_bottom)/2
*/
float sdf_text_baseline_bottom(float canvas_bottom_y, sdf_line_extents_t e);
float sdf_text_baseline_top(float canvas_top_y, sdf_line_extents_t e);
float sdf_text_baseline_center(float canvas_top_y, float canvas_height, sdf_line_extents_t e);

/* Scrolling and wrap helpers (3.5) */
/* Set position for absolute time: pos(t) = (x0,y0) + dir*speed*time_sec. */
void sdf_text_animate(sdf_text_t *t, const float time_sec);

/* Horizontal wrap helpers using total text advance width. */
void _sdf_text_wrap_left_to_right(sdf_text_t *t, float left_x, float right_x);
void _sdf_text_wrap_right_to_left(sdf_text_t *t, float left_x, float right_x);

/* Vertical wrap helpers using line extents (top/bottom). */
void _sdf_text_wrap_top_to_bottom(sdf_text_t *t, float top_y, float bottom_y);
void _sdf_text_wrap_bottom_to_top(sdf_text_t *t, float top_y, float bottom_y);

/* Kerning support (optional kerning.csv) */
typedef struct sdf_kern_pair_t {
    uint8_t left;   /* ASCII code */
    uint8_t right;  /* ASCII code */
    float   value;  /* kerning adjustment in pixels (added to advance of left when followed by right) */
} sdf_kern_pair_t;

/* Load kerning CSV: left,right,value (chars or numeric code). Returns 0 on success. */
int sdf_kerning_csv_load(const char *csv_path, sdf_kern_pair_t **out_pairs, size_t *out_count);
void sdf_kerning_pairs_free(sdf_kern_pair_t *pairs);

/* Lookup kerning between two ASCII codes; returns 0.0f if not found or kerning unavailable. */
float sdf_font_get_kerning(const sdf_font_t *font, unsigned left, unsigned right);

#ifdef __cplusplus
}
#endif

#endif /* TEXT_SDF_H */
