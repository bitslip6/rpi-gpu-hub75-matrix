// Clean includes (repaired)

#include "functions.h"
#include "pixels.h"
#include "util.h"
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "text_sdf.h"
#include "compositor.h"

/* Forward declaration for bilinear sampler used in font scaling and rendering
 */
static float sdf_sample_gray8_bilinear(const sdf_glyph_t *glyph, float fx,
                                       float fy);

/*
 * CSV loader for metrics:
 * char,advance_width,bearing_x,bearing_y,width,height,path Simple, robust
 * parser suitable for ASCII glyph packs used by the project.
 */
static char *dup_cstr(const char *s) {
  if (!s)
    return NULL;
  size_t n = strlen(s);
  char *p = (char *)malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

static char *trim_inplace(char *s) {
  if (!s)
    return s;
  while (*s && isspace((unsigned char)*s))
    s++;
  char *e = s + strlen(s);
  while (e > s && isspace((unsigned char)e[-1]))
    --e;
  *e = '\0';
  return s;
}

static int split_csv_line(char *line, char **fields, int max_fields) {
  int n = 0;
  char *p = line;
  while (*p && n < max_fields) {
    fields[n++] = p;
    char *c = strchr(p, ',');
    if (!c)
      break;
    *c = '\0';
    p = c + 1;
  }
  return n;
}

static unsigned parse_char_field(const char *t) {
  if (!t || !*t)
    return 0;
  if (t[1] == '\0') {
    /* single byte */
    return (unsigned)(unsigned char)t[0];
  }
  /* numeric code */
  return (unsigned)strtoul(t, NULL, 10);
}

int sdf_metrics_csv_load(const char *csv_path, sdf_metrics_row_t **out_rows,
                         size_t *out_count) {
  if (!csv_path || !out_rows || !out_count)
    return -1;
  *out_rows = NULL;
  *out_count = 0;

  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  FILE *f = fopen(csv_path, "rb");
  if (!f) {
    fprintf(stderr, "sdf_metrics_csv_load: failed to open %s%s. (%s)\n", cwd,
            csv_path, strerror(errno));
    return -1;
  }

  size_t cap = 128, n = 0;
  sdf_metrics_row_t *rows = (sdf_metrics_row_t *)calloc(cap, sizeof(*rows));
  if (!rows) {
    fclose(f);
    return -1;
  }

  char line[4096];
  int is_header = 1;
  while (fgets(line, sizeof(line), f)) {
    char *ln = trim_inplace(line);
    if (!*ln)
      continue;
    if (is_header) {
      is_header = 0;
      continue;
    }

    char *fields[7] = {0};
    int nf = split_csv_line(ln, fields, 7);
    if (nf < 7) {
      fprintf(stderr,
              "sdf_metrics_csv_load: skipping malformed row (got %d fields)\n",
              nf);
      continue;
    }

    sdf_metrics_row_t r;
    memset(&r, 0, sizeof(r));
    r.ch = parse_char_field(trim_inplace(fields[0]));
    r.advance_width = (float)strtod(trim_inplace(fields[1]), NULL);
    r.bearing_x = (float)strtod(trim_inplace(fields[2]), NULL);
    r.bearing_y = (float)strtod(trim_inplace(fields[3]), NULL);
    r.width = (int)strtol(trim_inplace(fields[4]), NULL, 10);
    r.height = (int)strtol(trim_inplace(fields[5]), NULL, 10);
    r.path = dup_cstr(trim_inplace(fields[6]));
    if (!r.path) {
      fprintf(stderr, "sdf_metrics_csv_load: OOM duplicating path\n");
      continue;
    }

    if (n == cap) {
      size_t ncap = cap * 2;
      sdf_metrics_row_t *tmp =
          (sdf_metrics_row_t *)realloc(rows, ncap * sizeof(*rows));
      if (!tmp) {
        fprintf(stderr, "sdf_metrics_csv_load: OOM grow\n");
        free(r.path);
        break;
      }
      /* zero the new part */
      memset(tmp + cap, 0, (ncap - cap) * sizeof(*rows));
      rows = tmp;
      cap = ncap;
    }
    rows[n++] = r;
  }

  fclose(f);
  if (n == 0) {
    free(rows);
    return -1;
  }
  *out_rows = rows;
  *out_count = n;
  return 0;
}

void sdf_metrics_rows_free(sdf_metrics_row_t *rows, size_t count) {
  if (!rows)
    return;
  for (size_t i = 0; i < count; ++i) {
    free(rows[i].path);
  }
  free(rows);
}

/* Utility: join font_dir and path if path is relative. Result written to out
 * buffer of size out_sz. */
static void join_path(char *out, size_t out_sz, const char *dir,
                      const char *path) {
  if (!out || out_sz == 0)
    return;
  if (!path || !*path) {
    out[0] = '\0';
    return;
  }
  if (path[0] == '/') {
    snprintf(out, out_sz, "%s", path);
    return;
  }
  size_t n = (size_t)snprintf(out, out_sz, "%s/%s", dir ? dir : ".", path);
  if (n >= out_sz) {
	debug(" [!] Error: font path exceeds max path len %d\n", out_sz);
  }
}

sdf_font_t *sdf_font_load(const char *font_dir) {
  if (!font_dir)
    return NULL;

  char metrics_path[1024];
  snprintf(metrics_path, sizeof(metrics_path), "%s/metrics.csv", font_dir);

  sdf_metrics_row_t *rows = NULL;
  size_t count = 0;
  if (sdf_metrics_csv_load(metrics_path, &rows, &count) != 0) {
    fprintf(stderr, "sdf_font_load: failed to load metrics %s\n", metrics_path);
    return NULL;
  }

  sdf_font_t *font = (sdf_font_t *)calloc(1, sizeof(sdf_font_t));
  if (!font) {
    sdf_metrics_rows_free(rows, count);
    return NULL;
  }
  font->box_valid = false;
  font->glyph_count = 0;
  font->dir = strdup(font_dir); /* ownership freed in sdf_font_free later */
  font->name = font->dir;       /* alias for now */

  for (size_t i = 0; i < count; ++i) {
    const sdf_metrics_row_t *r = &rows[i];
    unsigned ch = r->ch & 0xFFu;
    if (ch < 32 || ch > 126)
      continue; /* ASCII printable range */

    char path[1024];
    join_path(path, sizeof(path), font_dir, r->path);

    uint8_t *pixels = NULL;
    int w = 0, h = 0, stride = 0;
    if (image_read_gray8(path, &pixels, &w, &h, &stride) != 0) {
      fprintf(stderr, "sdf_font_load: failed to read image for '%c' (%u): %s\n",
              (char)ch, ch, path);
      continue;
    }
    if ((r->width > 0 && w != r->width) || (r->height > 0 && h != r->height)) {
      fprintf(
          stderr,
          "sdf_font_load: dim mismatch for '%c' (%u): CSV %dx%d, PNG %dx%d\n",
          (char)ch, ch, r->width, r->height, w, h);
    }

    /* Create glyph */
    sdf_glyph_t *g = (sdf_glyph_t *)calloc(1, sizeof(sdf_glyph_t));
    if (!g) {
      free(pixels);
      continue;
    }
    int cw = (w > 255) ? 255 : w;
    int chh = (h > 255) ? 255 : h;
    g->width = (uint8_t)cw;
    g->height = (uint8_t)chh;
    g->bearing_x = r->bearing_x;
    g->bearing_y = r->bearing_y;
    g->advance_width = r->advance_width;
    g->pixels = pixels; /* ownership to font */

    if (font->table[ch]) {
      /* Replace existing (unlikely) */
      free(font->table[ch]->pixels);
      free(font->table[ch]);
    }
    font->table[ch] = g;
    font->glyph_count++;
  }

  /* Create fallback box glyph based on computed line metrics */
  /* Compute line metrics from loaded glyphs */
  float line_top = 0.0f, line_bottom = 0.0f, adv_sum = 0.0f;
  int adv_cnt = 0;
  for (int c = 32; c <= 126; ++c) {
    sdf_glyph_t *g = font->table[c];
    if (!g)
      continue;
    if (g->bearing_y > line_top)
      line_top = g->bearing_y;
    float bottom = (float)g->height - g->bearing_y;
    if (bottom > line_bottom)
      line_bottom = bottom;
    adv_sum += g->advance_width;
    adv_cnt++;
  }
  font->line_top = line_top;
  font->line_bottom = line_bottom;
  font->line_height = line_top + line_bottom;
  int fallback_h = (int)(line_top + line_bottom + 0.5f);
  if (fallback_h <= 0)
    fallback_h = 16;
  if (fallback_h < 8)
    fallback_h = 8;
  if (fallback_h > 96)
    fallback_h = 96;
  int fallback_w = fallback_h; // square 
  uint8_t *pix = (uint8_t *)malloc((size_t)fallback_w * (size_t)fallback_h);
  if (pix) {
    /* Build a simple SDF ring (box frame) */
    const float cx = 0.5f * (float)fallback_w;
    const float cy = 0.5f * (float)fallback_h;
    const float m = 1.5f;      /* outer margin */
    const float t = 2.0f;      /* frame thickness */
    const float ax = cx - m;   /* outer half-size x */
    const float ay = cy - m;   /* outer half-size y */
    const float aix = ax - t;  /* inner half-size x */
    const float aiy = ay - t;  /* inner half-size y */
    const float spread = 4.0f; /* SDF ramp */
    for (int y = 0; y < fallback_h; ++y) {
      uint8_t *row = pix + (size_t)y * (size_t)fallback_w;
      for (int x = 0; x < fallback_w; ++x) {
        float px = (float)x + 0.5f - cx;
        float py = (float)y + 0.5f - cy;
        /* SDF to outer rectangle */
        float dx = fabsf(px) - ax;
        float dy = fabsf(py) - ay;
        float ox = fmaxf(dx, 0.0f), oy = fmaxf(dy, 0.0f);
        float outside = sqrtf(ox * ox + oy * oy);
        float inside = fminf(fmaxf(dx, dy), 0.0f);
        float sdf_outer = outside + inside; /* negative inside */
        /* SDF to inner rectangle */
        float dx2 = fabsf(px) - aix;
        float dy2 = fabsf(py) - aiy;
        float ox2 = fmaxf(dx2, 0.0f), oy2 = fmaxf(dy2, 0.0f);
        float outside2 = sqrtf(ox2 * ox2 + oy2 * oy2);
        float inside2 = fminf(fmaxf(dx2, dy2), 0.0f);
        float sdf_inner = outside2 + inside2; /* negative inside inner rect */
        /* Ring (frame) = outer minus inner => max(sdf_outer, -sdf_inner) */
        float sdf_ring = fmaxf(sdf_outer, -sdf_inner);
        /* Map to 8-bit: 128 at edge; >128 inside frame, <128 outside */
        float signed_inside = -sdf_ring; /* positive inside */
        float v = 128.0f + 120.0f * (signed_inside / spread);
        if (v < 0.0f)
          v = 0.0f;
        else if (v > 255.0f)
          v = 255.0f;
        row[x] = (uint8_t)(v + 0.5f);
      }
    }
    font->box.width = (uint8_t)((fallback_w > 255) ? 255 : fallback_w);
    font->box.height = (uint8_t)((fallback_h > 255) ? 255 : fallback_h);
    font->box.pixels = pix;
    font->box.bearing_x = 0.0f;
    font->box.bearing_y =
        (line_top > 0.0f) ? line_top : (0.8f * (float)fallback_h);
    font->box.advance_width =
        (adv_cnt > 0) ? (adv_sum / (float)adv_cnt) : (float)fallback_w;
    font->box_valid = true;
  } else {
    font->box_valid = false;
  }

  /* Attempt to load kerning.csv (optional) */
  {
    char kern_path[1024];
    join_path(kern_path, sizeof(kern_path), font_dir, "kerning.csv");
    sdf_kern_pair_t *pairs = NULL;
    size_t pcnt = 0;
    if (file_exists(kern_path) &&
        sdf_kerning_csv_load(kern_path, &pairs, &pcnt) == 0 && pcnt > 0) {
      font->kern_pairs = pairs;
      font->kern_count = pcnt;
    } else {
      font->kern_pairs = NULL;
      font->kern_count = 0;
    }
  }

  /* Ensure a non-drawing space glyph exists (ASCII 32). If assets omit it,
   * synthesize a metrics-only glyph so space advances without rendering the
   * fallback box. */
  if (font->table[32] == NULL) {
    float avg_adv =
        (adv_cnt > 0)
            ? (adv_sum / (float)adv_cnt)
            : ((font->line_height > 0.0f) ? 0.33f * font->line_height : 6.0f);
    float em_space =
        (font->line_height > 0.0f) ? 0.33f * font->line_height : avg_adv;
    float sp_adv = (avg_adv > 0.0f) ? fminf(avg_adv, em_space) : em_space;
    if (sp_adv < 1.0f)
      sp_adv = (font->line_height > 0.0f) ? 0.25f * font->line_height : 4.0f;
    sdf_glyph_t *sp = (sdf_glyph_t *)calloc(1, sizeof(sdf_glyph_t));
    if (sp) {
      sp->width = 0;
      sp->height = 0;
      sp->pixels = NULL;
      sp->bearing_x = 0.0f;
      sp->bearing_y = font->line_top; /* keep consistent baseline metrics */
      sp->advance_width = sp_adv;
      sp->ink_top = 0;
      sp->ink_bottom = 0;
      font->table[32] = sp;
      font->glyph_count++;
    }
  }

  sdf_metrics_rows_free(rows, count);
  return font;
}

void sdf_glyph_free(sdf_glyph_t *g) {
  if (!g)
    return;
  if (g->pixels) {
    free(g->pixels);
    g->pixels = NULL;
  }
  free(g);
}

void sdf_font_free(sdf_font_t *font) {
  if (!font)
    return;
  for (int i = 0; i < 256; ++i) {
    if (font->table[i]) {
      sdf_glyph_free(font->table[i]);
      font->table[i] = NULL;
    }
  }
  if (font->box_valid && font->box.pixels) {
    free(font->box.pixels);
    font->box.pixels = NULL;
    font->box_valid = false;
  }
  /* name is an alias to dir; don't double free */
  if (font->dir) {
    /* cast away const because we allocated it */
    free((void *)font->dir);
    font->dir = NULL;
    font->name = NULL;
  }
  if (font->kern_pairs) {
    sdf_kerning_pairs_free(font->kern_pairs);
    font->kern_pairs = NULL;
    font->kern_count = 0;
  }
  free(font);
}

/* ---------- Kerning support ---------- */

int sdf_kerning_csv_load(const char *csv_path, sdf_kern_pair_t **out_pairs,
                         size_t *out_count) {
  if (!csv_path || !out_pairs || !out_count)
    return -1;
  *out_pairs = NULL;
  *out_count = 0;
  FILE *f = fopen(csv_path, "rb");
  if (!f)
    return -1;

  size_t cap = 128, n = 0;
  sdf_kern_pair_t *pairs = (sdf_kern_pair_t *)malloc(cap * sizeof(*pairs));
  if (!pairs) {
    fclose(f);
    return -1;
  }

  char line[1024];
  int is_header = 1;
  while (fgets(line, sizeof(line), f)) {
    char *ln = trim_inplace(line);
    if (!*ln)
      continue;
    if (is_header) {
      is_header = 0;
      continue;
    }
    char *fields[3] = {0};
    int nf = split_csv_line(ln, fields, 3);
    if (nf < 3)
      continue;
    unsigned l = parse_char_field(trim_inplace(fields[0])) & 0xFFu;
    unsigned r = parse_char_field(trim_inplace(fields[1])) & 0xFFu;
    float v = (float)strtod(trim_inplace(fields[2]), NULL);
    if (n == cap) {
      size_t ncap = cap * 2;
      sdf_kern_pair_t *tmp =
          (sdf_kern_pair_t *)realloc(pairs, ncap * sizeof(*pairs));
      if (!tmp) {
        free(pairs);
        fclose(f);
        return -1;
      }
      pairs = tmp;
      cap = ncap;
    }
    pairs[n].left = (uint8_t)l;
    pairs[n].right = (uint8_t)r;
    pairs[n].value = v;
    n++;
  }
  fclose(f);
  if (n == 0) {
    free(pairs);
    return -1;
  }
  *out_pairs = pairs;
  *out_count = n;
  return 0;
}

void sdf_kerning_pairs_free(sdf_kern_pair_t *pairs) {
  if (pairs)
    free(pairs);
}

float sdf_font_get_kerning(const sdf_font_t *font, unsigned left,
                           unsigned right) {
  if (!font || !font->kern_pairs || font->kern_count == 0)
    return 0.0f;
  uint8_t l = (uint8_t)(left & 0xFFu);
  uint8_t r = (uint8_t)(right & 0xFFu);
  /* Linear scan; pair count expected to be modest. Could be optimized later. */
  for (size_t i = 0; i < font->kern_count; ++i) {
    const sdf_kern_pair_t *kp = &font->kern_pairs[i];
    if (kp->left == l && kp->right == r)
      return kp->value;
  }
  return 0.0f;
}

/* ---------- Font scaling support ---------- */

/* Bilinear sample on a raw grayscale buffer at continuous coords (fx, fy),
 * where pixel centers are at integer+0.5. Returns normalized [0,1]. */

static uint8_t *_resample_gray8_scale(const uint8_t *src, int sw, int sh,
                                      int dw, int dh) {
  if (!src || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
    return NULL;
  int dstride = dw;
  uint8_t *dst = (uint8_t *)malloc((size_t)dh * (size_t)dstride);
  if (!dst)
    return NULL;
  /* scale factor from dest to source */
  float sx = (float)sw / (float)dw;
  float sy = (float)sh / (float)dh;
  for (int y = 0; y < dh; ++y) {
    uint8_t *row = dst + (size_t)y * (size_t)dstride;
    float src_y = ((float)y + 0.5f) * sy; /* map center to source coords */
    for (int x = 0; x < dw; ++x) {
      float src_x = ((float)x + 0.5f) * sx;
      row[x] = sample_gray8_bilinear_buf(src, sw, sh, src_x, src_y);
    }
  }
  return dst;
}

static sdf_glyph_t *_scale_glyph(const sdf_glyph_t *g, float s) {
  if (!g || s <= 0.0f)
    return NULL;
  int sw = (int)g->width, sh = (int)g->height;
  /* Preserve non-drawing glyphs (e.g., synthesized space) as metrics-only */
  if (!g->pixels || sw == 0 || sh == 0) {
    sdf_glyph_t *ng = (sdf_glyph_t *)calloc(1, sizeof(sdf_glyph_t));
    if (!ng)
      return NULL;
    ng->width = 0;
    ng->height = 0;
    ng->pixels = NULL;
    ng->bearing_x = g->bearing_x * s;
    ng->bearing_y = g->bearing_y * s;
    ng->advance_width = g->advance_width * s;
    ng->ink_top = 0;
    ng->ink_bottom = 0;
    return ng;
  }
  int dw = (int)lrintf((float)sw * s);
  int dh = (int)lrintf((float)sh * s);
  if (dw < 1)
    dw = 1;
  if (dh < 1)
    dh = 1;
  if (dw > 255)
    dw = 255;
  if (dh > 255)
    dh = 255;
  uint8_t *pixels = _resample_gray8_scale(g->pixels, sw, sh, dw, dh);
  if (!pixels)
    return NULL;
  sdf_glyph_t *ng = (sdf_glyph_t *)calloc(1, sizeof(sdf_glyph_t));
  if (!ng) {
    free(pixels);
    return NULL;
  }
  ng->width = (uint8_t)dw;
  ng->height = (uint8_t)dh;
  ng->pixels = pixels;
  ng->bearing_x = g->bearing_x * s;
  ng->bearing_y = g->bearing_y * s;
  ng->advance_width = g->advance_width * s;
  return ng;
}

sdf_font_t *sdf_font_create_scaled(const sdf_font_t *base,
                                   float target_line_height_px) {
  if (!base || target_line_height_px <= 0.0f || base->line_height <= 0.0f)
    return NULL;
  float s = target_line_height_px / base->line_height;
  if (s <= 0.0f)
    s = 1.0f;
  sdf_font_t *f = (sdf_font_t *)calloc(1, sizeof(sdf_font_t));
  if (!f)
    return NULL;
  for (int i = 0; i < 256; ++i) {
    if (!base->table[i]) {
      f->table[i] = NULL;
      continue;
    }
    f->table[i] = _scale_glyph(base->table[i], s);
    if (f->table[i])
      f->glyph_count++;
  }
  if (base->box_valid) {
    sdf_glyph_t *scaled_box = _scale_glyph(&base->box, s);
    if (scaled_box) {
      f->box = *scaled_box; /* copy struct */
      free(scaled_box);     /* pixels remain owned by f->box */
      f->box_valid = true;
    }
  }
  /* kerning: copy and scale values */
  if (base->kern_pairs && base->kern_count > 0) {
    f->kern_pairs =
        (sdf_kern_pair_t *)malloc(base->kern_count * sizeof(sdf_kern_pair_t));
    if (f->kern_pairs) {
      f->kern_count = base->kern_count;
      for (size_t i = 0; i < base->kern_count; ++i) {
        f->kern_pairs[i] = base->kern_pairs[i];
        f->kern_pairs[i].value *= s;
      }
    }
  }
  /* identifiers: leave null to avoid free semantics confusion */
  f->name = NULL;
  f->dir = NULL;
  /* line metrics */
  f->line_top = base->line_top * s;
  f->line_bottom = base->line_bottom * s;
  f->line_height = base->line_height * s;
  return f;
}

sdf_font_t *sdf_font_load_scaled(const char *font_dir,
                                 float target_line_height_px) {
  sdf_font_t *base = sdf_font_load(font_dir);
  if (!base)
    return NULL;
  sdf_font_t *scaled = sdf_font_create_scaled(base, target_line_height_px);
  sdf_font_free(base);
  return scaled;
}
/* ---------------- Text object API ---------------- */

/* Forward declaration of internal shaping helper */
static void _sdf_text_update_shape(sdf_text_t *t);
static void _sdf_text_apply_size(sdf_text_t *t);

/* Forward declarations for helpers used in render before their full definitions
 */
static inline const sdf_glyph_t *font_glyph(const sdf_font_t *font,
                                            unsigned ch);
static float sdf_sample_gray8_bilinear(const sdf_glyph_t *glyph, float fx,
                                       float fy);

static void _normalize_dir(float *dx, float *dy) {
  float x = *dx, y = *dy;
  float len = sqrtf(x * x + y * y);
  if (len <= 0.00001f) {
    *dx = 1.0f;
    *dy = 0.0f;
    return;
  }
  *dx = x / len;
  *dy = y / len;
}

/* Public: recompute internal caches after attribute changes (kerning, tracking,
 * size, etc.). Ensures per-text scaling is applied and shaping arrays are
 * refreshed. */
void sdf_text_update(sdf_text_t *t, int32_t display_width) {
  if (!t)
    return;
  /* Apply size scaling from base_font if needed */
  _sdf_text_apply_size(t);
  /* Mark and refresh shaping */
  t->shape_dirty = true;
  _sdf_text_update_shape(t);

  /* Update cached rendered dimensions (in pixels) including effect expansion */
  {
    /* Base measured size from glyph metrics and shaping */
    vec2u sz = sdf_text_measure_px(t);
    /* Expand for visual effects that extend beyond glyph ink: outline/glow */
    float expand = 0.0f;
    if (t) {
      if (t->effects.glow_radius > expand)   expand = t->effects.glow_radius;
      if (t->effects.outline_width > expand) expand = t->effects.outline_width;
    }
    t->dimensions.x = sz.x+ (int)ceilf(expand);
    t->dimensions.y = sz.y + (int)ceilf(expand);

    // compute the text wrap time if the shape changes
    if (t->wrap) {
        int32_t maxlen = MAX(t->dimensions.x, t->dimensions.y);
        t->wrap_mod = ((float)maxlen / t->speed) + 1.0f;
    }
  }
  t->x0 = (float)display_width;
}

sdf_text_t *sdf_text_create(sdf_font_t *font, const char *text) {
  sdf_text_t *t = (sdf_text_t *)calloc(1, sizeof(sdf_text_t));
  if (!t)
    return NULL;
  t->font = font;
  t->base_font = font;
  if (text && *text) {
    t->text_len = strlen(text);
    t->text = (char *)malloc(t->text_len + 1);
    if (t->text) {
      memcpy(t->text, text, t->text_len + 1);
    }
  } else {
    t->text_len = 0;
    t->text = NULL;
  }

  /* Defaults */
  t->x = 0.0f;
  t->y = 0.0f;
  // base position for animation
  t->x0 = t->x;
  t->y0 = t->y;
  t->dir_x = -1.0f;
  t->dir_y = 0.0f;
  t->speed = 0.0f;
  t->angle = 0.0f;
  t->size_px = 0.0f; /* 0 => use current font line height */
  t->color = (RGBA){255, 255, 255, 255};
  t->alpha = 255;
  t->tracking = 3.0f;
  t->kerning_enable = false;
  t->wrap = true;
  t->softness = 0.1f;
  t->_glyph_indices = NULL;
  t->_advances = NULL;
  t->_ofs_x = NULL;
  t->_ofs_y = NULL;
  t->_glyph_count = 0;
  t->shape_dirty = true;
  /* Effects defaults */
  t->effects.glow_radius = 0.0f;
  t->effects.glow_color = t->color; // default: same as text color 
  t->effects.outline_width = 0.0f;
  t->effects.outline_smooth = 0.12f;
  t->effects.weight = 1.0f; // neutral 

  /* Default vertical alignment: baseline */

  t->valign = SDF_VALIGN_BOTTOM;

  /* Default orientation: horizontal */
  t->orient = SDF_ORIENT_HORIZONTAL;

  t->owns_font = false;

  return t;
}

void sdf_text_destroy(sdf_text_t *t) {
  if (!t)
    return;
  if (t->text)
    free(t->text);
  if (t->_glyph_indices)
    free(t->_glyph_indices);
  if (t->_advances)
    free(t->_advances);
  if (t->_ofs_x)
    free(t->_ofs_x);
  if (t->_ofs_y)
    free(t->_ofs_y);
  if (t->owns_font && t->font) {
    sdf_font_free(t->font);
    t->font = NULL;
  }
  free(t);
}

void sdf_text_set_position(sdf_text_t *t, float x, float y) {
  if (!t)
    return;
  t->x = x;
  t->y = y;
  /* update base position so absolute-time animation uses this as t=0 */
  t->x0 = x;
  t->y0 = y;
}

void sdf_text_set_direction(sdf_text_t *t, float dx, float dy) {
  if (!t)
    return;
  t->dir_x = dx;
  t->dir_y = dy;
  _normalize_dir(&t->dir_x, &t->dir_y);
}

void sdf_text_set_speed(sdf_text_t *t, float speed) {
  if (!t)
    return;
  t->speed = speed;
  t->shape_dirty = true;
}

void sdf_text_set_size_px(sdf_text_t *t, float size_px) {
  if (!t)
    return;
  t->size_px = size_px;
  /* Apply per-text scaling by creating a scaled font from base_font */
  _sdf_text_apply_size(t);
  t->shape_dirty = true;
}

void sdf_text_set_color(sdf_text_t *t, RGBA color) {
  if (!t)
    return;
  t->color = color;
}

void sdf_text_set_alpha(sdf_text_t *t, uint8_t a) {
  if (!t)
    return;
  t->alpha = a;
}

void sdf_text_set_tracking(sdf_text_t *t, float tracking) {
  if (!t)
    return;
  t->tracking = tracking;
  t->shape_dirty = true;
}

void sdf_text_set_kerning(sdf_text_t *t, bool enable) {
  if (!t)
    return;
  t->kerning_enable = enable;
  t->shape_dirty = true;
}

void sdf_text_set_wrap(sdf_text_t *t, bool enable) {
  if (!t)
    return;
  t->wrap = enable;
  t->shape_dirty = true;
}

void sdf_text_set_angle(sdf_text_t *t, float angle) {
  if (!t)
    return;
  t->angle = angle;
}

void sdf_text_set_text(sdf_text_t *t, const char *text) {
  if (!t)
    return;
  if (t->text) {
    free(t->text);
    t->text = NULL;
    t->text_len = 0;
  }
  if (text && *text) {
    t->text_len = strlen(text);
    t->text = (char *)malloc(t->text_len + 1);
    if (t->text) {
      memcpy(t->text, text, t->text_len + 1);
    }
  }
  t->shape_dirty = true;
}

void sdf_text_set_valign(sdf_text_t *t, sdf_valign_t valign) {
  if (!t)
    return;
  t->valign = valign;
  /* Changing valign affects vertical placement; mark shaping dirty. */
  t->shape_dirty = true;
}

void sdf_text_set_orientation(sdf_text_t *t, sdf_orientation_t orient) {
  if (!t)
    return;
  t->orient = orient;
  /* Orientation changes placement axes; require re-shaping */
  t->shape_dirty = true;
}

/**
 * @param stride - row stride in bytes (bytes per row, e.g., width * 4 for RGBA)
 * @param w - dst width in pixels
 * @param h - dst height in pixels
 */
void sdf_text_render(sdf_text_t *t, uint8_t *dst, int w, int h, int row_stride,
                     float time_sec) {
  // sanity guards
  if (!t) {
    debug("no text to render\n");
    return;
  }
  if (!t->font) {
    debug("no text font selected\n");
    return;
  }
  if (t->_glyph_count == 0) {
    debug("no glyphs (characters) to render\n");
    return;
  }
  if (!dst) {
    debug("no destination buffer\n");
    return;
  }
  if (w < 0 || h < 0 || row_stride < 0) {
    debug("invalid destination buffer dimensions\n");
    return;
  }

  // Honor size changes even if the field was written directly without using the
  // setter
  _sdf_text_apply_size(t);

  // Apply absolute-time animation. time_sec==0.0 is valid (start position).
  if ((t->dir_x != 0.0f || t->dir_y != 0.0f) && t->speed != 0.0f) {
    sdf_text_animate(t, time_sec);
  }

  // Ensure shaping/placement is up-to-date
  if (t->shape_dirty) {
    _sdf_text_update_shape(t);
  }

  const float baseline_y = t->y;

  int max_y = 0;
  int max_x = 0;
  // Main render loop over all glyphs
  for (size_t i = 0; i < t->_glyph_count; ++i) {
    // get glyph for character
    const unsigned ch = t->_glyph_indices[i] & 0xFFu;
    const sdf_glyph_t *g = font_glyph(t->font, ch);
    if (!g || !g->pixels || g->width == 0 || g->height == 0) {
      // TODO: fix rendering code 32 (space) to not use fallback box glyph
      // debug("degenerate glyph for char code %u, skipping\n", ch);
      continue;
    }

    // Top-left of glyph in canvas (floating), baseline origin is (t->x, baseline_y)
    const float gx0f = t->x + (float)t->_ofs_x[i];
    const float gy0f = baseline_y + (float)t->_ofs_y[i];

    // Compute clipped integer bounds on canvas, expand by glow/outline/weight
    float effect_pad = t->effects.glow_radius;
    if (t->effects.outline_width > effect_pad) {
        effect_pad = t->effects.outline_width;
    }
    effect_pad += 1.0f;
    int x0 = MAX(0, (int)floorf(gx0f - effect_pad));
    int y0 = MAX(0, (int)floorf(gy0f - (effect_pad)));
    int x1 = MIN(w, (int)ceilf(gx0f + (float)g->width + effect_pad));
    int y1 = MIN(h, (int)ceilf(gy0f + (float)g->height + effect_pad));


    // skip degenerate glyphs
    if (x0 >= x1 || y0 >= y1) {
      // debug("clipped glyph for char code %u is outside canvas, skipping\n",
      // ch);
      continue;
    }

    Normal W = Normal_clamp(0.5f + (1.0f - t->effects.weight));
    Normal I = W + t->effects.outline_width;
    Normal O = W - t->effects.outline_width;

    RGBA fill = t->color;
    RGBA stroke = t->effects.outline_color;
    RGBA glow = t->effects.glow_color;

    if (y1 > max_y) { max_y = y1; }
    if (x1 > max_x) { max_x = x1; }

    for (int iy = y0; iy < y1; ++iy) {
      uint8_t *p = dst + (iy * row_stride) + (x0 * 4);
      for (int ix = x0; ix < x1; ++ix) {
        // Local coords inside glyph (continuous), pixel center at +0.5
        const float fx = ((float)ix - gx0f) + 0.5f;
        const float fy = ((float)iy - gy0f) + 0.5f;
        const float sample = sdf_sample_gray8_bilinear(g, fx, fy);

        Normal a = sdf_coverage_smooth(sample,  W, t->softness);
        Normal a2 = sdf_outline_slab(sample, I, O, t->effects.outline_smooth);
        Normal a3 = sdf_glow_smooth(sample, I, t->effects.glow_radius);
	    Normal g = step_difference(a3, MAX(a, a2));

        fill.a = (uint8_t)(t->color.a * a);
        // printf("fill: %d, color.a:%d, = %f\n", fill.a, t->color.a, (double)a);
        stroke.a = (uint8_t)(t->effects.outline_color.a * a2);
        glow.a = (uint8_t)(t->effects.glow_color.a * g);
        
        composite_rgba((RGBA*)p, &glow,   (RGBA*)p);
        composite_rgba((RGBA*)p, &fill,   (RGBA*)p);
        composite_rgba((RGBA*)p, &stroke, (RGBA*)p);

        p += 4;
      }
    }
  }

  t->dimensions.y = max_y;
  t->dimensions.x = max_x;
  printf("MAX_Y: %dx%d\n", max_x, max_y);
}

/* Apply per-text font scaling to match t->size_px (interpreted as desired line
 * height). Creates and owns a scaled font derived from base_font. When size_px
 * <= 0, reverts to base_font.
 */
static void _sdf_text_apply_size(sdf_text_t *t) {
  if (!t)
    return;
  if (!t->base_font)
    t->base_font = t->font; // fallback
  if (t->size_px <= 0.0f) {
    if (t->owns_font && t->font && t->font != t->base_font) {
      sdf_font_free(t->font);
    }
    t->font = t->base_font;
    t->owns_font = false;
    return;
  }
  float target = t->size_px;
  float current = (t->font ? t->font->line_height : 0.0f);
  if (fabsf(current - target) < 0.01f)
    return; // already at desired size

  sdf_font_t *scaled = sdf_font_create_scaled(t->base_font, target);
  if (!scaled)
    return;
  if (t->owns_font && t->font && t->font != t->base_font) {
    sdf_font_free(t->font);
  }
  t->font = scaled;
  t->owns_font = true;
  t->shape_dirty = true;
}

/* ------------- Layout helpers ------------- */

static inline const sdf_glyph_t *font_glyph(const sdf_font_t *font,
                                            unsigned ch) {
  if (!font)
    return NULL;
  const sdf_glyph_t *g = NULL;
  if (ch < 256)
    g = font->table[ch];
  if (!g && font->box_valid)
    g = &font->box;
  return g;
}

sdf_line_extents_t sdf_text_compute_line_extents(const sdf_text_t *t) {
  sdf_line_extents_t e = {0, 0, 0};
  if (!t || !t->font || !t->text || t->text_len == 0)
    return e;

  float top = 0.0f;
  float bottom = 0.0f;
  const char *p = t->text;
  for (size_t i = 0; i < t->text_len && p[i] != '\0'; ++i) {
    unsigned ch = (unsigned char)p[i];
    const sdf_glyph_t *g = font_glyph(t->font, ch);
    if (!g)
      continue;
    if (g->bearing_y > top)
      top = g->bearing_y;
    float gb = (float)g->height - g->bearing_y;
    if (gb > bottom)
      bottom = gb;
  }
  e.top = top;
  e.bottom = bottom;
  e.height = top + bottom;
  return e;
}

/* Aggregate measurement of text width and height (in pixels). */
vec2u sdf_text_measure_px(sdf_text_t *t) {
  vec2u sz = {0, 0};
  if (!t)
    return sz;

  // Ensure font scaling and shaping are up-to-date
  _sdf_text_apply_size(t);
  if (t->shape_dirty) {
    _sdf_text_update_shape(t);
  }

  if (!t->font || t->_glyph_count == 0) {
    return sz; // empty
  }

  // Sum advances and track max glyph width if needed
  float total_adv = 0.0f;
  int max_gw = 0;
  for (size_t i = 0; i < t->_glyph_count; ++i) {
    total_adv += (t->_advances ? t->_advances[i] : 0.0f);
    unsigned ch = t->_glyph_indices ? (t->_glyph_indices[i] & 0xFFu) : 0u;
    const sdf_glyph_t *g = font_glyph(t->font, ch);
    if (g && (int)g->width > max_gw)
      max_gw = (int)g->width;
  }

  if (t->orient == SDF_ORIENT_HORIZONTAL) {
    sdf_line_extents_t e = sdf_text_compute_line_extents(t);
    sz.x = (int32_t)ceilf(total_adv);
    sz.y = (int32_t)ceilf(e.height);
  } else {
    // Vertical stack: height accumulates, width is max glyph width
    sz.x = (int32_t)(max_gw);
    sz.y = (int32_t)ceilf(total_adv);
  }
  return sz;
}

float sdf_text_baseline_bottom(float canvas_bottom_y, sdf_line_extents_t e) {
  /* Baseline sits above the bottom by line_bottom */
  return canvas_bottom_y - e.bottom;
}

float sdf_text_baseline_top(float canvas_top_y, sdf_line_extents_t e) {
  /* Baseline sits below the top by line_top */
  return canvas_top_y + e.top;
}

float sdf_text_baseline_center(float canvas_top_y, float canvas_height,
                               sdf_line_extents_t e) {
  /* Centered baseline relative to a box T..T+H */
  return canvas_top_y + 0.5f * (canvas_height + e.top - e.bottom);
}

/* ------------- Task 3.3: Per-glyph placement at baseline ------------- */

/* Internal helper: compute cached glyph indices, per-glyph rounded advances,
 * and per-glyph integer offsets (dx, dy) relative to a baseline origin at
 * (0,0). Final placement for glyph i is: x0 = baseline_x + ofs_x[i] y0 =
 * baseline_y + ofs_y[i] Where ofs_x[i] = pen_x + round(bearing_x), ofs_y[i] = -
 * round(bearing_y) And pen_x accumulates round(advance_width + tracking [+
 * kerning]).
 */
static inline int iroundf(float v) { return (int)lrintf(v); }

static void _sdf_text_update_shape(sdf_text_t *t) {
  if (!t)
    return;
  t->_glyph_count = 0;
  if (!t->font || !t->text || t->text_len == 0) {
    // Clear any previous caches
    if (t->_glyph_indices) {
      free(t->_glyph_indices);
      t->_glyph_indices = NULL;
    }
    if (t->_advances) {
      free(t->_advances);
      t->_advances = NULL;
    }
    if (t->_ofs_x) {
      free(t->_ofs_x);
      t->_ofs_x = NULL;
    }
    if (t->_ofs_y) {
      free(t->_ofs_y);
      t->_ofs_y = NULL;
    }
    t->shape_dirty = false;
    return;
  }

  size_t cap = t->text_len; /* ASCII v1: 1 byte per glyph */
  uint16_t *glyphs = (uint16_t *)malloc(cap * sizeof(uint16_t));
  float *advs = (float *)malloc(cap * sizeof(float));
  int *dxs = (int *)malloc(cap * sizeof(int));
  int *dys = (int *)malloc(cap * sizeof(int));
  if (!glyphs || !advs || !dxs || !dys) {
    free(glyphs);
    free(advs);
    free(dxs);
    free(dys);
    /* leave previous shape as-is */
    return;
  }

  /* Pass 1: collect glyph codes actually used (ASCII) */
  size_t n = 0;
  const char *p = t->text;
  for (size_t i = 0; i < t->text_len && p[i] != '\0'; ++i) {
    unsigned ch = (unsigned char)p[i];
    const sdf_glyph_t *g = font_glyph(t->font, ch);
    if (!g)
      continue; /* extremely unlikely if no fallback */
    glyphs[n++] = (uint16_t)(ch & 0xFFu);
  }

  /* Pass 2: compute adv[i] including tracking and optional kerning with next
   * glyph. Horizontal: use advance_width (+ kerning when enabled). Vertical:
   * advance along Y uses glyph pixel height (no kerning); tracking is extra
   * inter-glyph spacing. */
  for (size_t i = 0; i < n; ++i) {
    unsigned ch = glyphs[i];
    const sdf_glyph_t *g = font_glyph(t->font, ch);
    if (t->orient == SDF_ORIENT_HORIZONTAL) {
      float k = 0.0f;
      if (t->kerning_enable && i + 1 < n) {
        unsigned rn = glyphs[i + 1];
        k = sdf_font_get_kerning(t->font, ch, rn);
      }
      float adv_f = (g ? g->advance_width : 0.0f) + t->tracking + k;
      int adv_i = iroundf(adv_f);
      advs[i] = (float)adv_i;
    } else { // vertical
      int gh = g ? (int)g->height : 0;
      int adv_i = gh + iroundf(t->tracking);
      advs[i] = (float)adv_i;
    }
  }

  /* Pass 3: compute placement using orientation + valign */
  if (t->orient == SDF_ORIENT_HORIZONTAL) {
    /* Horizontal: compute per-string line extents for valign */
    int line_top_i = 0;    /* max(round(bearing_y)) over used glyphs */
    int line_bottom_i = 0; /* max(height - round(bearing_y)) over used glyphs */
    for (size_t i = 0; i < n; ++i) {
      const sdf_glyph_t *gg = font_glyph(t->font, glyphs[i]);
      if (!gg)
        continue;
      int iby = iroundf(gg->bearing_y);
      int gh = (int)gg->height;
      int ib = gh - iby;
      if (ib < 0)
        ib = 0;
      if (iby > line_top_i)
        line_top_i = iby;
      if (ib > line_bottom_i)
        line_bottom_i = ib;
    }
    int pen_x = 0;
    for (size_t i = 0; i < n; ++i) {
      unsigned ch = glyphs[i];
      const sdf_glyph_t *g = font_glyph(t->font, ch);
      int bx = g ? iroundf(g->bearing_x) : 0;
      int by = g ? iroundf(g->bearing_y) : 0;
      dxs[i] = pen_x + bx;
      if (t->valign == SDF_VALIGN_TOP) {
        dys[i] = -line_top_i;
      } else if (t->valign == SDF_VALIGN_BOTTOM) {
        dys[i] = g ? (line_bottom_i - (int)g->height) : 0;
      } else {
        dys[i] = -by;
      }
      pen_x += (int)advs[i];
    }

    

  } else { /* SDF_ORIENT_VERTICAL */
    /* Vertical stack: advance along Y; horizontal placement uses bearing_x
     * baseline. Valign is ignored here (future: introduce a horizontal
     * alignment option). */
    int pen_y = 0;
    for (size_t i = 0; i < n; ++i) {
      unsigned ch = glyphs[i];
      const sdf_glyph_t *g = font_glyph(t->font, ch);
      int bx = g ? iroundf(g->bearing_x) : 0;
      int by = g ? iroundf(g->bearing_y) : 0;
      dxs[i] = bx;
      dys[i] = pen_y - by;
      pen_y += (int)advs[i];
    }

    if (t->wrap) {
        t->wrap_mod = ((float)t->dimensions.y / t->speed) + 1.0f;
    }
  }

  /* Replace caches */
  if (t->_glyph_indices)
    free(t->_glyph_indices);
  if (t->_advances)
    free(t->_advances);
  if (t->_ofs_x)
    free(t->_ofs_x);
  if (t->_ofs_y)
    free(t->_ofs_y);
  t->_glyph_indices = glyphs;
  t->_advances = advs;
  t->_ofs_x = dxs;
  t->_ofs_y = dys;
  t->_glyph_count = n;
  t->shape_dirty = false;
  printf(" [!] updated text shape: glyphs=%zu\n", n);
}


/**
 * simple function to move the animation forward by time_sec
 * this has the effect of updating t->x and t->y by dir * speed * tmp_sec
 *  
 */
void sdf_text_animate(sdf_text_t *t, const float time_sec) {
  if (!t)
    return;
  /* Absolute time-based position: pos(t) = (x0,y0) + dir*speed*time */
  float tmp_sec = time_sec;
  if (tmp_sec > t->wrap_mod) {
    tmp_sec = fmodf(tmp_sec, t->wrap_mod);
  }
  t->x = t->x0 + t->dir_x * t->speed * tmp_sec;
  t->y = t->y0 + t->dir_y * t->speed * tmp_sec;
}


/* Bilinear sample on grayscale glyph image at continuous coords (fx, fy),
 * where pixel centers are at integer+0.5. Returns normalized [0,1]. */
static inline float sdf_sample_gray8_bilinear(const sdf_glyph_t *glyph,
                                              float fx, float fy) {
  if (!glyph->pixels || glyph->width <= 0 || glyph->height <= 0)
    return 0.0f;

  const float scale = 1.0f / 255.0f;
  uint8_t v = sample_gray8_bilinear_buf(glyph->pixels, glyph->width,
                                        glyph->height, fx, fy);
  return (float)v * scale;
}
