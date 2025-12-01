#include <threads.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "hub75gpu.h"
#include "pixels.h"
#include "gradient.h"
#include "mymath.h"

#include "debug.h"
#ifdef HAVE_LIBPNG
#include <png.h>
#endif


/* per-thread current scene */
#if defined(__STDC_NO_THREADS__)
#  error "need thread local storage support (_Thread_local)"
#endif

/* Allow compile-time flip of front-face convention if needed */
#ifndef FRONT_FACE_CCW
#define FRONT_FACE_CCW 1
#endif

/* Thread-local storage for the current scene being processed by this thread */
static _Thread_local hub75_display_t *tls_scene = NULL;
static _Thread_local scene3d_t *tls_current_os = NULL;
/* When true, skip per-vertex shadow sampling; per-pixel sampling will be applied in the raster loop */
static _Thread_local bool tls_use_pixel_shadows = false;

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

/* Forward decl for clip-space helper used in filled renderer culling */
static inline vec4 mat4_mul_point_clip_xyw(const mat4 m, const vec3 p);
static inline float tri_orientation_clip_xyw(vec4 v0, vec4 v1, vec4 v2);

/* --- Minimal struct to carry clip-space vertex + per-vertex color during clipping --- */
typedef struct {
    vec4 clip;   /* clip-space position (x,y,z,w) */
    RGB  color;  /* Gouraud per-vertex color */
    vec3 world;  /* world-space position for debug overlays */
} ClipVert;

/* Sutherland–Hodgman against near plane (OpenGL clip space): z + w >= 0 */
static inline int clip_polygon_near(const ClipVert *in, int in_count, ClipVert *out, int out_capacity) {
    if (in_count <= 0) return 0;
    int out_count = 0;

    ClipVert S = in[in_count - 1];
    float fS = S.clip.z + S.clip.w;
    for (int i = 0; i < in_count; ++i) {
        ClipVert E = in[i];
        float fE = E.clip.z + E.clip.w;
        bool Sin = (fS >= 0.0f);
        bool Ein = (fE >= 0.0f);
        if (Sin && Ein) {
            /* both inside: keep E */
            if (out_count < out_capacity) out[out_count++] = E;
        } else if (Sin && !Ein) {
            /* S in, E out: keep intersection */
            float denom = (fS - fE);
            float t = (fabsf(denom) > 1e-12f) ? (fS / denom) : 0.0f;
            if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
            ClipVert I;
            I.clip.x = S.clip.x + t * (E.clip.x - S.clip.x);
            I.clip.y = S.clip.y + t * (E.clip.y - S.clip.y);
            I.clip.z = S.clip.z + t * (E.clip.z - S.clip.z);
            I.clip.w = S.clip.w + t * (E.clip.w - S.clip.w);
            /* Interpolate color linearly along the edge */
            I.color.r = (uint8_t)lrintf((float)S.color.r + t * ((float)E.color.r - (float)S.color.r));
            I.color.g = (uint8_t)lrintf((float)S.color.g + t * ((float)E.color.g - (float)S.color.g));
            I.color.b = (uint8_t)lrintf((float)S.color.b + t * ((float)E.color.b - (float)S.color.b));
            /* Interpolate world-space for debug */
            I.world.x = S.world.x + t * (E.world.x - S.world.x);
            I.world.y = S.world.y + t * (E.world.y - S.world.y);
            I.world.z = S.world.z + t * (E.world.z - S.world.z);
            if (out_count < out_capacity) out[out_count++] = I;
        } else if (!Sin && Ein) {
            /* S out, E in: keep intersection then E */
            float denom = (fS - fE);
            float t = (fabsf(denom) > 1e-12f) ? (fS / denom) : 0.0f;
            if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
            ClipVert I;
            I.clip.x = S.clip.x + t * (E.clip.x - S.clip.x);
            I.clip.y = S.clip.y + t * (E.clip.y - S.clip.y);
            I.clip.z = S.clip.z + t * (E.clip.z - S.clip.z);
            I.clip.w = S.clip.w + t * (E.clip.w - S.clip.w);
            I.color.r = (uint8_t)lrintf((float)S.color.r + t * ((float)E.color.r - (float)S.color.r));
            I.color.g = (uint8_t)lrintf((float)S.color.g + t * ((float)E.color.g - (float)S.color.g));
            I.color.b = (uint8_t)lrintf((float)S.color.b + t * ((float)E.color.b - (float)S.color.b));
            I.world.x = S.world.x + t * (E.world.x - S.world.x);
            I.world.y = S.world.y + t * (E.world.y - S.world.y);
            I.world.z = S.world.z + t * (E.world.z - S.world.z);
            if (out_count < out_capacity) out[out_count++] = I;
            if (out_count < out_capacity) out[out_count++] = E;
        } else {
            /* both outside: keep nothing */
        }
        S = E; fS = fE;
    }
    return out_count;
}

/* ---------------- Shadows: column-major math helpers (match 3d.c) ---------------- */
typedef struct { int w, h; float *depth; mat4 VP; mat4 V; float z_bias; bool valid; } ShadowMap;
static _Thread_local ShadowMap *tls_shadow_maps = NULL;
static _Thread_local uint16_t tls_shadow_count = 0;
/* Debug: deferred shadowmap dump request (per-thread) */
static _Thread_local int tls_dump_sm_index = -1;
static _Thread_local char tls_dump_sm_path[256];

/* Simple 2x2 PCF sampler in shadow map NDC space */
static inline float sm_sample_pcf2x2(const ShadowMap *SM, float x_ndc, float y_ndc, float z_ndc) {
    if (!SM || !SM->valid || !SM->depth) return 1.0f;
    /* Map NDC [-1,1] to SM pixel coordinates [0, w-1]/[0, h-1] without rounding */
    float sx_f = (x_ndc * 0.5f + 0.5f) * (float)(SM->w - 1);
    float sy_f = (y_ndc * 0.5f + 0.5f) * (float)(SM->h - 1);
    int ix = (int)floorf(sx_f);
    int iy = (int)floorf(sy_f);
    float acc = 0.0f; int taps = 0;
    for (int dy = 0; dy < 2; ++dy) {
        int sy = iy + dy;
        if ((unsigned)sy >= (unsigned)SM->h) continue;
        for (int dx = 0; dx < 2; ++dx) {
            int sx = ix + dx;
            if ((unsigned)sx >= (unsigned)SM->w) continue;
            size_t sidx = (size_t)sy * (size_t)SM->w + (size_t)sx;
            float map_z = SM->depth[sidx];
            acc += (z_ndc > map_z + SM->z_bias) ? 0.0f : 1.0f;
            taps++;
        }
    }
    return (taps > 0) ? (acc / (float)taps) : 1.0f;
}

static inline mat4 cm_m4_identity(void){ mat4 r = { .m={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1} }; return r; }
static inline mat4 cm_m4_mul(mat4 a, mat4 b){
    mat4 r = (mat4){0};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            r.m[col*4 + row] =
                a.m[0*4 + row] * b.m[col*4 + 0] +
                a.m[1*4 + row] * b.m[col*4 + 1] +
                a.m[2*4 + row] * b.m[col*4 + 2] +
                a.m[3*4 + row] * b.m[col*4 + 3];
        }
    }
    return r;
}
static inline vec4 cm_m4_mul_point4(const mat4 m, const vec3 p){
    float x = m.m[0]*p.x + m.m[4]*p.y + m.m[8]*p.z + m.m[12];
    float y = m.m[1]*p.x + m.m[5]*p.y + m.m[9]*p.z + m.m[13];
    float z = m.m[2]*p.x + m.m[6]*p.y + m.m[10]*p.z + m.m[14];
    float w = m.m[3]*p.x + m.m[7]*p.y + m.m[11]*p.z + m.m[15];
    return (vec4){x,y,z,w};
}
static inline vec3 cm_m4_mul_point3(const mat4 m, const vec3 p){
    vec4 v = cm_m4_mul_point4(m, p);
    return (vec3){v.x, v.y, v.z};
}

/* Build 8 world-space corners of the camera view frustum */
static inline void camera_frustum_corners_ws(const camera_t *cam, vec3 out8[8]){
    /* Camera basis */
    vec3 fwd = v3_sub(cam->target, cam->position);
    float fl = sqrtf(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
    if (fl > 1e-6f) { fwd.x/=fl; fwd.y/=fl; fwd.z/=fl; } else { fwd = (vec3){0,0,-1}; }
    vec3 upn = (vec3){ cam->up.x, cam->up.y, cam->up.z };
    float ul = sqrtf(upn.x*upn.x + upn.y*upn.y + upn.z*upn.z);
    if (ul > 1e-6f) { upn.x/=ul; upn.y/=ul; upn.z/=ul; } else { upn = (vec3){0,1,0}; }
    vec3 right = v3_norm(v3_cross(fwd, upn));
    vec3 true_up = v3_cross(right, fwd);

    float zn = cam->z_near > 1e-6f ? cam->z_near : 0.01f;
    float zf = cam->z_far > zn ? cam->z_far : zn+1.0f;
    float tanY = tanf(cam->fov_y * 0.5f);
    float hN = 2.0f * tanY * zn;
    float wN = hN * cam->aspect;
    float hF = 2.0f * tanY * zf;
    float wF = hF * cam->aspect;

    vec3 cN = (vec3){ cam->position.x + fwd.x*zn, cam->position.y + fwd.y*zn, cam->position.z + fwd.z*zn };
    vec3 cF = (vec3){ cam->position.x + fwd.x*zf, cam->position.y + fwd.y*zf, cam->position.z + fwd.z*zf };
    vec3 rN = (vec3){ right.x*(wN*0.5f), right.y*(wN*0.5f), right.z*(wN*0.5f) };
    vec3 uN = (vec3){ true_up.x*(hN*0.5f), true_up.y*(hN*0.5f), true_up.z*(hN*0.5f) };
    vec3 rF = (vec3){ right.x*(wF*0.5f), right.y*(wF*0.5f), right.z*(wF*0.5f) };
    vec3 uF = (vec3){ true_up.x*(hF*0.5f), true_up.y*(hF*0.5f), true_up.z*(hF*0.5f) };

    /* Near plane */
    out8[0] = (vec3){ cN.x - rN.x - uN.x, cN.y - rN.y - uN.y, cN.z - rN.z - uN.z }; /* nbl */
    out8[1] = (vec3){ cN.x + rN.x - uN.x, cN.y + rN.y - uN.y, cN.z + rN.z - uN.z }; /* nbr */
    out8[2] = (vec3){ cN.x - rN.x + uN.x, cN.y - rN.y + uN.y, cN.z - rN.z + uN.z }; /* ntl */
    out8[3] = (vec3){ cN.x + rN.x + uN.x, cN.y + rN.y + uN.y, cN.z + rN.z + uN.z }; /* ntr */
    /* Far plane */
    out8[4] = (vec3){ cF.x - rF.x - uF.x, cF.y - rF.y - uF.y, cF.z - rF.z - uF.z }; /* fbl */
    out8[5] = (vec3){ cF.x + rF.x - uF.x, cF.y + rF.y - uF.y, cF.z + rF.z - uF.z }; /* fbr */
    out8[6] = (vec3){ cF.x - rF.x + uF.x, cF.y - rF.y + uF.y, cF.z - rF.z + uF.z }; /* ftl */
    out8[7] = (vec3){ cF.x + rF.x + uF.x, cF.y + rF.y + uF.y, cF.z + rF.z + uF.z }; /* ftr */
}

#ifdef HAVE_LIBPNG
static bool write_shadowmap_png(const char *filename, const ShadowMap *SM) {
    if (!SM || !SM->depth || SM->w <= 0 || SM->h <= 0) return false;
    /* Compute auto-contrast range over valid samples (z < 1e8) */
    int total = SM->w * SM->h;
    int valid = 0;
    float zmin = 1e9f, zmax = -1e9f;
    for (int i = 0; i < total; ++i) {
        float z = SM->depth[i];
        if (z < 1e8f) {
            valid++;
            if (z < zmin) zmin = z;
            if (z > zmax) zmax = z;
        }
    }
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return false; }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_write_struct(&png_ptr, NULL); fclose(fp); return false; }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp); return false;
    }
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, (png_uint_32)SM->w, (png_uint_32)SM->h,
                 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    uint8_t *row = (uint8_t*)malloc((size_t)SM->w);
    if (!row) { png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp); return false; }
    const bool use_autorange = (valid > 0 && zmax > zmin);
    /* Optional clamp to NDC range for visualization */
    const float vis_min = -1.0f, vis_max = 1.0f;
    for (int y = 0; y < SM->h; ++y) {
        for (int x = 0; x < SM->w; ++x) {
            float z = SM->depth[(size_t)y*(size_t)SM->w + (size_t)x];
            float v = 1.0f;
            if (z < 1e8f) {
                /* Clamp for display, then map */
                if (z < vis_min) z = vis_min; if (z > vis_max) z = vis_max;
                if (use_autorange) {
                    /* Auto-contrast to [0,1]; invert so closer-to-light appears brighter */
                    float t = (z - zmin) / (zmax - zmin);
                    if (t < 0.f) t = 0.f; if (t > 1.f) t = 1.f;
                    v = 1.0f - t;
                } else {
                    /* Fallback to fixed mapping [-1,1] -> [0,1] */
                    v = 1.0f - (z * 0.5f + 0.5f);
                    if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
                }
            } else {
                /* Unwritten -> pure white */
                v = 1.0f;
            }
            row[x] = (uint8_t)lrintf(v * 255.0f);
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return true;
}
#endif

hub75_error_t api_scene3d_dump_shadowmap_png(uint16_t light_index, const char *filepath) {
    if (!filepath) return HUB75_ERR_NULL_PARAM;
    tls_dump_sm_index = (int)light_index;
    snprintf(tls_dump_sm_path, sizeof(tls_dump_sm_path), "%s", filepath);
    return HUB75_OK;
}

/* Rasterize a world-space triangle into a directional light's shadow map (orthographic).
   Depth written is light-view z; lower is nearer to light (use min-compare). */
static void shadowmap_rasterize_triangle(ShadowMap *SM, vec3 v0w, vec3 v1w, vec3 v2w) {
    /* Write into the SM as long as the buffer exists; 'valid' will be set after population */
    if (!SM || !SM->depth || SM->w <= 0 || SM->h <= 0) return;
    /* Transform to clip for screen mapping and to light-view for depth */
    vec4 c0 = cm_m4_mul_point4(SM->VP, v0w);
    vec4 c1 = cm_m4_mul_point4(SM->VP, v1w);
    vec4 c2 = cm_m4_mul_point4(SM->VP, v2w);
    float iw0 = fabsf(c0.w) > 1e-6f ? 1.0f / c0.w : 1.0f;
    float iw1 = fabsf(c1.w) > 1e-6f ? 1.0f / c1.w : 1.0f;
    float iw2 = fabsf(c2.w) > 1e-6f ? 1.0f / c2.w : 1.0f;
    float x0 = c0.x * iw0, y0 = c0.y * iw0;
    float x1 = c1.x * iw1, y1 = c1.y * iw1;
    float x2 = c2.x * iw2, y2 = c2.y * iw2;
    /* Map to SM pixels */
    float sx0f = (x0 * 0.5f + 0.5f) * (float)(SM->w - 1);
    float sy0f = (y0 * 0.5f + 0.5f) * (float)(SM->h - 1);
    float sx1f = (x1 * 0.5f + 0.5f) * (float)(SM->w - 1);
    float sy1f = (y1 * 0.5f + 0.5f) * (float)(SM->h - 1);
    float sx2f = (x2 * 0.5f + 0.5f) * (float)(SM->w - 1);
    float sy2f = (y2 * 0.5f + 0.5f) * (float)(SM->h - 1);

    int sx0 = (int)floorf(sx0f + 0.5f);
    int sy0 = (int)floorf(sy0f + 0.5f);
    int sx1 = (int)floorf(sx1f + 0.5f);
    int sy1 = (int)floorf(sy1f + 0.5f);
    int sx2 = (int)floorf(sx2f + 0.5f);
    int sy2 = (int)floorf(sy2f + 0.5f);

    /* Compute bounding box */
    int minx = sx0, maxx = sx0, miny = sy0, maxy = sy0;
    if (sx1 < minx) minx = sx1;
    if (sx1 > maxx) maxx = sx1;
    if (sx2 < minx) minx = sx2;
    if (sx2 > maxx) maxx = sx2;
    if (sy1 < miny) miny = sy1;
    if (sy1 > maxy) maxy = sy1;
    if (sy2 < miny) miny = sy2;
    if (sy2 > maxy) maxy = sy2;
    if (maxx < 0 || maxy < 0 || minx >= SM->w || miny >= SM->h) return;
    if (minx < 0) minx = 0;
    if (miny < 0) miny = 0;
    if (maxx >= SM->w) maxx = SM->w - 1;
    if (maxy >= SM->h) maxy = SM->h - 1;

    /* Use light NDC depth (post-projection z/w) so depth test and bias are well-defined */
    float z0 = c0.z * iw0;
    float z1 = c1.z * iw1;
    float z2 = c2.z * iw2;

    /* Set up edge functions in float (sufficient for SM size) */
    float Ax = (float)(sy1 - sy2), Ay = (float)(sx2 - sx1), Ac = (float)(sx1*sy2 - sx2*sy1);
    float Bx = (float)(sy2 - sy0), By = (float)(sx0 - sx2), Bc = (float)(sx2*sy0 - sx0*sy2);
    float Cx = (float)(sy0 - sy1), Cy = (float)(sx1 - sx0), Cc = (float)(sx0*sy1 - sx1*sy0);
    float area2 = Ax * (float)sx0 + Ay * (float)sy0 + Ac;
    if (fabsf(area2) < 1e-6f) return; /* degenerate */
    if (area2 < 0.0f) { Ax=-Ax; Ay=-Ay; Ac=-Ac; Bx=-Bx; By=-By; Bc=-Bc; Cx=-Cx; Cy=-Cy; Cc=-Cc; area2 = -area2; }
    float inv_area = 1.0f / area2;

    for (int y = miny; y <= maxy; ++y) {
        float yf = (float)y + 0.5f;
        for (int x = minx; x <= maxx; ++x) {
            float xf = (float)x + 0.5f;
            float E0 = Ax*xf + Ay*yf + Ac;
            float E1 = Bx*xf + By*yf + Bc;
            float E2 = Cx*xf + Cy*yf + Cc;
            if (E0 >= 0.0f && E1 >= 0.0f && E2 >= 0.0f) {
                float w0 = E0 * inv_area;
                float w1 = E1 * inv_area;
                float w2 = 1.0f - w0 - w1;
                float z = w0*z0 + w1*z1 + w2*z2;
                size_t idx = (size_t)y * (size_t)SM->w + (size_t)x;
                if (z < SM->depth[idx]) SM->depth[idx] = z;
            }
        }
    }
}

/* -------- Helpers for filled renderer -------- */
static inline bool is_backface_ndc(const vec3 v0, const vec3 v1, const vec3 v2){
    float ax = v1.x - v0.x, ay = v1.y - v0.y;
    float bx = v2.x - v0.x, by = v2.y - v0.y;
    float area = ax * by - ay * bx;
    return (area < 0.0f);
}

/* Sample shadow map visibility at a world-space point for a given light index.
   Returns 1.0f if visible, 0.0f if occluded, or 1.0f if no shadow map exists. */
static float sample_shadow_visibility(uint16_t li, vec3 v_world) {
    if (!tls_shadow_maps || li >= tls_shadow_count) return 1.0f;
    ShadowMap *SM = &tls_shadow_maps[li];
    if (!SM || !SM->valid || !SM->depth) return 1.0f;
    vec4 cclip = cm_m4_mul_point4(SM->VP, v_world);
    float invw = (fabsf(cclip.w) > 1e-6f) ? (1.0f / cclip.w) : 1.0f;
    float x_ndc = cclip.x * invw;
    float y_ndc = cclip.y * invw;
    float z_ndc = cclip.z * invw;
    int sx = (int)((x_ndc * 0.5f + 0.5f) * (float)(SM->w - 1) + 0.5f);
    int sy = (int)((y_ndc * 0.5f + 0.5f) * (float)(SM->h - 1) + 0.5f);
    if ((unsigned)sx < (unsigned)SM->w && (unsigned)sy < (unsigned)SM->h) {
        size_t sidx = (size_t)sy * (size_t)SM->w + (size_t)sx;
    float map_z = SM->depth[sidx];
    return (z_ndc > map_z + SM->z_bias) ? 0.0f : 1.0f;
    }
    return 1.0f;
}

/* Compute shaded color for a face using face normal + centroid shadow test (matches prior behavior) */
static RGB compute_face_shaded_color(size_t face_index,
                                     const object_t *obj,
                                     const scene3d_lighting_t *lighting,
                                     const mat4 M_model,
                                     const float N3[9]){
    RGB base_rgb = (face_index < obj->edge_colors->length) ? obj->edge_colors->list[face_index] : (RGB){255,255,255};
    if (!lighting) return base_rgb;

    vec3 n_obj = (obj->normals && face_index < obj->normals->length) ? obj->normals->list[face_index] : (vec3){0,0,1};
    vec3 n_world = mat3_mul_vec3(N3, n_obj);
    float n_len = sqrtf(n_world.x*n_world.x + n_world.y*n_world.y + n_world.z*n_world.z);
    if (n_len > 1e-6f) { n_world.x/=n_len; n_world.y/=n_len; n_world.z/=n_len; }

    float br = base_rgb.r/255.0f, bg = base_rgb.g/255.0f, bb = base_rgb.b/255.0f;
    float lr = lighting->ambient.r, lg = lighting->ambient.g, lb = lighting->ambient.b;

    /* World-space centroid for shadow test */
    vec3 face = obj->faces->list[face_index];
    vec3 v0w = cm_m4_mul_point3(M_model, obj->verticies->list[(uint16_t)face.x]);
    vec3 v1w = cm_m4_mul_point3(M_model, obj->verticies->list[(uint16_t)face.y]);
    vec3 v2w = cm_m4_mul_point3(M_model, obj->verticies->list[(uint16_t)face.z]);
    vec3 cw = { (v0w.x+v1w.x+v2w.x)/3.0f, (v0w.y+v1w.y+v2w.y)/3.0f, (v0w.z+v1w.z+v2w.z)/3.0f };

    for (uint16_t li = 0; li < lighting->num_lights; ++li) {
        const light_t *L = &lighting->lights[li];
        if (L->intensity <= 0.0f) continue;
        if (L->type == LIGHT_DIRECTIONAL) {
            float Lx = -L->direction.x, Ly = -L->direction.y, Lz = -L->direction.z;
            float Llen = sqrtf(Lx*Lx + Ly*Ly + Lz*Lz);
            if (Llen > 1e-6f) { Lx/=Llen; Ly/=Llen; Lz/=Llen; }
            float ndotl = n_world.x*Lx + n_world.y*Ly + n_world.z*Lz;
            if (ndotl > 0.0f) {
                float vis = 1.0f;
                if (!tls_use_pixel_shadows && tls_shadow_maps && li < tls_shadow_count && obj->shadow_enabled && L->casts_shadows && L->shadow_enabled) {
                    ShadowMap *SM = &tls_shadow_maps[li];
                    if (SM && SM->valid && SM->depth) {
                        vec4 cclip = cm_m4_mul_point4(SM->VP, cw);
                        float invw = (fabsf(cclip.w) > 1e-6f) ? (1.0f / cclip.w) : 1.0f;
                        float x_ndc = cclip.x * invw;
                        float y_ndc = cclip.y * invw;
                        float z_ndc = cclip.z * invw;
                        int sx = (int)((x_ndc * 0.5f + 0.5f) * (float)(SM->w - 1) + 0.5f);
                        int sy = (int)((y_ndc * 0.5f + 0.5f) * (float)(SM->h - 1) + 0.5f);
                        if ((unsigned)sx < (unsigned)SM->w && (unsigned)sy < (unsigned)SM->h) {
                            size_t sidx = (size_t)sy * (size_t)SM->w + (size_t)sx;
                            float map_z = SM->depth[sidx];
                            if (z_ndc > map_z + SM->z_bias) {
                                vis = 0.0f;
                            }
                        }
                    }
                }
                if (vis > 0.0f) {
                    lr += L->color.r * L->intensity * ndotl;
                    lg += L->color.g * L->intensity * ndotl;
                    lb += L->color.b * L->intensity * ndotl;
                }
            }
        }
    }

    float cr = fminf(fmaxf(br * lr, 0.0f), 1.0f);
    float cg = fminf(fmaxf(bg * lg, 0.0f), 1.0f);
    float cb = fminf(fmaxf(bb * lb, 0.0f), 1.0f);
    RGB shaded_rgb;
    shaded_rgb.r = (uint8_t)(cr * 255.0f);
    shaded_rgb.g = (uint8_t)(cg * 255.0f);
    shaded_rgb.b = (uint8_t)(cb * 255.0f);
    return shaded_rgb;
}

/* Compute shaded color for a vertex using its world-space position and normal */
static RGB compute_vertex_shaded_color(RGB base_rgb,
                                       vec3 v_world,
                                       vec3 n_world,
                                       vec3 cam_pos,
                                       const object_t *obj,
                                       const scene3d_lighting_t *lighting){
    if (!lighting) return base_rgb;

    /* normalize normal */
    float nlen = sqrtf(n_world.x*n_world.x + n_world.y*n_world.y + n_world.z*n_world.z);
    if (nlen > 1e-6f) { n_world.x/=nlen; n_world.y/=nlen; n_world.z/=nlen; }

    float br = base_rgb.r/255.0f, bg = base_rgb.g/255.0f, bb = base_rgb.b/255.0f;
    /* Split diffuse/ambient (Ld) and specular (Ls) so specular is NOT modulated by albedo */
    float Ld_r = lighting->ambient.r, Ld_g = lighting->ambient.g, Ld_b = lighting->ambient.b;
    float Ls_r = 0.0f, Ls_g = 0.0f, Ls_b = 0.0f;

    /* view vector for specular */
    float Vx = cam_pos.x - v_world.x;
    float Vy = cam_pos.y - v_world.y;
    float Vz = cam_pos.z - v_world.z;
    float Vlen = sqrtf(Vx*Vx + Vy*Vy + Vz*Vz);
    if (Vlen > 1e-6f) { Vx/=Vlen; Vy/=Vlen; Vz/=Vlen; }

    const float ks = (obj ? obj->specular_strength : 0.0f);
    const float shin = (obj ? obj->specular_shininess : 32.0f);

    for (uint16_t li = 0; li < lighting->num_lights; ++li) {
        const light_t *L = &lighting->lights[li];
        if (L->intensity <= 0.0f) continue;
        if (L->type == LIGHT_DIRECTIONAL) {
            float Lx = -L->direction.x, Ly = -L->direction.y, Lz = -L->direction.z;
            float Llen = sqrtf(Lx*Lx + Ly*Ly + Lz*Lz);
            if (Llen > 1e-6f) { Lx/=Llen; Ly/=Llen; Lz/=Llen; }
            float ndotl = n_world.x*Lx + n_world.y*Ly + n_world.z*Lz;
            if (ndotl > 0.0f) {
                float vis = 1.0f;
                if (!tls_use_pixel_shadows && tls_shadow_maps && li < tls_shadow_count && obj->shadow_enabled && L->casts_shadows && L->shadow_enabled) {
                    ShadowMap *SM = &tls_shadow_maps[li];
                    if (SM && SM->valid && SM->depth) {
                        vec4 cclip = cm_m4_mul_point4(SM->VP, v_world);
                        float invw = (fabsf(cclip.w) > 1e-6f) ? (1.0f / cclip.w) : 1.0f;
                        float x_ndc = cclip.x * invw;
                        float y_ndc = cclip.y * invw;
                        float z_ndc = cclip.z * invw;
                        int sx = (int)((x_ndc * 0.5f + 0.5f) * (float)(SM->w - 1) + 0.5f);
                        int sy = (int)((y_ndc * 0.5f + 0.5f) * (float)(SM->h - 1) + 0.5f);
                        if ((unsigned)sx < (unsigned)SM->w && (unsigned)sy < (unsigned)SM->h) {
                            size_t sidx = (size_t)sy * (size_t)SM->w + (size_t)sx;
                            float map_z = SM->depth[sidx];
                            if (z_ndc > map_z + SM->z_bias) {
                                vis = 0.0f;
                            }
                        }
                    }
                }
                if (vis > 0.0f) {
                    /* Diffuse */
                    float diff = ndotl;
                    Ld_r += L->color.r * L->intensity * diff;
                    Ld_g += L->color.g * L->intensity * diff;
                    Ld_b += L->color.b * L->intensity * diff;

                    /* Blinn-Phong specular (per-vertex) */
                    if (ks > 0.0f) {
                        float Hx = Lx + Vx;
                        float Hy = Ly + Vy;
                        float Hz = Lz + Vz;
                        float Hlen = sqrtf(Hx*Hx + Hy*Hy + Hz*Hz);
                        if (Hlen > 1e-6f) { Hx/=Hlen; Hy/=Hlen; Hz/=Hlen; }
                        float ndoth = n_world.x*Hx + n_world.y*Hy + n_world.z*Hz;
                        if (ndoth > 0.0f) {
                            float spec = ks * powf(ndoth, shin) * L->intensity;
                            Ls_r += L->color.r * spec;
                            Ls_g += L->color.g * spec;
                            Ls_b += L->color.b * spec;
                        }
                    }
                }
            }
        }
    }

    /* Final color: albedo * (ambient+diffuse) + specular */
    float cr = fminf(fmaxf(br * Ld_r + Ls_r, 0.0f), 1.0f);
    float cg = fminf(fmaxf(bg * Ld_g + Ls_g, 0.0f), 1.0f);
    float cb = fminf(fmaxf(bb * Ld_b + Ls_b, 0.0f), 1.0f);
    RGB shaded_rgb;
    shaded_rgb.r = (uint8_t)(cr * 255.0f);
    shaded_rgb.g = (uint8_t)(cg * 255.0f);
    shaded_rgb.b = (uint8_t)(cb * 255.0f);
    return shaded_rgb;
}
static inline mat4 cm_m4_look_at(vec3 eye, vec3 target, vec3 up){
    vec3 f = v3_norm((vec3){ target.x - eye.x, target.y - eye.y, target.z - eye.z });
    vec3 s = v3_norm(v3_cross(f, up));
    vec3 u = v3_cross(s, f);
    mat4 r = cm_m4_identity();
    r.m[0] = s.x;   r.m[4] = s.y;   r.m[8]  = s.z;   r.m[12] = -(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    r.m[1] = u.x;   r.m[5] = u.y;   r.m[9]  = u.z;   r.m[13] = -(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    r.m[2] = -f.x;  r.m[6] = -f.y;  r.m[10] = -f.z;  r.m[14] =  (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    r.m[3] = 0.0f;  r.m[7] = 0.0f;  r.m[11] = 0.0f;  r.m[15] = 1.0f;
    return r;
}
static inline mat4 cm_m4_ortho(float l, float r, float b, float t, float n, float f){
    mat4 m = (mat4){0};
    m.m[0] = 2.0f/(r-l);
    m.m[5] = 2.0f/(t-b);
    /* Map z in [n,f] to NDC [-1,1]: z_ndc = (2z - (f+n)) / (f-n) */
    m.m[10] =  2.0f/(f-n);
    m.m[12] = -(r+l)/(r-l);
    m.m[13] = -(t+b)/(t-b);
    m.m[14] = -(f+n)/(f-n);
    m.m[15] = 1.0f;
    return m;
}
static inline vec3 pick_up_from_dir(vec3 dir){
    vec3 a = {0,1,0};
    float d = fabsf(v3_dot(dir, a));
    if (d > 0.9f) a = (vec3){1,0,0};
    return v3_norm(v3_cross(v3_cross(a, dir), dir));
}

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
static hub75_error_t api_clear() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }
    if (tls_scene->frame_buffer.data) {
        size_t img_size = (size_t)(tls_scene->frame_buffer.dimensions.x *
                                   tls_scene->frame_buffer.dimensions.y * 4);
        memset(tls_scene->frame_buffer.data, 0, img_size);
    }
    return HUB75_OK;
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
static hub75_error_t api_pixel(uint16_t x, uint16_t y, RGBA color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }
    draw_pixel(tls_scene, x, y, color);
    return HUB75_OK;
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
static hub75_error_t api_line(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }
    RGBA pixel = {color.r, color.g, color.b, 255};
    draw_line(tls_scene, x1, y1, x2, y2, pixel);
    return HUB75_OK;
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
static hub75_error_t api_line_aa(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }
    RGBA pixel = {color.r, color.g, color.b, 255};
    draw_line_aa(tls_scene, x1, y1, x2, y2, pixel);
    return HUB75_OK;
}


/**
 * @brief Begin a new frame for drawing operations
 * 
 * Acquires a new frame buffer from the ring buffer for drawing operations.
 * This must be called before any drawing operations and paired with api_frame_end().
 * Waits up to 10ms for a buffer to become available.
 * 
 * Sets tls_scene->frame_ready to false and updates tls_scene->frame_buffer.data pointer.
 */
static hub75_error_t api_frame_begin() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }
    if (!tls_scene->frame_ready) {
        debug("previous frame not released\n");
        return HUB75_ERR_INVALID_COORDS;  // reusing error code for invalid state
    }
    uint8_t *image = NULL;
    image = spsc_push_ptr_begin(tls_scene->ring_buf_mapper, 200);
    for(int i = 0; i < 10 && image == NULL; i++) {
        usleep(1000);
        image = spsc_push_ptr_begin(tls_scene->ring_buf_mapper, 200);
    }
    if (image == NULL) {
        debug("timed out waiting for frame buffer\n");
        return HUB75_ERR_OUT_OF_MEMORY;  // reusing error code for timeout/buffer unavailable
    }
    tls_scene->frame_ready = false;

    // Update frame_buffer.data (primary) and legacy image pointer (compatibility shim)
    tls_scene->frame_buffer.data = (RGBA*)image;
    tls_scene->image = image;  // TODO: remove once migration complete
    return HUB75_OK;
}

/**
 * @brief Complete the current frame and submit it for display
 * 
 * Finalizes the current frame and commits it to the ring buffer for display.
 * This must be called after api_frame_begin() and all drawing operations are complete.
 * 
 * Sets tls_scene->frame_ready to true and commits the buffer to the mapper.
 */
static hub75_error_t api_frame_end() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }

    if (tls_scene->frame_ready) {
        debug("begin frame not called\n");
        return HUB75_ERR_INVALID_COORDS;  // reusing error code for invalid state
    }

    tls_scene->frame_ready = true;
    spsc_push_ptr_commit(tls_scene->ring_buf_mapper);
    return HUB75_OK;
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
    const int32_t w = scene->frame_buffer.dimensions.x;
    const int32_t h = scene->frame_buffer.dimensions.y;
    if ((unsigned)y >= (unsigned)h) return;
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (x1 < 0 || x0 >= w) return;

    x0 = clamp_int(x0, 0, w - 1);
    x1 = clamp_int(x1, 0, w - 1);

    RGBA *row = scene->frame_buffer.data + (y * w);
    for (int x = x0; x <= x1; ++x) {
        row[x].r = c.r;
        row[x].g = c.g;
        row[x].b = c.b;
    }
}

/**
 * @brief Fill a horizontal span of pixels with RGBA color and alpha compositing
 * 
 * For alpha == 255 (fully opaque), uses fast direct assignment.
 * For alpha < 255, performs proper alpha compositing per pixel.
 */
static hub75_error_t fill_span_rgba(hub75_display_t *scene, int y, int x0, int x1, RGBA c) {
    const int32_t w = scene->frame_buffer.dimensions.x;
    const int32_t h = scene->frame_buffer.dimensions.y;
    if ((unsigned)y >= (unsigned)h) return HUB75_ERR_INVALID_COORDS;
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (x1 < 0 || x0 >= w) return HUB75_ERR_INVERTED_COORDS;

    x0 = clamp_int(x0, 0, w - 1);
    x1 = clamp_int(x1, 0, w - 1);

    RGBA *row = scene->frame_buffer.data + (y * w);
    
    // Fast path for fully opaque pixels
    if (c.a == 255) {
        for (int x = x0; x <= x1; ++x) {
            row[x] = c;
        }
    }
    // Alpha compositing for semi-transparent pixels
    else if (c.a > 0) {
        const uint32_t alpha = c.a;
        const uint32_t inv_alpha = 255 - alpha;
        
        for (int x = x0; x <= x1; ++x) {
            RGBA *dst = &row[x];
            dst->r = (uint8_t)((c.r * alpha + dst->r * inv_alpha) / 255);
            dst->g = (uint8_t)((c.g * alpha + dst->g * inv_alpha) / 255);
            dst->b = (uint8_t)((c.b * alpha + dst->b * inv_alpha) / 255);
            dst->a = (uint8_t)(alpha + (dst->a * inv_alpha) / 255);
        }
    }
    // alpha == 0: no-op, fully transparent

    return HUB75_OK;
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
void polygon_fill(hub75_display_t *scene, Polygonf_t *poly, RGBA color)
{
    if (!scene || !scene->frame_buffer.data || !poly || poly->num_points < 3) {
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
            fill_span_rgba(scene, y, x_start, x_end, color);
        }
    }
}

/* Gouraud-shaded triangle rasterization (normalized coords -> screen)
 * Integer edge functions for inside test + fixed-point color interpolation for speed.
 * Optional per-pixel Z-buffer test using provided z-buffer (uint16, near=0; far=65535).
 */
static inline void draw_triangle_gouraud(hub75_display_t *scene,
                                         const _TriFill *tri_in,
                                         uint16_t *zbuf,
                                         int zbuf_width) {
    if (!scene || !scene->frame_buffer.data || !tri_in) return;
    const Polygonf_t *poly = &tri_in->poly;
    if (poly->num_points != 3) return;
    const RGB *vcolor = tri_in->vcolor;
    const uint16_t *zv = tri_in->z16;
    const bool use_z = (zbuf != NULL);

    /* Fast path only allowed when Z-buffer is disabled */
    if (!use_z) {
        if (vcolor[0].r == vcolor[1].r && vcolor[0].g == vcolor[1].g && vcolor[0].b == vcolor[1].b &&
            vcolor[0].r == vcolor[2].r && vcolor[0].g == vcolor[2].g && vcolor[0].b == vcolor[2].b) {
                const RGBA tcolor = { vcolor[0].r, vcolor[0].g, vcolor[0].b, 255 };
            polygon_fill(scene, (Polygonf_t*)poly, tcolor);
            return;
        }
    }

    /* Convert to pixel space */
    int x[3], y[3];
    for (int i = 0; i < 3; ++i) {
        x[i] = norm_to_px(poly->points[i].x, scene->width);
        y[i] = norm_to_px(poly->points[i].y, scene->height);
    }

    /* Compute triangle bounding box */
    int minx = x[0], maxx = x[0];
    int miny = y[0], maxy = y[0];
    for (int i = 1; i < 3; ++i) {
        if (x[i] < minx) { minx = x[i]; }
        if (x[i] > maxx) { maxx = x[i]; }
        if (y[i] < miny) { miny = y[i]; }
        if (y[i] > maxy) { maxy = y[i]; }
    }
    /* Clamp to framebuffer */
    minx = clamp_int(minx, 0, scene->width - 1);
    maxx = clamp_int(maxx, 0, scene->width - 1);
    miny = clamp_int(miny, 0, scene->height - 1);
    maxy = clamp_int(maxy, 0, scene->height - 1);
    if (minx > maxx || miny > maxy) return;

    /* Integer edge functions E(x,y) = A*x + B*y + C
       Define edges for barycentric weights w0,w1,w2 corresponding to vertices v0,v1,v2.
       E0 is edge v1->v2 evaluated at (x,y), etc. */
    int x0 = x[0], y0 = y[0];
    int x1 = x[1], y1 = y[1];
    int x2 = x[2], y2 = y[2];

    int A0 = (y1 - y2), B0 = (x2 - x1), C0 = x1*y2 - x2*y1; /* w0 */
    int A1 = (y2 - y0), B1 = (x0 - x2), C1 = x2*y0 - x0*y2; /* w1 */
    int A2 = (y0 - y1), B2 = (x1 - x0), C2 = x0*y1 - x1*y0; /* w2 */

    int area2 = A0 * x0 + B0 * y0 + C0;
    if (area2 == 0) return; /* degenerate */
    if (area2 < 0) {
        /* Normalize so area is positive and inside test is >= 0 */
        A0 = -A0; B0 = -B0; C0 = -C0;
        A1 = -A1; B1 = -B1; C1 = -C1;
        A2 = -A2; B2 = -B2; C2 = -C2;
        area2 = -area2;
    }

    /* Top-left rule classification per edge (after normalization)
       Edge considered top-left if A>0 or (A==0 and B<0). */
    const int tl0 = (A0 > 0) || (A0 == 0 && B0 < 0);
    const int tl1 = (A1 > 0) || (A1 == 0 && B1 < 0);
    const int tl2 = (A2 > 0) || (A2 == 0 && B2 < 0);

    /* Fixed-point scale (8 fractional bits) for sampling at pixel centers (x+0.5,y+0.5) */
    const int FP = 8;
    const int ONE = 1 << FP;         /* 256 */
    const int HALF = ONE >> 1;       /* 128 */

    /* Precompute edge increments in fixed-point */
    int dE0dx = A0 * ONE, dE0dy = B0 * ONE;
    int dE1dx = A1 * ONE, dE1dy = B1 * ONE;
    int dE2dx = A2 * ONE, dE2dy = B2 * ONE;

    /* Evaluate edges at top-left sample point (minx+0.5, miny+0.5) in fixed-point */
    int X = (minx << FP) + HALF;
    int Y = (miny << FP) + HALF;
    int C0fp = C0 * ONE, C1fp = C1 * ONE, C2fp = C2 * ONE;
    int E0_row = A0 * X + B0 * Y + C0fp;
    int E1_row = A1 * X + B1 * Y + C1fp;
    int E2_row = A2 * X + B2 * Y + C2fp;

    /* Precompute color interpolation increments (fixed-point 8 fractional bits) */
    float inv_area2 = 1.0f / (float)area2;
    float r0 = (float)vcolor[0].r, g0 = (float)vcolor[0].g, b0 = (float)vcolor[0].b;
    float r1 = (float)vcolor[1].r, g1 = (float)vcolor[1].g, b1 = (float)vcolor[1].b;
    float r2 = (float)vcolor[2].r, g2 = (float)vcolor[2].g, b2 = (float)vcolor[2].b;

    float w2dx_f = -(float)A0 * inv_area2 - (float)A1 * inv_area2; /* since A2 = -A0 - A1 */
    float w2dy_f = -(float)B0 * inv_area2 - (float)B1 * inv_area2; /* since B2 = -B0 - B1 */
    float dw0dx_f = (float)A0 * inv_area2, dw0dy_f = (float)B0 * inv_area2;
    float dw1dx_f = (float)A1 * inv_area2, dw1dy_f = (float)B1 * inv_area2;

    float drdx_f = dw0dx_f*r0 + dw1dx_f*r1 + w2dx_f*r2;
    float dgdx_f = dw0dx_f*g0 + dw1dx_f*g1 + w2dx_f*g2;
    float dbdx_f = dw0dx_f*b0 + dw1dx_f*b1 + w2dx_f*b2;
    float drdy_f = dw0dy_f*r0 + dw1dy_f*r1 + w2dy_f*r2;
    float dgdy_f = dw0dy_f*g0 + dw1dy_f*g1 + w2dy_f*g2;
    float dbdy_f = dw0dy_f*b0 + dw1dy_f*b1 + w2dy_f*b2;

    int drdx = (int)lrintf(drdx_f * (float)ONE);
    int dgdx = (int)lrintf(dgdx_f * (float)ONE);
    int dbdx = (int)lrintf(dbdx_f * (float)ONE);
    int drdy = (int)lrintf(drdy_f * (float)ONE);
    int dgdy = (int)lrintf(dgdy_f * (float)ONE);
    int dbdy = (int)lrintf(dbdy_f * (float)ONE);

    /* Starting color at (minx+0.5, miny+0.5) using floats once, then fixed-point accumulation */
    float w0_row_f = ((float)(A0 * minx + B0 * miny) + (float)(A0 + B0) * 0.5f + (float)C0) * inv_area2;
    float w1_row_f = ((float)(A1 * minx + B1 * miny) + (float)(A1 + B1) * 0.5f + (float)C1) * inv_area2;
    float w2_row_f = 1.0f - w0_row_f - w1_row_f;
    int r_row = (int)lrintf((w0_row_f*r0 + w1_row_f*r1 + w2_row_f*r2) * (float)ONE);
    int g_row = (int)lrintf((w0_row_f*g0 + w1_row_f*g1 + w2_row_f*g2) * (float)ONE);
    int b_row = (int)lrintf((w0_row_f*b0 + w1_row_f*b1 + w2_row_f*b2) * (float)ONE);

    const int32_t fb_width = scene->frame_buffer.dimensions.x;

    /* Per-pixel shadow inputs using camera w for perspective-correct interpolation */
    const bool do_px_shadow = (tri_in->per_pixel_shadow && tls_shadow_maps && tri_in->sm_light_index < tls_shadow_count);
    const ShadowMap *SM = (do_px_shadow ? &tls_shadow_maps[tri_in->sm_light_index] : NULL);
    float tx0=0, ty0=0, tz0=0, tw0=0, iwc0=1.0f;
    float tx1=0, ty1=0, tz1=0, tw1=0, iwc1=1.0f;
    float tx2=0, ty2=0, tz2=0, tw2=0, iwc2=1.0f;
    float dtxdx=0, dtydx=0, dtzdx=0, dtwdx=0, diwcdx=0;
    float dtxdy=0, dtydy=0, dtzdy=0, dtwdy=0, diwcdy=0;
    float tx_row=0, ty_row=0, tz_row=0, tw_row=0, iwc_row=1.0f;
    if (do_px_shadow && SM && SM->valid && SM->depth) {
        /* Prepare attributes divided by camera clip w */
        vec4 lc0 = tri_in->sm_light_clip[0]; float wc0 = tri_in->cam_w[0]; float invwc0 = (fabsf(wc0)>1e-6f)? 1.0f/wc0 : 1.0f;
        vec4 lc1 = tri_in->sm_light_clip[1]; float wc1 = tri_in->cam_w[1]; float invwc1 = (fabsf(wc1)>1e-6f)? 1.0f/wc1 : 1.0f;
        vec4 lc2 = tri_in->sm_light_clip[2]; float wc2 = tri_in->cam_w[2]; float invwc2 = (fabsf(wc2)>1e-6f)? 1.0f/wc2 : 1.0f;
        tx0 = lc0.x * invwc0; ty0 = lc0.y * invwc0; tz0 = lc0.z * invwc0; tw0 = lc0.w * invwc0; iwc0 = invwc0;
        tx1 = lc1.x * invwc1; ty1 = lc1.y * invwc1; tz1 = lc1.z * invwc1; tw1 = lc1.w * invwc1; iwc1 = invwc1;
        tx2 = lc2.x * invwc2; ty2 = lc2.y * invwc2; tz2 = lc2.z * invwc2; tw2 = lc2.w * invwc2; iwc2 = invwc2;

        dtxdx = dw0dx_f*tx0 + dw1dx_f*tx1 + w2dx_f*tx2;
        dtydx = dw0dx_f*ty0 + dw1dx_f*ty1 + w2dx_f*ty2;
        dtzdx = dw0dx_f*tz0 + dw1dx_f*tz1 + w2dx_f*tz2;
        dtwdx = dw0dx_f*tw0 + dw1dx_f*tw1 + w2dx_f*tw2;
        diwcdx= dw0dx_f*iwc0+ dw1dx_f*iwc1+ w2dx_f*iwc2;
        dtxdy = dw0dy_f*tx0 + dw1dy_f*tx1 + w2dy_f*tx2;
        dtydy = dw0dy_f*ty0 + dw1dy_f*ty1 + w2dy_f*ty2;
        dtzdy = dw0dy_f*tz0 + dw1dy_f*tz1 + w2dy_f*tz2;
        dtwdy = dw0dy_f*tw0 + dw1dy_f*tw1 + w2dy_f*tw2;
        diwcdy= dw0dy_f*iwc0+ dw1dy_f*iwc1+ w2dy_f*iwc2;

        tx_row = w0_row_f*tx0 + w1_row_f*tx1 + w2_row_f*tx2;
        ty_row = w0_row_f*ty0 + w1_row_f*ty1 + w2_row_f*ty2;
        tz_row = w0_row_f*tz0 + w1_row_f*tz1 + w2_row_f*tz2;
        tw_row = w0_row_f*tw0 + w1_row_f*tw1 + w2_row_f*tw2;
        iwc_row= w0_row_f*iwc0+ w1_row_f*iwc1+ w2_row_f*iwc2;
    }

    /* Optional depth interpolation setup (use uint16 range in float domain) */
    int dzdx = 0, dzdy = 0, z_row = 0;
    if (use_z) {
        float z0 = (float)zv[0], z1 = (float)zv[1], z2 = (float)zv[2];
        float dzdx_f = dw0dx_f*z0 + dw1dx_f*z1 + w2dx_f*z2;
        float dzdy_f = dw0dy_f*z0 + dw1dy_f*z1 + w2dy_f*z2;
        dzdx = (int)lrintf(dzdx_f);
        dzdy = (int)lrintf(dzdy_f);
        z_row = (int)lrintf(w0_row_f*z0 + w1_row_f*z1 + w2_row_f*z2);
    }
    for (int py = miny; py <= maxy; ++py) {
        RGBA *p = scene->frame_buffer.data + py * fb_width + minx;
        uint16_t *pz = use_z ? (zbuf + (size_t)py * (size_t)zbuf_width + (size_t)minx) : NULL;
        int E0 = E0_row, E1 = E1_row, E2 = E2_row;
        int rfp = r_row, gfp = g_row, bfp = b_row;
        int zfp = z_row;
        for (int px = minx; px <= maxx; ++px) {
            if ( (E0 > 0 || (E0 == 0 && tl0)) &&
                 (E1 > 0 || (E1 == 0 && tl1)) &&
                 (E2 > 0 || (E2 == 0 && tl2)) ) {
                if (use_z) {
                    int zcl = zfp;
                    if (zcl < 0) zcl = 0; else if (zcl > 65535) zcl = 65535;
                    if (zcl < (int)*pz) {
                        int r8 = rfp >> FP; if (r8 < 0) r8 = 0; else if (r8 > 255) r8 = 255;
                        int g8 = gfp >> FP; if (g8 < 0) g8 = 0; else if (g8 > 255) g8 = 255;
                        int b8 = bfp >> FP; if (b8 < 0) b8 = 0; else if (b8 > 255) b8 = 255;
                        if (do_px_shadow && SM) {
                            float invw_sum = iwc_row;
                            if (fabsf(invw_sum) > 1e-6f) {
                                float x_lc = tx_row / invw_sum;
                                float y_lc = ty_row / invw_sum;
                                float z_lc = tz_row / invw_sum;
                                float w_lc = tw_row / invw_sum;
                                float x_ndc = x_lc / w_lc;
                                float y_ndc = y_lc / w_lc;
                                float z_ndc = z_lc / w_lc;
                                float vis = sm_sample_pcf2x2(SM, x_ndc, y_ndc, z_ndc);
                                r8 = (int)lrintf(vis * (float)r8);
                                g8 = (int)lrintf(vis * (float)g8);
                                b8 = (int)lrintf(vis * (float)b8);
                            }
                        }
                        p->r = (uint8_t)r8; p->g = (uint8_t)g8; p->b = (uint8_t)b8;
                        *pz = (uint16_t)zcl;
                    }
                } else {
                    int r8 = rfp >> FP; if (r8 < 0) r8 = 0; else if (r8 > 255) r8 = 255;
                    int g8 = gfp >> FP; if (g8 < 0) g8 = 0; else if (g8 > 255) g8 = 255;
                    int b8 = bfp >> FP; if (b8 < 0) b8 = 0; else if (b8 > 255) b8 = 255;
                    if (do_px_shadow && SM) {
                        float invw_sum = iwc_row;
                        if (fabsf(invw_sum) > 1e-6f) {
                            float x_lc = tx_row / invw_sum;
                            float y_lc = ty_row / invw_sum;
                            float z_lc = tz_row / invw_sum;
                            float w_lc = tw_row / invw_sum;
                            float x_ndc = x_lc / w_lc;
                            float y_ndc = y_lc / w_lc;
                            float z_ndc = z_lc / w_lc;
                            float vis = sm_sample_pcf2x2(SM, x_ndc, y_ndc, z_ndc);
                            r8 = (int)lrintf(vis * (float)r8);
                            g8 = (int)lrintf(vis * (float)g8);
                            b8 = (int)lrintf(vis * (float)b8);
                        }
                    }
                    p->r = (uint8_t)r8; p->g = (uint8_t)g8; p->b = (uint8_t)b8;
                }
            }
            E0 += dE0dx; E1 += dE1dx; E2 += dE2dx;
            rfp += drdx; gfp += dgdx; bfp += dbdx;
            if (use_z) { zfp += dzdx; pz++; }
            if (do_px_shadow && SM) { tx_row += dtxdx; ty_row += dtydx; tz_row += dtzdx; tw_row += dtwdx; iwc_row += diwcdx; }
            p++;
        }
        E0_row += dE0dy; E1_row += dE1dy; E2_row += dE2dy;
        r_row += drdy; g_row += dgdy; b_row += dbdy;
        if (use_z) { z_row += dzdy; }
        if (do_px_shadow && SM) { tx_row += dtxdy; ty_row += dtydy; tz_row += dtzdy; tw_row += dtwdy; iwc_row += diwcdy; }
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

    float a = 0.0;
    size_t n = poly->num_points;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        float xi = poly->points[i].x;
        float yi = poly->points[i].y;
        float xj = poly->points[j].x;
        float yj = poly->points[j].y;
        a += xj * yi - xi * yj;
    }

    /* treat tiny areas as degenerate to avoid jitter on almost-collinear input */
    const float eps = 1e-10f;
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
 * This is a wrapper around polygon_gradient() that uses the thread-local scene.
 * Checks that a scene is set before attempting to draw.
 */
hub75_error_t api_poly_gradient(Polygonf_t *poly, SimpleGradient gradient) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }

    polygon_gradient(tls_scene, poly, gradient);
    return HUB75_OK;
}


/**
 * @brief Fill a horizontal span with a simple gradient using the current thread-local scene
 */
hub75_error_t api_fill_gradient(int y, int x0, int x1, 
                                             const SimpleGradient *gradient, 
                                             int minx, int miny, int maxx, int maxy) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }

    gradient_fill(tls_scene, y, x0, x1, gradient, minx, miny, maxx, maxy);
    return HUB75_OK;
}

/**
 * @brief Draw a polygon using the current thread-local scene
 * 
 * @param poly Polygon with normalized coordinates to draw
 * @param color1 Color for counter-clockwise winding
 * @param color2 Color for clockwise winding
 * 
 * Public API function that draws a polygon using the current thread's scene.
 * This is a wrapper around polygon_fill() that uses the thread-local scene.
 * Checks that a scene is set before attempting to draw.
 */
hub75_error_t api_poly(Polygonf_t *poly, RGBA color1) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }

    polygon_fill(tls_scene, poly, color1);
    return HUB75_OK;
}

/**
 * @brief Request a graceful shutdown of the current scene's rendering
 */
hub75_error_t api_shutdown() {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return HUB75_ERR_NO_SCENE;
    }
    hub75_display_request_shutdown(tls_scene);
    return HUB75_OK;
}

void api_pixel_alpha (int x, int y, RGBA pixel) {
    if (tls_scene == NULL) {
        debug("no scene set\n");
        return;
    }
    draw_pixel_alpha(tls_scene, x, y, pixel);
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

object_t *api_new_plane(uint16_t width_segments, uint16_t height_segments, bool face_up) {
    /* Default previous behavior: plane faces +Y (face_up=true) */
    return object_plane(width_segments, height_segments, face_up);
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

    if (!tls_scene || !cam || !obj || !obj_xform) {
        debug("api_geo_render_wire: invalid parameters\n");
        return;
    }

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
        
        RGBA edge_pixel = {obj->edge_colors->list[i].r, obj->edge_colors->list[i].g, obj->edge_colors->list[i].b, 255};
        draw_line(tls_scene, (uint16_t)x1, (uint16_t)y1, (uint16_t)x2, (uint16_t)y2, edge_pixel);
    }

    if (front_face) free(front_face);
}


/* qsort comparator: sort by depth descending (far to near) */
static int _cmp_trifill_desc(const void *a, const void *b) {
    const _TriFill *ta = (const _TriFill*)a;
    const _TriFill *tb = (const _TriFill*)b;
    if (ta->depth < tb->depth) return 1;   /* b before a */
    if (ta->depth > tb->depth) return -1;  /* a before b */
    return 0;
}

void api_geo_render_filled(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene3d_lighting_t *lighting) {
    if (!tls_scene || !cam || !obj || !obj_xform || !obj->faces || !obj->verticies) {
        debug("api_geo_render_filled: invalid parameters\n");
        return;
    }
    mat4 mvp = camera_project(cam, obj_xform);

    /* Transform all vertices to NDC */
    transform_mesh_to_ndc(obj->verticies->list, obj->verticies->length, mvp, obj->rendered_vertices);

    /* Precompute world normal matrix only if lighting is provided */
    float N3[9];
    mat4 M_model = cm_m4_identity();
    if (lighting) {
        M_model = model_matrix(obj_xform);
        normal_matrix_from_model(M_model, N3);
    }

    /* Decide if we will use per-pixel shadowing for this object (pick first valid caster) */
    bool saved_tls_pp = tls_use_pixel_shadows;
    int selected_light = -1;
    if (obj->shadow_enabled && lighting && tls_shadow_maps && tls_shadow_count > 0) {
        for (uint16_t li = 0; li < lighting->num_lights; ++li) {
            const light_t *L = &lighting->lights[li];
            if (!L) continue;
            if (L->type != LIGHT_DIRECTIONAL) continue;
            if (!L->casts_shadows || !L->shadow_enabled) continue;
            if (li >= tls_shadow_count) continue;
            ShadowMap *SM = &tls_shadow_maps[li];
            if (!SM || !SM->valid || !SM->depth) continue;
            selected_light = (int)li;
            break;
        }
    }
    tls_use_pixel_shadows = (selected_light >= 0);

    size_t nf = obj->faces->length;
    /* Ensure vertex normals exist once */
    if (obj->vertex_normals && !obj->vertex_normals_ready) {
        object_build_vertex_normals(obj);
    }
    /* With near-plane clipping, a triangle can split into 2. Reserve 2x faces. */
    size_t needed_tri_capacity = nf * 2u;
    if (!obj->trifill_buffer || obj->trifill_capacity < needed_tri_capacity) {
        void *newbuf = realloc(obj->trifill_buffer, sizeof(_TriFill) * needed_tri_capacity);
        if (!newbuf) return;
        obj->trifill_buffer = newbuf;
        obj->trifill_capacity = needed_tri_capacity;
    }
    _TriFill *tri = (_TriFill*)obj->trifill_buffer;
    size_t tcount = 0;

    for (size_t i = 0; i < nf; i++) {
        vec3 face = obj->faces->list[i];
        /* Robust backface culling in clip-space XYW (pre-divide) */
        if (obj->cull_backface) {
            vec3 p0 = obj->verticies->list[(uint16_t)face.x];
            vec3 p1 = obj->verticies->list[(uint16_t)face.y];
            vec3 p2 = obj->verticies->list[(uint16_t)face.z];
            vec4 c0 = mat4_mul_point_clip_xyw(mvp, p0);
            vec4 c1 = mat4_mul_point_clip_xyw(mvp, p1);
            vec4 c2 = mat4_mul_point_clip_xyw(mvp, p2);
            float orient = tri_orientation_clip_xyw(c0, c1, c2);
            bool back = FRONT_FACE_CCW ? (orient <= 0.0f) : (orient >= 0.0f);
            if (back) continue;
        }
        /* Base material color per face (matches prior behavior) */
        RGB base_rgb = (i < obj->edge_colors->length) ? obj->edge_colors->list[i] : (RGB){255,255,255};
        /* Compute per-vertex world positions and normals, then shade (for original triangle) */
        RGB base_vcol[3];
        vec4 clip_in[3];
        uint16_t idx[3] = { (uint16_t)face.x, (uint16_t)face.y, (uint16_t)face.z };
        for (int vi = 0; vi < 3; ++vi) {
            vec3 vw = cm_m4_mul_point3(M_model, obj->verticies->list[idx[vi]]);
            vec3 n_obj = obj->vertex_normals && obj->vertex_normals->list ? obj->vertex_normals->list[idx[vi]] : (vec3){0,0,1};
            vec3 nw = mat3_mul_vec3(N3, n_obj);
            base_vcol[vi] = compute_vertex_shaded_color(base_rgb, vw, nw, cam->position, obj, lighting);
            /* clip-space position for clipping */
            clip_in[vi] = mat4_mul_point_clip_xyw(mvp, obj->verticies->list[idx[vi]]);
            /* world position into clip vertex for debug */
            /* Note: clip_in is vec4, but we store world in a parallel ClipVert below */
        }

        /* Build input polygon for near-plane clip */
        ClipVert poly_in[3];
        for (int vi = 0; vi < 3; ++vi) {
            poly_in[vi].clip = clip_in[vi];
            poly_in[vi].color = base_vcol[vi];
            /* pass world-space position for accurate debug sampling */
            poly_in[vi].world = cm_m4_mul_point3(M_model, obj->verticies->list[idx[vi]]);
        }
        ClipVert poly_out[8];
        int m = clip_polygon_near(poly_in, 3, poly_out, 8);
        if (m < 3) continue;

        /* Triangulate fan: (0,i,i+1) for i=1..m-2 */
        for (int k = 1; k + 1 < m; ++k) {
            ClipVert a = poly_out[0];
            ClipVert b = poly_out[k];
            ClipVert c = poly_out[k+1];

            _TriFill t;
            t.poly.num_points = 3;

            /* Convert to NDC and then to normalized 0..1 coords */
            vec3 ndc[3];
            vec4 cv[3] = { a.clip, b.clip, c.clip };
            RGB  col[3] = { a.color, b.color, c.color };
            for (int vi = 0; vi < 3; ++vi) {
                float iw = (fabsf(cv[vi].w) > 1e-12f) ? (1.0f / cv[vi].w) : 1.0f;
                ndc[vi].x = cv[vi].x * iw;
                ndc[vi].y = cv[vi].y * iw;
                ndc[vi].z = cv[vi].z * iw;
            }
            t.poly.points[0] = (vec2){ 0.5f * (ndc[0].x + 1.0f), 0.5f * (ndc[0].y + 1.0f) };
            t.poly.points[1] = (vec2){ 0.5f * (ndc[1].x + 1.0f), 0.5f * (ndc[1].y + 1.0f) };
            t.poly.points[2] = (vec2){ 0.5f * (ndc[2].x + 1.0f), 0.5f * (ndc[2].y + 1.0f) };
            t.vcolor[0] = col[0];
            t.vcolor[1] = col[1];
            t.vcolor[2] = col[2];
            /* Map NDC z [-1,1] -> [0..65535] for depth buffer */
            for (int vi = 0; vi < 3; ++vi) {
                float z01 = ndc[vi].z * 0.5f + 0.5f;
                int zi = (int)lrintf(z01 * 65535.0f);
                if (zi < 0) zi = 0; else if (zi > 65535) zi = 65535;
                t.z16[vi] = (uint16_t)zi;
            }
            /* Populate per-vertex shadow visibility for debug overlay (first matching light) */
            t.debug_vis_valid = false;
            if (tls_current_os && tls_current_os->debug.overlay_shadow_vis) {
                uint16_t li = tls_current_os->debug.shadow_vis_light;
                if (tls_shadow_maps && li < tls_shadow_count) {
                    /* Use world-space of the clipped triangle vertices for accurate sampling */
                    t.debug_vis[0] = (uint8_t)lrintf(sample_shadow_visibility(li, a.world) * 255.0f);
                    t.debug_vis[1] = (uint8_t)lrintf(sample_shadow_visibility(li, b.world) * 255.0f);
                    t.debug_vis[2] = (uint8_t)lrintf(sample_shadow_visibility(li, c.world) * 255.0f);
                    t.debug_vis_valid = true;
                }
            }
            /* Per-pixel shadow payload (single light index) */
            if (selected_light >= 0) {
                ShadowMap *SM = &tls_shadow_maps[(uint16_t)selected_light];
                if (SM && SM->valid && SM->depth) {
                    t.per_pixel_shadow = true;
                    t.sm_light_index = (uint8_t)selected_light;
                    vec3 wpos[3] = { a.world, b.world, c.world };
                    for (int vi = 0; vi < 3; ++vi) {
                        t.cam_w[vi] = cv[vi].w;
                        vec4 lc = cm_m4_mul_point4(SM->VP, wpos[vi]);
                        t.sm_light_clip[vi] = lc;
                    }
                } else {
                    t.per_pixel_shadow = false;
                }
            } else {
                t.per_pixel_shadow = false;
            }
            /* Use average ndc.z as triangle depth hint (optional, used only for debug prints) */
            t.depth = (ndc[0].z + ndc[1].z + ndc[2].z) / 3.0f;
            tri[tcount++] = t;
        }
    }

    /* With Z-buffering, triangle order doesn't matter; skip painter's sort entirely */
    for (size_t i = 0; i < tcount; ++i) {
        /*
        if (tls_scene && tls_scene->enhanced_debug) {
            printf("Filling triangle p1 (%.3f, %.3f), p2 (%.3f, %.3f), p3 (%.3f, %.3f) depth=%.3f\n",
                (double)tri[i].poly.points[0].x, (double)tri[i].poly.points[0].y,
                (double)tri[i].poly.points[1].x, (double)tri[i].poly.points[1].y,
                (double)tri[i].poly.points[2].x, (double)tri[i].poly.points[2].y,
                (double)tri[i].depth);
        }
                */
        /* Gouraud fill using per-vertex colors + Z-buffer if available (from scene3d) */
        uint16_t *zptr = NULL; int zw = 0;
        if (tls_current_os && tls_current_os->zbuffer_enabled) {
            zptr = tls_current_os->zbuf;
            zw = (int)(tls_current_os->zbuf_width);
        }
    draw_triangle_gouraud(tls_scene, &tri[i], zptr, zw);
        /* Debug overlay: shadow visibility per triangle (blend after main draw) */
        if (tls_current_os && tls_current_os->debug.overlay_shadow_vis && tri[i].debug_vis_valid) {
            RGB lit = tls_current_os->debug.shadow_vis_color_lit;
            RGB shd = tls_current_os->debug.shadow_vis_color_shadow;
            float strength = tls_current_os->debug.shadow_vis_strength;
            if (strength > 0.0f) {
                /* Rasterize a simple overlay with Z agreement check */
                const Polygonf_t *poly = &tri[i].poly;
                int x[3], y[3];
                for (int v = 0; v < 3; ++v) {
                    x[v] = norm_to_px(poly->points[v].x, tls_scene->width);
                    y[v] = norm_to_px(poly->points[v].y, tls_scene->height);
                }
                int minx = x[0], maxx = x[0], miny = y[0], maxy = y[0];
                for (int v = 1; v < 3; ++v) {
                    if (x[v] < minx) minx = x[v];
                    if (x[v] > maxx) maxx = x[v];
                    if (y[v] < miny) miny = y[v];
                    if (y[v] > maxy) maxy = y[v];
                }
                minx = clamp_int(minx, 0, tls_scene->width - 1);
                maxx = clamp_int(maxx, 0, tls_scene->width - 1);
                miny = clamp_int(miny, 0, tls_scene->height - 1);
                maxy = clamp_int(maxy, 0, tls_scene->height - 1);

                int x0 = x[0], y0 = y[0];
                int x1 = x[1], y1 = y[1];
                int x2 = x[2], y2 = y[2];
                int A0 = (y1 - y2), B0 = (x2 - x1), C0 = x1 * y2 - x2 * y1;
                int A1 = (y2 - y0), B1 = (x0 - x2), C1 = x2 * y0 - x0 * y2;
                int A2 = (y0 - y1), B2 = (x1 - x0), C2 = x0 * y1 - x1 * y0;
                int area2 = A0 * x0 + B0 * y0 + C0;
                if (area2 != 0) {
                    if (area2 < 0) {
                        A0 = -A0; B0 = -B0; C0 = -C0;
                        A1 = -A1; B1 = -B1; C1 = -C1;
                        A2 = -A2; B2 = -B2; C2 = -C2;
                        area2 = -area2;
                    }
                    /* Top-left rule */
                    const int tl0 = (A0 > 0) || (A0 == 0 && B0 < 0);
                    const int tl1 = (A1 > 0) || (A1 == 0 && B1 < 0);
                    const int tl2 = (A2 > 0) || (A2 == 0 && B2 < 0);
                    const int FP = 8, ONE = 1 << FP, HALF = ONE >> 1;
                    int dE0dx = A0 * ONE, dE0dy = B0 * ONE;
                    int dE1dx = A1 * ONE, dE1dy = B1 * ONE;
                    int dE2dx = A2 * ONE, dE2dy = B2 * ONE;
                    int X = (minx << FP) + HALF, Y = (miny << FP) + HALF;
                    int E0_row = A0 * X + B0 * Y + C0 * ONE;
                    int E1_row = A1 * X + B1 * Y + C1 * ONE;
                    int E2_row = A2 * X + B2 * Y + C2 * ONE;

                    float inv_area2 = 1.0f / (float)area2;
                    /* interpolate vis as scalar 0..255 */
                    float v0 = (float)tri[i].debug_vis[0];
                    float v1f = (float)tri[i].debug_vis[1];
                    float v2f = (float)tri[i].debug_vis[2];
                    float dw0dx = (float)A0 * inv_area2, dw0dy = (float)B0 * inv_area2;
                    float dw1dx = (float)A1 * inv_area2, dw1dy = (float)B1 * inv_area2;
                    float w2dx = -dw0dx - dw1dx;
                    float w2dy = -dw0dy - dw1dy;
                    float w0_row = ((float)(A0 * minx + B0 * miny) + (float)(A0 + B0) * 0.5f + (float)C0) * inv_area2;
                    float w1_row = ((float)(A1 * minx + B1 * miny) + (float)(A1 + B1) * 0.5f + (float)C1) * inv_area2;
                    float w2_row = 1.0f - w0_row - w1_row;
                    float vis_row = w0_row * v0 + w1_row * v1f + w2_row * v2f;

                    /* Depth interpolation for Z-agreement */
                    float z0 = (float)tri[i].z16[0];
                    float z1f = (float)tri[i].z16[1];
                    float z2f = (float)tri[i].z16[2];
                    float dzdx = dw0dx * z0 + dw1dx * z1f + w2dx * z2f;
                    float dzdy = dw0dy * z0 + dw1dy * z1f + w2dy * z2f;
                    float z_row = w0_row * z0 + w1_row * z1f + w2_row * z2f;

                    const int32_t fb_w = tls_scene->frame_buffer.dimensions.x;
                    for (int py = miny; py <= maxy; ++py) {
                        RGBA *p = tls_scene->frame_buffer.data + py * fb_w + minx;
                        uint16_t *pz = (zptr ? zptr + (size_t)py * (size_t)zw + (size_t)minx : NULL);
                        int E0 = E0_row, E1 = E1_row, E2 = E2_row;
                        float vis_fp = vis_row;
                        float zfp = z_row;
                        for (int px = minx; px <= maxx; ++px) {
                            if ((E0 > 0 || (E0 == 0 && tl0)) && (E1 > 0 || (E1 == 0 && tl1)) && (E2 > 0 || (E2 == 0 && tl2))) {
                                bool blend_ok = true;
                                if (pz) {
                                    int zcl = (int)lrintf(zfp);
                                    int zb = (int)*pz;
                                    if (zcl < zb - 3 || zcl > zb + 3) blend_ok = false;
                                }
                                if (blend_ok) {
                                    float v = vis_fp * (1.0f / 255.0f);
                                    float lr = (float)shd.r + v * ((float)lit.r - (float)shd.r);
                                    float lg = (float)shd.g + v * ((float)lit.g - (float)shd.g);
                                    float lb = (float)shd.b + v * ((float)lit.b - (float)shd.b);
                                    float s = strength, invs = 1.0f - s;
                                    float r = invs * (float)p->r + s * lr;
                                    float g = invs * (float)p->g + s * lg;
                                    float b = invs * (float)p->b + s * lb;
                                    if (r < 0) r = 0;
                                    if (r > 255) r = 255;
                                    if (g < 0) g = 0;
                                    if (g > 255) g = 255;
                                    if (b < 0) b = 0;
                                    if (b > 255) b = 255;
                                    p->r = (uint8_t)r; p->g = (uint8_t)g; p->b = (uint8_t)b;
                                }
                            }
                            E0 += dE0dx; E1 += dE1dx; E2 += dE2dx;
                            vis_fp += (dw0dx * v0 + dw1dx * v1f + w2dx * v2f);
                            zfp += dzdx;
                            p++; if (pz) ++pz;
                        }
                        E0_row += dE0dy; E1_row += dE1dy; E2_row += dE2dy;
                        vis_row += (dw0dy * v0 + dw1dy * v1f + w2dy * v2f);
                        z_row += dzdy;
                    }
                }
            }
        }
    }

    /* Restore per-pixel shadow TLS flag for other passes */
    tls_use_pixel_shadows = saved_tls_pp;

    /* trifill_buffer is persistent, do not free here */
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
    if (!tls_scene || !cam || !obj || !obj_xform || !obj->verticies || !obj->edges) {
        debug("api_geo_render_wire_clip_cull: invalid parameters\n");
        return;
    }

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

        /* Guard against malformed edge/color arrays */
        RGB edge_col = {255,255,255};
        if (obj->edge_colors && i < obj->edge_colors->length) {
            edge_col = obj->edge_colors->list[i];
        }
        /* Ensure edge indices are within vertex range; skip if out-of-bounds */
        if ((size_t)edge.x >= obj->verticies->length || (size_t)edge.y >= obj->verticies->length) {
            continue;
        }
        RGBA edge_pixel = {edge_col.r, edge_col.g, edge_col.b, 255};
        draw_line_aa(tls_scene, (uint16_t)x1, (uint16_t)y1, (uint16_t)x2, (uint16_t)y2, edge_pixel);
    }

    if (front_face) free(front_face);
    free(clip_xyw);
}

/* Render a list of object instances with per-object draw mode */
static void api_render_scene3d(const camera_t *cam, const scene3d_t *os, const scene3d_lighting_t *lighting) {
    if (!tls_scene || !cam || !os || !os->instances || os->count == 0) {
        debug("api_render_scene3d: invalid parameters\n");
        return;
    }
    /* Prefer explicitly provided lighting; else fall back to scene-owned lighting */
    const scene3d_lighting_t *L = lighting ? lighting : &os->lighting;
    /* Ensure Z-buffer exists and matches current framebuffer size; clear per frame */
    if (tls_scene && ((scene3d_t*)os)->zbuffer_enabled) {
        uint16_t W = tls_scene->width, H = tls_scene->height;
        bool need_alloc = (os->zbuf == NULL) || (os->zbuf_width != W) || (os->zbuf_height != H);
        if (need_alloc) {
            if (((scene3d_t*)os)->zbuf) free(((scene3d_t*)os)->zbuf);
            ((scene3d_t*)os)->zbuf = (uint16_t*)aligned_alloc(16, (size_t)W * (size_t)H * sizeof(uint16_t));
            ((scene3d_t*)os)->zbuf_width = W;
            ((scene3d_t*)os)->zbuf_height = H;
        }
        if (((scene3d_t*)os)->zbuf) {
            memset(((scene3d_t*)os)->zbuf, 0xFF, (size_t)W * (size_t)H * sizeof(uint16_t));
        }
    }
    /* ---------------- Build per-light shadow maps (centroid-based) ---------------- */
    tls_shadow_maps = NULL; tls_shadow_count = 0;
    /* --- Stable shadow map: cache VP/bias in light_t, only recompute if needed --- */
    if (L && L->num_lights > 0) {
        /* Compute camera frustum center for initial light aiming */
        vec3 frustum_ws[8];
        camera_frustum_corners_ws(cam, frustum_ws);
        vec3 bb_min = { +1e9f, +1e9f, +1e9f };
        vec3 bb_max = { -1e9f, -1e9f, -1e9f };
        for (int ci = 0; ci < 8; ++ci) {
            vec3 wp = frustum_ws[ci];
            if (wp.x < bb_min.x) bb_min.x = wp.x;
            if (wp.x > bb_max.x) bb_max.x = wp.x;
            if (wp.y < bb_min.y) bb_min.y = wp.y;
            if (wp.y > bb_max.y) bb_max.y = wp.y;
            if (wp.z < bb_min.z) bb_min.z = wp.z;
            if (wp.z > bb_max.z) bb_max.z = wp.z;
        }
        vec3 bb_center = { (bb_min.x+bb_max.x)*0.5f, (bb_min.y+bb_max.y)*0.5f, (bb_min.z+bb_max.z)*0.5f };
        vec3 bb_extent = { (bb_max.x-bb_min.x)*0.5f, (bb_max.y-bb_min.y)*0.5f, (bb_max.z-bb_min.z)*0.5f };
        float bb_radius = sqrtf(bb_extent.x*bb_extent.x + bb_extent.y*bb_extent.y + bb_extent.z*bb_extent.z);

        tls_shadow_count = L->num_lights;
        tls_shadow_maps = (ShadowMap*)calloc(tls_shadow_count, sizeof(ShadowMap));
        if (!tls_shadow_maps) { tls_shadow_count = 0; }

    const int SM_W = 512, SM_H = 512;
        for (uint16_t li = 0; li < L->num_lights; ++li) {
            light_t *Lt = &L->lights[li];
            ShadowMap *SM = &tls_shadow_maps[li];
            SM->valid = false;
            if (!Lt) continue;
            if (Lt->type != LIGHT_DIRECTIONAL) continue;
            if (Lt->intensity <= 0.0f) continue;
            if (!Lt->casts_shadows || !Lt->shadow_enabled) continue;

            /* Recompute shadow VP every frame for robustness in dynamic scenes */
            Lt->shadow_vp_valid = false;
            if (!Lt->shadow_vp_valid) {
                vec3 dir = v3_norm((vec3){ Lt->direction.x, Lt->direction.y, Lt->direction.z });
                if (fabsf(dir.x)+fabsf(dir.y)+fabsf(dir.z) < 1e-6f) dir = (vec3){0,-1,0};
                     /* For directional lights, derive an eye from direction instead of using position.
                         This avoids "underneath" views when a pose was provided for direction only. */
                    float dist = bb_radius * 2.5f;
                    /* For consistency with shading (which uses -L->direction), render SM looking along -dir */
                    vec3 eye = (vec3){ bb_center.x + dir.x*dist,
                                       bb_center.y + dir.y*dist,
                                       bb_center.z + dir.z*dist };
                    vec3 up = pick_up_from_dir((vec3){ -dir.x, -dir.y, -dir.z });
                mat4 V = cm_m4_look_at(eye, bb_center, up);

                /* Tight-fit: bound only casters intersecting the camera frustum, in light-view space */
                float lxmin=1e9f, lxmax=-1e9f, lymin=1e9f, lymax=-1e9f, lzmin=1e9f, lzmax=-1e9f;
                bool have_bounds = false;
                for (uint16_t oi = 0; oi < os->count; ++oi) {
                    const object_instance_t *inst_b = &os->instances[oi];
                    if (!inst_b || !inst_b->object || !inst_b->xform) continue;
                    /* For bounds, include any mesh (receivers and/or casters) so the SM covers tested pixels */
                    if (!inst_b->object->faces || !inst_b->object->verticies) continue;
                    /* MVP for camera-space culling */
                    mat4 mvp = camera_project(cam, inst_b->xform);
                    mat4 M   = model_matrix(inst_b->xform);
                    vec3 *Vtx= inst_b->object->verticies->list;
                    face_list_t *F = inst_b->object->faces;
                    for (uint16_t fi = 0; fi < F->length; ++fi) {
                        vec3 f = F->list[fi];
                        /* coarse visibility: any vertex inside camera NDC cube? */
                        vec4 c0c = cm_m4_mul_point4(mvp, Vtx[(uint16_t)f.x]);
                        vec4 c1c = cm_m4_mul_point4(mvp, Vtx[(uint16_t)f.y]);
                        vec4 c2c = cm_m4_mul_point4(mvp, Vtx[(uint16_t)f.z]);
                        float iw0c = (fabsf(c0c.w) > 1e-6f) ? (1.0f / c0c.w) : 1.0f;
                        float iw1c = (fabsf(c1c.w) > 1e-6f) ? (1.0f / c1c.w) : 1.0f;
                        float iw2c = (fabsf(c2c.w) > 1e-6f) ? (1.0f / c2c.w) : 1.0f;
                        vec3 n0 = { c0c.x * iw0c, c0c.y * iw0c, c0c.z * iw0c };
                        vec3 n1 = { c1c.x * iw1c, c1c.y * iw1c, c1c.z * iw1c };
                        vec3 n2 = { c2c.x * iw2c, c2c.y * iw2c, c2c.z * iw2c };
                        bool v0_in = (fabsf(n0.x) <= 1.0f && fabsf(n0.y) <= 1.0f && n0.z >= -1.0f && n0.z <= 1.0f);
                        bool v1_in = (fabsf(n1.x) <= 1.0f && fabsf(n1.y) <= 1.0f && n1.z >= -1.0f && n1.z <= 1.0f);
                        bool v2_in = (fabsf(n2.x) <= 1.0f && fabsf(n2.y) <= 1.0f && n2.z >= -1.0f && n2.z <= 1.0f);
                        if (!(v0_in || v1_in || v2_in)) continue;
                        /* include this triangle in light-view bounds */
                        vec3 v0w = cm_m4_mul_point3(M, Vtx[(uint16_t)f.x]);
                        vec3 v1w = cm_m4_mul_point3(M, Vtx[(uint16_t)f.y]);
                        vec3 v2w = cm_m4_mul_point3(M, Vtx[(uint16_t)f.z]);
                        vec3 l0 = cm_m4_mul_point3(V, v0w);
                        vec3 l1 = cm_m4_mul_point3(V, v1w);
                        vec3 l2 = cm_m4_mul_point3(V, v2w);
                        /* expand bounds */
                        if (l0.x < lxmin) lxmin = l0.x;
                        if (l0.x > lxmax) lxmax = l0.x;
                        if (l0.y < lymin) lymin = l0.y;
                        if (l0.y > lymax) lymax = l0.y;
                        if (l0.z < lzmin) lzmin = l0.z;
                        if (l0.z > lzmax) lzmax = l0.z;
                        if (l1.x < lxmin) lxmin = l1.x;
                        if (l1.x > lxmax) lxmax = l1.x;
                        if (l1.y < lymin) lymin = l1.y;
                        if (l1.y > lymax) lymax = l1.y;
                        if (l1.z < lzmin) lzmin = l1.z;
                        if (l1.z > lzmax) lzmax = l1.z;
                        if (l2.x < lxmin) lxmin = l2.x;
                        if (l2.x > lxmax) lxmax = l2.x;
                        if (l2.y < lymin) lymin = l2.y;
                        if (l2.y > lymax) lymax = l2.y;
                        if (l2.z < lzmin) lzmin = l2.z;
                        if (l2.z > lzmax) lzmax = l2.z;
                        have_bounds = true;
                    }
                }

                /* Fallback: if nothing intersected the camera frustum, use frustum-fit */
                if (!have_bounds) {
                    for (int ci=0; ci<8; ++ci) {
                        vec3 lv = cm_m4_mul_point3(V, frustum_ws[ci]);
                        if (lv.x < lxmin) { lxmin = lv.x; }
                        if (lv.x > lxmax) { lxmax = lv.x; }
                        if (lv.y < lymin) { lymin = lv.y; }
                        if (lv.y > lymax) { lymax = lv.y; }
                        if (lv.z < lzmin) { lzmin = lv.z; }
                        if (lv.z > lzmax) { lzmax = lv.z; }
                    }
                }

                /* Apply user-controlled zoom about the light-space center: zoom < 1 shrinks extents */
                float cx = (lxmin + lxmax) * 0.5f;
                float cy = (lymin + lymax) * 0.5f;
                float cz = (lzmin + lzmax) * 0.5f;
                float ex = (lxmax - lxmin) * 0.5f;
                float ey = (lymax - lymin) * 0.5f;
                float ez = (lzmax - lzmin) * 0.5f;
                float zoom = Lt->shadow_zoom;
                if (!(zoom > 0.0f)) zoom = 1.0f; /* guard */
                ex *= zoom; ey *= zoom; ez *= zoom;
                lxmin = cx - ex; lxmax = cx + ex;
                lymin = cy - ey; lymax = cy + ey;
                lzmin = cz - ez; lzmax = cz + ez;

                float span = fmaxf(fmaxf(lxmax-lxmin, lymax-lymin), lzmax-lzmin);
                float pad = 0.02f * span + 1e-4f; /* padding proportional to zoomed span */
                /* Flip Y in ortho so +Y in light space appears at the top of the shadow map image */
                mat4 P = cm_m4_ortho(lxmin-pad, lxmax+pad, lymax+pad, lymin-pad, lzmin-pad, lzmax+pad);
                Lt->shadow_V = V;
                Lt->shadow_P = P;
                Lt->shadow_VP = cm_m4_mul(P, V);
                Lt->shadow_z_bias = 0.0005f * (lzmax - lzmin) + 1e-5f; /* slightly reduced bias for tighter fit */
                Lt->shadow_vp_valid = true;
                if (tls_scene && tls_scene->enhanced_debug) {
                    printf("[shadow] li=%u %s bounds x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f] pad=%.3f bias=%.6f zoom=%.3f SM=%dx%d\n",
                        (unsigned)li, have_bounds ? "TIGHT" : "FRUSTUM",
                        (double)lxmin, (double)lxmax, (double)lymin, (double)lymax,
                        (double)lzmin, (double)lzmax, (double)pad, (double)Lt->shadow_z_bias, (double)zoom, SM_W, SM_H);
                }
            }
            SM->V  = Lt->shadow_V;
            SM->VP = Lt->shadow_VP;
            /* Convert world LV bias into NDC units using |P.m[10]| scaling */
            float proj_scale = fabsf(Lt->shadow_P.m[10]);
            SM->z_bias = proj_scale * Lt->shadow_z_bias;

            SM->w = SM_W; SM->h = SM_H;
            SM->depth = (float*)malloc((size_t)SM_W * (size_t)SM_H * sizeof(float));
            if (!SM->depth) { SM->valid = false; continue; }
            for (int i = 0; i < SM_W*SM_H; ++i) SM->depth[i] = 1e9f;

            int tri_count = 0;
            for (uint16_t oi = 0; oi < os->count; ++oi) {
                const object_instance_t *inst = &os->instances[oi];
                if (!inst || !inst->object || !inst->xform) continue;
                if (!inst->object->shadow_enabled || !inst->object->faces || !inst->object->verticies) continue;
                mat4 M = model_matrix(inst->xform);
                vec3 *Vtx = inst->object->verticies->list;
                face_list_t *F = inst->object->faces;
                for (uint16_t fi = 0; fi < F->length; ++fi) {
                    vec3 f = F->list[fi];
                    vec3 v0w = cm_m4_mul_point3(M, Vtx[(uint16_t)f.x]);
                    vec3 v1w = cm_m4_mul_point3(M, Vtx[(uint16_t)f.y]);
                    vec3 v2w = cm_m4_mul_point3(M, Vtx[(uint16_t)f.z]);
                    shadowmap_rasterize_triangle(SM, v0w, v1w, v2w);
                    tri_count++;
                }
            }
            SM->valid = true;
            if (tls_scene && tls_scene->enhanced_debug) {
                printf("[shadow] li=%u rasterized %d tris into SM\n", (unsigned)li, tri_count);
            }
        }
    }
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

    /* Debug overlay: screen-space checker drawn after rendering */
    if (tls_scene && os->debug.overlay_checker && tls_scene->frame_buffer.data) {
        const int W = tls_scene->frame_buffer.dimensions.x;
        const int H = tls_scene->frame_buffer.dimensions.y;
        const int tile = os->debug.checker_size > 0 ? os->debug.checker_size : 8;
        const float s = os->debug.checker_strength;
        const float invs = 1.0f - s;
        const float cr = (float)os->debug.checker_color.r;
        const float cg = (float)os->debug.checker_color.g;
        const float cb = (float)os->debug.checker_color.b;
        for (int y = 0; y < H; ++y) {
            RGBA *row = tls_scene->frame_buffer.data + y * W;
            int ty = (y / tile);
            for (int x = 0; x < W; ++x) {
                int tx = (x / tile);
                if (((tx + ty) & 1) == 0) {
                    RGBA *p = &row[x];
                    float r = (float)p->r;
                    float g = (float)p->g;
                    float b = (float)p->b;
                    r = invs * r + s * cr;
                    g = invs * g + s * cg;
                    b = invs * b + s * cb;
                    if (r < 0.f) r = 0.f;
                    if (r > 255.f) r = 255.f;
                    if (g < 0.f) g = 0.f;
                    if (g > 255.f) g = 255.f;
                    if (b < 0.f) b = 0.f;
                    if (b > 255.f) b = 255.f;
                    p->r = (uint8_t)r; p->g = (uint8_t)g; p->b = (uint8_t)b;
                }
            }
        }
    }

    /* If a shadow map dump was requested, write it now before cleanup */
#ifdef HAVE_LIBPNG
    if (tls_shadow_maps && tls_dump_sm_index >= 0 && (uint16_t)tls_dump_sm_index < tls_shadow_count) {
        ShadowMap *SM = &tls_shadow_maps[tls_dump_sm_index];
        if (SM && SM->valid && SM->depth) {
            /* Optional stats to help debug: how much of the SM was written? */
            if (tls_scene && tls_scene->enhanced_debug) {
                int total = SM->w * SM->h;
                int filled = 0;
                float minz = 1e9f, maxz = -1e9f;
                for (int i = 0; i < total; ++i) {
                    float z = SM->depth[i];
                    if (z < 1e8f) { /* treated as written */
                        filled++;
                        if (z < minz) minz = z;
                        if (z > maxz) maxz = z;
                    }
                }
                float cov = (total > 0) ? ((float)filled * 100.0f / (float)total) : 0.0f;
                printf("[shadow] dump li=%d coverage=%.1f%% z_ndc[min=%.4f max=%.4f] -> %s\n",
                       tls_dump_sm_index, (double)cov, (double)minz, (double)maxz,
                       (tls_dump_sm_path[0] ? tls_dump_sm_path : "shadowmap.png"));
            }
            write_shadowmap_png(tls_dump_sm_path[0] ? tls_dump_sm_path : "shadowmap.png", SM);
        }
    }
#endif
    tls_dump_sm_index = -1; tls_dump_sm_path[0] = '\0';

    /* Cleanup shadow maps */
    if (tls_shadow_maps) {
        for (uint16_t li = 0; li < tls_shadow_count; ++li) {
            if (tls_shadow_maps[li].depth) free(tls_shadow_maps[li].depth);
        }
        free(tls_shadow_maps);
        tls_shadow_maps = NULL; tls_shadow_count = 0;
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
                                  vec3 direction, RGBF color,
                                  float intensity, bool casts_shadows) {
    if (!l || !l->lights || index >= l->num_lights) return;
    light_t *L = &l->lights[index];
    L->type = LIGHT_DIRECTIONAL;
    L->direction = direction;
    L->color = color;
    L->intensity = intensity;
    L->casts_shadows = casts_shadows;
    L->shadow_enabled = casts_shadows; /* default runtime toggle aligns with casts_shadows */
    /* 1.0 = no zoom, <1.0 zooms in (tighter bounds), >1.0 zooms out */
    L->shadow_zoom = 0.5f;
}

/* -------- Object scene helpers (FFI-friendly) -------- */
/* ---- object_scene OO-style lighting helpers ---- */
static void os_set_ambient(scene3d_t *os, RGBF ambient) {
    if (!os) return;
    os->lighting.ambient = ambient;
}

static uint16_t os_add_directional(scene3d_t *os,
                                   vec3 direction,
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
    L->shadow_enabled = casts_shadows; /* enable by default if configured to cast */
    L->position = (vec3){0,0,0};
    /* Default zoom: 1.0 (no zoom); users can set <1.0 to tighten the fit further */
    L->shadow_zoom = 0.5f;
    L->range = 0.0f; L->inner_cos = 1.0f; L->outer_cos = 1.0f;
    os->lighting.num_lights = (uint16_t)(n + 1);
    return n;
}

/* Set a directional light using a camera-like pose (position + look_at) */
static void os_set_directional_pose(scene3d_t *os, uint16_t id,
                                    vec3 position, vec3 look_at) {
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
    L->direction = (vec3){ dx, dy, dz };
    /* Invalidate cached shadow view-projection when pose changes */
    L->shadow_vp_valid = false;
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
    /* Enable Z-buffer by default; allocate lazily on first render */
    os->zbuffer_enabled = true;
    os->zbuf = NULL;
    os->zbuf_width = os->zbuf_height = 0;
    /* Debug defaults */
    os->debug.overlay_checker = false;
    os->debug.checker_size = 8;
    os->debug.checker_strength = 0.3f;
    os->debug.checker_color = (RGB){255, 0, 255};
    os->debug.overlay_shadow_vis = false;
    os->debug.shadow_vis_light = 0;
    os->debug.shadow_vis_strength = 0.35f;
    os->debug.shadow_vis_color_lit = (RGB){0, 255, 0};
    os->debug.shadow_vis_color_shadow = (RGB){255, 0, 0};
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
 * @param value the value to wrap
 * @param the wrap "top" value
 * @return the wrapped value
 */
float api_wrap_float(float value, float wrap) {
    return (value > wrap) ? fmodf(value,  wrap) : value;
}


/* -------- SDF Text API wrappers -------- */
#include "text_sdf.h"

sdf_font_t* api_sdf_font_load(const char *font_dir) {
    return sdf_font_load(font_dir);
}

sdf_font_t* api_sdf_font_load_scaled(const char *font_dir, float target_line_height_px) {
    return sdf_font_load_scaled(font_dir, target_line_height_px);
}

void api_sdf_font_free(sdf_font_t *font) {
    sdf_font_free(font);
}

sdf_text_t* api_sdf_text_create(sdf_font_t *font, const char *text) {
    return sdf_text_create(font, text);
}

void api_sdf_text_destroy(sdf_text_t *t) {
    sdf_text_destroy(t);
}

void api_sdf_text_set_position(sdf_text_t *t, float x, float y) {
    sdf_text_set_position(t, x, y);
}

void api_sdf_text_set_direction(sdf_text_t *t, float dx, float dy) {
    sdf_text_set_direction(t, dx, dy);
}

void api_sdf_text_set_speed(sdf_text_t *t, float speed) {
    sdf_text_set_speed(t, speed);
}

void api_sdf_text_set_size_px(sdf_text_t *t, float size_px) {
    sdf_text_set_size_px(t, size_px);
}

void api_sdf_text_set_color(sdf_text_t *t, RGBA color) {
    sdf_text_set_color(t, color);
}

void api_sdf_text_set_alpha(sdf_text_t *t, uint8_t a) {
    sdf_text_set_alpha(t, a);
}

void api_sdf_text_set_tracking(sdf_text_t *t, float tracking) {
    sdf_text_set_tracking(t, tracking);
}

void api_sdf_text_set_text(sdf_text_t *t, const char *text) {
    sdf_text_set_text(t, text);
}

void api_sdf_text_update(sdf_text_t *t, const int32_t display_width) {
    sdf_text_update(t, display_width);
}

void api_sdf_text_render(sdf_text_t *t, uint8_t *dst, int w, int h, int stride, float time_sec) {
    sdf_text_render(t, dst, w, h, stride, time_sec);
}

vec2u api_sdf_text_measure_px(sdf_text_t *t) {
    return sdf_text_measure_px(t);
}


/**
 * @brief Function pointer table containing all drawing API functions
 * 
 * Static constant structure that maps function pointers to the implementation
 * functions. This table is returned by hub75_api() to provide the drawing API.
 */
static const hub75gpu_t api_table = {
    .version = HUB75_API_VERSION,
    .clear = api_clear,
    .pixel = api_pixel,
    .line = api_line,
    .line_aa = api_line_aa,
    .poly = api_poly,
    .poly_gradient = api_poly_gradient,
    .fill_gradient = api_fill_gradient,
    .frame_begin = api_frame_begin,
    .wrap = api_wrap_float,

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
    /* debug helpers */
    .scene3d_set_debug_checker = api_scene3d_set_debug_checker,
    .scene3d_set_debug_shadow_vis = api_scene3d_set_debug_shadow_vis,
    .scene3d_dump_shadowmap_png = api_scene3d_dump_shadowmap_png,
    .scene3d_set_shadowmap_zoom = api_scene3d_set_shadowmap_zoom,

    /* SDF text functions */
    .sdf_font_load = api_sdf_font_load,
    .sdf_font_load_scaled = api_sdf_font_load_scaled,
    .sdf_font_free = api_sdf_font_free,
    .sdf_text_create = api_sdf_text_create,
    .sdf_text_destroy = api_sdf_text_destroy,
    .sdf_text_set_position = api_sdf_text_set_position,
    .sdf_text_set_direction = api_sdf_text_set_direction,
    .sdf_text_set_speed = api_sdf_text_set_speed,
    .sdf_text_set_size_px = api_sdf_text_set_size_px,
    .sdf_text_set_color = api_sdf_text_set_color,
    .sdf_text_set_alpha = api_sdf_text_set_alpha,
    .sdf_text_set_tracking = api_sdf_text_set_tracking,
    .sdf_text_set_text = api_sdf_text_set_text,
    .sdf_text_update = api_sdf_text_update,
    .sdf_text_render = api_sdf_text_render,
    .sdf_text_measure_px = api_sdf_text_measure_px,
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
hub75_error_t api_scene3d_set_ambient(RGBF ambient) {
    if (!tls_current_os) return HUB75_ERR_NO_SCENE;
    if (!tls_current_os->set_ambient) return HUB75_ERR_NULL_PARAM;
    tls_current_os->set_ambient(tls_current_os, ambient);
    return HUB75_OK;
}
uint16_t api_scene3d_add_directional(vec3 direction, RGBF color, float intensity, bool casts_shadows) {
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

hub75_error_t api_scene3d_set_directional_pose(uint16_t id, vec3 position, vec3 look_at) {
    if (!tls_current_os) return HUB75_ERR_NO_SCENE;
    if (!tls_current_os->set_directional_pose) return HUB75_ERR_NULL_PARAM;
    tls_current_os->set_directional_pose(tls_current_os, id, position, look_at);
    return HUB75_OK;
}

/* Control shadow map tight-fit zoom for a light (1.0 = default; <1.0 zoom in; >1.0 zoom out) */
hub75_error_t api_scene3d_set_shadowmap_zoom(uint16_t light_index, float zoom) {
    if (!tls_current_os) return HUB75_ERR_NO_SCENE;
    if (!tls_current_os->lighting.lights || light_index >= tls_current_os->lighting.num_lights) return HUB75_ERR_INVALID_COORDS;
    if (!(zoom > 0.0f)) zoom = 1.0f; /* ignore non-positive values */
    light_t *L = &tls_current_os->lighting.lights[light_index];
    L->shadow_zoom = zoom;
    /* Force recompute of cached VP/bias next frame */
    L->shadow_vp_valid = false;
    return HUB75_OK;
}

/* -------- Debug helpers (thread-local current scene3d) -------- */
hub75_error_t api_scene3d_set_debug_checker(bool enabled, uint8_t tile_px, float strength, RGB color) {
    if (!tls_current_os) return HUB75_ERR_NO_SCENE;
    tls_current_os->debug.overlay_checker = enabled;
    if (tile_px == 0) tile_px = 8;
    tls_current_os->debug.checker_size = tile_px;
    if (strength < 0.0f) strength = 0.0f; else if (strength > 1.0f) strength = 1.0f;
    tls_current_os->debug.checker_strength = strength;
    tls_current_os->debug.checker_color = color;
    return HUB75_OK;
}

hub75_error_t api_scene3d_set_debug_shadow_vis(bool enabled, uint8_t light_index, float strength, RGB lit_color, RGB shadow_color) {
    if (!tls_current_os) return HUB75_ERR_NO_SCENE;
    tls_current_os->debug.overlay_shadow_vis = enabled;
    tls_current_os->debug.shadow_vis_light = light_index;
    if (strength < 0.0f) strength = 0.0f; else if (strength > 1.0f) strength = 1.0f;
    tls_current_os->debug.shadow_vis_strength = strength;
    tls_current_os->debug.shadow_vis_color_lit = lit_color;
    tls_current_os->debug.shadow_vis_color_shadow = shadow_color;
    return HUB75_OK;
}

/* -------- Object material helpers -------- */
void api_object_set_specular(object_t *obj, float strength, float shininess) {
    if (!obj) return;
    if (strength < 0.0f) strength = 0.0f;
    obj->specular_strength = strength;
    if (shininess < 1.0f) shininess = 1.0f;
    obj->specular_shininess = shininess;
}

void api_object_set_specular_strength(object_t *obj, float strength) {
    if (!obj) return;
    if (strength < 0.0f) strength = 0.0f;
    obj->specular_strength = strength;
}

void api_object_set_specular_shininess(object_t *obj, float shininess) {
    if (!obj) return;
    if (shininess < 1.0f) shininess = 1.0f;
    obj->specular_shininess = shininess;
}

