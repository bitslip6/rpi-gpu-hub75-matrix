#ifndef HUB75_RUNTIME_H
#define HUB75_RUNTIME_H

/* Additive, versioned embedding API. No call installs signals or exits.
 * Handles are one-shot. Callers must serialize destroy against all API calls.
 * Resources are retained by scenes, commands and active playback. */
#include "hub75gpu.h"

#define HUB75_RUNTIME_ABI 1u
#define HUB75_CAP_CPU 1u
#define HUB75_CAP_SHADER 2u
#define HUB75_CAP_VIDEO 4u
#define HUB75_CAP_GPIO 8u
#define HUB75_CAP_PNG 16u
#define HUB75_CAP_JPEG 32u
#define HUB75_CAP_NEON 64u

typedef enum {
    H75_OK = 0,
    H75_INVALID = -1,
    H75_STATE = -2,
    H75_TIMEOUT = -3,
    H75_UNSUPPORTED = -4,
    H75_IO = -5,
    H75_MEMORY = -6,
    H75_CANCELLED = -7,
    H75_BUSY = -8
} h75_status;
typedef struct {
    h75_status code;
    char message[512];
} h75_error;
typedef enum { H75_CREATED, H75_RUNNING, H75_STOPPING, H75_STOPPED, H75_CLOSED } h75_state;
typedef struct {
    uint32_t size, width, height, panel_width, panel_height, ports, chains;
    uint32_t bit_depth, fps, brightness, queue_depth, panel_order, mapping, tone_mapping;
    float gamma, dither, tone_level;
    bool offscreen, quant_dither, jitter;
    int32_t gpio_cpu; /* -1: no affinity; scheduling/system tuning is never automatic */
    bool video_preserve_aspect; /* Fit with centered black bars; default false */
} h75_config;
typedef struct h75_runtime h75_runtime;
typedef struct h75_resource h75_resource;
typedef enum {
    H75_IMAGE,
    H75_CAMERA,
    H75_TRANSFORM,
    H75_MESH,
    H75_SCENE,
    H75_FONT,
    H75_TEXT,
    H75_DRAWLIST
} h75_kind;
typedef enum {
    H75_CLEAR,
    H75_PIXEL,
    H75_LINE,
    H75_LINE_AA,
    H75_POLYGON,
    H75_GRADIENT,
    H75_BLIT,
    H75_RENDER_SCENE,
    H75_RENDER_TEXT
} h75_opcode;
/* Coordinates: pixel units for line/pixel/blit; polygon vertices normalized 0..1.
 * Blit p[0..3] = destination rectangle. A zero-sized rectangle means native size.
 * Resources are borrowed during synchronous execution, retained on list append. */
typedef struct {
    h75_opcode op;
    RGBA color;
    float p[8];
    Polygonf_t polygon;
    SimpleGradient gradient;
    h75_resource *a, *b;
} h75_command;
typedef struct {
    uint64_t submitted, consumed, mapped, scanned, dropped, queue_wait_ns, render_ns;
    uint64_t map_ns, copy_ns, scan_ns;     /* coarse inclusive stage totals, not per-pixel clocks */
    uint64_t latency_ns, latency_total_ns; /* last/total submit-to-logical-consumption latency */
    uint32_t queue_occupancy;
    h75_state state;
    bool playing, paused, ready;
} h75_stats;

uint32_t hub75_runtime_abi(void);
uint32_t hub75_runtime_capabilities(void);
const char *hub75_runtime_build(void);
void h75_config_default(h75_config *c);
h75_status h75_config_validate(const h75_config *c, h75_error *e);
h75_status h75_runtime_create(const h75_config *c, h75_runtime **out, h75_error *e);
h75_status h75_runtime_start(h75_runtime *r, h75_error *e);
void h75_runtime_request_stop(h75_runtime *r);
/* Lock-free lifecycle observation, including after a timed-out close. */
h75_state h75_runtime_get_state(h75_runtime *r);
/* Snapshot the latest asynchronous worker error without joining/clearing it. */
h75_status h75_runtime_error(h75_runtime *r, h75_error *e);
h75_status h75_runtime_wait(h75_runtime *r, int timeout_ms, h75_error *e);
h75_status h75_runtime_close(h75_runtime *r, int timeout_ms, h75_error *e);
void h75_runtime_destroy(h75_runtime *r);
h75_status h75_frame_acquire(h75_runtime *r, int timeout_ms, h75_error *e);
h75_status h75_frame_submit(h75_runtime *r, h75_error *e);
h75_status h75_frame_cancel(h75_runtime *r, h75_error *e);
h75_status h75_frame_command(h75_runtime *r, const h75_command *c, h75_error *e);
h75_status h75_frame_upload(h75_runtime *r, const void *data, size_t size, uint32_t pitch,
                            int channels, h75_error *e);
h75_status h75_frame_draw(h75_runtime *r, h75_resource *list, float time, h75_error *e);
h75_status h75_runtime_read(h75_runtime *r, void *rgba, size_t size, h75_error *e);
h75_status h75_runtime_flush(h75_runtime *r, int timeout_ms, h75_error *e);
void h75_runtime_stats(h75_runtime *r, h75_stats *out);
/* Content modes are mutually exclusive. stop_content joins the producer only.
 * wait_content reports EOF/error and leaves the last frame visible. */
h75_status h75_runtime_play(h75_runtime *r, h75_resource *list, h75_error *e);
h75_status h75_runtime_play_file(h75_runtime *r, const char *path, bool video, bool loop,
                                 float time_scale, h75_resource *overlay, h75_error *e);
h75_status h75_runtime_content_ready(h75_runtime *r, int timeout_ms, h75_error *e);
h75_status h75_runtime_pause(h75_runtime *r, bool paused, h75_error *e);
h75_status h75_runtime_stop_content(h75_runtime *r, int timeout_ms, h75_error *e);
h75_status h75_runtime_wait_content(h75_runtime *r, int timeout_ms, h75_error *e);

h75_resource *h75_resource_new(h75_kind kind, h75_error *e);
h75_resource *h75_image_new(uint32_t width, uint32_t height, h75_error *e);
h75_resource *h75_image_load(const char *path, h75_error *e);
h75_resource *h75_image_transform(h75_resource *source, uint32_t mapping, h75_error *e);
h75_status h75_camera_project(h75_resource *camera, h75_resource *transform, mat4 *out,
                              h75_error *e);
h75_resource *h75_mesh_primitive(const char *name, uint32_t x, uint32_t y, int mode, bool cull,
                                 h75_error *e);
h75_resource *h75_mesh_upload(const float *vertices, size_t nv, const uint32_t *edges, size_t ne,
                              const uint32_t *faces, size_t nf, h75_error *e);
h75_resource *h75_font_load(const char *path, float size, h75_error *e);
h75_resource *h75_text_new(h75_resource *font, const char *text, h75_error *e);
void h75_resource_retain(h75_resource *r);
void h75_resource_release(h75_resource *r);
h75_kind h75_resource_kind(const h75_resource *r);
h75_status h75_resource_set(h75_resource *r, const char *name, const float *v, size_t n,
                            h75_error *e);
/* v must have room for 16 floats; *n >=16 on input, actual length on output. */
h75_status h75_resource_get(h75_resource *r, const char *name, float *v, size_t *n, h75_error *e);
h75_status h75_text_string(h75_resource *r, const char *text, h75_error *e);
h75_status h75_scene_add(h75_resource *scene, h75_resource *mesh, h75_resource *transform,
                         vec3 angular_velocity, uint16_t *id, h75_error *e);
h75_status h75_scene_transforms(h75_resource *scene, const float *values, size_t count,
                                h75_error *e);
h75_status h75_scene_light(h75_resource *scene, vec3 direction, RGBF color, float intensity,
                           bool shadows, uint16_t *id, h75_error *e);
h75_status h75_scene_light_set(h75_resource *scene, uint16_t id, const char *name, const float *v,
                               size_t n, h75_error *e);
h75_status h75_drawlist_append(h75_resource *list, const h75_command *c, h75_error *e);
/* Geometry is a copied native float32 Nx2 (pixels) or Nx4 (lines) buffer. */
h75_status h75_drawlist_geometry(h75_resource *list, const float *xy, size_t count,
                                 unsigned columns, RGBA color, bool antialias, bool replace,
                                 h75_error *e);
/* Nonblocking submission. Queue contention returns accepted=false and counts a
 * dropped incoming frame; incompatible producer modes remain explicit errors. */
h75_status h75_runtime_try_draw(h75_runtime *r, h75_resource *list, float time, bool *accepted,
                                h75_error *e);
h75_status h75_image_upload(h75_resource *r, const void *data, size_t size, uint32_t width,
                            uint32_t height, uint32_t pitch, int channels, h75_error *e);
/* Export pins the allocation. Native reads/writes reject images with active
 * writable exports. Unexport exactly once; no resizing is supported. */
h75_status h75_image_export(h75_resource *r, image_buffer_t *out, h75_error *e);
void h75_image_unexport(h75_resource *r);
h75_status h75_image_copy(h75_resource *r, void *out, size_t size, h75_error *e);

#endif
