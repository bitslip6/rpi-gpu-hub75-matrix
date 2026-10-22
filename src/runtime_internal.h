#ifndef H75_RUNTIME_INTERNAL_H
#define H75_RUNTIME_INTERNAL_H
#include "hub75_runtime.h"
#include "text_sdf.h"
#include <stdarg.h>

/* Coarse resource lock: setters are applied between complete native frames.
 * Never taken by the GPIO scanner. Never held while acquiring the Python GIL. */
extern pthread_mutex_t h75_resources_mutex;
struct h75_resource {
    atomic_uint refs;
    h75_kind kind;
    unsigned exports;
    union {
        image_buffer_t *image;
        camera_t camera;
        transform_t transform;
        object_t *mesh;
        struct {
            scene3d_t *native;
            h75_resource **meshes, **transforms;
            vec3 *velocity;
            size_t capacity;
        } scene;
        sdf_font_t *font;
        struct {
            sdf_text_t *native;
            h75_resource *font;
            image_buffer_t *cache;
            int32_t cache_x, cache_y;
            bool dirty;
            vec3 sine; /* frequency, amplitude, phase speed */
        } text;
        struct {
            h75_command *commands;
            size_t count, capacity;
        } list;
    } u;
};
h75_status h75_fail(h75_error *e, h75_status status, const char *fmt, ...);
uint64_t h75_now_ns(void);
h75_status h75_copy_rgba(image_buffer_t *dst, const void *data, size_t size, uint32_t pitch,
                         int channels, h75_error *e);
h75_status h75_command_execute(hub75_display_t *display, const h75_command *c, float time,
                               h75_error *e);
h75_status h75_list_execute(hub75_display_t *display, h75_resource *list, float time, h75_error *e);
/* Optional backend, resolved before any function-table-by-value use. All EGL/
 * decoder handles stay on the producer thread. next returns 1 at EOF. */
typedef struct h75_media h75_media;
extern h75_status h75_media_open(h75_media **out, const char *path, bool video, uint32_t w,
                                 uint32_t h, bool preserve_aspect, const atomic_bool *stop, h75_error *e)
    __attribute__((weak));
extern int h75_media_next(h75_media *m, uint8_t *rgba, float time, float dt, float *pts,
                          h75_error *e) __attribute__((weak));
extern void h75_media_close(h75_media *m) __attribute__((weak));
#endif
