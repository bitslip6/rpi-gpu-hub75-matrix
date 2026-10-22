#include "runtime_internal.h"
#include "compositor.h"
#include "transformers.h"
#include "util.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t h75_resources_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool finite_values(const float *v, size_t n) {
    if (!v || n > 16)
        return false;
    for (size_t i = 0; i < n; ++i)
        if (!isfinite(v[i]) || fabsf(v[i]) > 1e7f)
            return false;
    return true;
}
static bool kind(h75_resource *r, h75_kind k) {
    return r && r->kind == k;
}

h75_resource *h75_resource_new(h75_kind k, h75_error *e) {
    if (k != H75_CAMERA && k != H75_TRANSFORM && k != H75_SCENE && k != H75_DRAWLIST) {
        h75_fail(e, H75_INVALID, "resource requires a specialized constructor");
        return NULL;
    }
    h75_resource *r = calloc(1, sizeof(*r));
    if (!r) {
        h75_fail(e, H75_MEMORY, "allocating resource");
        return NULL;
    }
    atomic_init(&r->refs, 1);
    r->kind = k;
    if (k == H75_CAMERA)
        r->u.camera = (camera_t){.position = {0, 0, 5},
                                 .target = {0, 0, 0},
                                 .up = {0, 1, 0},
                                 .fov_y = 1.04719755f,
                                 .aspect = 1,
                                 .z_near = 0.1f,
                                 .z_far = 100};
    if (k == H75_TRANSFORM)
        r->u.transform.scale = (vec3){1, 1, 1};
    if (k == H75_SCENE) {
        r->u.scene.native = api_object_scene_new(0);
        if (!r->u.scene.native) {
            free(r);
            h75_fail(e, H75_MEMORY, "allocating scene");
            return NULL;
        }
        r->u.scene.native->lighting.ambient = (RGBF){0.2f, 0.2f, 0.2f};
    }
    return r;
}
static h75_resource *wrap(h75_kind k, void *p, h75_error *e) {
    if (!p) {
        h75_fail(e, H75_MEMORY, "native allocation failed");
        return NULL;
    }
    h75_resource *r = calloc(1, sizeof(*r));
    if (!r) {
        h75_fail(e, H75_MEMORY, "allocating resource handle");
        return NULL;
    }
    atomic_init(&r->refs, 1);
    r->kind = k;
    /* Store through the correct union member, not pointer type punning. */
    if (k == H75_IMAGE)
        r->u.image = p;
    if (k == H75_MESH)
        r->u.mesh = p;
    if (k == H75_FONT)
        r->u.font = p;
    if (k == H75_TEXT)
        r->u.text.native = p;
    return r;
}
h75_resource *h75_image_new(uint32_t w, uint32_t h, h75_error *e) {
    if (!w || !h || w > 4096 || h > 4096 || (size_t)w * h * 4 > SPSC_MAX_ITEM_SIZE) {
        h75_fail(e, H75_INVALID, "image dimensions exceed supported allocation limits");
        return NULL;
    }
    image_buffer_t *im = image_buffer_new((int32_t)w, (int32_t)h);
    h75_resource *r = wrap(H75_IMAGE, im, e);
    if (!r)
        free(im);
    return r;
}
h75_resource *h75_image_load(const char *path, h75_error *e) {
    if (!path) {
        h75_fail(e, H75_INVALID, "null image path");
        return NULL;
    }
    uint8_t *p = NULL;
    int w = 0, h = 0, stride = 0;
    if (!path || image_read_rgba8(path, &p, &w, &h, &stride)) {
        h75_fail(e, H75_IO, "cannot load image: %s", path ? path : "(null)");
        return NULL;
    }
    h75_resource *r = h75_image_new((uint32_t)w, (uint32_t)h, e);
    if (r)
        for (int y = 0; y < h; y++)
            memcpy((uint8_t *)r->u.image->data + (size_t)y * (size_t)w * 4,
                   p + (size_t)y * (size_t)stride, (size_t)w * 4);
    free(p);
    return r;
}
h75_resource *h75_image_transform(h75_resource *source, uint32_t mapping, h75_error *e) {
    if (!kind(source, H75_IMAGE) || mapping > 3) {
        h75_fail(e, H75_INVALID, "expected image and flip/mirror mapping 0..3");
        return NULL;
    }
    image_buffer_t *src = source->u.image;
    h75_resource *out = h75_image_new((uint32_t)src->dimensions.x, (uint32_t)src->dimensions.y, e);
    if (!out)
        return NULL;
    pthread_mutex_lock(&h75_resources_mutex);
    if (source->exports) {
        pthread_mutex_unlock(&h75_resources_mutex);
        h75_resource_release(out);
        h75_fail(e, H75_BUSY, "release writable views before transforming images");
        return NULL;
    }
    hub75_display_t d = {
        .width = (uint16_t)src->dimensions.x, .height = (uint16_t)src->dimensions.y, .stride = 4};
    func_image_mapper_t maps[] = {NULL, flip_mapper, mirror_mapper, mirror_flip_mapper};
    if (mapping)
        maps[mapping]((uint8_t *)src->data, (uint8_t *)out->u.image->data, &d);
    else
        memcpy(out->u.image->data, src->data, (size_t)src->row_stride * (size_t)src->dimensions.y);
    pthread_mutex_unlock(&h75_resources_mutex);
    return out;
}
static bool scale_valid(const float *v) {
    for (unsigned i = 0; i < 3; i++)
        if (fabsf(v[i]) < 1e-4f || fabsf(v[i]) > 1e4f)
            return false;
    return true;
}
static bool camera_valid(const camera_t *c) {
    vec3 f = {c->target.x - c->position.x, c->target.y - c->position.y,
              c->target.z - c->position.z};
    vec3 cross = {f.y * c->up.z - f.z * c->up.y, f.z * c->up.x - f.x * c->up.z,
                  f.x * c->up.y - f.y * c->up.x};
    return cross.x * cross.x + cross.y * cross.y + cross.z * cross.z > 1e-10f &&
           c->fov_y >= 1e-3f && c->fov_y < 3.14f && c->aspect >= 1e-4f && c->aspect <= 1e4f &&
           c->z_near >= 1e-4f && c->z_far - c->z_near >= 1e-4f;
}
h75_status h75_camera_project(h75_resource *camera, h75_resource *transform, mat4 *out,
                              h75_error *e) {
    if (!kind(camera, H75_CAMERA) || !kind(transform, H75_TRANSFORM) || !out)
        return h75_fail(e, H75_INVALID, "expected camera, transform and matrix output");
    pthread_mutex_lock(&h75_resources_mutex);
    h75_status s = H75_OK;
    if (!camera_valid(&camera->u.camera))
        s = h75_fail(e, H75_INVALID, "camera view direction and up vector are degenerate");
    else
        *out = camera_project(&camera->u.camera, &transform->u.transform);
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
h75_resource *h75_mesh_primitive(const char *name, uint32_t x, uint32_t y, int mode, bool cull,
                                 h75_error *e) {
    object_t *o = NULL;
    if (!name || (mode != DRAW_WIRE && mode != DRAW_FILLED))
        goto invalid;
    if (!strcmp(name, "cube"))
        o = object_cube(mode, cull);
    else if (!strcmp(name, "tetrahedron"))
        o = object_tetrahedron(mode, cull);
    else if (!strcmp(name, "octahedron"))
        o = object_octahedron(mode, cull);
    else if (!strcmp(name, "pyramid"))
        o = object_pyramid();
    else if (!strcmp(name, "cylinder")) {
        if (x < 3 || x > 32)
            goto invalid;
        o = object_cylinder((uint16_t)x, mode, cull);
    } else if (!strcmp(name, "sphere")) {
        if (x > 4)
            goto invalid;
        o = object_sphere((uint16_t)x);
    } else if (!strcmp(name, "torus")) {
        if (x < 3 || y < 3 || x > 16 || y > 16)
            goto invalid;
        o = object_torus((uint16_t)x, (uint16_t)y);
    } else if (!strcmp(name, "plane")) {
        if (!x || !y || x > 128 || y > 128)
            goto invalid;
        o = object_plane((uint16_t)x, (uint16_t)y, true);
    } else
        goto invalid;
    if (o) {
        o->draw_mode = mode;
        o->cull_backface = cull;
    }
    h75_resource *r = wrap(H75_MESH, o, e);
    if (!r)
        object_free(o);
    return r;
invalid:
    h75_fail(e, H75_INVALID, "invalid primitive, draw mode or complexity");
    return NULL;
}
h75_resource *h75_mesh_upload(const float *v, size_t nv, const uint32_t *edges, size_t ne,
                              const uint32_t *faces, size_t nf, h75_error *e) {
    if (!v || !nv || nv > 65535 || ne > 65535 || nf > 65535 || (ne && !edges) || (nf && !faces))
        goto invalid;
    for (size_t i = 0; i < nv * 3; i++)
        if (!isfinite(v[i]) || fabsf(v[i]) > 1e4f)
            goto invalid;
    for (size_t i = 0; i < ne * 2; i++)
        if (edges[i] >= nv)
            goto invalid;
    for (size_t i = 0; i < nf * 3; i++)
        if (faces[i] >= nv)
            goto invalid;
    object_t *o = object_new((uint16_t)nv, (uint16_t)ne, (uint16_t)nf);
    if (!o) {
        h75_fail(e, H75_MEMORY, "allocating mesh");
        return NULL;
    }
    memcpy(o->verticies->list, v, nv * sizeof(vec3));
    for (size_t i = 0; i < o->edge_colors->length; i++)
        o->edge_colors->list[i] = (RGB){255, 255, 255};
    for (size_t i = 0; i < ne; i++) {
        o->edges->list[i] = (vec2){(float)edges[i * 2], (float)edges[i * 2 + 1]};
        o->edge_colors->list[i] = (RGB){255, 255, 255};
    }
    for (size_t i = 0; i < nf; i++)
        o->faces->list[i] =
            (vec3){(float)faces[i * 3], (float)faces[i * 3 + 1], (float)faces[i * 3 + 2]};
    o->draw_mode = nf ? DRAW_FILLED : DRAW_WIRE;
    h75_resource *r = wrap(H75_MESH, o, e);
    if (!r)
        object_free(o);
    return r;
invalid:
    h75_fail(e, H75_INVALID, "invalid mesh dimensions, indices or nonfinite vertices");
    return NULL;
}
h75_resource *h75_font_load(const char *path, float size, h75_error *e) {
    if (!path || !isfinite(size) || size < 0 || size > 256) {
        h75_fail(e, H75_INVALID, "invalid font size");
        return NULL;
    }
#ifndef HAVE_LIBPNG
    h75_fail(e, H75_UNSUPPORTED, "SDF text requires a build with libpng");
    return NULL;
#endif
    sdf_font_t *f = size > 0 ? sdf_font_load_scaled(path, size) : sdf_font_load(path);
    if (!f) {
        h75_fail(e, H75_IO, "cannot load SDF font: %s", path);
        return NULL;
    }
    h75_resource *r = wrap(H75_FONT, f, e);
    if (!r)
        sdf_font_free(f);
    return r;
}
h75_resource *h75_text_new(h75_resource *font, const char *text, h75_error *e) {
    if (!kind(font, H75_FONT) || !text || strlen(text) > 65535) {
        h75_fail(e, H75_INVALID, "invalid font or text length");
        return NULL;
    }
    sdf_text_t *t = sdf_text_create(font->u.font, text);
    if (t && text[0] && !t->text) {
        sdf_text_destroy(t);
        h75_fail(e, H75_MEMORY, "allocating text string");
        return NULL;
    }
    h75_resource *r = wrap(H75_TEXT, t, e);
    if (!r)
        sdf_text_destroy(t);
    else {
        r->u.text.font = font;
        r->u.text.dirty = true;
        h75_resource_retain(font);
    }
    return r;
}
void h75_resource_retain(h75_resource *r) {
    if (r)
        atomic_fetch_add_explicit(&r->refs, 1, memory_order_relaxed);
}
void h75_resource_release(h75_resource *r) {
    if (!r || atomic_fetch_sub_explicit(&r->refs, 1, memory_order_acq_rel) != 1)
        return;
    switch (r->kind) {
    case H75_IMAGE:
        free(r->u.image);
        break;
    case H75_MESH:
        object_free(r->u.mesh);
        break;
    case H75_FONT:
        sdf_font_free(r->u.font);
        break;
    case H75_TEXT:
        free(r->u.text.cache);
        sdf_text_destroy(r->u.text.native);
        h75_resource_release(r->u.text.font);
        break;
    case H75_SCENE:
        for (size_t i = 0; i < r->u.scene.native->count; i++) {
            h75_resource_release(r->u.scene.meshes[i]);
            h75_resource_release(r->u.scene.transforms[i]);
        }
        api_object_scene_free(r->u.scene.native);
        free(r->u.scene.meshes);
        free(r->u.scene.transforms);
        free(r->u.scene.velocity);
        break;
    case H75_DRAWLIST:
        for (size_t i = 0; i < r->u.list.count; i++) {
            h75_resource_release(r->u.list.commands[i].a);
            h75_resource_release(r->u.list.commands[i].b);
        }
        free(r->u.list.commands);
        break;
    default:
        break;
    }
    free(r);
}
h75_kind h75_resource_kind(const h75_resource *r) {
    return r->kind;
}

/* Property dispatch deliberately validates before narrowing or entering legacy
 * rasterizers. This TU is compiled without fast-math. */
static h75_status property(h75_resource *r, const char *name, float *v, size_t *n, bool set,
                           h75_error *e) {
    float *p = NULL;
    size_t count = 0;
    if (!r || !name)
        goto invalid;
    if (set && !finite_values(v, *n))
        goto invalid;
    if (r->kind == H75_TRANSFORM) {
        if (!strcmp(name, "position")) {
            p = &r->u.transform.position.x;
            count = 3;
        }
        if (!strcmp(name, "rotation")) {
            p = &r->u.transform.rotation.x;
            count = 3;
        }
        if (!strcmp(name, "scale")) {
            p = &r->u.transform.scale.x;
            count = 3;
        }
        if (set && !strcmp(name, "scale") && *n == 3 && !scale_valid(v))
            goto invalid;
    }
    if (r->kind == H75_CAMERA) {
        camera_t *c = &r->u.camera;
        if (!strcmp(name, "position")) {
            p = &c->position.x;
            count = 3;
        }
        if (!strcmp(name, "target")) {
            p = &c->target.x;
            count = 3;
        }
        if (!strcmp(name, "up")) {
            p = &c->up.x;
            count = 3;
        }
        if (!strcmp(name, "fov_y")) {
            p = &c->fov_y;
            count = 1;
            if (set && (v[0] < 1e-3f || v[0] >= 3.14f))
                goto invalid;
        }
        if (!strcmp(name, "aspect")) {
            p = &c->aspect;
            count = 1;
            if (set && (v[0] < 1e-4f || v[0] > 1e4f))
                goto invalid;
        }
        if (!strcmp(name, "z_near")) {
            p = &c->z_near;
            count = 1;
            if (set && (v[0] < 1e-4f || c->z_far - v[0] < 1e-4f))
                goto invalid;
        }
        if (!strcmp(name, "z_far")) {
            p = &c->z_far;
            count = 1;
            if (set && v[0] - c->z_near < 1e-4f)
                goto invalid;
        }
    }
    if (r->kind == H75_SCENE && !strcmp(name, "ambient")) {
        if (set && *n == 3 &&
            (v[0] < 0 || v[0] > 1 || v[1] < 0 || v[1] > 1 || v[2] < 0 || v[2] > 1))
            goto invalid;
        p = &r->u.scene.native->lighting.ambient.r;
        count = 3;
    }
    if (r->kind == H75_MESH) {
        object_t *o = r->u.mesh;
        if (!strcmp(name, "specular_strength")) {
            p = &o->specular_strength;
            count = 1;
            if (set && v[0] < 0)
                goto invalid;
        }
        if (!strcmp(name, "specular_shininess")) {
            p = &o->specular_shininess;
            count = 1;
            if (set && v[0] < 1)
                goto invalid;
        }
        if (!strcmp(name, "draw_mode")) {
            if (set) {
                if (*n != 1 || (v[0] != 0 && v[0] != 1))
                    goto invalid;
                o->draw_mode = (int)v[0];
            } else {
                v[0] = (float)o->draw_mode;
                *n = 1;
            }
            return H75_OK;
        }
        if (!strcmp(name, "cull_backface") || !strcmp(name, "shadow_enabled")) {
            bool *b = !strcmp(name, "cull_backface") ? &o->cull_backface : &o->shadow_enabled;
            if (set) {
                if (*n != 1 || (v[0] != 0 && v[0] != 1))
                    goto invalid;
                *b = v[0] != 0;
            } else {
                v[0] = *b;
                *n = 1;
            }
            return H75_OK;
        }
        if (!strcmp(name, "color") && set && *n == 3) {
            for (int j = 0; j < 3; j++)
                if (v[j] < 0 || v[j] > 255 || v[j] != floorf(v[j]))
                    goto invalid;
            for (size_t i = 0; i < o->edge_colors->length; i++)
                o->edge_colors->list[i] = (RGB){(uint8_t)v[0], (uint8_t)v[1], (uint8_t)v[2]};
            return H75_OK;
        }
    }
    if (r->kind == H75_IMAGE && !strcmp(name, "dimensions") && !set) {
        v[0] = (float)r->u.image->dimensions.x;
        v[1] = (float)r->u.image->dimensions.y;
        *n = 2;
        return H75_OK;
    }
    if (r->kind == H75_TEXT) {
        sdf_text_t *t = r->u.text.native;
        if (!strcmp(name, "dimensions") && !set) {
            vec2u d = sdf_text_measure_px(t);
            v[0] = (float)d.x;
            v[1] = (float)d.y;
            *n = 2;
            return H75_OK;
        }
        if (!set) {
            if (!strcmp(name, "position")) {
                v[0] = t->x0;
                v[1] = t->y0;
                *n = 2;
                return H75_OK;
            }
            if (!strcmp(name, "direction")) {
                v[0] = t->dir_x;
                v[1] = t->dir_y;
                *n = 2;
                return H75_OK;
            }
            if (!strcmp(name, "speed")) {
                v[0] = t->speed;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "size")) {
                v[0] = t->size_px;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "tracking")) {
                v[0] = t->tracking;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "angle")) {
                v[0] = t->angle;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "weight")) {
                v[0] = t->effects.weight;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "outline_width")) {
                v[0] = t->effects.outline_width;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "wrap")) {
                v[0] = t->wrap;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "kerning")) {
                v[0] = t->kerning_enable;
                *n = 1;
                return H75_OK;
            }
            if (!strcmp(name, "sine")) {
                memcpy(v, &r->u.text.sine, sizeof(vec3));
                *n = 3;
                return H75_OK;
            }
            if (!strcmp(name, "color") || !strcmp(name, "outline_color")) {
                RGBA c = !strcmp(name, "color") ? t->color : t->effects.outline_color;
                v[0] = c.r;
                v[1] = c.g;
                v[2] = c.b;
                v[3] = c.a;
                *n = 4;
                return H75_OK;
            }
        }
        if (set) {
            if (!strcmp(name, "position") && *n == 2)
                sdf_text_set_position(t, v[0], v[1]);
            else if (!strcmp(name, "direction") && *n == 2)
                sdf_text_set_direction(t, v[0], v[1]);
            else if (!strcmp(name, "speed") && *n == 1)
                sdf_text_set_speed(t, v[0]);
            else if (!strcmp(name, "size") && *n == 1 && v[0] > 0 && v[0] <= 256)
                sdf_text_set_size_px(t, v[0]);
            else if (!strcmp(name, "tracking") && *n == 1 && fabsf(v[0]) <= 256)
                sdf_text_set_tracking(t, v[0]);
            else if (!strcmp(name, "angle") && *n == 1) {
                if (v[0] != 0)
                    return h75_fail(e, H75_UNSUPPORTED,
                                    "the native SDF renderer does not implement text rotation");
                sdf_text_set_angle(t, v[0]);
            } else if (!strcmp(name, "wrap") && *n == 1)
                sdf_text_set_wrap(t, v[0] != 0);
            else if (!strcmp(name, "kerning") && *n == 1)
                sdf_text_set_kerning(t, v[0] != 0);
            else if (!strcmp(name, "sine") && *n == 3 && fabsf(v[0]) <= 100 && fabsf(v[1]) <= 4096)
                r->u.text.sine = (vec3){v[0], v[1], v[2]};
            else if (!strcmp(name, "weight") && *n == 1 && v[0] > 0 && v[0] <= 4)
                t->effects.weight = v[0];
            else if (!strcmp(name, "outline_width") && *n == 1 && v[0] >= 0 && v[0] <= 1)
                t->effects.outline_width = v[0];
            else if ((!strcmp(name, "color") || !strcmp(name, "outline_color")) && *n == 4) {
                for (int j = 0; j < 4; j++)
                    if (v[j] < 0 || v[j] > 255 || v[j] != floorf(v[j]))
                        goto invalid;
                RGBA c = {(uint8_t)v[0], (uint8_t)v[1], (uint8_t)v[2], (uint8_t)v[3]};
                if (!strcmp(name, "color"))
                    sdf_text_set_color(t, c);
                else
                    t->effects.outline_color = c;
            } else
                goto invalid;
            if (strcmp(name, "position") && strcmp(name, "direction") && strcmp(name, "speed") &&
                strcmp(name, "sine"))
                r->u.text.dirty = true;
            return H75_OK;
        }
    }
    if (!p || (set && *n != count))
        goto invalid;
    if (set)
        memcpy(p, v, count * sizeof(float));
    else {
        memcpy(v, p, count * sizeof(float));
        *n = count;
    }
    return H75_OK;
invalid:
    return h75_fail(e, H75_INVALID, "invalid %s property or value", name ? name : "resource");
}
h75_status h75_resource_set(h75_resource *r, const char *name, const float *v, size_t n,
                            h75_error *e) {
    float copy[16];
    if (!v || n == 0 || n > 16)
        return h75_fail(e, H75_INVALID, "invalid property arity");
    memcpy(copy, v, n * sizeof(float));
    pthread_mutex_lock(&h75_resources_mutex);
    h75_status s = property(r, name, copy, &n, true, e);
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
h75_status h75_resource_get(h75_resource *r, const char *name, float *v, size_t *n, h75_error *e) {
    if (!v || !n || *n < 16)
        return h75_fail(e, H75_INVALID, "property output requires capacity for 16 floats");
    pthread_mutex_lock(&h75_resources_mutex);
    h75_status s = property(r, name, v, n, false, e);
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
h75_status h75_text_string(h75_resource *r, const char *text, h75_error *e) {
    if (!kind(r, H75_TEXT) || !text || strlen(text) > 65535)
        return h75_fail(e, H75_INVALID, "invalid text");
    char *copy = strdup(text);
    if (!copy)
        return h75_fail(e, H75_MEMORY, "copying text string");
    pthread_mutex_lock(&h75_resources_mutex);
    sdf_text_t *t = r->u.text.native;
    free(t->text);
    t->text = copy;
    t->text_len = strlen(copy);
    t->shape_dirty = true;
    r->u.text.dirty = true;
    pthread_mutex_unlock(&h75_resources_mutex);
    return H75_OK;
}
h75_status h75_scene_add(h75_resource *r, h75_resource *mesh, h75_resource *transform,
                         vec3 velocity, uint16_t *id, h75_error *e) {
    if (!kind(r, H75_SCENE) || !kind(mesh, H75_MESH) || !kind(transform, H75_TRANSFORM) || !id ||
        !finite_values(&velocity.x, 3))
        return h75_fail(e, H75_INVALID, "scene requires a mesh and transform");
    pthread_mutex_lock(&h75_resources_mutex);
    size_t n = r->u.scene.native->count;
    h75_status status = H75_OK;
    if (n >= 65535) {
        status = h75_fail(e, H75_INVALID, "scene instance limit");
        goto done;
    }
    if (n == r->u.scene.capacity) {
        size_t cap = n ? n * 2 : 8;
        h75_resource **meshes = calloc(cap, sizeof(*meshes)),
                     **transforms = calloc(cap, sizeof(*transforms));
        vec3 *vel = calloc(cap, sizeof(*vel));
        if (!meshes || !transforms || !vel) {
            free(meshes);
            free(transforms);
            free(vel);
            status = h75_fail(e, H75_MEMORY, "growing scene");
            goto done;
        }
        if (n) {
            memcpy(meshes, r->u.scene.meshes, n * sizeof(*meshes));
            memcpy(transforms, r->u.scene.transforms, n * sizeof(*transforms));
            memcpy(vel, r->u.scene.velocity, n * sizeof(*vel));
        }
        free(r->u.scene.meshes);
        free(r->u.scene.transforms);
        free(r->u.scene.velocity);
        r->u.scene.meshes = meshes;
        r->u.scene.transforms = transforms;
        r->u.scene.velocity = vel;
        r->u.scene.capacity = cap;
    }
    *id = r->u.scene.native->add_object(r->u.scene.native, mesh->u.mesh, &transform->u.transform);
    if (*id == UINT16_MAX) {
        status = h75_fail(e, H75_MEMORY, "growing native scene");
        goto done;
    }
    h75_resource_retain(mesh);
    h75_resource_retain(transform);
    r->u.scene.meshes[n] = mesh;
    r->u.scene.transforms[n] = transform;
    r->u.scene.velocity[n] = velocity;
done:
    pthread_mutex_unlock(&h75_resources_mutex);
    return status;
}
h75_status h75_scene_transforms(h75_resource *r, const float *values, size_t count, h75_error *e) {
    if (!kind(r, H75_SCENE) || !values || count > 65535)
        return h75_fail(e, H75_INVALID, "invalid transform batch");
    for (size_t i = 0; i < count; i++) {
        if (!finite_values(values + i * 9, 9) || !scale_valid(values + i * 9 + 6))
            return h75_fail(e, H75_INVALID,
                            "transforms require finite components and nonzero scales");
    }
    pthread_mutex_lock(&h75_resources_mutex);
    h75_status s = H75_OK;
    if (count != r->u.scene.native->count)
        s = h75_fail(e, H75_INVALID, "transform batch length must equal scene instance count");
    else
        for (size_t i = 0; i < count; i++)
            memcpy(&r->u.scene.transforms[i]->u.transform, values + i * 9, sizeof(transform_t));
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}

h75_status h75_scene_light(h75_resource *r, vec3 dir, RGBF col, float intensity, bool shadows,
                           uint16_t *id, h75_error *e) {
    if (!kind(r, H75_SCENE) || !id || !finite_values(&dir.x, 3) || !finite_values(&col.r, 3) ||
        !isfinite(intensity) || intensity < 0 || intensity > 1e4f ||
        dir.x * dir.x + dir.y * dir.y + dir.z * dir.z < 1e-10f || col.r < 0 || col.r > 1 ||
        col.g < 0 || col.g > 1 || col.b < 0 || col.b > 1)
        return h75_fail(e, H75_INVALID, "invalid directional light");
    pthread_mutex_lock(&h75_resources_mutex);
    *id = r->u.scene.native->add_directional(r->u.scene.native, dir, col, intensity, shadows);
    pthread_mutex_unlock(&h75_resources_mutex);
    return *id == UINT16_MAX ? h75_fail(e, H75_MEMORY, "adding light") : H75_OK;
}
h75_status h75_scene_light_set(h75_resource *r, uint16_t id, const char *name, const float *v,
                               size_t n, h75_error *e) {
    if (!kind(r, H75_SCENE) || !name || !finite_values(v, n))
        return h75_fail(e, H75_INVALID, "invalid light property");
    pthread_mutex_lock(&h75_resources_mutex);
    light_t *l = r->u.scene.native->get_directional(r->u.scene.native, id);
    h75_status s = H75_OK;
    if (!l)
        s = h75_fail(e, H75_INVALID, "invalid light id");
    else if (!strcmp(name, "direction") && n == 3 &&
             v[0] * v[0] + v[1] * v[1] + v[2] * v[2] >= 1e-10f) {
        l->direction = (vec3){v[0], v[1], v[2]};
        l->shadow_vp_valid = false;
    } else if (!strcmp(name, "color") && n == 3 && v[0] >= 0 && v[0] <= 1 && v[1] >= 0 &&
               v[1] <= 1 && v[2] >= 0 && v[2] <= 1)
        l->color = (RGBF){v[0], v[1], v[2]};
    else if (!strcmp(name, "intensity") && n == 1 && v[0] >= 0 && v[0] <= 1e4f)
        l->intensity = v[0];
    else if (!strcmp(name, "shadow_zoom") && n == 1 && v[0] >= 1e-4f && v[0] <= 1e4f)
        l->shadow_zoom = v[0];
    else if (!strcmp(name, "shadow_enabled") && n == 1 && (v[0] == 0 || v[0] == 1))
        l->shadow_enabled = v[0] != 0;
    else
        s = h75_fail(e, H75_INVALID, "invalid light property");
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
static h75_status command_validate(const h75_command *c, h75_error *e) {
    if (!c || c->op < H75_CLEAR || c->op > H75_RENDER_TEXT || !finite_values(c->p, 8))
        goto invalid;
    if (c->op == H75_BLIT && !kind(c->a, H75_IMAGE))
        goto invalid;
    if (c->op == H75_RENDER_SCENE && (!kind(c->a, H75_SCENE) || !kind(c->b, H75_CAMERA)))
        goto invalid;
    if (c->op == H75_RENDER_TEXT && !kind(c->a, H75_TEXT))
        goto invalid;
    if (c->op == H75_PIXEL || c->op == H75_LINE || c->op == H75_LINE_AA) {
        for (int i = 0; i < (c->op == H75_PIXEL ? 2 : 4); i++)
            if (c->p[i] < 0 || c->p[i] > 65535 || c->p[i] != floorf(c->p[i]))
                goto invalid;
    }
    if (c->op == H75_POLYGON || c->op == H75_GRADIENT) {
        if (c->polygon.num_points < 3 || c->polygon.num_points > MAX_POLY_POINTS)
            goto invalid;
        for (size_t i = 0; i < c->polygon.num_points; i++)
            if (!isfinite(c->polygon.points[i].x) || !isfinite(c->polygon.points[i].y) ||
                c->polygon.points[i].x < 0 || c->polygon.points[i].x > 1 ||
                c->polygon.points[i].y < 0 || c->polygon.points[i].y > 1)
                goto invalid;
        if (c->gradient.direction < GRADIENT_HORIZONTAL ||
            c->gradient.direction > GRADIENT_RADIAL || c->gradient.easing < EASE_LINEAR ||
            c->gradient.easing > EASE_IN_OUT_QUAD)
            goto invalid;
    }
    return H75_OK;
invalid:
    return h75_fail(e, H75_INVALID, "invalid drawing command or resource");
}
h75_status h75_drawlist_append(h75_resource *r, const h75_command *c, h75_error *e) {
    if (!kind(r, H75_DRAWLIST))
        return h75_fail(e, H75_INVALID, "expected draw list");
    h75_status s = command_validate(c, e);
    if (s)
        return s;
    pthread_mutex_lock(&h75_resources_mutex);
    if (r->u.list.count == r->u.list.capacity) {
        size_t cap = r->u.list.capacity ? r->u.list.capacity * 2 : 16;
        if (cap > 1048576) {
            s = h75_fail(e, H75_INVALID, "draw list command limit");
            goto done;
        }
        h75_command *cmds = realloc(r->u.list.commands, cap * sizeof(*cmds));
        if (!cmds) {
            s = h75_fail(e, H75_MEMORY, "growing draw list");
            goto done;
        }
        r->u.list.commands = cmds;
        r->u.list.capacity = cap;
    }
    r->u.list.commands[r->u.list.count++] = *c;
    h75_resource_retain(c->a);
    h75_resource_retain(c->b);
done:
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
h75_status h75_drawlist_geometry(h75_resource *r, const float *xy, size_t count, unsigned columns,
                                 RGBA color, bool antialias, bool replace, h75_error *e) {
    if (!kind(r, H75_DRAWLIST) || (columns != 2 && columns != 4) || count > 1048576 ||
        (!xy && count))
        return h75_fail(e, H75_INVALID, "expected bounded Nx2 points or Nx4 lines");
    for (size_t i = 0; i < count * columns; i++)
        if (!isfinite(xy[i]) || xy[i] < 0 || xy[i] > 65535 || xy[i] != floorf(xy[i]))
            return h75_fail(e, H75_INVALID, "geometry coordinates must be integers in [0,65535]");
    pthread_mutex_lock(&h75_resources_mutex);
    size_t base = replace ? 0 : r->u.list.count, need = base + count;
    h75_status s = H75_OK;
    if (need > 1048576) {
        s = h75_fail(e, H75_INVALID, "draw list command limit");
        goto done;
    }
    if (need > r->u.list.capacity) {
        size_t capacity = r->u.list.capacity * 2;
        if (capacity < need)
            capacity = need;
        if (capacity > 1048576)
            capacity = 1048576;
        h75_command *commands = realloc(r->u.list.commands, capacity * sizeof(*commands));
        if (!commands) {
            s = h75_fail(e, H75_MEMORY, "growing geometry batch");
            goto done;
        }
        r->u.list.commands = commands;
        r->u.list.capacity = capacity;
    }
    if (replace)
        for (size_t i = 0; i < r->u.list.count; i++) {
            h75_resource_release(r->u.list.commands[i].a);
            h75_resource_release(r->u.list.commands[i].b);
        }
    for (size_t i = 0; i < count; i++) {
        h75_command *command = &r->u.list.commands[base + i];
        *command = (h75_command){.op = columns == 2 ? H75_PIXEL
                                       : antialias  ? H75_LINE_AA
                                                    : H75_LINE,
                                 .color = color};
        memcpy(command->p, xy + i * columns, columns * sizeof(float));
    }
    r->u.list.count = need;
done:
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
/* Rasterize SDF only when text/style changes. Cache the allocation bounds,
 * never the dimensions overwritten by sdf_text_render(). */
static h75_status text_cache(h75_resource *r, h75_error *e) {
    if (r->u.text.cache && !r->u.text.dirty)
        return H75_OK;
    sdf_text_t *t = r->u.text.native;
    const float x = t->x, y = t->y, x0 = t->x0, y0 = t->y0;
    sdf_text_update(t, 0);
    t->x = x;
    t->y = y;
    t->x0 = x0;
    t->y0 = y0;
    if (t->shape_dirty)
        return h75_fail(e, H75_MEMORY, "shaping cached text");
    int minx = 0, miny = 0, maxx = 1, maxy = 1;
    const int pad = 2 + (int)ceilf(fmaxf(t->effects.glow_radius, t->effects.outline_width));
    for (size_t i = 0; i < t->_glyph_count; i++) {
        sdf_glyph_t *g = t->font->table[t->_glyph_indices[i] & 255u];
        if (!g && t->font->box_valid)
            g = &t->font->box;
        if (!g)
            continue;
        int gx = t->_ofs_x[i], gy = t->_ofs_y[i];
        if (gx - pad < minx)
            minx = gx - pad;
        if (gy - pad < miny)
            miny = gy - pad;
        if (gx + g->width + pad > maxx)
            maxx = gx + g->width + pad;
        if (gy + g->height + pad > maxy)
            maxy = gy + g->height + pad;
    }
    int64_t w = (int64_t)maxx - minx, h = (int64_t)maxy - miny;
    if (w <= 0 || h <= 0 || w > 65535 || h > 4096 || w * h * 4 > SPSC_MAX_ITEM_SIZE)
        return h75_fail(e, H75_INVALID, "cached text exceeds image allocation limits");
    image_buffer_t *im = image_buffer_new((int32_t)w, (int32_t)h);
    if (!im)
        return h75_fail(e, H75_MEMORY, "allocating cached SDF text");
    sdf_text_t copy = *t;
    copy.x = (float)-minx;
    copy.y = (float)-miny;
    copy.speed = 0;
    copy.wrap = false;
    sdf_text_render(&copy, (uint8_t *)im->data, (int)w, (int)h, (int)im->row_stride, 0);
    free(r->u.text.cache);
    r->u.text.cache = im;
    r->u.text.cache_x = minx;
    r->u.text.cache_y = miny;
    r->u.text.dirty = false;
    return H75_OK;
}

/* Caller holds h75_resources_mutex. Every display TLS binding is on the actual
 * executing thread, and is cleared on exit (including failed commands). */
h75_status h75_command_execute(hub75_display_t *d, const h75_command *c, float time, h75_error *e) {
    h75_status s = command_validate(c, e);
    if (s)
        return s;
    hub75gpu_t api = hub75_api(d);
    switch (c->op) {
    case H75_CLEAR:
        for (size_t i = 0; i < (size_t)d->width * d->height; i++)
            d->frame_buffer.data[i] = c->color;
        break;
    case H75_PIXEL:
        if (c->p[0] >= d->width || c->p[1] >= d->height)
            s = h75_fail(e, H75_INVALID, "pixel outside framebuffer");
        else
            api.pixel((uint16_t)c->p[0], (uint16_t)c->p[1], c->color);
        break;
    case H75_LINE:
        draw_line(d, (uint16_t)c->p[0], (uint16_t)c->p[1], (uint16_t)c->p[2], (uint16_t)c->p[3],
                  c->color);
        break;
    case H75_LINE_AA:
        draw_line_aa(d, (uint16_t)c->p[0], (uint16_t)c->p[1], (uint16_t)c->p[2], (uint16_t)c->p[3],
                     c->color);
        break;
    case H75_POLYGON: {
        Polygonf_t p = c->polygon;
        polygon_fill(d, &p, c->color);
        break;
    }
    case H75_GRADIENT: {
        Polygonf_t p = c->polygon;
        polygon_gradient(d, &p, c->gradient);
        break;
    }
    case H75_BLIT: {
        if (c->a->exports) {
            s = h75_fail(e, H75_BUSY, "release writable image views before drawing");
            break;
        }
        image_buffer_t *src = c->a->u.image;
        float w = c->p[2], h = c->p[3];
        if (w == 0)
            w = (float)src->dimensions.x;
        if (h == 0)
            h = (float)src->dimensions.y;
        if (w <= 0 || h <= 0) {
            s = h75_fail(e, H75_INVALID, "blit dimensions must be positive");
            break;
        }
        composite_rgba_over_rgba(&d->frame_buffer, src,
                                 (vec4){c->p[0], c->p[1], c->p[0] + w, c->p[1] + h},
                                 (vec4){0, 0, (float)src->dimensions.x, (float)src->dimensions.y});
        break;
    }
    case H75_RENDER_SCENE: {
        scene3d_t *scene = c->a->u.scene.native;
        if (!camera_valid(&c->b->u.camera)) {
            s = h75_fail(e, H75_INVALID, "camera view direction and up vector are degenerate");
            break;
        }
        transform_t *animated = calloc(scene->count ? scene->count : 1, sizeof(*animated));
        if (!animated) {
            s = h75_fail(e, H75_MEMORY, "animation transforms");
            break;
        }
        for (size_t i = 0; i < scene->count; i++) {
            animated[i] = c->a->u.scene.transforms[i]->u.transform;
            vec3 velocity = c->a->u.scene.velocity[i];
            animated[i].rotation.x += velocity.x * time;
            animated[i].rotation.y += velocity.y * time;
            animated[i].rotation.z += velocity.z * time;
            scene->instances[i].xform = &animated[i];
        }
        api.render_scene3d(&c->b->u.camera, scene, NULL);
        if (hub75_api_render_status() != HUB75_OK)
            s = h75_fail(e, H75_MEMORY, "native scene rendering allocation failed");
        for (size_t i = 0; i < scene->count; i++)
            scene->instances[i].xform = &c->a->u.scene.transforms[i]->u.transform;
        free(animated);
        break;
    }
    case H75_RENDER_TEXT: {
        s = text_cache(c->a, e);
        if (s)
            break;
        sdf_text_t *t = c->a->u.text.native;
        image_buffer_t *im = c->a->u.text.cache;
        float x = t->x0 + t->dir_x * t->speed * time + (float)c->a->u.text.cache_x;
        float y = t->y0 + t->dir_y * t->speed * time + (float)c->a->u.text.cache_y;
        if (t->wrap && t->speed != 0) {
            if (t->dir_x != 0) {
                float period = (float)(d->width + im->dimensions.x);
                x = fmodf(x + (float)im->dimensions.x, period);
                if (x < 0)
                    x += period;
                x -= (float)im->dimensions.x;
            }
            if (t->dir_y != 0) {
                float period = (float)(d->height + im->dimensions.y);
                y = fmodf(y + (float)im->dimensions.y, period);
                if (y < 0)
                    y += period;
                y -= (float)im->dimensions.y;
            }
        }
        vec3 sine = c->a->u.text.sine;
        if (sine.y != 0)
            composite_rgba_sine_over_rgba(&d->frame_buffer, im, x, y, sine.x, sine.y,
                                          time * sine.z);
        else
            composite_rgba_over_rgba(
                &d->frame_buffer, im,
                (vec4){x, y, x + (float)im->dimensions.x, y + (float)im->dimensions.y},
                (vec4){0, 0, (float)im->dimensions.x, (float)im->dimensions.y});
        break;
    }
    }
    hub75_api(NULL);
    return s;
}
h75_status h75_list_execute(hub75_display_t *d, h75_resource *list, float time, h75_error *e) {
    if (!kind(list, H75_DRAWLIST) || !isfinite(time))
        return h75_fail(e, H75_INVALID, "expected draw list and finite time");
    for (size_t i = 0; i < list->u.list.count; i++) {
        h75_status s = h75_command_execute(d, &list->u.list.commands[i], time, e);
        if (s)
            return s;
    }
    return H75_OK;
}
h75_status h75_copy_rgba(image_buffer_t *dst, const void *data, size_t size, uint32_t pitch,
                         int channels, h75_error *e) {
    uint32_t w = (uint32_t)dst->dimensions.x, h = (uint32_t)dst->dimensions.y;
    if (!data || !w || !h || (channels != 3 && channels != 4) ||
        pitch < (uint64_t)w * (unsigned)channels ||
        (uint64_t)(h - 1) * pitch + (uint64_t)w * (unsigned)channels > size)
        return h75_fail(e, H75_INVALID, "buffer shape/length/pitch does not match destination");
    if (channels == 4 && pitch == (size_t)w * 4) {
        memcpy(dst->data, data, (size_t)w * h * 4);
        return H75_OK;
    }
    for (uint32_t y = 0; y < h; y++) {
        RGBA *out = dst->data + (size_t)y * w;
        const uint8_t *src = (const uint8_t *)data + (size_t)y * pitch;
        if (channels == 4)
            memcpy(out, src, (size_t)w * 4);
        else
            for (uint32_t x = 0; x < w; x++)
                out[x] = (RGBA){src[x * 3], src[x * 3 + 1], src[x * 3 + 2], 255};
    }
    return H75_OK;
}
h75_status h75_image_upload(h75_resource *r, const void *data, size_t size, uint32_t w, uint32_t h,
                            uint32_t pitch, int channels, h75_error *e) {
    if (!kind(r, H75_IMAGE) || !data || w != (uint32_t)r->u.image->dimensions.x ||
        h != (uint32_t)r->u.image->dimensions.y || (channels != 3 && channels != 4) ||
        pitch < (uint64_t)w * (unsigned)channels ||
        (uint64_t)(h - 1) * pitch + (uint64_t)w * (unsigned)channels > size)
        return h75_fail(e, H75_INVALID, "buffer shape/length/pitch does not match image");
    pthread_mutex_lock(&h75_resources_mutex);
    h75_status s = H75_OK;
    if (r->exports)
        s = h75_fail(e, H75_BUSY, "image has active writable views");
    else
        s = h75_copy_rgba(r->u.image, data, size, pitch, channels, e);
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
h75_status h75_image_export(h75_resource *r, image_buffer_t *out, h75_error *e) {
    if (!kind(r, H75_IMAGE) || !out)
        return h75_fail(e, H75_INVALID, "only images export buffers");
    pthread_mutex_lock(&h75_resources_mutex);
    r->exports++;
    *out = *r->u.image;
    pthread_mutex_unlock(&h75_resources_mutex);
    return H75_OK;
}
void h75_image_unexport(h75_resource *r) {
    pthread_mutex_lock(&h75_resources_mutex);
    if (r->exports)
        r->exports--;
    pthread_mutex_unlock(&h75_resources_mutex);
}
h75_status h75_image_copy(h75_resource *r, void *out, size_t size, h75_error *e) {
    if (!kind(r, H75_IMAGE) || !out ||
        size != (size_t)r->u.image->dimensions.x * (size_t)r->u.image->dimensions.y * 4)
        return h75_fail(e, H75_INVALID, "image copy size mismatch");
    pthread_mutex_lock(&h75_resources_mutex);
    h75_status s = H75_OK;
    if (r->exports)
        s = h75_fail(e, H75_BUSY, "image has active writable views");
    else
        memcpy(out, r->u.image->data, size);
    pthread_mutex_unlock(&h75_resources_mutex);
    return s;
}
