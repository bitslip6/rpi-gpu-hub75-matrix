#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "hub75gpu.h"

// Module-wide API table, initialized by hub75_init(scene)
static hub75gpu_t g_api;
static int g_api_inited = 0;

// Capsule names for type safety
#define CAPS_SCENE       "hub75gpu.hub75_display"
#define CAPS_CAMERA      "hub75gpu.camera"
#define CAPS_TRANSFORM   "hub75gpu.transform"
#define CAPS_OBJECT      "hub75gpu.object"
#define CAPS_OBJECT_SCENE "hub75gpu.scene3d"

// Helpers
static void* cap_get_ptr(PyObject *cap, const char *name) {
    if (!PyCapsule_CheckExact(cap)) {
        PyErr_SetString(PyExc_TypeError, "expected capsule");
        return NULL;
    }
    void *ptr = PyCapsule_GetPointer(cap, name);
    if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "capsule has wrong name or NULL pointer");
        return NULL;
    }
    return ptr;
}

// Proper PyCapsule destructors (signature: void(PyObject*))
static void capsule_free_camera(PyObject *cap) {
    void *ptr = PyCapsule_GetPointer(cap, CAPS_CAMERA);
    if (ptr) free(ptr);
}
static void capsule_free_transform(PyObject *cap) {
    void *ptr = PyCapsule_GetPointer(cap, CAPS_TRANSFORM);
    if (ptr) free(ptr);
}

static PyObject* cap_from_ptr(void *ptr, const char *name, PyCapsule_Destructor dtor) {
    if (!ptr) Py_RETURN_NONE;
    return PyCapsule_New(ptr, name, dtor);
}

static int require_api(void) {
    if (!g_api_inited) {
        PyErr_SetString(PyExc_RuntimeError, "hub75 API not initialized. Call hub75_init(scene) first.");
        return 0;
    }
    return 1;
}

// ---------------- Python Display type ----------------
typedef struct {
    PyObject_HEAD
    hub75_display_t *s;  // borrowed/owned by C runtime (no free here unless API provides)
} PyDisplay;

static PyTypeObject PyDisplayType; // forward

static inline int is_display(PyObject *obj) {
    return PyObject_TypeCheck(obj, &PyDisplayType);
}

static inline hub75_display_t* scene_from_display(PyObject *obj) {
    return ((PyDisplay*)obj)->s;
}

static hub75_display_t* scene_from_any(PyObject *obj) {
    if (is_display(obj)) return scene_from_display(obj);
    if (PyCapsule_CheckExact(obj)) return (hub75_display_t*)cap_get_ptr(obj, CAPS_SCENE);
    PyErr_SetString(PyExc_TypeError, "expected Display or Scene capsule");
    return NULL;
}

static PyObject* display_from_ptr(hub75_display_t *s) {
    if (!s) Py_RETURN_NONE;
    PyDisplay *self = PyObject_New(PyDisplay, &PyDisplayType);
    if (!self) return NULL;
    self->s = s;
    return (PyObject*)self;
}

static void pydisplay_dealloc(PyDisplay *self) {
    // If a proper free function exists in C API, call it here.
    // e.g., hub75_display_free(self->s);
    PyObject_Del((PyObject*)self);
}

// Macros to define Display.getset properties
#define DEF_PROP_U32(name, field, cast_t) \
static PyObject* disp_get_##name(PyObject *self, void *closure) { \
    hub75_display_t *s = ((PyDisplay*)self)->s; \
    return PyLong_FromUnsignedLong((unsigned long)(s->field)); \
} \
static int disp_set_##name(PyObject *self, PyObject *value, void *closure) { \
    if (!value) return -1; \
    unsigned long v = PyLong_AsUnsignedLong(value); \
    if (PyErr_Occurred()) return -1; \
    hub75_display_t *s = ((PyDisplay*)self)->s; \
    s->field = (cast_t)v; \
    return 0; \
}

#define DEF_PROP_BOOL(name, field) \
static PyObject* disp_get_##name(PyObject *self, void *closure) { \
    hub75_display_t *s = ((PyDisplay*)self)->s; \
    if (s->field) Py_RETURN_TRUE; else Py_RETURN_FALSE; \
} \
static int disp_set_##name(PyObject *self, PyObject *value, void *closure) { \
    if (!value) return -1; \
    int v = PyObject_IsTrue(value); \
    if (v < 0) return -1; \
    hub75_display_t *s = ((PyDisplay*)self)->s; \
    s->field = v ? true : false; \
    return 0; \
}

#define DEF_PROP_FLOAT(name, field) \
static PyObject* disp_get_##name(PyObject *self, void *closure) { \
    hub75_display_t *s = ((PyDisplay*)self)->s; \
    return PyFloat_FromDouble((double)(s->field)); \
} \
static int disp_set_##name(PyObject *self, PyObject *value, void *closure) { \
    if (!value) return -1; \
    double v = PyFloat_AsDouble(value); \
    if (PyErr_Occurred()) return -1; \
    hub75_display_t *s = ((PyDisplay*)self)->s; \
    s->field = (float)v; \
    return 0; \
}

// Define properties
DEF_PROP_U32(width, width, uint16_t)
DEF_PROP_U32(height, height, uint16_t)
DEF_PROP_U32(panel_width, panel_width, uint16_t)
DEF_PROP_U32(panel_height, panel_height, uint16_t)
DEF_PROP_U32(stride, stride, uint8_t)
DEF_PROP_U32(num_ports, num_ports, uint8_t)
DEF_PROP_U32(num_chains, num_chains, uint8_t)
DEF_PROP_U32(bit_depth, bit_depth, uint8_t)
DEF_PROP_U32(brightness, brightness, uint8_t)
DEF_PROP_U32(panel_order, panel_order, panel_order_t)
DEF_PROP_U32(fps, fps, uint16_t)
DEF_PROP_BOOL(auto_fps, auto_fps)
DEF_PROP_FLOAT(gamma, gamma)
DEF_PROP_FLOAT(red_gamma, red_gamma)
DEF_PROP_FLOAT(green_gamma, green_gamma)
DEF_PROP_FLOAT(blue_gamma, blue_gamma)
DEF_PROP_FLOAT(red_linear, red_linear)
DEF_PROP_FLOAT(green_linear, green_linear)
DEF_PROP_FLOAT(blue_linear, blue_linear)
DEF_PROP_FLOAT(dither, dither)
DEF_PROP_FLOAT(tone_level, tone_level)
DEF_PROP_BOOL(quant_dither, quant_dither)
DEF_PROP_BOOL(jitter_brightness, jitter_brightness)
DEF_PROP_U32(motion_blur_frames, motion_blur_frames, uint8_t)
DEF_PROP_BOOL(show_fps, show_fps)
DEF_PROP_BOOL(enhanced_debug, enhanced_debug)

// Read-only getters
static PyObject* disp_get_render_width(PyObject *self, void *closure) {
    hub75_display_t *s = ((PyDisplay*)self)->s; return PyLong_FromUnsignedLong((unsigned long)s->render_width);
}
static PyObject* disp_get_render_height(PyObject *self, void *closure) {
    hub75_display_t *s = ((PyDisplay*)self)->s; return PyLong_FromUnsignedLong((unsigned long)s->render_height);
}
static PyObject* disp_get_frame_index(PyObject *self, void *closure) {
    hub75_display_t *s = ((PyDisplay*)self)->s; return PyLong_FromUnsignedLong((unsigned long)s->frame_index);
}
static PyObject* disp_get_do_render(PyObject *self, void *closure) {
    hub75_display_t *s = ((PyDisplay*)self)->s; if (s->do_render) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

// shader_file str or None
static PyObject* disp_get_shader_file(PyObject *self, void *closure) {
    hub75_display_t *s = ((PyDisplay*)self)->s; if (!s->shader_file) Py_RETURN_NONE; return PyUnicode_FromString(s->shader_file);
}
static int disp_set_shader_file(PyObject *self, PyObject *value, void *closure) {
    hub75_display_t *s = ((PyDisplay*)self)->s;
    if (s->shader_file) { free(s->shader_file); s->shader_file = NULL; }
    if (!value || value == Py_None) return 0;
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "shader_file must be str or None");
        return -1;
    }
    PyObject *b = PyUnicode_AsUTF8String(value); if (!b) return -1;
    const char *c = PyBytes_AsString(b); if (!c) { Py_DECREF(b); return -1; }
    s->shader_file = strdup(c);
    Py_DECREF(b);
    return 0;
}

static PyGetSetDef Display_getset[] = {
    {"width", disp_get_width, disp_set_width, "total width", NULL},
    {"height", disp_get_height, disp_set_height, "total height", NULL},
    {"panel_width", disp_get_panel_width, disp_set_panel_width, "panel width", NULL},
    {"panel_height", disp_get_panel_height, disp_set_panel_height, "panel height", NULL},
    {"stride", disp_get_stride, disp_set_stride, "bytes per pixel", NULL},
    {"num_ports", disp_get_num_ports, disp_set_num_ports, "number of HUB75 ports", NULL},
    {"num_chains", disp_get_num_chains, disp_set_num_chains, "panels per chain", NULL},
    {"bit_depth", disp_get_bit_depth, disp_set_bit_depth, "PWM bit depth", NULL},
    {"brightness", disp_get_brightness, disp_set_brightness, "brightness", NULL},
    {"panel_order", disp_get_panel_order, disp_set_panel_order, "panel color order", NULL},
    {"fps", disp_get_fps, disp_set_fps, "target frames per second", NULL},
    {"auto_fps", disp_get_auto_fps, disp_set_auto_fps, "auto fps mode", NULL},
    {"gamma", disp_get_gamma, disp_set_gamma, "gamma", NULL},
    {"red_gamma", disp_get_red_gamma, disp_set_red_gamma, "red gamma", NULL},
    {"green_gamma", disp_get_green_gamma, disp_set_green_gamma, "green gamma", NULL},
    {"blue_gamma", disp_get_blue_gamma, disp_set_blue_gamma, "blue gamma", NULL},
    {"red_linear", disp_get_red_linear, disp_set_red_linear, "red linear", NULL},
    {"green_linear", disp_get_green_linear, disp_set_green_linear, "green linear", NULL},
    {"blue_linear", disp_get_blue_linear, disp_set_blue_linear, "blue linear", NULL},
    {"dither", disp_get_dither, disp_set_dither, "dithering strength", NULL},
    {"tone_level", disp_get_tone_level, disp_set_tone_level, "tone mapping level", NULL},
    {"quant_dither", disp_get_quant_dither, disp_set_quant_dither, "quantization dither", NULL},
    {"jitter_brightness", disp_get_jitter_brightness, disp_set_jitter_brightness, "jitter brightness", NULL},
    {"motion_blur_frames", disp_get_motion_blur_frames, disp_set_motion_blur_frames, "motion blur frames", NULL},
    {"show_fps", disp_get_show_fps, disp_set_show_fps, "show FPS overlay", NULL},
    {"enhanced_debug", disp_get_enhanced_debug, disp_set_enhanced_debug, "extra debug", NULL},
    {"render_width", disp_get_render_width, NULL, "render width (read-only)", NULL},
    {"render_height", disp_get_render_height, NULL, "render height (read-only)", NULL},
    {"frame_index", disp_get_frame_index, NULL, "frame index (read-only)", NULL},
    {"do_render", disp_get_do_render, NULL, "do_render flag (read-only)", NULL},
    {"shader_file", disp_get_shader_file, disp_set_shader_file, "shader/video file path", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject PyDisplayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "hub75gpu.Display",
    .tp_basicsize = sizeof(PyDisplay),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)pydisplay_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Hub75 display configuration and runtime",
    .tp_getset = Display_getset,
};

// new_scene() -> Scene capsule
static PyObject* py_new_scene(PyObject *self, PyObject *args) {
    hub75_display_t *s = hub75_display_new();
    if (!s) {
        PyErr_SetString(PyExc_MemoryError, "new_scene failed");
        return NULL;
    }
    return display_from_ptr(s);
}

// parse_args(argv: list[str]) -> Scene capsule
static PyObject* py_parse_args(PyObject *self, PyObject *args) {
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;
    if (!PySequence_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "expected a sequence of strings");
        return NULL;
    }

    Py_ssize_t n = PySequence_Size(seq);
    if (n < 0) return NULL;

    // Build argv with a fake program name at argv[0]
    int argc = (int)(n + 1);
    char **argv = (char**)calloc((size_t)argc, sizeof(char*));
    if (!argv) {
        PyErr_NoMemory();
        return NULL;
    }
    argv[0] = strdup("hub75gpu.py");

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        if (!item) { /* cleanup */ goto fail; }
        if (!PyUnicode_Check(item)) {
            Py_DECREF(item);
            PyErr_SetString(PyExc_TypeError, "all argv items must be str");
            goto fail;
        }
        PyObject *utf8 = PyUnicode_AsUTF8String(item);
        Py_DECREF(item);
        if (!utf8) goto fail;
        const char *s = PyBytes_AsString(utf8);
        if (!s) { Py_DECREF(utf8); goto fail; }
        argv[i+1] = strdup(s);
        Py_DECREF(utf8);
        if (!argv[i+1]) { PyErr_NoMemory(); goto fail; }
    }

    // Call into C parser
    hub75_display_t *scene = hub75_display_parse_args(argc, argv);
    if (!scene) {
        // parser may exit on error; if it returns NULL, treat as failure
        PyErr_SetString(PyExc_RuntimeError, "hub75_display_parse_args failed");
        goto fail;
    }
    // Ensure shader_file (if set) does not reference temporary argv memory
    if (scene->shader_file) {
        char *copy = strdup(scene->shader_file);
        if (copy) scene->shader_file = copy;
    }

    // Cleanup argv strings
    for (int i = 0; i < argc; ++i) { if (argv[i]) free(argv[i]); }
    free(argv);
    return display_from_ptr(scene);

fail:
    if (argv) {
        for (int i = 0; i < argc; ++i) { if (argv[i]) free(argv[i]); }
        free(argv);
    }
    return NULL;
}

// hub75_init(scene_capsule)
static PyObject* py_hub75_init(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;
    g_api = hub75_api(s);
    g_api_inited = 1;
    Py_RETURN_NONE;
}

// geo_camera() -> camera capsule
static PyObject* py_geo_camera(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    camera_t *cam = g_api.geo_camera();
    return cap_from_ptr(cam, CAPS_CAMERA, capsule_free_camera);
}

// geo_transform() -> transform capsule
static PyObject* py_geo_transform(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    transform_t *x = g_api.geo_transform();
    return cap_from_ptr(x, CAPS_TRANSFORM, capsule_free_transform);
}

// geo_cube(mode:int, cull_backface:bool) -> object capsule
static PyObject* py_geo_cube(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    int mode_int = 0; int cull = 1;
    if (!PyArg_ParseTuple(args, "ip", &mode_int, &cull)) return NULL;
    object_draw_mode_t mode = (object_draw_mode_t)mode_int;
    object_t *obj = g_api.geo_cube(mode, cull ? true : false);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL /* no free available */);
}

// geo_object(nv, ne, nf) -> object capsule
static PyObject* py_geo_object(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int nv, ne, nf;
    if (!PyArg_ParseTuple(args, "III", &nv, &ne, &nf)) return NULL;
    object_t *obj = g_api.geo_object((uint16_t)nv, (uint16_t)ne, (uint16_t)nf);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}

static PyObject* py_geo_tetrahedron(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    object_t *obj = g_api.geo_tetrahedron();
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_octahedron(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    object_t *obj = g_api.geo_octahedron();
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_pyramid(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    object_t *obj = g_api.geo_pyramid();
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_cylinder(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int seg;
    if (!PyArg_ParseTuple(args, "I", &seg)) return NULL;
    object_t *obj = g_api.geo_cylinder((uint16_t)seg);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_sphere(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int subs;
    if (!PyArg_ParseTuple(args, "I", &subs)) return NULL;
    object_t *obj = g_api.geo_sphere((uint16_t)subs);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_torus(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int maj, min;
    if (!PyArg_ParseTuple(args, "II", &maj, &min)) return NULL;
    object_t *obj = g_api.geo_torus((uint16_t)maj, (uint16_t)min);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_plane(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int w, h;
    if (!PyArg_ParseTuple(args, "II", &w, &h)) return NULL;
    object_t *obj = g_api.geo_plane((uint16_t)w, (uint16_t)h);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}

// scene_new(count) -> object_scene capsule
static PyObject* py_scene_new(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int count;
    if (!PyArg_ParseTuple(args, "I", &count)) return NULL;
    scene3d_t *os = g_api.scene3d_new((uint16_t)count);
    return cap_from_ptr(os, CAPS_OBJECT_SCENE, NULL /* freed via api_object_scene_free in C land by user if needed */);
}

static PyObject* py_scene_set_current(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    scene3d_t *os = (scene3d_t*)cap_get_ptr(cap, CAPS_OBJECT_SCENE);
    if (!os) return NULL;
    g_api.scene3d_set_current(os);
    Py_RETURN_NONE;
}

static PyObject* py_scene_clear_current(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    g_api.scene3d_clear_current();
    Py_RETURN_NONE;
}

static PyObject* py_scene_set_ambient(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    float r,g,b;
    if (!PyArg_ParseTuple(args, "fff", &r,&g,&b)) return NULL;
    RGBF c = {r,g,b};
    g_api.scene3d_set_ambient(c);
    Py_RETURN_NONE;
}

static PyObject* py_scene_add_directional(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    float dx,dy,dz, cr,cg,cb, intensity; int casts;
    if (!PyArg_ParseTuple(args, "fffffffp", &dx,&dy,&dz, &cr,&cg,&cb, &intensity, &casts)) return NULL;
    light_vec3 dir = {dx,dy,dz};
    RGBF col = {cr,cg,cb};
    uint16_t id = g_api.scene3d_add_directional(dir, col, intensity, casts ? true : false);
    return PyLong_FromUnsignedLong((unsigned long)id);
}

static PyObject* py_scene_add_object(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *o_cap=NULL, *t_cap=NULL;
    if (!PyArg_ParseTuple(args, "OO", &o_cap, &t_cap)) return NULL;
    object_t *obj = (object_t*)cap_get_ptr(o_cap, CAPS_OBJECT);
    transform_t *x = (transform_t*)cap_get_ptr(t_cap, CAPS_TRANSFORM);
    if (!obj || !x) return NULL;
    uint16_t id = g_api.scene3d_add_object(obj, x);
    return PyLong_FromUnsignedLong((unsigned long)id);
}

static PyObject* py_scene_get_object(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int id;
    if (!PyArg_ParseTuple(args, "I", &id)) return NULL;
    object_t *obj = g_api.scene3d_get_object((uint16_t)id);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}

static PyObject* py_scene_get_transform(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int id;
    if (!PyArg_ParseTuple(args, "I", &id)) return NULL;
    transform_t *x = g_api.scene3d_get_transform((uint16_t)id);
    return cap_from_ptr(x, CAPS_TRANSFORM, NULL);
}

// render_scene(camera, object_scene) -> None (uses scene-owned lighting)
static PyObject* py_render_scene(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *c_cap=NULL, *os_cap=NULL;
    if (!PyArg_ParseTuple(args, "OO", &c_cap, &os_cap)) return NULL;
    camera_t *cam = (camera_t*)cap_get_ptr(c_cap, CAPS_CAMERA);
    scene3d_t *os = (scene3d_t*)cap_get_ptr(os_cap, CAPS_OBJECT_SCENE);
    if (!cam || !os) return NULL;
    g_api.render_scene3d(cam, os, NULL);
    Py_RETURN_NONE;
}

// render_wire(camera, object, transform)
static PyObject* py_render_wire(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *c_cap=NULL, *o_cap=NULL, *t_cap=NULL;
    if (!PyArg_ParseTuple(args, "OOO", &c_cap, &o_cap, &t_cap)) return NULL;
    camera_t *cam = (camera_t*)cap_get_ptr(c_cap, CAPS_CAMERA);
    object_t *obj = (object_t*)cap_get_ptr(o_cap, CAPS_OBJECT);
    transform_t *x = (transform_t*)cap_get_ptr(t_cap, CAPS_TRANSFORM);
    if (!cam || !obj || !x) return NULL;
    g_api.render_wire(cam, obj, x, NULL);
    Py_RETURN_NONE;
}

// render_filled(camera, object, transform)
static PyObject* py_render_filled(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *c_cap=NULL, *o_cap=NULL, *t_cap=NULL;
    if (!PyArg_ParseTuple(args, "OOO", &c_cap, &o_cap, &t_cap)) return NULL;
    camera_t *cam = (camera_t*)cap_get_ptr(c_cap, CAPS_CAMERA);
    object_t *obj = (object_t*)cap_get_ptr(o_cap, CAPS_OBJECT);
    transform_t *x = (transform_t*)cap_get_ptr(t_cap, CAPS_TRANSFORM);
    if (!cam || !obj || !x) return NULL;
    g_api.render_filled(cam, obj, x, NULL);
    Py_RETURN_NONE;
}

// frame_begin(), frame_end()
static PyObject* py_frame_begin(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    g_api.frame_begin();
    Py_RETURN_NONE;
}
static PyObject* py_frame_end(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    g_api.frame_end();
    Py_RETURN_NONE;
}

// clear()
static PyObject* py_clear(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    g_api.clear();
    Py_RETURN_NONE;
}

// pixel(x,y,(r,g,b))
static PyObject* py_pixel(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int x,y; unsigned int r,g,b;
    if (!PyArg_ParseTuple(args, "II(III)", &x,&y, &r,&g,&b)) return NULL;
    RGB c = {(uint8_t)r,(uint8_t)g,(uint8_t)b};
    g_api.pixel((uint16_t)x, (uint16_t)y, c);
    Py_RETURN_NONE;
}

// line(x0,y0,x1,y1,(r,g,b))
static PyObject* py_line(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int x0,y0,x1,y1; unsigned int r,g,b;
    if (!PyArg_ParseTuple(args, "IIII(III)", &x0,&y0,&x1,&y1, &r,&g,&b)) return NULL;
    RGB c = {(uint8_t)r,(uint8_t)g,(uint8_t)b};
    g_api.line((uint16_t)x0,(uint16_t)y0,(uint16_t)x1,(uint16_t)y1,c);
    Py_RETURN_NONE;
}

// hub75_display_start(scene)
static PyObject* py_display_start(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;
    hub75_display_start(s);
    Py_RETURN_NONE;
}

// hub75_display_run(scene) -> blocks until shutdown; releases GIL while running
static PyObject* py_display_run(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;
    Py_BEGIN_ALLOW_THREADS
    hub75_display_run(s);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

// hub75_display_wait(scene)
static PyObject* py_display_wait(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;
    hub75_display_wait(s);
    Py_RETURN_NONE;
}

// hub75_display_request_shutdown(scene)
static PyObject* py_display_request_shutdown(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;
    hub75_display_request_shutdown(s);
    Py_RETURN_NONE;
}

// signal_handler_install()
static PyObject* py_signal_install(PyObject *self, PyObject *args) {
    signal_handler_install();
    Py_RETURN_NONE;
}

// Start the RGB->BCM mapper thread in C
static PyObject* py_start_mapper(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;
    int rc = pthread_create(&s->mapper_thread, NULL, mapper_thread_main, s);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "failed to start mapper thread");
        return NULL;
    }
    Py_RETURN_NONE;
}

// Helper to check simple suffix
static int ends_with(const char *s, const char *suffix) {
    if (!s || !suffix) return 0;
    size_t ls = strlen(s), lf = strlen(suffix);
    if (lf > ls) return 0;
    return (strcmp(s + (ls - lf), suffix) == 0);
}

// Start the renderer thread (shader or video) in C
static PyObject* py_start_renderer(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    hub75_display_t *s = scene_from_any(cap);
    if (!s) return NULL;

    void *(*fn)(void*) = render_shader; // default to shader
    if (s->shader_file && !ends_with(s->shader_file, ".glsl")) {
        fn = render_video_fn;
    }
    int rc = pthread_create(&s->render_thread, NULL, fn, s);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "failed to start render thread");
        return NULL;
    }
    Py_RETURN_NONE;
}

// ---- Transform & Camera field setters (for CPU 3D control) ----
static PyObject* py_transform_set_pos(PyObject *self, PyObject *args) {
    PyObject *cap=NULL; float x,y,z;
    if (!PyArg_ParseTuple(args, "Offf", &cap, &x,&y,&z)) return NULL;
    transform_t *t = (transform_t*)cap_get_ptr(cap, CAPS_TRANSFORM);
    if (!t) return NULL;
    t->position = (vec3){x,y,z};
    Py_RETURN_NONE;
}
static PyObject* py_transform_set_rot(PyObject *self, PyObject *args) {
    PyObject *cap=NULL; float x,y,z;
    if (!PyArg_ParseTuple(args, "Offf", &cap, &x,&y,&z)) return NULL;
    transform_t *t = (transform_t*)cap_get_ptr(cap, CAPS_TRANSFORM);
    if (!t) return NULL;
    t->rotation = (vec3){x,y,z};
    Py_RETURN_NONE;
}
static PyObject* py_transform_set_scale(PyObject *self, PyObject *args) {
    PyObject *cap=NULL; float x,y,z;
    if (!PyArg_ParseTuple(args, "Offf", &cap, &x,&y,&z)) return NULL;
    transform_t *t = (transform_t*)cap_get_ptr(cap, CAPS_TRANSFORM);
    if (!t) return NULL;
    t->scale = (vec3){x,y,z};
    Py_RETURN_NONE;
}
static PyObject* py_camera_set_all(PyObject *self, PyObject *args) {
    PyObject *cap=NULL;
    float px,py,pz, tx,ty,tz, ux,uy,uz, fov, aspect, zn, zf;
    if (!PyArg_ParseTuple(args, "Offfffffffffff", &cap,
                          &px,&py,&pz, &tx,&ty,&tz, &ux,&uy,&uz, &fov, &aspect, &zn, &zf)) return NULL;
    camera_t *c = (camera_t*)cap_get_ptr(cap, CAPS_CAMERA);
    if (!c) return NULL;
    c->position = (vec3){px,py,pz};
    c->target   = (vec3){tx,ty,tz};
    c->up       = (vec3){ux,uy,uz};
    c->fov_y    = fov;
    c->aspect   = aspect;
    c->z_near   = zn;
    c->z_far    = zf;
    Py_RETURN_NONE;
}


static PyMethodDef Hub75Methods[] = {
    {"new_scene", (PyCFunction)py_new_scene, METH_VARARGS, "Create a default Display and return it."},
    {"parse_args", (PyCFunction)py_parse_args, METH_VARARGS, "Parse argv (list[str]) into a new Display and return it."},
    {"hub75_init", (PyCFunction)py_hub75_init, METH_VARARGS, "Initialize API with a scene capsule (sets current scene)."},
    {"display_start", (PyCFunction)py_display_start, METH_VARARGS, "Validate and allocate buffers for the display (hub75_display_start)."},
    {"display_run", (PyCFunction)py_display_run, METH_VARARGS, "Run the HUB75 driver loop (blocks until shutdown)."},
    {"display_wait", (PyCFunction)py_display_wait, METH_VARARGS, "Join mapper/render threads and return."},
    {"request_shutdown", (PyCFunction)py_display_request_shutdown, METH_VARARGS, "Signal render threads to shut down (hub75_display_request_shutdown)."},
    {"signal_handler_install", (PyCFunction)py_signal_install, METH_NOARGS, "Install C-level SIGINT/SIGTERM handlers for graceful shutdown."},
    {"start_mapper", (PyCFunction)py_start_mapper, METH_VARARGS, "Start the RGB->BCM mapper thread in C."},
    {"start_renderer", (PyCFunction)py_start_renderer, METH_VARARGS, "Start the renderer thread (shader/video) in C."},
    
    {"transform_set_pos", (PyCFunction)py_transform_set_pos, METH_VARARGS, "Set transform.position (x,y,z)."},
    {"transform_set_rot", (PyCFunction)py_transform_set_rot, METH_VARARGS, "Set transform.rotation (x,y,z radians)."},
    {"transform_set_scale", (PyCFunction)py_transform_set_scale, METH_VARARGS, "Set transform.scale (x,y,z)."},
    {"camera_set_all", (PyCFunction)py_camera_set_all, METH_VARARGS, "Set camera fields: pos, target, up, fov_y, aspect, znear, zfar."},

    {"geo_camera", (PyCFunction)py_geo_camera, METH_NOARGS, "Create a new camera (capsule)."},
    {"geo_transform", (PyCFunction)py_geo_transform, METH_NOARGS, "Create a new transform (capsule)."},
    {"geo_object", (PyCFunction)py_geo_object, METH_VARARGS, "Create a raw object with counts."},
    {"geo_cube", (PyCFunction)py_geo_cube, METH_VARARGS, "Create a cube object."},
    {"geo_tetrahedron", (PyCFunction)py_geo_tetrahedron, METH_NOARGS, "Create a tetrahedron object."},
    {"geo_octahedron", (PyCFunction)py_geo_octahedron, METH_NOARGS, "Create an octahedron object."},
    {"geo_pyramid", (PyCFunction)py_geo_pyramid, METH_NOARGS, "Create a pyramid object."},
    {"geo_cylinder", (PyCFunction)py_geo_cylinder, METH_VARARGS, "Create a cylinder object."},
    {"geo_sphere", (PyCFunction)py_geo_sphere, METH_VARARGS, "Create a sphere object."},
    {"geo_torus", (PyCFunction)py_geo_torus, METH_VARARGS, "Create a torus object."},
    {"geo_plane", (PyCFunction)py_geo_plane, METH_VARARGS, "Create a plane object."},

    {"scene_new", (PyCFunction)py_scene_new, METH_VARARGS, "Create a new object_scene (capsule)."},
    {"scene_set_current", (PyCFunction)py_scene_set_current, METH_VARARGS, "Set current object_scene for convenience wrappers."},
    {"scene_clear_current", (PyCFunction)py_scene_clear_current, METH_NOARGS, "Clear current object_scene."},
    {"scene_set_ambient", (PyCFunction)py_scene_set_ambient, METH_VARARGS, "Set ambient color (r,g,b floats)."},
    {"scene_add_directional", (PyCFunction)py_scene_add_directional, METH_VARARGS, "Add directional light; returns id."},
    {"scene_add_object", (PyCFunction)py_scene_add_object, METH_VARARGS, "Add object+transform; returns id."},
    {"scene_get_object", (PyCFunction)py_scene_get_object, METH_VARARGS, "Get object capsule by id."},
    {"scene_get_transform", (PyCFunction)py_scene_get_transform, METH_VARARGS, "Get transform capsule by id."},

    {"render_scene", (PyCFunction)py_render_scene, METH_VARARGS, "Render full scene using scene-owned lighting."},
    {"render_wire", (PyCFunction)py_render_wire, METH_VARARGS, "Render a single object in wireframe."},
    {"render_filled", (PyCFunction)py_render_filled, METH_VARARGS, "Render a single object filled."},

    {"frame_begin", (PyCFunction)py_frame_begin, METH_NOARGS, "Begin a frame."},
    {"frame_end", (PyCFunction)py_frame_end, METH_NOARGS, "End a frame."},
    {"clear", (PyCFunction)py_clear, METH_NOARGS, "Clear the framebuffer."},
    {"pixel", (PyCFunction)py_pixel, METH_VARARGS, "Draw a pixel (x,y,(r,g,b) ints)."},
    {"line", (PyCFunction)py_line, METH_VARARGS, "Draw a line (x0,y0,x1,y1,(r,g,b) ints)."},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef Hub75Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "hub75gpu",
    .m_doc = "Python bindings for hub75gpu_t API",
    .m_size = -1,
    .m_methods = Hub75Methods,
};

PyMODINIT_FUNC PyInit_hub75gpu(void) {
    PyObject *m = PyModule_Create(&Hub75Module);
    if (!m) return NULL;
    if (PyType_Ready(&PyDisplayType) < 0) return NULL;
    Py_INCREF(&PyDisplayType);
    if (PyModule_AddObject(m, "Display", (PyObject*)&PyDisplayType) < 0) {
        Py_DECREF(&PyDisplayType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
