#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "hub75gpu.h"

// Module-wide API table, initialized by hub75_init(scene)
static hub75gpu_t g_api;
static int g_api_inited = 0;

// Capsule names for type safety
#define CAPS_SCENE       "hub75gpu.scene_info"
#define CAPS_CAMERA      "hub75gpu.camera"
#define CAPS_TRANSFORM   "hub75gpu.transform"
#define CAPS_OBJECT      "hub75gpu.object"
#define CAPS_OBJECT_SCENE "hub75gpu.object_scene"

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

// new_scene() -> Scene capsule
static PyObject* py_new_scene(PyObject *self, PyObject *args) {
    scene_info *s = scene_new();
    if (!s) {
        PyErr_SetString(PyExc_MemoryError, "new_scene failed");
        return NULL;
    }
    return cap_from_ptr(s, CAPS_SCENE, NULL /* no generic free available for scene */);
}

// hub75_init(scene_capsule)
static PyObject* py_hub75_init(PyObject *self, PyObject *args) {
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    scene_info *s = (scene_info*)cap_get_ptr(cap, CAPS_SCENE);
    if (!s) return NULL;
    g_api = hub75gpu(s);
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
    int mode_int = 0; int cull = 1;
    if (!PyArg_ParseTuple(args, "ip", &mode_int, &cull)) return NULL;
    object_draw_mode_t mode = (object_draw_mode_t)mode_int;
    object_t *obj = g_api.geo_tetrahedron(mode, cull ? true : false);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}
static PyObject* py_geo_octahedron(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    int mode_int = 0; int cull = 1;
    if (!PyArg_ParseTuple(args, "ip", &mode_int, &cull)) return NULL;
    object_draw_mode_t mode = (object_draw_mode_t)mode_int;
    object_t *obj = g_api.geo_octahedron(mode, cull ? true : false);
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
    object_scene_t *os = g_api.scene_new((uint16_t)count);
    return cap_from_ptr(os, CAPS_OBJECT_SCENE, NULL /* freed via api_object_scene_free in C land by user if needed */);
}

static PyObject* py_scene_set_current(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *cap = NULL;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    object_scene_t *os = (object_scene_t*)cap_get_ptr(cap, CAPS_OBJECT_SCENE);
    if (!os) return NULL;
    g_api.scene_set_current(os);
    Py_RETURN_NONE;
}

static PyObject* py_scene_clear_current(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    g_api.scene_clear_current();
    Py_RETURN_NONE;
}

static PyObject* py_scene_set_ambient(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    float r,g,b;
    if (!PyArg_ParseTuple(args, "fff", &r,&g,&b)) return NULL;
    RGBF c = {r,g,b};
    g_api.scene_set_ambient(c);
    Py_RETURN_NONE;
}

static PyObject* py_scene_add_directional(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    float dx,dy,dz, cr,cg,cb, intensity; int casts;
    if (!PyArg_ParseTuple(args, "fffffffp", &dx,&dy,&dz, &cr,&cg,&cb, &intensity, &casts)) return NULL;
    light_vec3 dir = {dx,dy,dz};
    RGBF col = {cr,cg,cb};
    uint16_t id = g_api.scene_add_directional(dir, col, intensity, casts ? true : false);
    return PyLong_FromUnsignedLong((unsigned long)id);
}

static PyObject* py_scene_add_object(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *o_cap=NULL, *t_cap=NULL;
    if (!PyArg_ParseTuple(args, "OO", &o_cap, &t_cap)) return NULL;
    object_t *obj = (object_t*)cap_get_ptr(o_cap, CAPS_OBJECT);
    transform_t *x = (transform_t*)cap_get_ptr(t_cap, CAPS_TRANSFORM);
    if (!obj || !x) return NULL;
    uint16_t id = g_api.scene_add_object(obj, x);
    return PyLong_FromUnsignedLong((unsigned long)id);
}

static PyObject* py_scene_get_object(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int id;
    if (!PyArg_ParseTuple(args, "I", &id)) return NULL;
    object_t *obj = g_api.scene_get_object((uint16_t)id);
    return cap_from_ptr(obj, CAPS_OBJECT, NULL);
}

static PyObject* py_scene_get_transform(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    unsigned int id;
    if (!PyArg_ParseTuple(args, "I", &id)) return NULL;
    transform_t *x = g_api.scene_get_transform((uint16_t)id);
    return cap_from_ptr(x, CAPS_TRANSFORM, NULL);
}

// render_scene(camera, object_scene) -> None (uses scene-owned lighting)
static PyObject* py_render_scene(PyObject *self, PyObject *args) {
    if (!require_api()) return NULL;
    PyObject *c_cap=NULL, *os_cap=NULL;
    if (!PyArg_ParseTuple(args, "OO", &c_cap, &os_cap)) return NULL;
    camera_t *cam = (camera_t*)cap_get_ptr(c_cap, CAPS_CAMERA);
    object_scene_t *os = (object_scene_t*)cap_get_ptr(os_cap, CAPS_OBJECT_SCENE);
    if (!cam || !os) return NULL;
    g_api.render_scene(cam, os, NULL);
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

static PyMethodDef Hub75Methods[] = {
    {"new_scene", (PyCFunction)py_new_scene, METH_VARARGS, "Create a default scene_info and return a capsule."},
    {"hub75_init", (PyCFunction)py_hub75_init, METH_VARARGS, "Initialize API with a scene capsule (sets current scene)."},

    {"geo_camera", (PyCFunction)py_geo_camera, METH_NOARGS, "Create a new camera (capsule)."},
    {"geo_transform", (PyCFunction)py_geo_transform, METH_NOARGS, "Create a new transform (capsule)."},
    {"geo_object", (PyCFunction)py_geo_object, METH_VARARGS, "Create a raw object with counts."},
    {"geo_cube", (PyCFunction)py_geo_cube, METH_VARARGS, "Create a cube object."},
    {"geo_tetrahedron", (PyCFunction)py_geo_tetrahedron, METH_VARARGS, "Create a tetrahedron object."},
    {"geo_octahedron", (PyCFunction)py_geo_octahedron, METH_VARARGS, "Create an octahedron object."},
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
    return PyModule_Create(&Hub75Module);
}
