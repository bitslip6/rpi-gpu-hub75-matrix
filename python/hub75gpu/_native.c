#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "hub75_runtime.h"

typedef struct {
    PyObject *resource_type, *runtime_type;
    PyObject *error, *state_error, *busy_error, *unsupported_error;
} module_state;
typedef struct {
    PyObject_HEAD h75_resource *ptr;
    unsigned exports;
} Resource;
typedef struct {
    PyObject_HEAD h75_runtime *ptr;
    size_t bytes;
    uint32_t width, height;
    uint64_t lease_serial;
    unsigned long lease_thread;
    bool python_lease;
} Runtime;
static int mesh_buffer(PyObject *obj, Py_buffer *v, const char *format, int columns);
static module_state *state(PyObject *o) {
    return PyType_GetModuleState(Py_TYPE(o));
}
static PyObject *raise_native(module_state *s, h75_status code, h75_error *e) {
    PyObject *exc = s ? s->error : NULL;
    if (code == H75_INVALID)
        exc = PyExc_ValueError;
    else if (code == H75_TIMEOUT)
        exc = PyExc_TimeoutError;
    else if (code == H75_MEMORY)
        exc = PyExc_MemoryError;
    else if (code == H75_STATE || code == H75_CANCELLED)
        exc = s ? s->state_error : NULL;
    else if (code == H75_BUSY)
        exc = s ? s->busy_error : NULL;
    else if (code == H75_UNSUPPORTED)
        exc = s ? s->unsupported_error : NULL;
    else if (code == H75_IO)
        exc = PyExc_OSError;
    PyErr_SetString(exc ? exc : PyExc_RuntimeError,
                    e && e->message[0] ? e->message : "native operation failed");
    return NULL;
}
static h75_resource *resource_ptr(PyObject *o, module_state *s) {
    if (!s || !s->resource_type) {
        PyErr_SetString(PyExc_RuntimeError, "native module is unavailable");
        return NULL;
    }
    if (!PyObject_TypeCheck(o, (PyTypeObject *)s->resource_type)) {
        PyErr_SetString(PyExc_TypeError, "expected native resource");
        return NULL;
    }
    h75_resource *p = ((Resource *)o)->ptr;
    if (!p)
        PyErr_SetString(s->state_error, "resource is closed");
    return p;
}
static PyObject *wrap(module_state *s, h75_resource *p, h75_error *e) {
    if (!p)
        return raise_native(s, e->code, e);
    if (!s || !s->resource_type) {
        h75_resource_release(p);
        PyErr_SetString(PyExc_RuntimeError, "native module is unavailable");
        return NULL;
    }
    PyTypeObject *type = (PyTypeObject *)s->resource_type;
    Resource *r = (Resource *)type->tp_alloc(type, 0);
    if (!r) {
        h75_resource_release(p);
        return NULL;
    }
    r->ptr = p;
    r->exports = 0;
    return (PyObject *)r;
}
static int uint32_arg(PyObject *o, uint32_t *out) {
    PyObject *i = PyNumber_Index(o);
    if (!i)
        return 0;
    unsigned long long n = PyLong_AsUnsignedLongLong(i);
    Py_DECREF(i);
    if (PyErr_Occurred())
        return 0;
    if (n > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "integer exceeds native uint32 range");
        return 0;
    }
    *out = (uint32_t)n;
    return 1;
}
static int floats(PyObject *o, float *out, size_t min, size_t max, size_t *n) {
    PyObject *seq = PySequence_Fast(o, "expected a numeric sequence");
    if (!seq)
        return 0;
    Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    if (count < (Py_ssize_t)min || count > (Py_ssize_t)max) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "wrong number of components");
        return 0;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        double v = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return 0;
        }
        if (!isfinite(v) || fabs(v) > 1e7) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "components must be finite, bounded float32 values");
            return 0;
        }
        out[i] = (float)v;
    }
    Py_DECREF(seq);
    *n = (size_t)count;
    return 1;
}
static int color(PyObject *o, RGBA *out, int channels) {
    PyObject *seq = PySequence_Fast(o, "expected RGB or RGBA color");
    if (!seq)
        return 0;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n != channels) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "wrong color channel count");
        return 0;
    }
    uint8_t v[4] = {0, 0, 0, 255};
    for (int i = 0; i < channels; i++) {
        uint32_t value;
        if (!uint32_arg(PySequence_Fast_GET_ITEM(seq, i), &value)) {
            Py_DECREF(seq);
            return 0;
        }
        if (value > 255) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "color channels must be 0..255");
            return 0;
        }
        v[i] = (uint8_t)value;
    }
    *out = (RGBA){v[0], v[1], v[2], v[3]};
    Py_DECREF(seq);
    return 1;
}
static int command(PyObject *args, module_state *s, h75_command *c) {
    int op;
    PyObject *col, *params, *points, *gradient, *a, *b;
    memset(c, 0, sizeof(*c));
    if (!PyArg_ParseTuple(args, "iOOOOOO", &op, &col, &params, &points, &gradient, &a, &b))
        return 0;
    c->op = op;
    size_t n = 0;
    if (!color(col, &c->color, 4) || !floats(params, c->p, 0, 8, &n))
        return 0;
    PyObject *seq = PySequence_Fast(points, "expected polygon points");
    if (!seq)
        return 0;
    Py_ssize_t count = PySequence_Fast_GET_SIZE(seq);
    if (count > MAX_POLY_POINTS) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "too many polygon vertices");
        return 0;
    }
    c->polygon.num_points = (size_t)count;
    for (Py_ssize_t i = 0; i < count; i++)
        if (!floats(PySequence_Fast_GET_ITEM(seq, i), &c->polygon.points[i].x, 2, 2, &n)) {
            Py_DECREF(seq);
            return 0;
        }
    Py_DECREF(seq);
    if (gradient != Py_None) {
        PyObject *start, *end;
        int dir, ease;
        RGBA x, y;
        if (!PyArg_ParseTuple(gradient, "OOii", &start, &end, &dir, &ease) ||
            !color(start, &x, 3) || !color(end, &y, 3))
            return 0;
        if (dir < 0 || dir > 3 || ease < 0 || ease > 3) {
            PyErr_SetString(PyExc_ValueError, "invalid gradient enum");
            return 0;
        }
        c->gradient = (SimpleGradient){.start_color = {x.r, x.g, x.b},
                                       .end_color = {y.r, y.g, y.b},
                                       .direction = dir,
                                       .easing = ease};
    }
    if (a != Py_None) {
        c->a = resource_ptr(a, s);
        if (!c->a)
            return 0;
    }
    if (b != Py_None) {
        c->b = resource_ptr(b, s);
        if (!c->b)
            return 0;
    }
    return 1;
}
static int native_traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT((PyObject *)Py_TYPE(self));
    return 0;
}
static void resource_dealloc(Resource *r) {
    PyObject_GC_UnTrack(r);
    h75_resource *p = r->ptr;
    r->ptr = NULL;
    Py_BEGIN_ALLOW_THREADS
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    PyTypeObject *type = Py_TYPE(r);
    type->tp_free((PyObject *)r);
    Py_DECREF(type);
}
static PyObject *resource_close(Resource *r, PyObject *ignored) {
    (void)ignored;
    if (r->exports) {
        PyErr_SetString(PyExc_BufferError, "release image memoryviews before close");
        return NULL;
    }
    h75_resource *p = r->ptr;
    r->ptr = NULL;
    Py_BEGIN_ALLOW_THREADS
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
static PyObject *resource_set(Resource *r, PyObject *args) {
    const char *name;
    PyObject *values;
    float v[16];
    size_t n;
    if (!PyArg_ParseTuple(args, "sO", &name, &values) || !floats(values, v, 1, 16, &n))
        return NULL;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_resource_set(p, name, v, n, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *resource_get(Resource *r, PyObject *arg) {
    const char *name = PyUnicode_AsUTF8(arg);
    if (!name)
        return NULL;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    float v[16];
    size_t n = 16;
    Py_BEGIN_ALLOW_THREADS
        s = h75_resource_get(p, name, v, &n, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    if (n == 1)
        return PyFloat_FromDouble(v[0]);
    PyObject *tuple = PyTuple_New((Py_ssize_t)n);
    if (!tuple)
        return NULL;
    for (size_t i = 0; i < n; i++) {
        PyObject *f = PyFloat_FromDouble(v[i]);
        if (!f) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, (Py_ssize_t)i, f);
    }
    return tuple;
}
static PyObject *resource_append(Resource *r, PyObject *arg) {
    module_state *st = state((PyObject *)r);
    h75_command c;
    if (!command(arg, st, &c))
        return NULL;
    h75_resource *p = resource_ptr((PyObject *)r, st);
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_resource_retain(c.a);
    h75_resource_retain(c.b);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_drawlist_append(p, &c, &e);
        h75_resource_release(p);
        h75_resource_release(c.a);
        h75_resource_release(c.b);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    Py_RETURN_NONE;
}
static PyObject *resource_text(Resource *r, PyObject *arg) {
    const char *text = PyUnicode_AsUTF8(arg);
    if (!text)
        return NULL;
    if (strlen(text) != (size_t)PyUnicode_GET_LENGTH(arg)) { /* ASCII-only native atlas contract. */
        PyErr_SetString(PyExc_ValueError,
                        "SDF text currently supports ASCII only (no embedded NUL)");
        return NULL;
    }
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_text_string(p, text, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *resource_add(Resource *r, PyObject *args) {
    PyObject *mesh, *transform, *velocity;
    float v[3];
    size_t n;
    if (!PyArg_ParseTuple(args, "OOO", &mesh, &transform, &velocity) ||
        !floats(velocity, v, 3, 3, &n))
        return NULL;
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr((PyObject *)r, st), *a = resource_ptr(mesh, st),
                 *b = resource_ptr(transform, st);
    if (!p || !a || !b)
        return NULL;
    h75_resource_retain(p);
    h75_resource_retain(a);
    h75_resource_retain(b);
    h75_error e = {0};
    h75_status s;
    uint16_t id;
    Py_BEGIN_ALLOW_THREADS
        s = h75_scene_add(p, a, b, (vec3){v[0], v[1], v[2]}, &id, &e);
        h75_resource_release(p);
        h75_resource_release(a);
        h75_resource_release(b);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    return PyLong_FromUnsignedLong(id);
}
static PyObject *resource_light(Resource *r, PyObject *args) {
    PyObject *direction, *col;
    float v[3], rgb[3], intensity;
    size_t n;
    int shadows;
    if (!PyArg_ParseTuple(args, "OOfp", &direction, &col, &intensity, &shadows) ||
        !floats(direction, v, 3, 3, &n) || !floats(col, rgb, 3, 3, &n))
        return NULL;
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr((PyObject *)r, st);
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    uint16_t id;
    Py_BEGIN_ALLOW_THREADS
        s = h75_scene_light(p, (vec3){v[0], v[1], v[2]}, (RGBF){rgb[0], rgb[1], rgb[2]}, intensity,
                            shadows, &id, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    return PyLong_FromUnsignedLong(id);
}
static PyObject *resource_light_set(Resource *r, PyObject *args) {
    PyObject *index, *values;
    const char *name;
    uint32_t id;
    float v[16];
    size_t n;
    if (!PyArg_ParseTuple(args, "OsO", &index, &name, &values) || !uint32_arg(index, &id) ||
        !floats(values, v, 1, 16, &n))
        return NULL;
    if (id >= UINT16_MAX) {
        PyErr_SetString(PyExc_ValueError, "light id out of range");
        return NULL;
    }
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr((PyObject *)r, st);
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_scene_light_set(p, (uint16_t)id, name, v, n, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    Py_RETURN_NONE;
}
static int image_buffer(PyObject *obj, Py_buffer *view, uint32_t w, uint32_t h, uint32_t pitch,
                        int channels) {
    if (PyObject_GetBuffer(obj, view, PyBUF_FORMAT | PyBUF_STRIDES | PyBUF_ND) < 0)
        return 0;
    if (view->itemsize != 1 || !view->format || strcmp(view->format, "B") ||
        !PyBuffer_IsContiguous(view, 'C'))
        goto invalid;
    for (int i = 0; i < view->ndim; i++)
        if (view->strides && view->strides[i] <= 0)
            goto invalid;
    if (view->ndim != 1 &&
        !(view->ndim == 3 && view->shape[0] == h && view->shape[1] == w &&
          view->shape[2] == channels && pitch == (uint64_t)w * (unsigned)channels))
        goto invalid;
    return 1;
invalid:
    PyBuffer_Release(view);
    PyErr_SetString(PyExc_ValueError, "expected contiguous uint8 bytes or HxWxC image; "
                                      "arbitrary/negative/broadcast strides are unsupported");
    return 0;
}
static PyObject *resource_upload(Resource *r, PyObject *args) {
    PyObject *data, *wo, *ho, *po;
    int channels;
    uint32_t w, h, pitch;
    if (!PyArg_ParseTuple(args, "OOOOi", &data, &wo, &ho, &po, &channels) || !uint32_arg(wo, &w) ||
        !uint32_arg(ho, &h) || !uint32_arg(po, &pitch))
        return NULL;
    Py_buffer view;
    if (!image_buffer(data, &view, w, h, pitch, channels))
        return NULL;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p) {
        PyBuffer_Release(&view);
        return NULL;
    }
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_image_upload(p, view.buf, (size_t)view.len, w, h, pitch, channels, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *resource_bytes(Resource *r, PyObject *ignored) {
    (void)ignored;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p)
        return NULL;
    float dims[16];
    size_t n = 16;
    h75_error e = {0};
    h75_status s;
    h75_resource_retain(p);
    Py_BEGIN_ALLOW_THREADS
        s = h75_resource_get(p, "dimensions", dims, &n, &e);
    Py_END_ALLOW_THREADS
    if (s) {
        h75_resource_release(p);
        return raise_native(state((PyObject *)r), s, &e);
    }
    if (h75_resource_kind(p) != H75_IMAGE) {
        h75_resource_release(p);
        PyErr_SetString(PyExc_TypeError, "only images have bytes");
        return NULL;
    }
    size_t bytes = (size_t)dims[0] * (size_t)dims[1] * 4;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)bytes);
    if (!out) {
        h75_resource_release(p);
        return NULL;
    }
    char *dest = PyBytes_AS_STRING(out);
    Py_BEGIN_ALLOW_THREADS
        s = h75_image_copy(p, dest, bytes, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s) {
        Py_DECREF(out);
        return raise_native(state((PyObject *)r), s, &e);
    }
    return out;
}
static int resource_getbuffer(Resource *r, Py_buffer *view, int flags) {
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr((PyObject *)r, st);
    if (!p)
        return -1;
    image_buffer_t im;
    h75_error e = {0};
    h75_status s = h75_image_export(p, &im, &e);
    if (s) {
        raise_native(st, s, &e);
        return -1;
    }
    if (PyBuffer_FillInfo(view, (PyObject *)r, im.data, (Py_ssize_t)im.row_stride * im.dimensions.y,
                          0, flags) < 0) {
        h75_image_unexport(p);
        return -1;
    }
    r->exports++;
    return 0;
}
static void resource_releasebuffer(Resource *r, Py_buffer *view) {
    (void)view;
    if (r->exports) {
        r->exports--;
        h75_image_unexport(r->ptr);
    }
}
static PyObject *resource_transforms(Resource *r, PyObject *arg) {
    Py_buffer view;
    if (!mesh_buffer(arg, &view, "f", 9))
        return NULL;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p) {
        PyBuffer_Release(&view);
        return NULL;
    }
    float *values = PyMem_Malloc((size_t)view.len ? (size_t)view.len : 1);
    if (!values) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    memcpy(values, view.buf, (size_t)view.len);
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_scene_transforms(p, values, (size_t)view.len / 36, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    PyMem_Free(values);
    PyBuffer_Release(&view);
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *resource_geometry(Resource *r, PyObject *args) {
    PyObject *obj, *col;
    int columns, aa, replace;
    if (!PyArg_ParseTuple(args, "OiOpp", &obj, &columns, &col, &aa, &replace))
        return NULL;
    if (columns != 2 && columns != 4) {
        PyErr_SetString(PyExc_ValueError, "geometry needs 2 or 4 columns");
        return NULL;
    }
    RGBA rgba;
    if (!color(col, &rgba, 4))
        return NULL;
    Py_buffer view = {0};
    if (!mesh_buffer(obj, &view, "f", columns))
        return NULL;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p) {
        PyBuffer_Release(&view);
        return NULL;
    }
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_drawlist_geometry(p, view.buf, (size_t)view.len / ((size_t)columns * 4),
                                  (unsigned)columns, rgba, aa, replace, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *resource_reset(Resource *r, PyObject *ignored) {
    (void)ignored;
    h75_resource *p = resource_ptr((PyObject *)r, state((PyObject *)r));
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_drawlist_geometry(p, NULL, 0, 2, (RGBA){0}, false, true, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyMethodDef resource_methods[] = {
    {"close", (PyCFunction)resource_close, METH_NOARGS, NULL},
    {"set", (PyCFunction)resource_set, METH_VARARGS, NULL},
    {"get", (PyCFunction)resource_get, METH_O, NULL},
    {"append", (PyCFunction)resource_append, METH_O, NULL},
    {"text", (PyCFunction)resource_text, METH_O, NULL},
    {"add", (PyCFunction)resource_add, METH_VARARGS, NULL},
    {"light", (PyCFunction)resource_light, METH_VARARGS, NULL},
    {"light_set", (PyCFunction)resource_light_set, METH_VARARGS, NULL},
    {"upload", (PyCFunction)resource_upload, METH_VARARGS, NULL},
    {"tobytes", (PyCFunction)resource_bytes, METH_NOARGS, NULL},
    {"transforms", (PyCFunction)resource_transforms, METH_O, NULL},
    {"geometry", (PyCFunction)resource_geometry, METH_VARARGS, NULL},
    {"reset", (PyCFunction)resource_reset, METH_NOARGS, NULL},
    {NULL}};
static PyType_Slot resource_slots[] = {{Py_tp_dealloc, resource_dealloc},
                                       {Py_tp_traverse, native_traverse},
                                       {Py_tp_methods, resource_methods},
                                       {Py_bf_getbuffer, resource_getbuffer},
                                       {Py_bf_releasebuffer, resource_releasebuffer},
                                       {0, NULL}};
static PyType_Spec resource_spec = {"hub75gpu._native.Resource", sizeof(Resource), 0,
                                    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, resource_slots};

static int parse_config(PyObject *dict, h75_config *c) {
    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "configuration must be a dictionary");
        return 0;
    }
    h75_config_default(c);
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict, &pos, &key, &val)) {
        const char *k = PyUnicode_AsUTF8(key);
        if (!k)
            return 0;
#define U(name)                                                                                    \
    if (!strcmp(k, #name)) {                                                                       \
        if (!uint32_arg(val, &c->name))                                                            \
            return 0;                                                                              \
        continue;                                                                                  \
    }
        U(width)
        U(height)
        U(panel_width)
        U(panel_height)
        U(ports)
        U(chains)
        U(bit_depth) U(fps) U(brightness) U(queue_depth) U(panel_order) U(mapping) U(tone_mapping)
#undef U
#define F(name)                                                                                    \
    if (!strcmp(k, #name)) {                                                                       \
        double x = PyFloat_AsDouble(val);                                                          \
        if (PyErr_Occurred())                                                                      \
            return 0;                                                                              \
        c->name = (float)x;                                                                        \
        continue;                                                                                  \
    }
            F(gamma) F(dither) F(tone_level)
#undef F
#define B(name)                                                                                    \
    if (!strcmp(k, #name)) {                                                                       \
        if (!PyBool_Check(val)) {                                                                  \
            PyErr_SetString(PyExc_TypeError, #name " must be bool");                               \
            return 0;                                                                              \
        }                                                                                          \
        c->name = val == Py_True;                                                                  \
        continue;                                                                                  \
    }
                B(offscreen) B(quant_dither) B(jitter) B(video_preserve_aspect)
#undef B
                    if (!strcmp(k, "gpio_cpu")) {
            long x = PyLong_AsLong(val);
            if (PyErr_Occurred())
                return 0;
            if (x < -1 || x > INT32_MAX) {
                PyErr_SetString(PyExc_ValueError, "invalid gpio_cpu");
                return 0;
            }
            c->gpio_cpu = (int32_t)x;
            continue;
        }
        PyErr_Format(PyExc_TypeError, "unknown configuration option: %s", k);
        return 0;
    }
    return 1;
}
static PyObject *runtime_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject *dict;
    static char *keys[] = {"config", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keys, &dict))
        return NULL;
    h75_config c;
    if (!parse_config(dict, &c))
        return NULL;
    h75_error e = {0};
    h75_runtime *p = NULL;
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_create(&c, &p, &e);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(PyType_GetModuleState(type), s, &e);
    Runtime *r = (Runtime *)type->tp_alloc(type, 0);
    if (!r) {
        h75_runtime_destroy(p);
        return NULL;
    }
    r->ptr = p;
    r->bytes = (size_t)c.width * c.height * 4;
    r->width = c.width;
    r->height = c.height;
    return (PyObject *)r;
}
static void runtime_dealloc(Runtime *r) {
    PyObject_GC_UnTrack(r);
    Py_BEGIN_ALLOW_THREADS
        h75_runtime_destroy(r->ptr);
    Py_END_ALLOW_THREADS
    PyTypeObject *type = Py_TYPE(r);
    type->tp_free((PyObject *)r);
    Py_DECREF(type);
}
static PyObject *runtime_finish(Runtime *r, bool submit) {
    uint64_t serial = r->lease_serial;
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = submit ? h75_frame_submit(r->ptr, &e) : h75_frame_cancel(r->ptr, &e);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    if (r->lease_serial == serial)
        r->python_lease = false;
    Py_RETURN_NONE;
}
static PyObject *runtime_submit(Runtime *r, PyObject *ignored) {
    (void)ignored;
    return runtime_finish(r, true);
}
static PyObject *runtime_cancel(Runtime *r, PyObject *ignored) {
    (void)ignored;
    return runtime_finish(r, false);
}
static PyObject *runtime_lease_serial(Runtime *r, PyObject *ignored) {
    (void)ignored;
    return PyLong_FromUnsignedLongLong(r->lease_serial);
}
static PyObject *runtime_cancel_serial(Runtime *r, PyObject *arg) {
    unsigned long long serial = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred())
        return NULL;
    if (!r->python_lease || r->lease_serial != serial ||
        r->lease_thread != PyThread_get_thread_ident())
        Py_RETURN_NONE;
    r->python_lease = false;
    if (h75_runtime_get_state(r->ptr) == H75_RUNNING) {
        Py_BEGIN_ALLOW_THREADS
            h75_frame_cancel(r->ptr, NULL);
        Py_END_ALLOW_THREADS
    }
    Py_RETURN_NONE;
}
static PyObject *runtime_start(Runtime *r, PyObject *ignored) {
    (void)ignored;
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_start(r->ptr, &e);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    if (PyErr_CheckSignals() < 0) {
        Py_BEGIN_ALLOW_THREADS
            h75_runtime_close(r->ptr, -1, NULL);
        Py_END_ALLOW_THREADS
        return NULL;
    }
    Py_RETURN_NONE;
}
#define TIMED_METHOD(name, call)                                                                   \
    static PyObject *runtime_##name(Runtime *r, PyObject *arg) {                                   \
        long ms = PyLong_AsLong(arg);                                                              \
        if (PyErr_Occurred())                                                                      \
            return NULL;                                                                           \
        if (ms < -1 || ms > INT_MAX) {                                                             \
            PyErr_SetString(PyExc_ValueError, "invalid millisecond timeout");                      \
            return NULL;                                                                           \
        }                                                                                          \
        h75_error e = {0};                                                                         \
        h75_status s;                                                                              \
        Py_BEGIN_ALLOW_THREADS                                                                     \
            s = call(r->ptr, (int)ms, &e);                                                         \
        Py_END_ALLOW_THREADS                                                                       \
        if (s)                                                                                     \
            return raise_native(state((PyObject *)r), s, &e);                                      \
        if (PyErr_CheckSignals() < 0)                                                              \
            return NULL;                                                                           \
        Py_RETURN_NONE;                                                                            \
    }
TIMED_METHOD(close, h75_runtime_close)
TIMED_METHOD(wait, h75_runtime_wait)
TIMED_METHOD(wait_content, h75_runtime_wait_content)
TIMED_METHOD(stop_content, h75_runtime_stop_content)
TIMED_METHOD(flush, h75_runtime_flush)
TIMED_METHOD(ready, h75_runtime_content_ready)
#undef TIMED_METHOD
static PyObject *runtime_pause(Runtime *r, PyObject *arg) {
    if (!PyBool_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "paused must be bool");
        return NULL;
    }
    bool paused = arg == Py_True;
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_pause(r->ptr, paused, &e);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *runtime_acquire(Runtime *r, PyObject *arg) {
    long ms = PyLong_AsLong(arg);
    if (PyErr_Occurred())
        return NULL;
    if (ms < 0 || ms > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "invalid frame timeout");
        return NULL;
    }
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_frame_acquire(r->ptr, (int)ms, &e);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    r->lease_serial++;
    r->lease_thread = PyThread_get_thread_ident();
    r->python_lease = true;
    if (PyErr_CheckSignals() < 0) {
        r->python_lease = false;
        Py_BEGIN_ALLOW_THREADS
            h75_frame_cancel(r->ptr, NULL);
        Py_END_ALLOW_THREADS
        return NULL;
    }
    Py_RETURN_NONE;
}
static PyObject *runtime_stop(Runtime *r, PyObject *ignored) {
    (void)ignored;
    h75_runtime_request_stop(r->ptr);
    Py_RETURN_NONE;
}
static PyObject *runtime_command(Runtime *r, PyObject *arg) {
    module_state *st = state((PyObject *)r);
    h75_command c;
    if (!command(arg, st, &c))
        return NULL;
    h75_resource_retain(c.a);
    h75_resource_retain(c.b);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_frame_command(r->ptr, &c, &e);
        h75_resource_release(c.a);
        h75_resource_release(c.b);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    Py_RETURN_NONE;
}
static PyObject *runtime_upload(Runtime *r, PyObject *args) {
    PyObject *data, *po;
    int channels;
    uint32_t pitch;
    if (!PyArg_ParseTuple(args, "OOi", &data, &po, &channels) || !uint32_arg(po, &pitch))
        return NULL;
    Py_buffer view;
    if (!image_buffer(data, &view, r->width, r->height, pitch, channels))
        return NULL;
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_frame_upload(r->ptr, view.buf, (size_t)view.len, pitch, channels, &e);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *runtime_draw(Runtime *r, PyObject *args) {
    PyObject *obj;
    float time = 0;
    if (!PyArg_ParseTuple(args, "Of", &obj, &time))
        return NULL;
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr(obj, st);
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_frame_draw(r->ptr, p, time, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    Py_RETURN_NONE;
}
static PyObject *runtime_try_draw(Runtime *r, PyObject *args) {
    PyObject *obj;
    float time;
    if (!PyArg_ParseTuple(args, "Of", &obj, &time))
        return NULL;
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr(obj, st);
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    bool accepted = false;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_try_draw(r->ptr, p, time, &accepted, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    return PyBool_FromLong(accepted);
}
static PyObject *runtime_read(Runtime *r, PyObject *ignored) {
    (void)ignored;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)r->bytes);
    if (!out)
        return NULL;
    h75_error e = {0};
    h75_status s;
    char *dest = PyBytes_AS_STRING(out);
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_read(r->ptr, dest, r->bytes, &e);
    Py_END_ALLOW_THREADS
    if (s) {
        Py_DECREF(out);
        return raise_native(state((PyObject *)r), s, &e);
    }
    return out;
}
static PyObject *runtime_stats(Runtime *r, PyObject *ignored) {
    (void)ignored;
    h75_stats st;
    Py_BEGIN_ALLOW_THREADS
        h75_runtime_stats(r->ptr, &st);
    Py_END_ALLOW_THREADS
    return Py_BuildValue(
        "{sK,sK,sK,sK,sK,sK,sK,sI,si,sO,sO,sO,sK,sK,sK,sK,sK}", "submitted",
        (unsigned long long)st.submitted, "consumed", (unsigned long long)st.consumed, "mapped",
        (unsigned long long)st.mapped, "scanned", (unsigned long long)st.scanned, "dropped",
        (unsigned long long)st.dropped, "queue_wait_ns", (unsigned long long)st.queue_wait_ns,
        "render_ns", (unsigned long long)st.render_ns, "queue_occupancy", st.queue_occupancy,
        "state", st.state, "playing", st.playing ? Py_True : Py_False, "paused",
        st.paused ? Py_True : Py_False, "ready", st.ready ? Py_True : Py_False, "map_ns",
        (unsigned long long)st.map_ns, "copy_ns", (unsigned long long)st.copy_ns, "scan_ns",
        (unsigned long long)st.scan_ns, "latency_ns", (unsigned long long)st.latency_ns,
        "latency_total_ns", (unsigned long long)st.latency_total_ns);
}
static PyObject *runtime_play(Runtime *r, PyObject *obj) {
    module_state *st = state((PyObject *)r);
    h75_resource *p = resource_ptr(obj, st);
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_play(r->ptr, p, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    Py_RETURN_NONE;
}
static PyObject *runtime_play_file(Runtime *r, PyObject *args) {
    const char *path;
    int video, loop;
    float scale;
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "sppfO", &path, &video, &loop, &scale, &obj))
        return NULL;
    h75_resource *overlay = NULL;
    if (obj != Py_None) {
        overlay = resource_ptr(obj, state((PyObject *)r));
        if (!overlay)
            return NULL;
        h75_resource_retain(overlay);
    }
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_play_file(r->ptr, path, video, loop, scale, overlay, &e);
        h75_resource_release(overlay);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(state((PyObject *)r), s, &e);
    Py_RETURN_NONE;
}
static PyObject *runtime_error(Runtime *r, PyObject *ignored) {
    (void)ignored;
    h75_error e = {0};
    h75_status s;
    Py_BEGIN_ALLOW_THREADS
        s = h75_runtime_error(r->ptr, &e);
    Py_END_ALLOW_THREADS
    if (!s)
        Py_RETURN_NONE;
    raise_native(state((PyObject *)r), s, &e);
#if PY_VERSION_HEX >= 0x030C0000
    return PyErr_GetRaisedException();
#else
    PyObject *type = NULL, *value = NULL, *tb = NULL;
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);
    Py_XDECREF(type);
    Py_XDECREF(tb);
    if (PyErr_Occurred()) {
        Py_XDECREF(value);
        return NULL;
    }
    if (!value)
        return PyErr_NoMemory();
    return value;
#endif
}
static PyObject *runtime_state(Runtime *r, PyObject *ignored) {
    (void)ignored;
    return PyLong_FromLong(h75_runtime_get_state(r->ptr));
}
static PyMethodDef runtime_methods[] = {
    {"start", (PyCFunction)runtime_start, METH_NOARGS, NULL},
    {"close", (PyCFunction)runtime_close, METH_O, NULL},
    {"acquire", (PyCFunction)runtime_acquire, METH_O, NULL},
    {"lease_serial", (PyCFunction)runtime_lease_serial, METH_NOARGS, NULL},
    {"cancel_serial", (PyCFunction)runtime_cancel_serial, METH_O, NULL},
    {"submit", (PyCFunction)runtime_submit, METH_NOARGS, NULL},
    {"cancel", (PyCFunction)runtime_cancel, METH_NOARGS, NULL},
    {"command", (PyCFunction)runtime_command, METH_O, NULL},
    {"draw", (PyCFunction)runtime_draw, METH_VARARGS, NULL},
    {"try_draw", (PyCFunction)runtime_try_draw, METH_VARARGS, NULL},
    {"read", (PyCFunction)runtime_read, METH_NOARGS, NULL},
    {"upload", (PyCFunction)runtime_upload, METH_VARARGS, NULL},
    {"stats", (PyCFunction)runtime_stats, METH_NOARGS, NULL},
    {"state", (PyCFunction)runtime_state, METH_NOARGS, NULL},
    {"error", (PyCFunction)runtime_error, METH_NOARGS, NULL},
    {"stop", (PyCFunction)runtime_stop, METH_NOARGS, NULL},
    {"wait", (PyCFunction)runtime_wait, METH_O, NULL},
    {"wait_content", (PyCFunction)runtime_wait_content, METH_O, NULL},
    {"stop_content", (PyCFunction)runtime_stop_content, METH_O, NULL},
    {"flush", (PyCFunction)runtime_flush, METH_O, NULL},
    {"pause", (PyCFunction)runtime_pause, METH_O, NULL},
    {"ready", (PyCFunction)runtime_ready, METH_O, NULL},
    {"play", (PyCFunction)runtime_play, METH_O, NULL},
    {"play_file", (PyCFunction)runtime_play_file, METH_VARARGS, NULL},
    {NULL}};
static PyType_Slot runtime_slots[] = {{Py_tp_new, runtime_new},
                                      {Py_tp_dealloc, runtime_dealloc},
                                      {Py_tp_traverse, native_traverse},
                                      {Py_tp_methods, runtime_methods},
                                      {0, NULL}};
static PyType_Spec runtime_spec = {"hub75gpu._native.Runtime", sizeof(Runtime), 0,
                                   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, runtime_slots};

static PyObject *new_resource(PyObject *m, PyObject *arg) {
    long k = PyLong_AsLong(arg);
    if (PyErr_Occurred())
        return NULL;
    if (k < 0 || k > H75_DRAWLIST) {
        PyErr_SetString(PyExc_ValueError, "invalid resource kind");
        return NULL;
    }
    h75_error e = {0};
    return wrap(PyModule_GetState(m), h75_resource_new((h75_kind)k, &e), &e);
}
static PyObject *new_image(PyObject *m, PyObject *args) {
    PyObject *a, *b;
    uint32_t w, h;
    if (!PyArg_ParseTuple(args, "OO", &a, &b) || !uint32_arg(a, &w) || !uint32_arg(b, &h))
        return NULL;
    h75_error e = {0};
    h75_resource *p;
    Py_BEGIN_ALLOW_THREADS
        p = h75_image_new(w, h, &e);
    Py_END_ALLOW_THREADS
    return wrap(PyModule_GetState(m), p, &e);
}
static PyObject *load_image(PyObject *m, PyObject *arg) {
    Py_ssize_t size;
    const char *path = PyUnicode_AsUTF8AndSize(arg, &size);
    if (!path)
        return NULL;
    if (strlen(path) != (size_t)size) {
        PyErr_SetString(PyExc_ValueError, "embedded NUL in image path");
        return NULL;
    }
    h75_error e = {0};
    h75_resource *p;
    Py_BEGIN_ALLOW_THREADS
        p = h75_image_load(path, &e);
    Py_END_ALLOW_THREADS
    return wrap(PyModule_GetState(m), p, &e);
}
static PyObject *transform_image(PyObject *m, PyObject *args) {
    PyObject *obj, *value;
    uint32_t mapping;
    if (!PyArg_ParseTuple(args, "OO", &obj, &value) || !uint32_arg(value, &mapping))
        return NULL;
    h75_resource *p = resource_ptr(obj, PyModule_GetState(m));
    if (!p)
        return NULL;
    h75_resource_retain(p);
    h75_error e = {0};
    h75_resource *out;
    Py_BEGIN_ALLOW_THREADS
        out = h75_image_transform(p, mapping, &e);
        h75_resource_release(p);
    Py_END_ALLOW_THREADS
    return wrap(PyModule_GetState(m), out, &e);
}
static PyObject *project(PyObject *m, PyObject *args) {
    PyObject *camera, *transform;
    if (!PyArg_ParseTuple(args, "OO", &camera, &transform))
        return NULL;
    module_state *st = PyModule_GetState(m);
    h75_resource *a = resource_ptr(camera, st), *b = resource_ptr(transform, st);
    if (!a || !b)
        return NULL;
    h75_resource_retain(a);
    h75_resource_retain(b);
    h75_error e = {0};
    h75_status s;
    mat4 matrix;
    Py_BEGIN_ALLOW_THREADS
        s = h75_camera_project(a, b, &matrix, &e);
        h75_resource_release(a);
        h75_resource_release(b);
    Py_END_ALLOW_THREADS
    if (s)
        return raise_native(st, s, &e);
    PyObject *out = PyTuple_New(16);
    if (!out)
        return NULL;
    for (int i = 0; i < 16; i++) {
        PyObject *v = PyFloat_FromDouble(matrix.m[i]);
        if (!v) {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, v);
    }
    return out;
}
static PyObject *new_mesh(PyObject *m, PyObject *args) {
    const char *name;
    PyObject *a, *b;
    uint32_t x, y;
    int mode, cull;
    if (!PyArg_ParseTuple(args, "sOOip", &name, &a, &b, &mode, &cull) || !uint32_arg(a, &x) ||
        !uint32_arg(b, &y))
        return NULL;
    h75_error e = {0};
    h75_resource *p;
    Py_BEGIN_ALLOW_THREADS
        p = h75_mesh_primitive(name, x, y, mode, cull, &e);
    Py_END_ALLOW_THREADS
    return wrap(PyModule_GetState(m), p, &e);
}
static int mesh_buffer(PyObject *obj, Py_buffer *v, const char *format, int columns) {
    if (PyObject_GetBuffer(obj, v, PyBUF_FORMAT | PyBUF_STRIDES | PyBUF_ND) < 0)
        return 0;
    const char *f = v->format;
    if (f && (*f == '@' || *f == '='))
        f++;
    if (v->itemsize != 4 || ((uintptr_t)v->buf % 4) || !f || strcmp(f, format) ||
        !PyBuffer_IsContiguous(v, 'C') || v->len % (columns * 4) ||
        !(v->ndim == 1 || (v->ndim == 2 && v->shape[1] == columns)))
        goto invalid;
    for (int i = 0; i < v->ndim; i++)
        if (v->strides && v->strides[i] <= 0)
            goto invalid;
    if (v->len / (columns * 4) > 65535)
        goto invalid;
    return 1;
invalid:
    PyBuffer_Release(v);
    PyErr_SetString(PyExc_ValueError,
                    "buffer must be aligned contiguous native float32/uint32 data "
                    "with the specified shape (at most 65535 rows)");
    return 0;
}
static PyObject *upload_mesh(PyObject *m, PyObject *args) {
    PyObject *v, *edges, *faces;
    if (!PyArg_ParseTuple(args, "OOO", &v, &edges, &faces))
        return NULL;
    Py_buffer bv = {0}, be = {0}, bf = {0};
    float *vertices = NULL;
    uint32_t *eidx = NULL, *fidx = NULL;
    h75_resource *p = NULL;
    h75_error e = {0};
    if (!mesh_buffer(v, &bv, "f", 3))
        return NULL;
    if (edges != Py_None && !mesh_buffer(edges, &be, "I", 2))
        goto done;
    if (faces != Py_None && !mesh_buffer(faces, &bf, "I", 3))
        goto done;
    vertices = PyMem_Malloc((size_t)bv.len ? (size_t)bv.len : 1);
    eidx = PyMem_Malloc((size_t)be.len ? (size_t)be.len : 1);
    fidx = PyMem_Malloc((size_t)bf.len ? (size_t)bf.len : 1);
    if (!vertices || !eidx || !fidx) {
        PyErr_NoMemory();
        goto done;
    }
    memcpy(vertices, bv.buf, (size_t)bv.len);
    if (be.len)
        memcpy(eidx, be.buf, (size_t)be.len);
    if (bf.len)
        memcpy(fidx, bf.buf, (size_t)bf.len);
    Py_BEGIN_ALLOW_THREADS
        p = h75_mesh_upload(vertices, (size_t)bv.len / 12, eidx, (size_t)be.len / 8, fidx,
                            (size_t)bf.len / 12, &e);
    Py_END_ALLOW_THREADS
done:
    PyBuffer_Release(&bv);
    if (be.obj)
        PyBuffer_Release(&be);
    if (bf.obj)
        PyBuffer_Release(&bf);
    PyMem_Free(vertices);
    PyMem_Free(eidx);
    PyMem_Free(fidx);
    if (PyErr_Occurred())
        return NULL;
    return wrap(PyModule_GetState(m), p, &e);
}
static PyObject *new_font(PyObject *m, PyObject *args) {
    const char *path;
    float size;
    if (!PyArg_ParseTuple(args, "sf", &path, &size))
        return NULL;
    h75_error e = {0};
    h75_resource *p;
    Py_BEGIN_ALLOW_THREADS
        p = h75_font_load(path, size, &e);
    Py_END_ALLOW_THREADS
    return wrap(PyModule_GetState(m), p, &e);
}
static PyObject *new_text(PyObject *m, PyObject *args) {
    PyObject *font, *text;
    if (!PyArg_ParseTuple(args, "OO", &font, &text))
        return NULL;
    const char *str = PyUnicode_AsUTF8(text);
    if (!str)
        return NULL;
    if (strlen(str) != (size_t)PyUnicode_GET_LENGTH(text)) {
        PyErr_SetString(PyExc_ValueError, "SDF text supports ASCII only");
        return NULL;
    }
    h75_resource *f = resource_ptr(font, PyModule_GetState(m));
    if (!f)
        return NULL;
    h75_resource_retain(f);
    h75_error e = {0};
    h75_resource *p;
    Py_BEGIN_ALLOW_THREADS
        p = h75_text_new(f, str, &e);
        h75_resource_release(f);
    Py_END_ALLOW_THREADS
    return wrap(PyModule_GetState(m), p, &e);
}
static PyObject *capabilities(PyObject *m, PyObject *ignored) {
    (void)m;
    (void)ignored;
    uint32_t c = hub75_runtime_capabilities();
    return Py_BuildValue(
        "{sI,ss,sO,sO,sO,sO,sO,sO,sO,ss,sO,sO,sO}", "abi", hub75_runtime_abi(), "build",
        hub75_runtime_build(), "cpu", Py_True, "gpio", (c & HUB75_CAP_GPIO) ? Py_True : Py_False,
        "shader", (c & HUB75_CAP_SHADER) ? Py_True : Py_False, "video",
        (c & HUB75_CAP_VIDEO) ? Py_True : Py_False, "png", (c & HUB75_CAP_PNG) ? Py_True : Py_False,
        "jpeg", (c & HUB75_CAP_JPEG) ? Py_True : Py_False, "text",
        (c & HUB75_CAP_PNG) ? Py_True : Py_False, "backend", (c & HUB75_CAP_SHADER) ? "gpu" : "cpu",
        "offscreen", Py_True, "retained", Py_True, "neon",
        (c & HUB75_CAP_NEON) ? Py_True : Py_False);
}
static PyMethodDef methods[] = {{"resource", new_resource, METH_O, NULL},
                                {"image", new_image, METH_VARARGS, NULL},
                                {"load_image", load_image, METH_O, NULL},
                                {"transform_image", transform_image, METH_VARARGS, NULL},
                                {"project", project, METH_VARARGS, NULL},
                                {"primitive", new_mesh, METH_VARARGS, NULL},
                                {"mesh", upload_mesh, METH_VARARGS, NULL},
                                {"font", new_font, METH_VARARGS, NULL},
                                {"text", new_text, METH_VARARGS, NULL},
                                {"capabilities", capabilities, METH_NOARGS, NULL},
                                {NULL}};
static int traverse(PyObject *m, visitproc visit, void *arg) {
    module_state *s = PyModule_GetState(m);
    Py_VISIT(s->resource_type);
    Py_VISIT(s->runtime_type);
    Py_VISIT(s->error);
    Py_VISIT(s->state_error);
    Py_VISIT(s->busy_error);
    Py_VISIT(s->unsupported_error);
    return 0;
}
static int clear(PyObject *m) {
    module_state *s = PyModule_GetState(m);
    Py_CLEAR(s->resource_type);
    Py_CLEAR(s->runtime_type);
    Py_CLEAR(s->error);
    Py_CLEAR(s->state_error);
    Py_CLEAR(s->busy_error);
    Py_CLEAR(s->unsupported_error);
    return 0;
}
static int execute(PyObject *m) {
    if (hub75_runtime_abi() != HUB75_RUNTIME_ABI) {
        PyErr_SetString(PyExc_ImportError, "incompatible hub75 native runtime ABI");
        return -1;
    }
#ifdef Py_GIL_DISABLED
    PyErr_SetString(PyExc_ImportError, "free-threaded Python is not supported");
    return -1;
#endif
    if (PyInterpreterState_GetID(PyInterpreterState_Get()) != 0) {
        PyErr_SetString(PyExc_ImportError, "subinterpreters are not supported");
        return -1;
    }
    module_state *s = PyModule_GetState(m);
    s->resource_type = PyType_FromModuleAndSpec(m, &resource_spec, NULL);
    s->runtime_type = PyType_FromModuleAndSpec(m, &runtime_spec, NULL);
    s->error = PyErr_NewException("hub75gpu.Error", PyExc_RuntimeError, NULL);
    if (!s->resource_type || !s->runtime_type || !s->error)
        return -1;
    s->state_error = PyErr_NewException("hub75gpu.StateError", s->error, NULL);
    s->busy_error = PyErr_NewException("hub75gpu.BusyError", s->error, NULL);
    s->unsupported_error = PyErr_NewException("hub75gpu.UnsupportedError", s->error, NULL);
    if (!s->state_error || !s->busy_error || !s->unsupported_error)
        return -1;
#define ADD(name, value)                                                                           \
    if (PyModule_AddObjectRef(m, name, value) < 0)                                                 \
    return -1
    ADD("Resource", s->resource_type);
    ADD("Runtime", s->runtime_type);
    ADD("Error", s->error);
    ADD("StateError", s->state_error);
    ADD("BusyError", s->busy_error);
    ADD("UnsupportedError", s->unsupported_error);
#undef ADD
    return 0;
}
static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, execute},
#if PY_VERSION_HEX >= 0x030C0000
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
    {0, NULL}};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "hub75gpu._native",
    .m_doc = "Safe native HUB75 runtime (no import-time hardware access).",
    .m_size = sizeof(module_state),
    .m_methods = methods,
    .m_slots = slots,
    .m_traverse = traverse,
    .m_clear = clear};
PyMODINIT_FUNC PyInit__native(void) {
    return PyModuleDef_Init(&module);
}
