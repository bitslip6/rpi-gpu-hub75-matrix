// scene_transform.c
// Build: gcc -std=c17 -O2 scene_transform.c -lm -o scene_transform

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "hub75gpu.h"

#ifndef M_PI
#define M_PI 3.14159265f
#endif


/* =========================
   Small math helpers
   ========================= */

static inline vec3 vec3_add(vec3 a, vec3 b) { return (vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline vec3 vec3_sub(vec3 a, vec3 b) { return (vec3){a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline vec3 vec3_scale(vec3 a, float s) { return (vec3){a.x*s, a.y*s, a.z*s}; }

static inline float vec3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 vec3_cross(vec3 a, vec3 b) {
    return (vec3){ a.y*b.z - a.z*b.y,
                   a.z*b.x - a.x*b.z,
                   a.x*b.y - a.y*b.x };
}
static inline vec3 vec3_norm(vec3 v) {
    float d = sqrtf(vec3_dot(v, v));
    return d > 0.0f ? vec3_scale(v, 1.0f/d) : (vec3){0,0,0};
}

/* =========================
   Matrix helpers
   ========================= */

static inline mat4 mat4_identity(void) {
    mat4 r = { .m = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    }};
    return r;
}

static inline mat4 mat4_mul(mat4 a, mat4 b) {
    /* Column-major matrix multiply: r = a * b */
    mat4 r = {0};
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

static inline vec3 mat4_mul_point(const mat4 m, const vec3 p) {
    /* Column-major multiply with column vector: clip = M * [p.x, p.y, p.z, 1]^T */
    float x = m.m[0]*p.x + m.m[4]*p.y + m.m[8]*p.z + m.m[12];
    float y = m.m[1]*p.x + m.m[5]*p.y + m.m[9]*p.z + m.m[13];
    float z = m.m[2]*p.x + m.m[6]*p.y + m.m[10]*p.z + m.m[14];

    const float w = m.m[3]*p.x + m.m[7]*p.y + m.m[11]*p.z + m.m[15];
    /* Branchless safe divide by w: clamp |w| to epsilon and preserve sign */
    const float eps = 1e-8f;
    const float denom = copysignf(fmaxf(fabsf(w), eps), w);
    const float inv_w = 1.0f / denom;
    x *= inv_w; y *= inv_w; z *= inv_w;
    return (vec3){x, y, z};
}

/* =========================
   Transform builders
   ========================= */

static inline mat4 mat4_translate(const vec3 t) {
    /* Column-major translation: last column holds translation */
    mat4 r = mat4_identity();
    r.m[12]  = t.x;
    r.m[13]  = t.y;
    r.m[14] = t.z;
    return r;
}

static inline mat4 mat4_rotate_x(const float a) {
    float c = cosf(a), s = sinf(a);
    mat4 r = mat4_identity();
    r.m[5] = c;  r.m[6] = -s;
    r.m[9] = s;  r.m[10]= c;
    return r;
}
static inline mat4 mat4_rotate_y(const float a) {
    float c = cosf(a), s = sinf(a);
    mat4 r = mat4_identity();
    r.m[0] = c;  r.m[2] = s;
    r.m[8] = -s; r.m[10]= c;
    return r;
}
static inline mat4 mat4_rotate_z(const float a) {
    float c = cosf(a), s = sinf(a);
    mat4 r = mat4_identity();
    r.m[0] = c;  r.m[1] = -s;
    r.m[4] = s;  r.m[5] = c;
    return r;
}

static inline mat4 mat4_scale(vec3 s) {
    mat4 r = mat4_identity();
    r.m[0] = s.x;
    r.m[5] = s.y;
    r.m[10]= s.z;
    return r;
}

/* perspective matrix, right-handed, depth in [+znear, +zfar] */
static inline mat4 mat4_perspective(const float fovy_radians, const float aspect, const float znear, const float zfar) {
    /* Column-major, right-handed, OpenGL-style NDC z in [-1,1] */
    float f = 1.0f / tanf(fovy_radians * 0.5f);
    mat4 r = (mat4){0};
    r.m[0]  = f / aspect;                      /* col0,row0 */
    r.m[5]  = f;                               /* col1,row1 */
    r.m[10] = (zfar + znear) / (znear - zfar); /* col2,row2 */
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar); /* col3,row2 */
    r.m[11] = -1.0f;                           /* col2,row3 */
    /* r.m[15] stays 0 */
    return r;
}

/* view matrix using eye, target, up, right-handed */
static inline mat4 mat4_look_at(const vec3 eye, const vec3 target, const vec3 up) {
    /* Column-major view matrix */
    vec3 f = vec3_norm(vec3_sub(target, eye)); /* forward */
    vec3 s = vec3_norm(vec3_cross(f, up));     /* right */
    vec3 u = vec3_cross(s, f);                 /* true up */

    mat4 r = mat4_identity();
    /* basis vectors in columns 0..2 */
    r.m[0] = s.x;   r.m[4] = s.y;   r.m[8]  = s.z;   r.m[12] = -vec3_dot(s, eye);
    r.m[1] = u.x;   r.m[5] = u.y;   r.m[9]  = u.z;   r.m[13] = -vec3_dot(u, eye);
    r.m[2] = -f.x;  r.m[6] = -f.y;  r.m[10] = -f.z;  r.m[14] =  vec3_dot(f, eye);
    r.m[3] = 0.0f;  r.m[7] = 0.0f;  r.m[11] = 0.0f;  r.m[15] = 1.0f;
    return r;
}


/* compose model matrix as T * Rz * Ry * Rx * S for clarity */
static inline mat4 transform_model_matrix(const transform_t *t) {
    mat4 s = mat4_scale(t->scale.x == 0 && t->scale.y == 0 && t->scale.z == 0
                        ? (vec3){1,1,1} : t->scale);
    mat4 rx = mat4_rotate_x(t->rotation.x);
    mat4 ry = mat4_rotate_y(t->rotation.y);
    mat4 rz = mat4_rotate_z(t->rotation.z);
    mat4 tr = mat4_translate(t->position);

    /* order: translate then rotate then scale for classic object-local rotation */
    mat4 r = mat4_mul(rz, mat4_mul(ry, rx));
    return mat4_mul(tr, mat4_mul(r, s));
}

static inline mat4 camera_view_matrix(const camera_t *c) {
    return mat4_look_at(c->position, c->target, c->up);
}

static inline mat4 camera_proj_matrix(const camera_t *c) {
    return mat4_perspective(c->fov_y, c->aspect, c->z_near, c->z_far);
}

/* =========================
   Object management
   ========================= */

vert_list_t* vert_list_new(uint16_t num_vertices) {
    size_t total = sizeof(vert_list_t) + sizeof(vec3) * num_vertices;
    vert_list_t *vlist = calloc(1, total);
    if (!vlist) return NULL;
    vlist->length = num_vertices;
    vlist->list = (vec3*)(vlist + 1);  /* Point to memory after the struct */
    return vlist;
}

edge_list_t* edge_list_new(const uint16_t num_edges) {
    size_t total = sizeof(edge_list_t) + sizeof(vec2) * num_edges;
    edge_list_t *elist = calloc(1, total);
    if (!elist) return NULL;
    elist->length = num_edges;
    elist->list = (vec2*)(elist + 1);  /* Point to memory after the struct */
    return elist;
}

color_list_t* color_list_new(const uint16_t length) {
    size_t total = sizeof(color_list_t) + sizeof(RGB) * length;
    color_list_t *clist = calloc(1, total);
    if (!clist) return NULL;
    clist->length = length;
    clist->list = (RGB*)(clist + 1);  /* Point to memory after the struct */
    return clist;
}

face_list_t* face_list_new(const uint16_t num_faces) {
    size_t total = sizeof(face_list_t) + sizeof(vec3) * num_faces;
    face_list_t *flist = calloc(1, total);
    if (!flist) return NULL;
    flist->length = num_faces;
    flist->list = (vec3*)(flist + 1);  /* Point to memory after the struct */
    return flist;
}

normal_list_t* normal_list_new(const uint16_t num_normals) {
    size_t total = sizeof(normal_list_t) + sizeof(vec3) * num_normals;
    normal_list_t *nlist = calloc(1, total);
    if (!nlist) return NULL;
    nlist->length = num_normals;
    nlist->list = (vec3*)(nlist + 1);  /* Point to memory after the struct */
    return nlist;
}

object_t* object_new(const uint16_t num_vertices, const uint16_t num_edges, const uint16_t num_faces) {
    size_t total = sizeof(object_t);
    total += sizeof(vec3) * num_vertices; /* space for rendered vertices */

    object_t *obj = calloc(1, total);
    if (!obj) return NULL;

    obj->verticies = vert_list_new(num_vertices);
    obj->rendered_vertices = calloc(num_vertices, sizeof(vec3));
    obj->edge_colors = color_list_new(num_edges);  /* Should be num_edges, not num_vertices */
    obj->edges = edge_list_new(num_edges);
    obj->faces = face_list_new(num_faces);
    obj->normals = normal_list_new(num_faces);

    obj->trifill_buffer = NULL;
    obj->trifill_capacity = 0;

    if (!obj->verticies || !obj->rendered_vertices || !obj->edge_colors ||
        !obj->edges || !obj->faces || !obj->normals) {
        /* Cleanup on failure */
        if (obj->verticies) free(obj->verticies);
        if (obj->rendered_vertices) free(obj->rendered_vertices);
        if (obj->edge_colors) free(obj->edge_colors);
        if (obj->edges) free(obj->edges);
        if (obj->faces) free(obj->faces);
        if (obj->normals) free(obj->normals);
        if (obj->trifill_buffer) free(obj->trifill_buffer);
        free(obj);
        return NULL;
    }
    /* Allocate trifill buffer for filled rendering */
    if (num_faces > 0) {
        obj->trifill_buffer = malloc(sizeof(_TriFill) * num_faces);
        if (!obj->trifill_buffer) {
            /* Cleanup on failure */
            if (obj->verticies) free(obj->verticies);
            if (obj->rendered_vertices) free(obj->rendered_vertices);
            if (obj->edge_colors) free(obj->edge_colors);
            if (obj->edges) free(obj->edges);
            if (obj->faces) free(obj->faces);
            if (obj->normals) free(obj->normals);
            free(obj);
            return NULL;
        }
        obj->trifill_capacity = num_faces;
    }
    /* Defaults */
    obj->cull_backface = true;
    obj->draw_mode = DRAW_WIRE;
    obj->shadow_enabled = true; /* default: objects cast/receive shadows */
    return obj;
}


/** 
 * @brief Create a unit cube object centered at origin
 */
object_t* object_cube(const object_draw_mode_t mode, const bool cull_backface) {
    object_t *obj = object_new(8, 12, 12);  /* 8 vertices, 12 edges, 12 triangles (2 per face) */
    if (!obj) return NULL;

    /* fill vertices */
    vec3 *v = obj->verticies->list;
    v[0] = (vec3){-1,-1,-1}; v[1] = (vec3){+1,-1,-1};
    v[2] = (vec3){+1,+1,-1}; v[3] = (vec3){-1,+1,-1};
    v[4] = (vec3){-1,-1,+1}; v[5] = (vec3){+1,-1,+1};
    v[6] = (vec3){+1,+1,+1}; v[7] = (vec3){-1,+1,+1};

    /* fill edges */
    vec2 *e = obj->edges->list;
    e[0]  = (vec2){0,1};  e[1]  = (vec2){1,2};
    e[2]  = (vec2){2,3};  e[3]  = (vec2){3,0};
    e[4]  = (vec2){4,5};  e[5]  = (vec2){5,6};
    e[6]  = (vec2){6,7};  e[7]  = (vec2){7,4};
    e[8]  = (vec2){0,4};  e[9]  = (vec2){1,5};
    e[10] = (vec2){2,6};  e[11] = (vec2){3,7};

    /* fill faces (triangles) - 2 triangles per cube face */
    vec3 *f = obj->faces->list;
    /*
     * Ensure all triangle windings are consistent with outward normals.
     * Using right-handed coordinates and row-major math, a front-facing
     * triangle (as seen from outside the cube) should have its vertices
     * wound such that the computed normal points outward.
     */
    /* front face (z=-1) */  f[0] = (vec3){0,2,1};  f[1] = (vec3){0,3,2};
    /* back face (z=+1) */   f[2] = (vec3){5,7,4};  f[3] = (vec3){5,6,7};
    /* left face (x=-1) */   f[4] = (vec3){4,3,0};  f[5] = (vec3){4,7,3};
    /* right face (x=+1) */  f[6] = (vec3){1,6,5};  f[7] = (vec3){1,2,6};
    /* bottom face (y=-1) */ f[8] = (vec3){4,1,5};  f[9] = (vec3){4,0,1};
    /* top face (y=+1) */    f[10]= (vec3){3,6,2};  f[11]= (vec3){3,7,6};

    /* fill face normals */
    vec3 *n = obj->normals->list;
    n[0] = n[1] = (vec3){0,0,-1};   /* front face */
    n[2] = n[3] = (vec3){0,0,+1};   /* back face */
    n[4] = n[5] = (vec3){-1,0,0};   /* left face */
    n[6] = n[7] = (vec3){+1,0,0};   /* right face */
    n[8] = n[9] = (vec3){0,-1,0};   /* bottom face */
    n[10]= n[11]= (vec3){0,+1,0};   /* top face */

    /* fill edge colors - make all edges white */
    RGB *c = obj->edge_colors->list;
    for (int i = 0; i < 12; i++) {
        c[i] = (RGB){255, 255, 255};  /* white edges */
    }

    obj->draw_mode = mode;
    obj->cull_backface = cull_backface;
    return obj;
}

/** 
 * @brief Create a tetrahedron (triangular pyramid) object centered at origin
 */
object_t* object_tetrahedron(const object_draw_mode_t mode, const bool cull_backface) {
    object_t *obj = object_new(4, 6, 4);  /* 4 vertices, 6 edges, 4 triangular faces */
    if (!obj) return NULL;

    /* fill vertices - regular tetrahedron inscribed in unit sphere */
    vec3 *v = obj->verticies->list;
    float a = 1.0f / sqrtf(3.0f);  /* 1/sqrt(3) for unit tetrahedron */
    v[0] = (vec3){ a,  a,  a};  /* top front right */
    v[1] = (vec3){-a, -a,  a};  /* bottom back right */
    v[2] = (vec3){-a,  a, -a};  /* top back left */
    v[3] = (vec3){ a, -a, -a};  /* bottom front left */

    /* fill edges - connect each vertex to every other */
    vec2 *e = obj->edges->list;
    e[0] = (vec2){0,1};  e[1] = (vec2){0,2};
    e[2] = (vec2){0,3};  e[3] = (vec2){1,2};
    e[4] = (vec2){1,3};  e[5] = (vec2){2,3};

    /* fill faces (triangles) */
    vec3 *f = obj->faces->list;
    f[0] = (vec3){0,1,3};  /* front face */
    f[1] = (vec3){0,3,2};  /* right face */
    f[2] = (vec3){0,2,1};  /* top face */
    f[3] = (vec3){1,2,3};  /* bottom face */

    /* Ensure outward CCW winding and compute face normals */
    {
        /* Helper to enforce outward-facing normals (centroid test) */
        if (obj->faces && obj->normals && obj->verticies) {
            vec3 *V = obj->verticies->list;
            for (uint16_t i = 0; i < obj->faces->length; ++i) {
                uint16_t i0 = (uint16_t)f[i].x;
                uint16_t i1 = (uint16_t)f[i].y;
                uint16_t i2 = (uint16_t)f[i].z;
                vec3 v0 = V[i0], v1 = V[i1], v2 = V[i2];
                vec3 e1 = vec3_sub(v1, v0);
                vec3 e2 = vec3_sub(v2, v0);
                vec3 nn = vec3_cross(e1, e2);
                vec3 c  = (vec3){ (v0.x+v1.x+v2.x)/3.0f, (v0.y+v1.y+v2.y)/3.0f, (v0.z+v1.z+v2.z)/3.0f };
                float d = vec3_dot(nn, c);
                if (d < 0.0f) {
                    /* Flip winding to make normal outward */
                    float tmp = f[i].y; f[i].y = f[i].z; f[i].z = tmp;
                    /* Recompute normal */
                    i1 = (uint16_t)f[i].y; i2 = (uint16_t)f[i].z;
                    v1 = V[i1]; v2 = V[i2];
                    e1 = vec3_sub(v1, v0);
                    e2 = vec3_sub(v2, v0);
                    nn = vec3_cross(e1, e2);
                }
                obj->normals->list[i] = vec3_norm(nn);
            }
        }
    }

    /* fill edge colors - make all edges white */
    RGB *c = obj->edge_colors->list;
    for (int i = 0; i < 6; i++) {
        c[i] = (RGB){255, 255, 255};  /* white edges */
    }

    obj->draw_mode = mode;
    obj->cull_backface = cull_backface;

    return obj;
}

/** 
 * @brief Create an octahedron object centered at origin
 */
object_t* object_octahedron(object_draw_mode_t mode, bool cull_backface) {
    object_t *obj = object_new(6, 12, 8);  /* 6 vertices, 12 edges, 8 triangular faces */
    if (!obj) return NULL;

    /* fill vertices - octahedron with vertices on coordinate axes */
    vec3 *v = obj->verticies->list;
    v[0] = (vec3){ 1,  0,  0};  /* +X */
    v[1] = (vec3){-1,  0,  0};  /* -X */
    v[2] = (vec3){ 0,  1,  0};  /* +Y */
    v[3] = (vec3){ 0, -1,  0};  /* -Y */
    v[4] = (vec3){ 0,  0,  1};  /* +Z */
    v[5] = (vec3){ 0,  0, -1};  /* -Z */

    /* fill edges - connect vertices to form 8 triangular faces */
    vec2 *e = obj->edges->list;
    /* top pyramid edges (connecting +Y to ±X, ±Z) */
    e[0] = (vec2){2,0};  e[1] = (vec2){2,1};
    e[2] = (vec2){2,4};  e[3] = (vec2){2,5};
    /* bottom pyramid edges (connecting -Y to ±X, ±Z) */
    e[4] = (vec2){3,0};  e[5] = (vec2){3,1};
    e[6] = (vec2){3,4};  e[7] = (vec2){3,5};
    /* middle ring edges */
    e[8] = (vec2){0,4};  e[9] = (vec2){4,1};
    e[10]= (vec2){1,5};  e[11]= (vec2){5,0};

    /* edge colors white */
    for (int i = 0; i < 12; ++i) obj->edge_colors->list[i] = (RGB){255,255,255};

    /* faces (8 triangles) */
    vec3 *f = obj->faces->list;
    f[0] = (vec3){2,0,4};
    f[1] = (vec3){2,4,1};
    f[2] = (vec3){2,1,5};
    f[3] = (vec3){2,5,0};
    f[4] = (vec3){3,0,5};
    f[5] = (vec3){3,5,1};
    f[6] = (vec3){3,1,4};
    f[7] = (vec3){3,4,0};

    /* enforce outward CCW and compute normals */
    if (obj->faces && obj->normals && obj->verticies) {
        vec3 *V = obj->verticies->list;
        for (uint16_t i = 0; i < obj->faces->length; ++i) {
            uint16_t i0 = (uint16_t)f[i].x;
            uint16_t i1 = (uint16_t)f[i].y;
            uint16_t i2 = (uint16_t)f[i].z;
            vec3 v0 = V[i0], v1 = V[i1], v2 = V[i2];
            vec3 e1 = vec3_sub(v1, v0);
            vec3 e2 = vec3_sub(v2, v0);
            vec3 nn = vec3_cross(e1, e2);
            vec3 c  = (vec3){ (v0.x+v1.x+v2.x)/3.0f, (v0.y+v1.y+v2.y)/3.0f, (v0.z+v1.z+v2.z)/3.0f };
            float d = vec3_dot(nn, c);
            if (d < 0.0f) {
                float tmp = f[i].y; f[i].y = f[i].z; f[i].z = tmp;
                i1 = (uint16_t)f[i].y; i2 = (uint16_t)f[i].z;
                v1 = V[i1]; v2 = V[i2];
                e1 = vec3_sub(v1, v0);
                e2 = vec3_sub(v2, v0);
                nn = vec3_cross(e1, e2);
            }
            obj->normals->list[i] = vec3_norm(nn);
        }
    }

    obj->draw_mode = mode;
    obj->cull_backface = cull_backface;
    return obj;
}

/** 
 * @brief Create a pyramid (square base) object centered at origin
 */
object_t* object_pyramid(void) {
    object_t *obj = object_new(5, 8, 6);  /* 5 vertices, 8 edges, 6 triangular faces */
    if (!obj) return NULL;

    // fill vertices 
    vec3 *v = obj->verticies->list;
    // square base at y = -1 
    v[0] = (vec3){-1, -1, -1};  // base: back left
    v[1] = (vec3){ 1, -1, -1};  // base: back right
    v[2] = (vec3){ 1, -1,  1};  // base: front right
    v[3] = (vec3){-1, -1,  1};  // base: front left
    // apex at top 
    v[4] = (vec3){ 0,  1,  0};  // apex

    // fill edges 
    vec2 *e = obj->edges->list;
    // base edges
    e[0] = (vec2){0,1};  e[1] = (vec2){1,2};
    e[2] = (vec2){2,3};  e[3] = (vec2){3,0};
    // edges from base to apex
    e[4] = (vec2){0,4};  e[5] = (vec2){1,4};
    e[6] = (vec2){2,4};  e[7] = (vec2){3,4};

    return obj;
}

/** 
 * @brief Create a cylinder object centered at origin (wireframe approximation)
 * @param segments Number of segments around the circumference (minimum 3)
 */
object_t* object_cylinder(const uint16_t segments, const object_draw_mode_t mode, const bool cull_backface) {
    uint16_t seg = segments;
    if (seg < 3) seg = 3;
    if (seg > 32) seg = 32;  // practical limit for wireframe 

    uint16_t num_vertices = seg * 2;  // top and bottom circles 
    uint16_t num_edges = seg * 3;     // top circle + bottom circle + vertical lines 
    uint16_t num_faces = seg * 4;     // top/bottom caps + side triangles
    
    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    // fill vertices 
    vec3 *v = obj->verticies->list;
    for (uint16_t i = 0; i < seg; ++i) {
        float angle = 2.0f * (float)M_PI * (float)i / (float)seg;
        float x = cosf(angle);
        float z = sinf(angle);
        
        v[i] = (vec3){x, 1.0f, z};              // top circle
        v[i + seg] = (vec3){x, -1.0f, z};  // bottom circle 
    }

    // fill edges
    vec2 *e = obj->edges->list;
    uint16_t edge_idx = 0;
    
    // top circle edges 
    for (uint16_t i = 0; i < seg; ++i) {
        e[edge_idx++] = (vec2){(float)i, (float)((i + 1) % seg)};
    }
    
    // bottom circle edges 
    for (uint16_t i = 0; i < seg; ++i) {
        uint16_t bottom_i = i + seg;
        uint16_t bottom_next = (uint16_t)((i + 1) % seg) + seg;
        e[edge_idx++] = (vec2){bottom_i, bottom_next};
    }
    
    // vertical edges connecting top to bottom 
    for (uint16_t i = 0; i < seg; ++i) {
        e[edge_idx++] = (vec2){i, i + seg};
    }
    /* edge colors white */
    for (uint16_t i = 0; i < num_edges; ++i) obj->edge_colors->list[i] = (RGB){255,255,255};

    /* faces: sides (2*seg) and caps (2*(seg-2)) */
    vec3 *F = obj->faces->list;
    uint16_t fcount = 0;
    for (uint16_t i = 0; i < seg; ++i) {
        uint16_t inext = (uint16_t)((i + 1) % seg);
        uint16_t t0 = i, t1 = inext;
        uint16_t b0 = (uint16_t)(i + seg);
        uint16_t b1 = (uint16_t)(inext + seg);
        F[fcount++] = (vec3){ (float)t0, (float)b0, (float)t1 };
        F[fcount++] = (vec3){ (float)t1, (float)b0, (float)b1 };
    }
    for (uint16_t i = 1; i + 1 < seg; ++i) {
        F[fcount++] = (vec3){ 0.0f, (float)i, (float)(i+1) };
    }
    uint16_t b_anchor = seg;
    for (uint16_t i = 1; i + 1 < seg; ++i) {
        F[fcount++] = (vec3){ (float)b_anchor, (float)(seg + i + 1), (float)(seg + i) };
    }

    if (obj->normals && obj->verticies) {
        vec3 *V = obj->verticies->list;
        for (uint16_t i = 0; i < fcount; ++i) {
            uint16_t i0 = (uint16_t)F[i].x;
            uint16_t i1 = (uint16_t)F[i].y;
            uint16_t i2 = (uint16_t)F[i].z;
            vec3 v0 = V[i0], v1 = V[i1], v2 = V[i2];
            vec3 e1v = vec3_sub(v1, v0);
            vec3 e2v = vec3_sub(v2, v0);
            vec3 nn = vec3_cross(e1v, e2v);
            vec3 c  = (vec3){ (v0.x+v1.x+v2.x)/3.0f, (v0.y+v1.y+v2.y)/3.0f, (v0.z+v1.z+v2.z)/3.0f };
            float d = vec3_dot(nn, c);
            if (d < 0.0f) {
                float tmp = F[i].y; F[i].y = F[i].z; F[i].z = tmp;
                i1 = (uint16_t)F[i].y; i2 = (uint16_t)F[i].z;
                v1 = V[i1]; v2 = V[i2];
                e1v = vec3_sub(v1, v0);
                e2v = vec3_sub(v2, v0);
                nn = vec3_cross(e1v, e2v);
            }
            obj->normals->list[i] = vec3_norm(nn);
        }
    }

    obj->faces->length = fcount;
    obj->draw_mode = mode;
    obj->cull_backface = cull_backface;

    return obj;
}

/* ---- Helpers for icosphere generation (sphere) ---- */
typedef struct {
    uint16_t a;
    uint16_t b;
    uint16_t mid;
    uint8_t  used;
} MidEntry;

typedef struct {
    uint32_t cap;
    MidEntry *tab;
} MidCache;

static inline uint32_t pair_key_u32(uint16_t a, uint16_t b){ return ((uint32_t)a << 16) | (uint32_t)b; }
static inline uint32_t hash_u32(uint32_t k){
    /* Simple integer hash (xorshift mix) */
    k ^= k >> 16; k *= 0x7feb352dU; k ^= k >> 15; k *= 0x846ca68bU; k ^= k >> 16;
    return k;
}

/* Add a normalized vertex to the array (no realloc expected due to pre-sizing) */
static void sphere_add_vertex(vec3 **Vptr, uint16_t *vcount, uint16_t *vcap, vec3 p){
    (void)vcap; /* capacity preallocated by caller */
    float d = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    if (d > 1e-9f){ p.x/=d; p.y/=d; p.z/=d; }
    (*Vptr)[*vcount] = p;
    (*vcount)++;
}

/* Look up or create midpoint vertex between indices v0 and v1 */
static uint16_t mid_lookup(MidCache *mc, vec3 **Vptr, uint16_t *vcount, uint16_t *vcap, uint16_t v0, uint16_t v1){
    (void)vcap; /* pre-sized */
    uint16_t a = (v0 < v1) ? v0 : v1;
    uint16_t b = (v0 < v1) ? v1 : v0;
    if (!mc->tab || mc->cap == 0){
        /* Fallback: just create */
        vec3 vm = { ((*Vptr)[a].x + (*Vptr)[b].x)*0.5f,
                    ((*Vptr)[a].y + (*Vptr)[b].y)*0.5f,
                    ((*Vptr)[a].z + (*Vptr)[b].z)*0.5f };
        sphere_add_vertex(Vptr, vcount, vcap, vm);
        return (uint16_t)(*vcount - 1);
    }
    uint32_t key = pair_key_u32(a, b);
    uint32_t mask = mc->cap - 1u;
    uint32_t h = hash_u32(key);
    for (uint32_t probe = 0; probe < mc->cap; ++probe){
        uint32_t idx = (h + probe) & mask;
        MidEntry *e = &mc->tab[idx];
        if (!e->used){
            /* Insert */
            vec3 vm = { ((*Vptr)[a].x + (*Vptr)[b].x)*0.5f,
                        ((*Vptr)[a].y + (*Vptr)[b].y)*0.5f,
                        ((*Vptr)[a].z + (*Vptr)[b].z)*0.5f };
            sphere_add_vertex(Vptr, vcount, vcap, vm);
            e->a = a; e->b = b; e->mid = (uint16_t)(*vcount - 1); e->used = 1;
            return e->mid;
        }
        if (e->a == a && e->b == b){
            return e->mid;
        }
    }
    /* Table full: fallback create */
    vec3 vm = { ((*Vptr)[a].x + (*Vptr)[b].x)*0.5f,
                ((*Vptr)[a].y + (*Vptr)[b].y)*0.5f,
                ((*Vptr)[a].z + (*Vptr)[b].z)*0.5f };
    sphere_add_vertex(Vptr, vcount, vcap, vm);
    return (uint16_t)(*vcount - 1);
}

typedef struct {
    uint16_t a;
    uint16_t b;
    uint8_t  used;
} EdgeEntry;

typedef struct {
    uint32_t cap;
    EdgeEntry *tab;
} EdgeSet;

static void add_edge_unique(EdgeSet *es, vec2 *E, uint16_t *ecount, uint16_t i0, uint16_t i1){
    uint16_t a = (i0 < i1) ? i0 : i1;
    uint16_t b = (i0 < i1) ? i1 : i0;
    if (!es->tab || es->cap == 0){
        E[(*ecount)++] = (vec2){i0, i1};
        return;
    }
    uint32_t key = pair_key_u32(a, b);
    uint32_t mask = es->cap - 1u;
    uint32_t h = hash_u32(key);
    for (uint32_t probe = 0; probe < es->cap; ++probe){
        uint32_t idx = (h + probe) & mask;
        EdgeEntry *ee = &es->tab[idx];
        if (!ee->used){
            ee->a = a; ee->b = b; ee->used = 1;
            E[(*ecount)++] = (vec2){i0, i1};
            return;
        }
        if (ee->a == a && ee->b == b){
            return; /* already added */
        }
    }
    /* Table full: append anyway */
    E[(*ecount)++] = (vec2){i0, i1};
}

/** 
 * @brief Create a sphere object (geodesic approximation with icosphere)
 * @param subdivisions Number of subdivision levels (0-4 recommended)
 */
object_t* object_sphere(uint16_t subdivisions) {
    /* Clamp subdivisions for memory/perf */
    if (subdivisions > 4) subdivisions = 4;

    /* Capacity estimates for icosphere */
    uint16_t num_vertices = 12;           /* 10*4^0 + 2 */
    uint16_t num_faces    = 20;           /* base icosahedron */
    uint16_t num_edges    = 30;           /* base */
    for (uint16_t i = 0; i < subdivisions; ++i) {
        num_vertices += num_edges;        /* one new vertex per split edge */
        num_faces    *= 4;                /* each face -> 4 */
        num_edges    *= 4;                /* each edge -> 4 */
    }

    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    /* Build normalized icosahedron */
    const float phi = (1.0f + sqrtf(5.0f)) * 0.5f;
    const float inv_len = 1.0f / sqrtf(1.0f + phi*phi);
    vec3 base_v[12] = {
        { inv_len,  phi*inv_len, 0}, { -inv_len,  phi*inv_len, 0},
        { inv_len, -phi*inv_len, 0}, { -inv_len, -phi*inv_len, 0},
        { 0,  inv_len,  phi*inv_len}, { 0, -inv_len,  phi*inv_len},
        { 0,  inv_len, -phi*inv_len}, { 0, -inv_len, -phi*inv_len},
        {  phi*inv_len, 0,  inv_len}, { -phi*inv_len, 0,  inv_len},
        {  phi*inv_len, 0, -inv_len}, { -phi*inv_len, 0, -inv_len}
    };

    /* Faces of icosahedron mapped to our vertex order (see comment above) */
    uint16_t faces_idx[20][3] = {
        {1,9,4}, {1,4,0}, {1,0,6}, {1,6,11}, {1,11,9},
        {0,4,8}, {4,9,5}, {9,11,3}, {11,6,7}, {6,0,10},
        {2,8,5}, {2,5,3}, {2,3,7}, {2,7,10}, {2,10,8},
        {5,8,4}, {3,5,9}, {7,3,11}, {10,7,6}, {8,10,0}
    };

    vec3 *V = obj->verticies->list;
    vec3 *N = obj->normals->list;
    vec3 *F = obj->faces->list; /* actually vec3 of indices (x,y,z) */

    uint16_t vcount = 0; uint16_t vcap = num_vertices;
    for (int i = 0; i < 12; ++i) {
        sphere_add_vertex(&V, &vcount, &vcap, base_v[i]);
    }

    /* Initialize faces */
    uint16_t fcount = 20;
    for (int i = 0; i < 20; ++i) {
        F[i] = (vec3){ (float)faces_idx[i][0], (float)faces_idx[i][1], (float)faces_idx[i][2] };
    }

    /* Midpoint cache to avoid duplicate vertices on shared edges */
    MidCache mc = {0};
    mc.cap = (uint32_t)(num_edges * 2u + 64u);
    mc.tab = (mc.cap ? calloc(mc.cap, sizeof(MidEntry)) : NULL);

    /* Subdivide faces */
    for (uint16_t s = 0; s < subdivisions; ++s) {
        uint16_t old_fcount = fcount;
        vec3 *oldF = malloc(old_fcount * sizeof(vec3));
        if (!oldF) break;
        for (uint16_t i = 0; i < old_fcount; ++i) oldF[i] = F[i];
        fcount = 0;
        for (uint16_t i = 0; i < old_fcount; ++i) {
            uint16_t v0 = (uint16_t)oldF[i].x;
            uint16_t v1 = (uint16_t)oldF[i].y;
            uint16_t v2 = (uint16_t)oldF[i].z;
            uint16_t a = mid_lookup(&mc, &V, &vcount, &vcap, v0, v1);
            uint16_t b = mid_lookup(&mc, &V, &vcount, &vcap, v1, v2);
            uint16_t c = mid_lookup(&mc, &V, &vcount, &vcap, v2, v0);
            /* Four new faces */
            F[fcount++] = (vec3){ (float)v0, (float)a, (float)c };
            F[fcount++] = (vec3){ (float)v1, (float)b, (float)a };
            F[fcount++] = (vec3){ (float)v2, (float)c, (float)b };
            F[fcount++] = (vec3){ (float)a,  (float)b, (float)c };
        }
        free(oldF);
    }

    /* Build unique edges from faces */
    vec2 *E = obj->edges->list;
    uint16_t ecount = 0;
    EdgeSet es = {0};
    es.cap = (uint32_t)num_edges * 2u + 64u;
    es.tab = (es.cap ? calloc(es.cap, sizeof(EdgeEntry)) : NULL);

    for (uint16_t i = 0; i < fcount; ++i) {
        uint16_t i0 = (uint16_t)F[i].x, i1 = (uint16_t)F[i].y, i2 = (uint16_t)F[i].z;
        add_edge_unique(&es, E, &ecount, i0, i1);
        add_edge_unique(&es, E, &ecount, i1, i2);
        add_edge_unique(&es, E, &ecount, i2, i0);
    }

    /* Face normals (flat) */
    for (uint16_t i = 0; i < fcount && i < obj->normals->length; ++i) {
        uint16_t i0 = (uint16_t)F[i].x, i1 = (uint16_t)F[i].y, i2 = (uint16_t)F[i].z;
        vec3 e1 = vec3_sub(V[i1], V[i0]);
        vec3 e2 = vec3_sub(V[i2], V[i0]);
        N[i] = vec3_norm(vec3_cross(e1, e2));
    }

    if (mc.tab) free(mc.tab);
    if (es.tab) free(es.tab);

    obj->verticies->length = vcount;
    obj->faces->length = fcount;
    obj->edges->length = ecount;
    return obj;
}

/** 
 * @brief Create a torus object centered at origin
 * @param major_segments Number of segments around the major radius (minimum 3)
 * @param minor_segments Number of segments around the minor radius (minimum 3)
 */
/* Helper: centerline point on the major circle corresponding to vertex p */
static inline vec3 torus_centerline(vec3 p, float R) {
    float rxy = sqrtf(p.x*p.x + p.z*p.z);
    if (rxy < 1e-6f) {
        return (vec3){ R, 0.0f, 0.0f };
    }
    float ux = p.x / rxy, uz = p.z / rxy;
    return (vec3){ R * ux, 0.0f, R * uz };
}

object_t* object_torus(uint16_t major_segments, uint16_t minor_segments) {
    if (major_segments < 3) major_segments = 3;
    if (minor_segments < 3) minor_segments = 3;
    if (major_segments > 16) major_segments = 16;  /* practical limits */
    if (minor_segments > 16) minor_segments = 16;

    uint16_t num_vertices = (uint16_t)(major_segments * minor_segments);
    uint16_t num_edges = (uint16_t)(major_segments * minor_segments * 2);  /* both directions */
    uint16_t num_faces = (uint16_t)(major_segments * minor_segments * 2);  /* 2 triangles per quad */
    
    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    float major_radius = 1.0f;
    float minor_radius = 0.3f;

    /* fill vertices */
    vec3 *v = obj->verticies->list;
    for (uint16_t i = 0; i < major_segments; ++i) {
        float major_angle = 2.0f * (float)M_PI * (float)i / (float)major_segments;
        float major_x = cosf(major_angle);
        float major_z = sinf(major_angle);
        
        for (uint16_t j = 0; j < minor_segments; ++j) {
            float minor_angle = 2.0f * (float)M_PI * (float)j / (float)minor_segments;
            float minor_radius_offset = minor_radius * cosf(minor_angle);
            float y = minor_radius * sinf(minor_angle);
            
            uint16_t idx = (uint16_t)(i * minor_segments + j);
            v[idx] = (vec3){
                (major_radius + minor_radius_offset) * major_x,
                y,
                (major_radius + minor_radius_offset) * major_z
            };
        }
    }

    /* fill edges */
    vec2 *e = obj->edges->list;
    uint16_t edge_idx = 0;
    
    for (uint16_t i = 0; i < major_segments; ++i) {
        for (uint16_t j = 0; j < minor_segments; ++j) {
            uint16_t current = (uint16_t)(i * minor_segments + j);
            uint16_t next_major = (uint16_t)(((i + 1) % major_segments) * minor_segments + j);
            uint16_t next_minor = (uint16_t)(i * minor_segments + ((j + 1) % minor_segments));
            
            /* edge along major direction */
            e[edge_idx++] = (vec2){current, next_major};
            /* edge along minor direction */
            e[edge_idx++] = (vec2){current, next_minor};
        }
    }

    /* set edge colors to white */
    for (uint16_t i = 0; i < obj->edges->length; ++i) obj->edge_colors->list[i] = (RGB){255,255,255};

    /* fill faces: two triangles per quad on (major,minor) grid */
    vec3 *F = obj->faces->list;
    uint16_t fcount = 0;
    for (uint16_t i = 0; i < major_segments; ++i) {
        uint16_t inext = (uint16_t)((i + 1) % major_segments);
        for (uint16_t j = 0; j < minor_segments; ++j) {
            uint16_t jnext = (uint16_t)((j + 1) % minor_segments);
            uint16_t i0 = (uint16_t)(i * minor_segments + j);
            uint16_t i1 = (uint16_t)(inext * minor_segments + j);
            uint16_t i2 = (uint16_t)(i * minor_segments + jnext);
            uint16_t i3 = (uint16_t)(inext * minor_segments + jnext);
            /* initial guess for winding; will enforce outward below */
            F[fcount++] = (vec3){ (float)i0, (float)i1, (float)i2 };
            F[fcount++] = (vec3){ (float)i1, (float)i3, (float)i2 };
        }
    }

    /* enforce outward CCW using torus tube outward direction and compute face normals */
    if (obj->normals && obj->verticies) {
        vec3 *V = obj->verticies->list;
        for (uint16_t i = 0; i < fcount && i < obj->normals->length; ++i) {
            uint16_t ia = (uint16_t)F[i].x;
            uint16_t ib = (uint16_t)F[i].y;
            uint16_t ic = (uint16_t)F[i].z;
            vec3 a = V[ia], b = V[ib], c = V[ic];
            vec3 e1v = vec3_sub(b, a);
            vec3 e2v = vec3_sub(c, a);
            vec3 nn = vec3_cross(e1v, e2v);
            /* Approximate outward direction for torus: average (p - C(u)) where
               C(u) is the centerline point on the major circle at the same angle u. */
            vec3 ca = torus_centerline(a, major_radius);
            vec3 cb = torus_centerline(b, major_radius);
            vec3 cc = torus_centerline(c, major_radius);
            vec3 oa = vec3_sub(a, ca);
            vec3 ob = vec3_sub(b, cb);
            vec3 oc = vec3_sub(c, cc);
            vec3 oavg = (vec3){ (oa.x+ob.x+oc.x)/3.0f, (oa.y+ob.y+oc.y)/3.0f, (oa.z+ob.z+oc.z)/3.0f };
            float d = vec3_dot(nn, oavg);
            if (d < 0.0f) {
                /* flip winding */
                float tmp = F[i].y; F[i].y = F[i].z; F[i].z = tmp;
                ib = (uint16_t)F[i].y; ic = (uint16_t)F[i].z;
                b = V[ib]; c = V[ic];
                e1v = vec3_sub(b, a);
                e2v = vec3_sub(c, a);
                nn = vec3_cross(e1v, e2v);
            }
            /* normalize */
            float len = sqrtf(nn.x*nn.x + nn.y*nn.y + nn.z*nn.z);
            if (len > 1e-6f) { nn.x/=len; nn.y/=len; nn.z/=len; }
            obj->normals->list[i] = nn;
        }
    }

    obj->faces->length = fcount;

    return obj;
}

/** 
 * @brief Create a plane object (grid) centered at origin
 * @param width_segments Number of segments along X axis
 * @param height_segments Number of segments along Z axis
 */
object_t* object_plane(uint16_t width_segments, uint16_t height_segments) {
    if (width_segments < 1) width_segments = 1;
    if (height_segments < 1) height_segments = 1;
    
    uint16_t num_vertices = (uint16_t)((width_segments + 1) * (height_segments + 1));
    uint16_t num_edges = (uint16_t)(width_segments * (height_segments + 1) + height_segments * (width_segments + 1));
    uint16_t num_faces = (uint16_t)(width_segments * height_segments * 2);  /* 2 triangles per quad */
    
    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    /* fill vertices */
    vec3 *v = obj->verticies->list;
    for (uint16_t z = 0; z <= height_segments; ++z) {
        for (uint16_t x = 0; x <= width_segments; ++x) {
            uint16_t idx = (uint16_t)(z * (width_segments + 1) + x);
            v[idx] = (vec3){
                2.0f * (float)x / (float)width_segments - 1.0f,   /* -1 to +1 */
                0.0f,                                             /* y = 0 (flat) */
                2.0f * (float)z / (float)height_segments - 1.0f   /* -1 to +1 */
            };
        }
    }

    /* fill edges */
    vec2 *e = obj->edges->list;
    uint16_t edge_idx = 0;
    
    /* horizontal edges */
    for (uint16_t z = 0; z <= height_segments; ++z) {
        for (uint16_t x = 0; x < width_segments; ++x) {
            uint16_t current = (uint16_t)(z * (width_segments + 1) + x);
            uint16_t next = current + 1;
            e[edge_idx++] = (vec2){current, next};
        }
    }
    
    /* vertical edges */
    for (uint16_t z = 0; z < height_segments; ++z) {
        for (uint16_t x = 0; x <= width_segments; ++x) {
            uint16_t current = (uint16_t)(z * (width_segments + 1) + x);
            uint16_t below = (uint16_t)((z + 1) * (width_segments + 1) + x);
            e[edge_idx++] = (vec2){current, below};
        }
    }

    /* fill faces (two triangles per quad), ensure CCW when viewed from +Y */
    vec3 *F = obj->faces->list;
    uint16_t fcount = 0;
    for (uint16_t z = 0; z < height_segments; ++z) {
        for (uint16_t x = 0; x < width_segments; ++x) {
            uint16_t i00 = (uint16_t)(z * (width_segments + 1) + x);
            uint16_t i10 = (uint16_t)(z * (width_segments + 1) + (x + 1));
            uint16_t i01 = (uint16_t)((z + 1) * (width_segments + 1) + x);
            uint16_t i11 = (uint16_t)((z + 1) * (width_segments + 1) + (x + 1));
            /* Winding for +Y normals: (i00,i11,i10) and (i00,i01,i11) */
            F[fcount++] = (vec3){ (float)i00, (float)i11, (float)i10 };
            F[fcount++] = (vec3){ (float)i00, (float)i01, (float)i11 };
        }
    }

    /* edge colors white */
    for (uint16_t i = 0; i < obj->edges->length; ++i) obj->edge_colors->list[i] = (RGB){255,255,255};

    /* compute face normals; if normal.y < 0, flip triangle to face +Y */
    if (obj->normals && obj->verticies) {
        vec3 *V = obj->verticies->list;
        for (uint16_t i = 0; i < fcount && i < obj->normals->length; ++i) {
            uint16_t i0 = (uint16_t)F[i].x;
            uint16_t i1 = (uint16_t)F[i].y;
            uint16_t i2 = (uint16_t)F[i].z;
            vec3 v0 = V[i0], v1 = V[i1], v2 = V[i2];
            vec3 e1v = vec3_sub(v1, v0);
            vec3 e2v = vec3_sub(v2, v0);
            vec3 nn = vec3_cross(e1v, e2v);
            if (nn.y < 0.0f) {
                /* flip to make normal point upward */
                float tmp = F[i].y; F[i].y = F[i].z; F[i].z = tmp;
                i1 = (uint16_t)F[i].y; i2 = (uint16_t)F[i].z;
                v1 = V[i1]; v2 = V[i2];
                e1v = vec3_sub(v1, v0);
                e2v = vec3_sub(v2, v0);
                nn = vec3_cross(e1v, e2v);
            }
            /* normalize */
            float len = sqrtf(nn.x*nn.x + nn.y*nn.y + nn.z*nn.z);
            if (len > 1e-6f) { nn.x/=len; nn.y/=len; nn.z/=len; }
            obj->normals->list[i] = nn;
        }
    }

    obj->faces->length = fcount;

    return obj;
}


/* =========================
   World → clip pipeline
   ========================= */

void transform_mesh_to_ndc(const vec3 *in_vertices, const size_t n,
                                  const mat4 mvp, vec3 *out_ndc) {
    for (size_t i = 0; i < n; ++i) {
        out_ndc[i] = mat4_mul_point(mvp, in_vertices[i]); /* perspective divide inside */
    }
}

mat4 camera_project(const camera_t *cam, const transform_t *obj_xform) {
    mat4 model = transform_model_matrix(obj_xform);
    mat4 view  = camera_view_matrix(cam);
    mat4 proj  = camera_proj_matrix(cam);
    mat4 mv    = mat4_mul(view, model);
    mat4 mvp   = mat4_mul(proj, mv);
    return mvp;
}

/* =========================
   Exported normal utilities
   ========================= */

mat4 model_matrix(const transform_t *t) {
    return transform_model_matrix(t);
}

/* Extract upper-left 3x3 from a 4x4 (column-major) into 3x3 (column-major) */
static inline void mat3_from_mat4(const mat4 m, float out[9]) {
    out[0] = m.m[0];  out[1] = m.m[1];  out[2] = m.m[2];
    out[3] = m.m[4];  out[4] = m.m[5];  out[5] = m.m[6];
    out[6] = m.m[8];  out[7] = m.m[9];  out[8] = m.m[10];
}

/* Compute inverse of a 3x3 column-major matrix. Returns false if singular. */
static bool mat3_inverse(const float a[9], float inv_out[9]) {
    /* Map to row-major names for readability */
    float m00 = a[0], m01 = a[3], m02 = a[6];
    float m10 = a[1], m11 = a[4], m12 = a[7];
    float m20 = a[2], m21 = a[5], m22 = a[8];

    float c00 =  (m11*m22 - m12*m21);
    float c01 = -(m10*m22 - m12*m20);
    float c02 =  (m10*m21 - m11*m20);
    float c10 = -(m01*m22 - m02*m21);
    float c11 =  (m00*m22 - m02*m20);
    float c12 = -(m00*m21 - m01*m20);
    float c20 =  (m01*m12 - m02*m11);
    float c21 = -(m00*m12 - m02*m10);
    float c22 =  (m00*m11 - m01*m10);

    float det = m00*c00 + m01*c01 + m02*c02;
    if (fabsf(det) < 1e-8f) return false;
    float inv_det = 1.0f / det;

    /* inverse = (1/det) * adjugate = (1/det) * transpose(C) */
    /* store back in column-major */
    inv_out[0] = c00 * inv_det;  inv_out[1] = c10 * inv_det;  inv_out[2] = c20 * inv_det;
    inv_out[3] = c01 * inv_det;  inv_out[4] = c11 * inv_det;  inv_out[5] = c21 * inv_det;
    inv_out[6] = c02 * inv_det;  inv_out[7] = c12 * inv_det;  inv_out[8] = c22 * inv_det;
    return true;
}

static inline void mat3_transpose(const float in[9], float out[9]) {
    out[0] = in[0]; out[1] = in[3]; out[2] = in[6];
    out[3] = in[1]; out[4] = in[4]; out[5] = in[7];
    out[6] = in[2]; out[7] = in[5]; out[8] = in[8];
}

void normal_matrix_from_model(const mat4 model, float out3x3[9]) {
    float m3[9];
    mat3_from_mat4(model, m3);
    float inv3[9];
    if (!mat3_inverse(m3, inv3)) {
        /* fallback to identity */
        out3x3[0]=1; out3x3[1]=0; out3x3[2]=0;
        out3x3[3]=0; out3x3[4]=1; out3x3[5]=0;
        out3x3[6]=0; out3x3[7]=0; out3x3[8]=1;
        return;
    }
    mat3_transpose(inv3, out3x3);
}

vec3 mat3_mul_vec3(const float M[9], vec3 v) {
    return (vec3){
        M[0]*v.x + M[3]*v.y + M[6]*v.z,
        M[1]*v.x + M[4]*v.y + M[7]*v.z,
        M[2]*v.x + M[5]*v.y + M[8]*v.z
    };
}

void object_free(object_t *obj) {
    if (!obj) return;
    if (obj->verticies) free(obj->verticies);
    if (obj->rendered_vertices) free(obj->rendered_vertices);
    if (obj->edge_colors) free(obj->edge_colors);
    if (obj->edges) free(obj->edges);
    if (obj->faces) free(obj->faces);
    if (obj->normals) free(obj->normals);
    if (obj->trifill_buffer) free(obj->trifill_buffer);
    free(obj);
}
