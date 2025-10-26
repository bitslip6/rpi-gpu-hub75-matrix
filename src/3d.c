
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
    mat4 r = {0};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            r.m[row*4 + col] =
                a.m[row*4 + 0] * b.m[0*4 + col] +
                a.m[row*4 + 1] * b.m[1*4 + col] +
                a.m[row*4 + 2] * b.m[2*4 + col] +
                a.m[row*4 + 3] * b.m[3*4 + col];
        }
    }
    return r;
}

static inline vec3 mat4_mul_point(const mat4 m, const vec3 p) {
    float x = m.m[0]*p.x + m.m[1]*p.y + m.m[2]*p.z + m.m[3];
    float y = m.m[4]*p.x + m.m[5]*p.y + m.m[6]*p.z + m.m[7];
    float z = m.m[8]*p.x + m.m[9]*p.y + m.m[10]*p.z + m.m[11];
    float w = m.m[12]*p.x + m.m[13]*p.y + m.m[14]*p.z + m.m[15];
    if (w != 0.0f) { x /= w; y /= w; z /= w; }
    return (vec3){x, y, z};
}

/* =========================
   Transform builders
   ========================= */

static inline mat4 mat4_translate(const vec3 t) {
    mat4 r = mat4_identity();
    r.m[3]  = t.x;
    r.m[7]  = t.y;
    r.m[11] = t.z;
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
    float f = 1.0f / tanf(fovy_radians * 0.5f);
    mat4 r = {0};
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = (2.0f * zfar * znear) / (znear - zfar);
    r.m[14] = -1.0f;
    return r;
}

/* view matrix using eye, target, up, right-handed */
static inline mat4 mat4_look_at(const vec3 eye, const vec3 target, const vec3 up) {
    vec3 f = vec3_norm(vec3_sub(target, eye)); /* forward */
    vec3 s = vec3_norm(vec3_cross(f, up));     /* right */
    vec3 u = vec3_cross(s, f);                 /* true up */

    /* row-major view */
    mat4 r = { .m = {
        s.x, u.x, -f.x, 0.0f,
        s.y, u.y, -f.y, 0.0f,
        s.z, u.z, -f.z, 0.0f,
        -vec3_dot(s, eye), -vec3_dot(u, eye), vec3_dot(f, eye), 1.0f
    }};
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
    return vlist;
}

edge_list_t* edge_list_new(const uint16_t num_edges) {
    size_t total = sizeof(edge_list_t) + sizeof(vec2) * num_edges;
    edge_list_t *elist = calloc(1, total);
    if (!elist) return NULL;
    elist->length = num_edges;
    return elist;
}

color_list_t* color_list_new(const uint16_t length) {
    size_t total = sizeof(edge_list_t) + sizeof(RGB) * length;
    color_list_t *clist = calloc(1, total);
    if (!clist) return NULL;
    clist->length = length;
    return clist;
}

face_list_t* face_list_new(const uint16_t num_faces) {
    size_t total = sizeof(face_list_t) + sizeof(vec3) * num_faces;
    face_list_t *flist = calloc(1, total);
    if (!flist) return NULL;
    flist->length = num_faces;
    return flist;
}

normal_list_t* normal_list_new(const uint16_t num_normals) {
    size_t total = sizeof(normal_list_t) + sizeof(vec3) * num_normals;
    normal_list_t *nlist = calloc(1, total);
    if (!nlist) return NULL;
    nlist->length = num_normals;
    return nlist;
}

object_t* object_new(const uint16_t num_vertices, const uint16_t num_edges, const uint16_t num_faces) {
    size_t total = sizeof(object_t);
    total += sizeof(vec3) * num_vertices; /* space for rendered vertices */

    object_t *obj = calloc(1, total);
    if (!obj) return NULL;
    obj->verticies = vert_list_new(num_vertices);
    obj->rendered_vertices = calloc(num_vertices, sizeof(vec3));
    obj->edge_colors = color_list_new(num_vertices);
    obj->edges = edge_list_new(num_edges);
    obj->faces = face_list_new(num_faces);
    obj->normals = normal_list_new(num_faces);
    if (!obj->edges || !obj->faces || !obj->normals) {
        return NULL;
    }

    return obj;
}


/** 
 * @brief Create a unit cube object centered at origin
 */
object_t* object_cube(void) {
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
    /* front face (z=-1) */  f[0] = (vec3){0,1,2};  f[1] = (vec3){0,2,3};
    /* back face (z=+1) */   f[2] = (vec3){5,4,7};  f[3] = (vec3){5,7,6};
    /* left face (x=-1) */   f[4] = (vec3){4,0,3};  f[5] = (vec3){4,3,7};
    /* right face (x=+1) */  f[6] = (vec3){1,5,6};  f[7] = (vec3){1,6,2};
    /* bottom face (y=-1) */ f[8] = (vec3){4,5,1};  f[9] = (vec3){4,1,0};
    /* top face (y=+1) */    f[10]= (vec3){3,2,6};  f[11]= (vec3){3,6,7};

    /* fill face normals */
    vec3 *n = obj->normals->list;
    n[0] = n[1] = (vec3){0,0,-1};   /* front face */
    n[2] = n[3] = (vec3){0,0,+1};   /* back face */
    n[4] = n[5] = (vec3){-1,0,0};   /* left face */
    n[6] = n[7] = (vec3){+1,0,0};   /* right face */
    n[8] = n[9] = (vec3){0,-1,0};   /* bottom face */
    n[10]= n[11]= (vec3){0,+1,0};   /* top face */

    return obj;
}

/** 
 * @brief Create a tetrahedron (triangular pyramid) object centered at origin
 */
object_t* object_tetrahedron(void) {
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

    /* calculate face normals */
    vec3 *n = obj->normals->list;
    for (int i = 0; i < 4; i++) {
        vec3 v1 = vec3_sub(v[(int)f[i].y], v[(int)f[i].x]);
        vec3 v2 = vec3_sub(v[(int)f[i].z], v[(int)f[i].x]);
        n[i] = vec3_norm(vec3_cross(v1, v2));
    }

    return obj;
}

/** 
 * @brief Create an octahedron object centered at origin
 */
object_t* object_octahedron(void) {
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
object_t* object_cylinder(uint16_t segments) {
    if (segments < 3) segments = 3;
    if (segments > 32) segments = 32;  // practical limit for wireframe 

    uint16_t num_vertices = segments * 2;  // top and bottom circles 
    uint16_t num_edges = segments * 3;     // top circle + bottom circle + vertical lines 
    uint16_t num_faces = segments * 4;     // top/bottom caps + side triangles
    
    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    // fill vertices 
    vec3 *v = obj->verticies->list;
    for (uint16_t i = 0; i < segments; ++i) {
        float angle = 2.0f * (float)M_PI * (float)i / (float)segments;
        float x = cosf(angle);
        float z = sinf(angle);
        
        v[i] = (vec3){x, 1.0f, z};              // top circle
        v[i + segments] = (vec3){x, -1.0f, z};  // bottom circle 
    }

    // fill edges
    vec2 *e = obj->edges->list;
    uint16_t edge_idx = 0;
    
    // top circle edges 
    for (uint16_t i = 0; i < segments; ++i) {
        e[edge_idx++] = (vec2){i, (i + 1) % segments};
    }
    
    // bottom circle edges 
    for (uint16_t i = 0; i < segments; ++i) {
        uint16_t bottom_i = i + segments;
        uint16_t bottom_next = ((i + 1) % segments) + segments;
        e[edge_idx++] = (vec2){bottom_i, bottom_next};
    }
    
    // vertical edges connecting top to bottom 
    for (uint16_t i = 0; i < segments; ++i) {
        e[edge_idx++] = (vec2){i, i + segments};
    }

    return obj;
}

/* Helper function to add vertex if not exists, return index */
static uint16_t sphere_add_vertex(vec3 **vertices, uint16_t *vertex_count, uint16_t *capacity, vec3 vertex) {
    /* Normalize vertex to unit sphere */
    vertex = vec3_norm(vertex);
    
    /* Check if vertex already exists (within tolerance) */
    const float tolerance = 1e-6f;
    for (uint16_t i = 0; i < *vertex_count; i++) {
        vec3 diff = vec3_sub((*vertices)[i], vertex);
        if (vec3_dot(diff, diff) < tolerance * tolerance) {
            return i;  /* Found existing vertex */
        }
    }
    
    /* Add new vertex */
    if (*vertex_count >= *capacity) {
        /* Should not happen with pre-calculated capacity */
        return *vertex_count;
    }
    
    (*vertices)[*vertex_count] = vertex;
    return (*vertex_count)++;
}

/* Helper function to get midpoint between two vertices */
static vec3 sphere_get_midpoint(vec3 a, vec3 b) {
    vec3 mid = vec3_add(a, vec3_scale(vec3_sub(b, a), 0.5f));
    return vec3_norm(mid);  /* Project to unit sphere */
}

/** 
 * @brief Create a sphere object (geodesic approximation with icosphere)
 * @param subdivisions Number of subdivision levels (0-4 recommended)
 */
object_t* object_sphere(uint16_t subdivisions) {
    /* Limit subdivisions for memory and performance reasons */
    if (subdivisions > 4) subdivisions = 4;
    
    /* Calculate number of vertices and edges after subdivision */
    uint16_t base_vertices = 12;
    uint16_t base_edges = 30;
    
    /* Each subdivision level roughly quadruples the triangle count */
    /* For icosphere: vertices ≈ 10 * 4^level + 2, edges ≈ 30 * 4^level */
    uint16_t num_vertices = base_vertices;
    uint16_t num_edges = base_edges;
    
    for (uint16_t i = 0; i < subdivisions; i++) {
        /* Each edge split adds one new vertex, each triangle split adds 3 new edges per original edge */
        num_vertices += num_edges;  /* One new vertex per edge split */
        num_edges *= 4;             /* Each edge becomes 4 edges after subdivision */
    }
    
    uint16_t num_faces = 20; /* icosahedron has 20 triangular faces */
    for (uint16_t i = 0; i < subdivisions; i++) {
        num_faces *= 4;  /* Each subdivision quadruples triangle count */
    }
    
    /* Create object with calculated capacity */
    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    /* Golden ratio for icosahedron construction */
    float phi = (1.0f + sqrtf(5.0f)) / 2.0f;  /* golden ratio */
    float inv_len = 1.0f / sqrtf(1.0f + phi * phi);
    
    /* Start with icosahedron vertices */
    vec3 icosahedron_vertices[12] = {
        { inv_len,  phi * inv_len,  0},
        {-inv_len,  phi * inv_len,  0},
        { inv_len, -phi * inv_len,  0},
        {-inv_len, -phi * inv_len,  0},
        { 0,  inv_len,  phi * inv_len},
        { 0, -inv_len,  phi * inv_len},
        { 0,  inv_len, -phi * inv_len},
        { 0, -inv_len, -phi * inv_len},
        { phi * inv_len,  0,  inv_len},
        {-phi * inv_len,  0,  inv_len},
        { phi * inv_len,  0, -inv_len},
        {-phi * inv_len,  0, -inv_len}
    };
    
    /* Icosahedron edges (as vertex index pairs) */
    uint16_t icosahedron_edges[30][2] = {
        {0,1},   {0,4},   {0,6},   {0,8},   {0,10},
        {1,4},   {1,6},   {1,9},   {1,11},  {2,3},
        {2,5},   {2,7},   {2,8},   {2,10},  {3,5},
        {3,7},   {3,9},   {3,11},  {4,5},   {4,8},
        {4,9},   {5,8},   {5,9},   {6,7},   {6,10},
        {6,11},  {7,10},  {7,11},  {8,10},  {9,11}
    };
    
    /* Initialize with base icosahedron */
    vec3 *vertices = obj->verticies->list;
    vec2 *edges = obj->edges->list;
    
    uint16_t vertex_count = 0;
    uint16_t edge_count = 0;
    uint16_t vertex_capacity = num_vertices;
    
    /* Add initial vertices */
    for (int i = 0; i < 12; i++) {
        sphere_add_vertex(&vertices, &vertex_count, &vertex_capacity, icosahedron_vertices[i]);
    }
    
    /* Add initial edges */
    for (int i = 0; i < 30; i++) {
        edges[edge_count++] = (vec2){icosahedron_edges[i][0], icosahedron_edges[i][1]};
    }
    
    /* Perform subdivisions */
    for (uint16_t level = 0; level < subdivisions; level++) {
        uint16_t old_edge_count = edge_count;
        vec2 *old_edges = malloc(old_edge_count * sizeof(vec2));
        if (!old_edges) break;  /* Out of memory, return what we have */
        
        /* Copy current edges */
        for (uint16_t i = 0; i < old_edge_count; i++) {
            old_edges[i] = edges[i];
        }
        
        /* Reset edge count for new subdivision */
        edge_count = 0;
        
        /* Subdivide each edge */
        for (uint16_t i = 0; i < old_edge_count; i++) {
            uint16_t v1 = (uint16_t)old_edges[i].x;
            uint16_t v2 = (uint16_t)old_edges[i].y;
            
            /* Get midpoint vertex */
            vec3 midpoint = sphere_get_midpoint(vertices[v1], vertices[v2]);
            uint16_t v_mid = sphere_add_vertex(&vertices, &vertex_count, &vertex_capacity, midpoint);
            
            /* Create two new edges from the split */
            if (edge_count < num_edges) {
                edges[edge_count++] = (vec2){v1, v_mid};
            }
            if (edge_count < num_edges) {
                edges[edge_count++] = (vec2){v_mid, v2};
            }
        }
        
        /* For a proper geodesic sphere, we should also add edges between midpoints of triangle edges
         * This is a simplified version that creates a more spherical shape by subdivision */
        
        free(old_edges);
    }
    
    /* Update actual counts in the object */
    obj->verticies->length = vertex_count;
    obj->edges->length = edge_count;
    
    return obj;
}

/** 
 * @brief Create a torus object centered at origin
 * @param major_segments Number of segments around the major radius (minimum 3)
 * @param minor_segments Number of segments around the minor radius (minimum 3)
 */
object_t* object_torus(uint16_t major_segments, uint16_t minor_segments) {
    if (major_segments < 3) major_segments = 3;
    if (minor_segments < 3) minor_segments = 3;
    if (major_segments > 16) major_segments = 16;  /* practical limits */
    if (minor_segments > 16) minor_segments = 16;

    uint16_t num_vertices = major_segments * minor_segments;
    uint16_t num_edges = major_segments * minor_segments * 2;  /* both directions */
    uint16_t num_faces = major_segments * minor_segments * 2;  /* 2 triangles per quad */
    
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
            
            uint16_t idx = i * minor_segments + j;
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
            uint16_t current = i * minor_segments + j;
            uint16_t next_major = ((i + 1) % major_segments) * minor_segments + j;
            uint16_t next_minor = i * minor_segments + ((j + 1) % minor_segments);
            
            /* edge along major direction */
            e[edge_idx++] = (vec2){current, next_major};
            /* edge along minor direction */
            e[edge_idx++] = (vec2){current, next_minor};
        }
    }

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
    
    uint16_t num_vertices = (width_segments + 1) * (height_segments + 1);
    uint16_t num_edges = width_segments * (height_segments + 1) + height_segments * (width_segments + 1);
    uint16_t num_faces = width_segments * height_segments * 2;  /* 2 triangles per quad */
    
    object_t *obj = object_new(num_vertices, num_edges, num_faces);
    if (!obj) return NULL;

    /* fill vertices */
    vec3 *v = obj->verticies->list;
    for (uint16_t z = 0; z <= height_segments; ++z) {
        for (uint16_t x = 0; x <= width_segments; ++x) {
            uint16_t idx = z * (width_segments + 1) + x;
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
            uint16_t current = z * (width_segments + 1) + x;
            uint16_t next = current + 1;
            e[edge_idx++] = (vec2){current, next};
        }
    }
    
    /* vertical edges */
    for (uint16_t z = 0; z < height_segments; ++z) {
        for (uint16_t x = 0; x <= width_segments; ++x) {
            uint16_t current = z * (uint16_t)((width_segments + 1) + x);
            uint16_t below = (uint16_t)((z + 1) * (width_segments + 1) + x);
            e[edge_idx++] = (vec2){current, below};
        }
    }

    return obj;
}


/* =========================
   World → clip pipeline
   ========================= */

void transform_mesh_to_ndc(const vec3 *in_vertices, size_t n,
                                  mat4 mvp, vec3 *out_ndc) {
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
