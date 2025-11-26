#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

 #include "spsc.h"

#ifndef __HUB75GPU_H__
#define __HUB75GPU_H__

#define MAX_BITS 64
#define MAX_PANEL_TYPES 8
#define MAX_PANELS     24
#define FPS_MAX 200
#define FPS_MIN 1

#ifndef MAX_POLY_POINTS
#define MAX_POLY_POINTS 32
#endif



#ifdef ADA_HAT
    #define ADDRESS_TYPE "ADAFRUIT_HAT"
#else
    #ifdef ADA_3HAT
        #define ADDRESS_TYPE "ADA_3HAT"
    #else
        #define ADDRESS_TYPE "HZELLER_HAT"
    #endif
#endif

/**
 * @brief just a float, should be normalized to 0-1
 */
typedef float Normal;
typedef float NormalSigned;

#define NormalSAT 1.0f;
#define NormalZERO 0.0f;
#define NormalSignedZERO 0.0f;
#define NormalSignedSAT 1.0f;
#define NormalSignedNEGSAT -1.0f;

inline Normal Normal_clamp(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return (Normal)x;
}

inline NormalSigned NormalSigned_clamp(float x) {
    if (x < -1.0f) return -1.0f;
    if (x > 1.0f) return 1.0f;
    return (NormalSigned)x;
}



/**
 * @brief pointer to a single 24bpp RGB pixel (3 bytes)
 * RGB *pixel = (RGB *)(image + offset)
 */
typedef struct {
    uint8_t r;
    uint8_t g; 
    uint8_t b; 
} RGB;

/**
 * @brief pointer to a single 24bpp RGBA pixel (4 bytes)
 * RGBA *pixel = (RGBA *)(image + offset)
 */
typedef struct {
    uint8_t r;
    uint8_t g; 
    uint8_t b; 
    uint8_t a; 
} RGBA;


/**
 * @brief pointer to a single 24bpp RGB pixel normalized as floats (0-1)
 * RGBF *pixel_norm = normalize_rgb((RGB *)(image + offset))
 */
typedef struct {
    Normal r;
    Normal g; 
    Normal b; 
} RGBF;

/**
 * @brief pointer to a single 24bpp RGB pixel normalized as floats (0-1)
 * RGBF *pixel_norm = normalize_rgb((RGB *)(image + offset))
 */
typedef struct {
    Normal h;
    Normal s; 
    Normal l; 
} HSLF;

// int32 vec2
typedef struct { int32_t x, y; } vec2u;
// int32 vec3
typedef struct { int32_t x, y, z; } vec3u;
// int32 vec4
typedef struct { int32_t x, y, z, w; } vec4u;

// float vec2
typedef struct { float x, y; } vec2;
// float vec3
typedef struct { float x, y, z; } vec3;
// float vec4
typedef struct { float x, y, z, w; } vec4;
typedef struct { float m[16]; }   mat4;  // column-major, m[col*4 + row]

typedef struct {
    vec3 position;     // world position 
    vec3 rotation;     // Euler angles in radians, x=pitch, y=yaw, z=roll 
    vec3 scale;        // per-axis scale 
} transform_t;



typedef struct
{
    Normal x;
    Normal y;

} Pointf_t;

typedef struct
{
    Pointf_t points[MAX_POLY_POINTS];
    size_t num_points;

} Polygonf_t;

typedef enum { POLY_DEGENERATE = 0, POLY_CW = 1, POLY_CCW = 2 } poly_winding_t;

typedef struct panel_rgb_scale {
    uint8_t red_q8;
    uint8_t green_q8;
    uint8_t blue_q8;
} panel_rgb_scale;

typedef struct panel_rgb_offset {
    int8_t red_q8;
    int8_t green_q8;
    int8_t blue_q8;
} panel_rgb_offset;

typedef struct image_buffer_t {
    vec2u dimensions;
    uint32_t row_stride;
    RGBA *data;
} image_buffer_t;


/**
 * @brief Gradient direction for simple gradients
 */
typedef enum {
    GRADIENT_HORIZONTAL,    /**< Left to right */
    GRADIENT_VERTICAL,      /**< Top to bottom */
    GRADIENT_DIAGONAL,      /**< Top-left to bottom-right */
    GRADIENT_RADIAL         /**< From center outward */
} gradient_direction_t;

/**
 * @brief Easing functions for smooth gradient transitions
 */
typedef enum {
    EASE_LINEAR,            /**< Linear interpolation */
    EASE_IN_QUAD,          /**< Quadratic ease-in */
    EASE_OUT_QUAD,         /**< Quadratic ease-out */
    EASE_IN_OUT_QUAD       /**< Quadratic ease-in-out */
} easing_function_t;

/**
 * @brief Simple two-color gradient definition
 * 
 * A simplified gradient system that takes two colors, a direction,
 * and an easing function for smooth color transitions.
 */
typedef struct {
    RGB start_color;                /**< Starting color of the gradient */
    RGB end_color;                  /**< Ending color of the gradient */
    gradient_direction_t direction; /**< Direction of the gradient */
    easing_function_t easing;       /**< Easing function for transitions */
} SimpleGradient;


// panel order describes which logical color drives panel wires R,G,B respectively
typedef enum {
    PANEL_RGB = 0, PANEL_RBG, PANEL_GRB, PANEL_GBR, PANEL_BRG, PANEL_BGR
} panel_order_t;


// self referencing function pointers need this defined first
struct hub75_display;

// void map_byte_image_to_pwm(uint8_t *image, const scene_info *scene, uint8_t fps_sync) {
typedef void (*func_tone_mapper_t)(const RGBF *in, RGBF *out, const float level);
typedef uint8_t *(*func_image_mapper_t)(const uint8_t *image_in, uint8_t *image_out, const struct hub75_display *scene);
typedef uint8_t *(image_mapper_t)(const uint8_t *image_in, uint8_t *image_out, const struct hub75_display *scene);




/**
 * @brief everything to define the scene and panel configuration 
 * This is a kind of global configuration for the entire system 
 */
typedef struct hub75_display {
    /** @brief the total width of the image in pixels */
    uint16_t width;
    /** @brief the total height of the image in pixels */
    uint16_t height;
    /** @brief the number of bytes per pixel in the drawing buffers (3 for RGB, 4 for RGBA) */
    uint8_t  stride;

    /** @brief used by the image mappers and the renderer to support mapping of images to single ports */
    uint16_t render_width;
    uint16_t render_height;

    /** the RGB color order to render pixels out to the panels */
    panel_order_t panel_order;
    
    /** @brief single panel width in pixels */
    uint16_t panel_width;

    /** @brief single panel height in pixels */
    uint16_t panel_height;

    /** @brief number of ports connected to the PI (1-3) */
    uint8_t num_ports;

    /** @brief number of bits per color channel (8-64) */
    uint8_t bit_depth;

    /** @brief brightness level (0-255) */
    uint8_t brightness;

    /** @brief array of red fractional brightness, panel type 0, type 1, ... */
    panel_rgb_scale  panel_scale[MAX_PANEL_TYPES];
    /** @brief array of red brightness offsets, panel type 0, type 1, ... */
    panel_rgb_offset panel_offset[MAX_PANEL_TYPES];

    /** @brief array panel types for each output */
    uint8_t panel_types[MAX_PANELS];
    /** @brief number of unique panel types for panel_scale and panel_offset */
    uint8_t num_panel_types;


    /** @brief dithering strength. (0-10) 0 is off, improves simulated color in dark areas but reduces image sharpness */
    float dither;
    /** @brief toggle flag if quant dithering should be applied */
    bool quant_dither;

    /** @brief number of panels connected to each chain on the port (1-8) */
    uint8_t num_chains;

    spsc_semring_t *ring_buf_renderer;
    spsc_semring_t *ring_buf_mapper;

    /** @brief frame buffer for CPU drawing functions (new unified structure) */
    image_buffer_t frame_buffer;

    /** @brief pointer to a single frame for CPU drawing functions
     *  @deprecated Use frame_buffer.data instead. Kept for compatibility during migration.
     */
    uint8_t *image;


    /** @brief accumulator for quantization errors */
    int32_t *accum;

    /** @brief LUT for RGB to quant error */
    uint16_t *quant_errors_lut;

    /** @brief a shader file to render on the GPU */
    char *shader_file;


	/** 
     * @brief the tone mapping function to use, if null no tone mapping applied
     * @see aces_tone_map
     */
    func_tone_mapper_t tone_mapper;

	/** 
     * @brief the tone mapping function to use, if null no tone mapping applied
     * @see aces_tone_map
     */
    func_image_mapper_t image_mapper;

    /**
     * @brief  the target frame rate:
     * maximum frame rate is: 9600 / bpp / (panel_width / 16)
     */
    uint16_t fps;
    bool auto_fps;

	/**
     * @brief gamma correction value to use for pwm scaling. if 0 - no gamma is applied
     */
	float gamma;

    /**
     * @brief optional parameter passed to the tone mapper to determine strength
     */
    float tone_level;

    bool jitter_brightness;

    uint8_t motion_blur_frames;

    float red_gamma;
    float green_gamma;
    float blue_gamma;
    Normal red_linear;
    Normal green_linear;
    Normal blue_linear;

    /**
     * @brief boolean flag to indicate that render_forever should exit.
     */
    volatile _Atomic bool do_render;

    /**
     * set to true to show the FPS on the screen
     */
    bool show_fps;

    /**
     * When true, enable extra verbose debug prints in CPU rendering paths
     * (camera, transform, per-edge/triangle diagnostics). Default: false.
     */
    bool enhanced_debug;

    /**
     * @brief current frame index, increments every frame rendered
     */
    uint32_t frame_index;

    pthread_t render_thread;
    pthread_t mapper_thread;

    bool frame_ready;

    uint16_t latch_blank_cycles;
    bool rising_edge;
    
} hub75_display_t;


/**
 * @brief this function takes the image data and maps it to the bcm signal.
 * 
 * if scene->tone_mapper is updated, new bcm bit masks will be created.
 * 
 * @param scene the scene information
 * @param image the image to map to the scene bcm data. if NULL scene->image will be used
 */
void hub75_display_map_image_to_bcm(const hub75_display_t *scene, uint8_t *image);

void *mapper_thread_main(void *arg);

/**
 * must be called on the main thread to start the renderer. it never returns
 */
void *hub75_display_run(const hub75_display_t *scene);

/**
 * @brief render the shader arg->shader_file shader on the GPU
 */
void *render_shader(void *arg);
void *render_shader_old(void *arg);
void *render_shader_minimal(void *arg);

/**
 * @brief pass this function to your pthread_create() call to render a video file
 * will render the video file pointed to by scene->shader_file until
 * scene->do_render is false;
 * 
 * @param arg 
 * @return void* 
 */
void* render_video_fn(void *arg);

/**
 * @brief pass this function to your pthread_create() call to render a video file
 * will render the video file pointed to by scene->shader_file until
 * scene->do_render is false; returns once the video is done rendering
 * 
 * @param arg 
 * @return void* 
 */
bool hub_render_video(hub75_display_t *scene, const char *filename);

/**
 * @brief count number of times this function is called, 1 every second output
 * the number of times called and reset the counter. This function can not
 * be called from multiple locations. It is not thread safe.
 * 
 * @param target_fps - target a sleep time to achieve this fps
 * @return long - returns sleep time in microseconds
 */
float calculate_fps(const uint16_t target_fps, const bool show_fps);


// graceful shutdown helpers
void hub75_display_request_shutdown(struct hub75_display *scene);
void hub75_display_wait(struct hub75_display *scene);

void draw_polygon_fill(hub75_display_t *scene, Polygonf_t *poly, RGBA color);
void gradient_polygon(hub75_display_t *scene, Polygonf_t *poly, SimpleGradient gradient);

/* Easing function implementations */
float apply_easing(float t, easing_function_t easing);


void hub_line_aa(hub75_display_t *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, const RGB color);
void hub_line(hub75_display_t *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, RGB color);

/**
 * @brief parse command line arguments and create a valid scene
 * 
 * @param argc command line argument count
 * @param argv command line arguments
 * @return scene_info* the created scene information
 */
hub75_display_t *hub75_display_parse_args(int argc, char **argv);

/**
 * @brief create a default scene
 * 
 * @return scene_info* the created scene information
 */
hub75_display_t *hub75_display_new();

/**
 * @brief allocate memory for a new image_buffer_t of requested dimensions
 * heap allocation the caller must free
 *
 */
image_buffer_t *image_buffer_new(int32_t width, int32_t height);

/**
 * check the scene and start the rendering threads if everying is ok
 */
void hub75_display_start(hub75_display_t *scene);

/**
 * @brief install signal handlers for graceful shutdown on SIGINT/SIGTERM
 * 
 */
void signal_handler_install(void);


/* --------------------------------------------------------------
 * OPTIONAL: Public function table for FFI (e.g. Python / Rust / Go)
 * --------------------------------------------------------------
 * This lightweight indirection lets foreign language bindings obtain
 * stable pointers to the drawing helpers without relying on parsing
 * multiple headers.  Versioning can be added later by extending the
 * struct (always append new fields) and bumping hub75_api.version.
 */
/*
typedef struct hub75_api {
    uint32_t version;      // struct version for compatibility 
    scene_info *scene;     // active scene (set via hub75_get_api(scene)) 

    // Scene creation helpers (do NOT use internal scene pointer) 
    scene_info *(*new_scene)(void);
    scene_info *(*parse_scene)(int argc, char **argv);

    // Lifecycle operating on api->scene (scene must be set) 
    void (*start)(void);
    void (*request_shutdown)(void);
    void (*wait_shutdown)(void);
    void (*map_image)(uint8_t *image); // NULL -> use scene->image 

    // Timing (independent of scene except show_fps flag consumed inside) 
    unsigned long (*fps_calculate)(uint16_t target_fps, bool show_fps);

    // Drawing primitives (use api->scene internally)
    void (*pixel)(int x, int y, RGB pixel);
    void (*pixel_factor)(int x, int y, RGB pixel, float factor);
    void (*pixel_alpha)(int x, int y, RGBA pixel);
    void (*fill)(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color);
    void (*line)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB color);
    void (*line_aa)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB color);
    void (*triangle)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color);
    void (*triangle_aa)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color);
    void (*circle)(uint16_t cx, uint16_t cy, uint16_t radius, RGB color);
    int (*fps_get)();
} hub75_api;
 */

// Obtain & initialize (or re-point) the singleton API to a scene. 
// const hub75_api *hub75_get_api(scene_info *scene);

// poly_winding_t polygon_winding(const Polygonf_t *poly);



/* --------------------------------------------------------------
 * Lighting types and scene lighting configuration
 * (definitions appear before scene_info usage)
 * -------------------------------------------------------------- */
typedef enum {
    LIGHT_DIRECTIONAL = 0,
    LIGHT_POINT       = 1,
    LIGHT_SPOT        = 2
} light_type_t;

/* local 3-float vector for lighting (avoid dependency on vec3 defined later) */
// typedef struct { float x, y, z; } vec3;

typedef struct light_t {
    light_type_t type;     /* light category */

    /* Common parameters */
    RGBF   color;          /* light color (0..1 per channel) */
    float  intensity;      /* scalar multiplier for brightness */
    bool   casts_shadows;  /* whether this light should cast shadows */
    bool   shadow_enabled; /* runtime toggle to enable/disable shadowing for this light */

    /* Geometric parameters (interpreted by type) */
    vec3 position;   /* for point/spot lights */
    vec3 direction;  /* for directional/spot lights */

    /* Optional falloff / cone controls (POINT/SPOT) */
    float  range;          /* effective radius for point/spot; 0 => infinite */
    float  inner_cos;      /* spot inner cone (cosine of angle); 1 => no cone */
    float  outer_cos;      /* spot outer cone (cosine of angle); must be <= inner_cos */
    /* Shadow map view/proj cache for stability */
    mat4 shadow_V, shadow_P, shadow_VP;
    float shadow_z_bias;
    bool shadow_vp_valid;
    /* Debug/quality control: zoom factor for tight-fit SM bounds (1=original size, <1 zoom-in) */
    float shadow_zoom;
} light_t;

typedef struct scene3d_lighting_t {
    RGBF     ambient;      // ambient light color (0..1 per channel)
    uint16_t num_lights;   // number of active lights
    light_t *lights;       // dynamic array of lights (NULL when num_lights == 0)

    void (*set_directional)(uint16_t index, vec3 direction,
                                RGBF color, float intensity, bool casts_shadows);
} scene3d_lighting_t;



typedef struct {
    vec3 position;     // camera position 
    vec3 target;       // look-at target 
    vec3 up;           // usually {0,1,0} 
    float fov_y;       // radians 
    float aspect;      // width / height 
    float z_near, z_far;
} camera_t;

typedef struct {
    uint16_t length;
    vec3 *list;
} vert_list_t;

typedef struct {
    uint16_t length;
    RGB *list;
} color_list_t;

typedef struct {
    uint16_t length;
    vec2 *list;
} edge_list_t;

typedef struct {
    uint16_t length;
    vec3 *list;  /* triangle indices: each vec3 contains 3 vertex indices (x,y,z) */
} face_list_t;

typedef struct {
    uint16_t length;
    vec3 *list;  /* normal vectors: one per face for flat shading */
} normal_list_t;

/* Per-object drawing mode */
typedef enum {
    DRAW_WIRE = 0,
    DRAW_FILLED = 1,
} object_draw_mode_t;

typedef struct {
    vert_list_t *verticies;
    edge_list_t *edges;
    face_list_t *faces;       /* NEW: triangle definitions */
    normal_list_t *normals;   /* NEW: face normals */
    normal_list_t *vertex_normals; /* NEW: smooth per-vertex normals */
    bool vertex_normals_ready;     /* computed once flag */
    color_list_t *edge_colors;

    vec3 *rendered_vertices;
    bool cull_backface;    /* toggle backface culling for wireframe/fill */
    object_draw_mode_t draw_mode; /* how to render this object */
    bool shadow_enabled;   /* runtime toggle to mark this object as casting/receiving shadows */

    /* Persistent buffer for filled triangle rendering */
    void *trifill_buffer; /* actually _TriFill*, but opaque here */
    size_t trifill_capacity;

    /* Simple material for specular highlights (Blinn-Phong) */
    float specular_strength;   /* scales specular contribution (0..1 typical) */
    float specular_shininess;  /* Blinn shininess exponent (e.g., 8..128) */
} object_t;

typedef struct {
    float depth;      /* average NDC z */
    Polygonf_t poly;  /* 3 points normalized to [0,1] */
    RGB vcolor[3];    /* per-vertex shaded color */
    uint16_t z16[3];  /* per-vertex depth mapped from NDC [-1,1] -> [0..65535] */
    /* Optional per-vertex debug visibility (0..255). Valid when a debug overlay uses it. */
    uint8_t debug_vis[3];
    bool debug_vis_valid;
    /* Per-pixel shadow sampling payload for one light */
    bool  per_pixel_shadow;
    uint8_t sm_light_index;
    vec4  sm_light_clip[3];  /* light clip coords per vertex (SM->VP * world) */
    float cam_w[3];          /* camera clip w per vertex (for perspective-correct interp) */
} _TriFill;



object_t* object_cube(const object_draw_mode_t mode, const bool cull_backface);
object_t* object_tetrahedron(const object_draw_mode_t mode, const bool cull_backface);
object_t* object_octahedron(const object_draw_mode_t mode, const bool cull_backface);
object_t* object_pyramid(void);
object_t* object_cylinder(const uint16_t segments, const object_draw_mode_t mode, const bool cull_backface);
object_t* object_sphere(uint16_t subdivisions);
object_t* object_torus(uint16_t major_segments, uint16_t minor_segments);
object_t* object_plane(uint16_t width_segments, uint16_t height_segments, bool face_up);
mat4 camera_project(const camera_t *cam, const transform_t *obj_xform);
void transform_mesh_to_ndc(const vec3 *in_vertices, size_t n, mat4 mvp, vec3 *out_ndc);
object_t* object_new(uint16_t num_vertices, uint16_t num_edges, uint16_t num_faces);
/* Build smooth per-vertex normals from faces (averaged and normalized) */
void object_build_vertex_normals(object_t *obj);
/* Deform a plane mesh with a time-based rolling sine wave (y displacement, along +X) */
void plane_apply_sine_wave(object_t *plane, float time_sec);

/* 3D math utilities (normals) */
/* Build model matrix (T * Rz * Ry * Rx * S) */
mat4 model_matrix(const transform_t *t);
/* Extract world-space normal matrix from model (inverse-transpose of upper-left 3x3) */
void normal_matrix_from_model(const mat4 model, float out3x3[9]);
/* Multiply a 3x3 (column-major) with a vec3 */
vec3 mat3_mul_vec3(const float M[9], vec3 v);

/* Scene of object instances */
typedef struct {
    object_t *object;             /* mesh + material/state */
    transform_t *xform;           /* object transform (external owner) */
} object_instance_t;

typedef struct scene3d_t {
    uint16_t count;               /* number of active instances */
    uint16_t capacity;            /* allocated capacity for instances */
    object_instance_t *instances; /* array of count instances (owned by scene) */

    /* Embedded lighting owned by the scene (no separate lighting object needed) */
    scene3d_lighting_t lighting;     /* ambient + dynamic array of lights */

    /* OO-style helpers (method-like function pointers for ease of use / FFI) */
    void (*set_ambient)(struct scene3d_t *os, RGBF ambient);
    uint16_t (*add_directional)(struct scene3d_t *os,
                                vec3 direction,
                                RGBF color,
                                float intensity,
                                bool casts_shadows);

    /* Convenience: set position and look_at for a directional light; computes direction */
    void (*set_directional_pose)(struct scene3d_t *os, uint16_t id,
                                 vec3 position, vec3 look_at);

    light_t *(*get_directional)(struct scene3d_t *os, uint16_t id);
    /* Object management helpers */
    uint16_t (*add_object)(struct scene3d_t *os, object_t *obj, transform_t *xform);
    object_t *(*get_object)(struct scene3d_t *os, uint16_t id);
    transform_t *(*get_transform)(struct scene3d_t *os, uint16_t id);

    /* Z-buffer owned by the scene for CPU rasterizer */
    uint16_t *zbuf;
    bool zbuffer_enabled;
    uint16_t zbuf_width;
    uint16_t zbuf_height;

    /* Debug/diagnostic visualization toggles */
    struct {
        bool overlay_checker;      /* draw a screen-space checker overlay after rendering */
        uint8_t checker_size;      /* tile size in pixels (default 8) */
        float checker_strength;    /* 0..1 blend toward checker_color (default 0.3) */
        RGB checker_color;         /* overlay color (default magenta) */

        bool overlay_shadow_vis;   /* overlay shadow visibility (lit vs shadow) */
        uint8_t shadow_vis_light;  /* which light index to visualize */
        float shadow_vis_strength; /* 0..1 blend of overlay */
        RGB shadow_vis_color_lit;  /* color tint where visible */
        RGB shadow_vis_color_shadow; /* color tint where shadowed */
    } debug;
} scene3d_t;


void object_free(object_t *obj);

typedef struct {
    void (*clear)();
    void (*pixel)(const uint16_t x, const uint16_t y, RGB c);
    void (*pixel_factor)(int x, int y, RGB pixel, float factor);
    void (*pixel_alpha)(int x, int y, RGBA pixel);
    void (*line)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c);
    void (*line_aa)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c);
    void (*poly)(Polygonf_t *poly, RGBA color1);
    void (*poly_gradient)(Polygonf_t *poly, SimpleGradient gradient);
    void (*fill_gradient)(int y, int x0, int x1, const SimpleGradient *gradient, 
                            int minx, int miny, int maxx, int maxy);
    void (*shutdown)();
    void (*frame_begin)();
    void (*frame_end)();

    mat4 (*geo_project)(camera_t *cam, transform_t *obj_xform);
    void (*render_wire)(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene3d_lighting_t *lighting);
    void (*render_filled)(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene3d_lighting_t *lighting);
    void (*render_scene3d)(const camera_t *cam, const scene3d_t *scene, const scene3d_lighting_t *lighting);

    // camera and scene lighting functions
    camera_t* (*geo_camera)();
    transform_t* (*geo_transform)();
    

    // geometry creation functions
    object_t* (*geo_object)(const uint16_t num_vertices, const uint16_t num_edges, const uint16_t num_faces);
    object_t* (*geo_cube)(const object_draw_mode_t mode, const bool cull_backface);
    object_t* (*geo_tetrahedron)(const object_draw_mode_t mode, const bool cull_backface);
    object_t* (*geo_octahedron)(const object_draw_mode_t mode, const bool cull_backface);
    object_t* (*geo_pyramid)();
    object_t* (*geo_cylinder)(const uint16_t segments, const object_draw_mode_t mode, const bool cull_backface);
    object_t* (*geo_sphere)(uint16_t subdivisions);
    object_t* (*geo_torus)(uint16_t major_segments, uint16_t minor_segments);
    object_t* (*geo_plane)(uint16_t width_segments, uint16_t height_segments, bool face_up);

    /* Convenience scene3d wrappers (avoid passing scene3d repeatedly) */
    scene3d_t* (*scene3d_new)(uint16_t count);
    void (*scene3d_set_current)(scene3d_t *os);
    void (*scene3d_clear_current)(void);
    void (*scene3d_set_ambient)(RGBF ambient);
    uint16_t (*scene3d_add_directional)(vec3 direction, RGBF color, float intensity, bool casts_shadows);
    void (*scene3d_set_directional_pose)(uint16_t id, vec3 position, vec3 look_at);
    light_t* (*scene3d_get_directional)(uint16_t id);
    uint16_t (*scene3d_add_object)(object_t *obj, transform_t *xform);
    object_t* (*scene3d_get_object)(uint16_t id);
    transform_t* (*scene3d_get_transform)(uint16_t id);

    /* Debug helpers */
    void (*scene3d_set_debug_checker)(bool enabled, uint8_t tile_px, float strength, RGB color);
    void (*scene3d_set_debug_shadow_vis)(bool enabled, uint8_t light_index, float strength, RGB lit_color, RGB shadow_color);
    /* Debug: dump a light's shadow map to a PNG file for inspection */
    void (*scene3d_dump_shadowmap_png)(uint16_t light_index, const char *filepath);
    /* Debug/quality: set an explicit zoom on the shadow map fit for a light */
    void (*scene3d_set_shadowmap_zoom)(uint16_t light_index, float zoom);

} hub75gpu_t;

/* set the current thread's scene and get an API whose functions do not take a scene */
hub75gpu_t hub75_api(hub75_display_t *s);

/* -------- Python/FFI-friendly helper API -------- */
camera_t *api_new_camera(void);
transform_t *api_new_transform(void);

scene3d_lighting_t *api_lighting_new(uint16_t num_lights, RGBF ambient);
void api_lighting_free(scene3d_lighting_t *l);
void api_lighting_set_ambient(scene3d_lighting_t *l, RGBF color);
void api_lighting_set_directional(scene3d_lighting_t *l, uint16_t index,
                                  vec3 direction,
                                  RGBF color,
                                  float intensity, bool casts_shadows);

scene3d_t *api_object_scene_new(uint16_t count);
void api_object_scene_set(scene3d_t *os, uint16_t index, object_t *obj, transform_t *xform);
void api_object_scene_free(scene3d_t *os);
void api_object_set_draw_mode(object_t *obj, object_draw_mode_t mode);
void api_render_geo(const camera_t *cam, const scene3d_t *os, const scene3d_lighting_t *lighting);

/* Object material helpers */
void api_object_set_specular(object_t *obj, float strength, float shininess);
void api_object_set_specular_strength(object_t *obj, float strength);
void api_object_set_specular_shininess(object_t *obj, float shininess);

/* Convenience scene3d wrappers (thread-local current scene3d) */
void api_scene3d_set_current(scene3d_t *os);
void api_scene3d_clear_current(void);
void api_scene3d_set_ambient(RGBF ambient);
uint16_t api_scene3d_add_directional(vec3 direction, RGBF color, float intensity, bool casts_shadows);
uint16_t api_scene3d_add_object(object_t *obj, transform_t *xform);
object_t* api_scene3d_get_object(uint16_t id);
light_t* api_scene3d_get_directional(uint16_t id);
transform_t* api_scene3d_get_transform(uint16_t id);
void api_scene3d_set_directional_pose(uint16_t id, vec3 position, vec3 look_at);

/* Debug/diagnostic helpers */
void api_scene3d_set_debug_checker(bool enabled, uint8_t tile_px, float strength, RGB color);
void api_scene3d_set_debug_shadow_vis(bool enabled, uint8_t light_index, float strength, RGB lit_color, RGB shadow_color);
/* Debug: request dumping the current frame's shadow map for a light to a PNG file */
void api_scene3d_dump_shadowmap_png(uint16_t light_index, const char *filepath);
/* Debug/quality: control zoom for tight-fit shadow map bounds per light (1=default, <1 zoom in) */
void api_scene3d_set_shadowmap_zoom(uint16_t light_index, float zoom);

/* Common web colors (RGBF normalized 0..1) */
#define COLOR_BLACK        (RGBF){ 0.0f, 0.0f, 0.0f }
#define COLOR_WHITE        (RGBF){ 1.0f, 1.0f, 1.0f }
#define COLOR_RED          (RGBF){ 1.0f, 0.0f, 0.0f }
#define COLOR_GREEN        (RGBF){ 0.0f, 0.501961f, 0.0f }       /* CSS green (0,128,0) */
#define COLOR_LIME         (RGBF){ 0.0f, 1.0f, 0.0f }           /* CSS lime (0,255,0) */
#define COLOR_BLUE         (RGBF){ 0.0f, 0.0f, 1.0f }
#define COLOR_CYAN         (RGBF){ 0.0f, 1.0f, 1.0f }
#define COLOR_MAGENTA      (RGBF){ 1.0f, 0.0f, 1.0f }
#define COLOR_YELLOW       (RGBF){ 1.0f, 1.0f, 0.0f }
#define COLOR_ORANGE       (RGBF){ 1.0f, 0.647059f, 0.0f }      /* (255,165,0) */
#define COLOR_PURPLE       (RGBF){ 0.5f, 0.0f, 0.5f }           /* (128,0,128) */
#define COLOR_PINK         (RGBF){ 1.0f, 0.752941f, 0.796078f } /* (255,192,203) */
#define COLOR_TEAL         (RGBF){ 0.0f, 0.501961f, 0.501961f } /* (0,128,128) */
#define COLOR_NAVY         (RGBF){ 0.0f, 0.0f, 0.501961f }      /* (0,0,128) */
#define COLOR_MAROON       (RGBF){ 0.501961f, 0.0f, 0.0f }      /* (128,0,0) */
#define COLOR_OLIVE        (RGBF){ 0.501961f, 0.501961f, 0.0f } /* (128,128,0) */
#define COLOR_SILVER       (RGBF){ 0.752941f, 0.752941f, 0.752941f } /* (192,192,192) */
#define COLOR_GREY         (RGBF){ 0.501961f, 0.501961f, 0.501961f } /* (128,128,128) */
#define COLOR_LIGHT_GREY   (RGBF){ 0.827451f, 0.827451f, 0.827451f } /* (211,211,211) */
#define COLOR_DARK_GREY    (RGBF){ 0.25f, 0.25f, 0.25f }
#define COLOR_DARK_DARK_GREY (RGBF){ 0.12f, 0.12f, 0.12f }
#define COLOR_BROWN        (RGBF){ 0.647059f, 0.164706f, 0.164706f } /* (165,42,42) */
#define COLOR_GOLD         (RGBF){ 1.0f, 0.843137f, 0.0f }           /* (255,215,0) */
#define COLOR_INDIGO       (RGBF){ 0.294118f, 0.0f, 0.509804f }      /* (75,0,130) */
#define COLOR_VIOLET       (RGBF){ 0.933333f, 0.509804f, 0.933333f } /* (238,130,238) */
#define COLOR_CORAL        (RGBF){ 1.0f, 0.498039f, 0.313725f }      /* (255,127,80) */
#define COLOR_TURQUOISE    (RGBF){ 0.250980f, 0.878431f, 0.815686f } /* (64,224,208) */
#define COLOR_SALMON       (RGBF){ 0.980392f, 0.501961f, 0.447059f } /* (250,128,114) */
#define COLOR_SKY_BLUE     (RGBF){ 0.529412f, 0.807843f, 0.921569f } /* (135,206,235) */
#define COLOR_DEEPSKY_BLUE (RGBF){ 0.0f, 0.749020f, 1.0f }           /* (0,191,255) */
#define COLOR_ORANGERED    (RGBF){ 1.0f, 0.270588f, 0.0f }           /* (255,69,0) */
#define COLOR_HOTPINK      (RGBF){ 1.0f, 0.411765f, 0.705882f }      /* (255,105,180) */

/* Common aliases */
#define COLOR_AQUA         COLOR_CYAN
#define COLOR_FUCHSIA      COLOR_MAGENTA
#define COLOR_GRAY         COLOR_GREY
#define COLOR_LIGHT_GRAY   COLOR_LIGHT_GREY
#define COLOR_DARK_GRAY    COLOR_DARK_GREY

#endif
