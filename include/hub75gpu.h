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
struct scene_info;

// void map_byte_image_to_pwm(uint8_t *image, const scene_info *scene, uint8_t fps_sync) {
typedef void (*func_tone_mapper_t)(const RGBF *in, RGBF *out, const float level);
typedef uint8_t *(*func_image_mapper_t)(const uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene);
typedef uint8_t *(image_mapper_t)(const uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene);

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
typedef struct { float x, y, z; } light_vec3;

typedef struct light_t {
    light_type_t type;     /* light category */

    /* Common parameters */
    RGBF   color;          /* light color (0..1 per channel) */
    float  intensity;      /* scalar multiplier for brightness */
    bool   casts_shadows;  /* whether this light should cast shadows */

    /* Geometric parameters (interpreted by type) */
    light_vec3 position;   /* for point/spot lights */
    light_vec3 direction;  /* for directional/spot lights */

    /* Optional falloff / cone controls (POINT/SPOT) */
    float  range;          /* effective radius for point/spot; 0 => infinite */
    float  inner_cos;      /* spot inner cone (cosine of angle); 1 => no cone */
    float  outer_cos;      /* spot outer cone (cosine of angle); must be <= inner_cos */
} light_t;

typedef struct scene_lighting_t {
    RGBF     ambient;      /* ambient light color (0..1 per channel) */
    uint16_t num_lights;   /* number of active lights */
    light_t *lights;       /* dynamic array of lights (NULL when num_lights == 0) */
} scene_lighting_t;




/**
 * @brief everything to define the scene and panel configuration 
 * This is a kind of global configuration for the entire system 
 */
typedef struct scene_info {
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

    /** @brief pointer to a single frame for CPU drawing functions */
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
     * @brief current frame index, increments every frame rendered
     */
    uint32_t frame_index;

    pthread_t render_thread;
    pthread_t mapper_thread;

    bool frame_ready;
    
} scene_info;


/**
 * @brief this function takes the image data and maps it to the bcm signal.
 * 
 * if scene->tone_mapper is updated, new bcm bit masks will be created.
 * 
 * @param scene the scene information
 * @param image the image to map to the scene bcm data. if NULL scene->image will be used
 */
void map_byte_image_to_bcm(const scene_info *scene, uint8_t *image);

void *mapper_thread_main(void *arg);

/**
 * must be called on the main thread to start the renderer. it never returns
 */
void *render_forever(const scene_info *scene);

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
bool hub_render_video(scene_info *scene, const char *filename);

/**
 * @brief count number of times this function is called, 1 every second output
 * the number of times called and reset the counter. This function can not
 * be called from multiple locations. It is not thread safe.
 * 
 * @param target_fps - target a sleep time to achieve this fps
 * @return long - returns sleep time in microseconds
 */
unsigned long calculate_fps(const uint16_t target_fps, const bool show_fps);


// graceful shutdown helpers
void hub75_request_shutdown(struct scene_info *scene);
void hub75_wait_shutdown(struct scene_info *scene);

void draw_polygon_fill(scene_info *scene, Polygonf_t *poly, RGB color);
void gradient_polygon(scene_info *scene, Polygonf_t *poly, SimpleGradient gradient);

/* Easing function implementations */
float apply_easing(float t, easing_function_t easing);


void hub_line_aa(scene_info *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, const RGB color);
void hub_line(scene_info *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, RGB color);

/**
 * @brief parse command line arguments and create a valid scene
 * 
 * @param argc command line argument count
 * @param argv command line arguments
 * @return scene_info* the created scene information
 */
scene_info *parse_scene(int argc, char **argv);

/**
 * @brief create a default scene
 * 
 * @return scene_info* the created scene information
 */
scene_info *new_scene();

/**
 * check the scene and start the rendering threads if everying is ok
 */
void start_scene(scene_info *scene);

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

typedef struct { float x, y; } vec2;
typedef struct { float x, y, z; } vec3;
typedef struct { float m[16]; }   mat4;  // column-major, m[col*4 + row]

typedef struct {
    vec3 position;     // world position 
    vec3 rotation;     // Euler angles in radians, x=pitch, y=yaw, z=roll 
    vec3 scale;        // per-axis scale 
} transform_t;

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

typedef struct {
    vert_list_t *verticies;
    edge_list_t *edges;
    face_list_t *faces;       /* NEW: triangle definitions */
    normal_list_t *normals;   /* NEW: face normals */
    color_list_t *edge_colors;

    vec3 *rendered_vertices;
    bool cull_backface;    /* toggle backface culling for wireframe/fill */
} object_t;



object_t* object_cube(void);
object_t* object_tetrahedron(void);
object_t* object_octahedron(void);
object_t* object_pyramid(void);
object_t* object_cylinder(uint16_t segments);
object_t* object_sphere(uint16_t subdivisions);
object_t* object_torus(uint16_t major_segments, uint16_t minor_segments);
object_t* object_plane(uint16_t width_segments, uint16_t height_segments);
mat4 camera_project(const camera_t *cam, const transform_t *obj_xform);
void transform_mesh_to_ndc(const vec3 *in_vertices, size_t n, mat4 mvp, vec3 *out_ndc);
object_t* object_new(uint16_t num_vertices, uint16_t num_edges, uint16_t num_faces);

/* 3D math utilities (normals) */
/* Build model matrix (T * Rz * Ry * Rx * S) */
mat4 model_matrix(const transform_t *t);
/* Extract world-space normal matrix from model (inverse-transpose of upper-left 3x3) */
void normal_matrix_from_model(const mat4 model, float out3x3[9]);
/* Multiply a 3x3 (column-major) with a vec3 */
vec3 mat3_mul_vec3(const float M[9], vec3 v);


typedef struct {
    void (*clear)();
    void (*pixel)(const uint16_t x, const uint16_t y, RGB c);
    void (*pixel_factor)(int x, int y, RGB pixel, float factor);
    void (*pixel_alpha)(int x, int y, RGBA pixel);
    void (*line)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c);
    void (*line_aa)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c);
    void (*poly)(Polygonf_t *poly, RGB color1);
    void (*poly_gradient)(Polygonf_t *poly, SimpleGradient gradient);
    void (*fill_gradient)(int y, int x0, int x1, const SimpleGradient *gradient, 
                            int minx, int miny, int maxx, int maxy);
    void (*shutdown)();
    void (*begin_frame)();
    void (*end_frame)();

    mat4 (*geo_render)(camera_t *cam, transform_t *obj_xform);
    mat4 (*geo_project)(camera_t *cam, transform_t *obj_xform);
    void (*geo_render_wire)(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene_lighting_t *lighting);
    void (*geo_render_filled)(const camera_t *cam, object_t *obj, const transform_t *obj_xform, const scene_lighting_t *lighting);

    camera_t* (*geo_camera)();
    transform_t* (*geo_transform)();
    object_t* (*geo_object)(const uint16_t num_vertices, const uint16_t num_edges, const uint16_t num_faces);
    object_t* (*geo_cube)();
    object_t* (*geo_tetrahedron)();
    object_t* (*geo_octahedron)();
    object_t* (*geo_pyramid)();
    object_t* (*geo_cylinder)(uint16_t segments);
    object_t* (*geo_sphere)(uint16_t subdivisions);
    object_t* (*geo_torus)(uint16_t major_segments, uint16_t minor_segments);
    object_t* (*geo_plane)(uint16_t width_segments, uint16_t height_segments);

} hub75gpu_t;

/* set the current thread's scene and get an API whose functions do not take a scene */
hub75gpu_t hub75gpu(scene_info *s);

#endif
