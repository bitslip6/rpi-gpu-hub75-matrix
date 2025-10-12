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
#define FPS_MAX 240
#define FPS_MIN 1

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
 * @brief a gradient function defines the direction of the gradient
 * you can implement your own gradient function and pass it to the gradient struct
 */
typedef float (*Gradient_func)(uint16_t p1, uint16_t p2, uint16_t p3, uint16_t p4, float r1, float r2);


/**
 * @brief define a gradient between two colors, the blending will be defined
 * in the direction of type
 * 
 */
typedef struct {
    RGB colorA1;
    RGB colorA2;
    RGB colorB1;
    RGB colorB2;
    Gradient_func type;
} Gradient;



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

    /* flag to indicate a new frame is ready (i think we can just signal on bcm_ptr ... */
    _Atomic(unsigned) frame_ready;

    /** @brief pointer to a single frame for CPU drawing functions */
    uint8_t *image;

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
    
} scene_info;

/**
 * @brief this function takes the image data and maps it to the bcm signal.
 * 
 * if scene->tone_mapper is updated, new bcm bit masks will be created.
 * 
 * @param scene the scene information
 * @param image the image to map to the scene bcm data. if NULL scene->image will be used
 */
void map_byte_image_to_bcm(const scene_info *scene, const uint8_t *image);

/**
 * must be called on the main thread to start the renderer. it never returns
 */
void *render_forever(const scene_info *scene);

/**
 * @brief render the shader arg->shader_file shader on the GPU
 */
void *render_shader(void *arg);

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
long calculate_fps(const uint16_t target_fps, const bool show_fps);


// graceful shutdown helpers
void render_loop_shutdown(struct scene_info *scene);
void hub75_request_shutdown(struct scene_info *scene);
void hub75_wait_shutdown(struct scene_info *scene);


/**
 * @brief draw an unfilled anti-aliased triangle using Xiolin Wu's line drawing algorithm
 * 
 * @param scene 
 * @param x0  p0 x
 * @param y0  p0 y
 * @param x1  p1 x
 * @param y1  p2 y
 * @param x2  p3 x
 * @param y2  p3 y
 * @param color 
 */
void hub_triangle_aa(scene_info *scene, int x0, int y0, int x1, int y1, int x2, int y2, RGB color);

void hub_line_aa(scene_info *scene, const int x0, const int y0, const int x1, const int y1, const RGB color);
void hub_line(scene_info *scene, const int x0, const int y0, const int x1, const int y1, const RGB color);

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

/* --------------------------------------------------------------
 * OPTIONAL: Public function table for FFI (e.g. Python / Rust / Go)
 * --------------------------------------------------------------
 * This lightweight indirection lets foreign language bindings obtain
 * stable pointers to the drawing helpers without relying on parsing
 * multiple headers.  Versioning can be added later by extending the
 * struct (always append new fields) and bumping hub75_api.version.
 */
typedef struct hub75_api {
    uint32_t version;      /* struct version for compatibility */
    scene_info *scene;     /* active scene (set via hub75_get_api(scene)) */

    /* Scene creation helpers (do NOT use internal scene pointer) */
    scene_info *(*new_scene)(void);
    scene_info *(*parse_scene)(int argc, char **argv);

    /* Lifecycle operating on api->scene (scene must be set) */
    void (*start)(void);
    void (*request_shutdown)(void);
    void (*wait_shutdown)(void);
    void (*map_image)(uint8_t *image); /* NULL -> use scene->image */

    /* Timing (independent of scene except show_fps flag consumed inside) */
    long (*calculate_fps)(uint16_t target_fps, bool show_fps);

    /* Drawing primitives (use api->scene internally) */
    void (*pixel)(int x, int y, RGB pixel);
    void (*pixel_factor)(int x, int y, RGB pixel, float factor);
    void (*pixel_alpha)(int x, int y, RGBA pixel);
    void (*fill)(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color);
    void (*line)(int x0, int y0, int x1, int y1, RGB color);
    void (*line_aa)(int x0, int y0, int x1, int y1, RGB color);
    void (*triangle)(int x0, int y0, int x1, int y1, int x2, int y2, RGB color);
    void (*triangle_aa)(int x0, int y0, int x1, int y1, int x2, int y2, RGB color);
    void (*circle)(uint16_t cx, uint16_t cy, uint16_t radius, RGB color);
    void (*fill_grad)(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, Gradient gradient);
} hub75_api;

/* Obtain & initialize (or re-point) the singleton API to a scene. */
const hub75_api *hub75_get_api(scene_info *scene);


#endif