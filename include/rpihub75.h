#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

#ifdef RENDER_USE_PBO
#include <GLES3/gl3.h>
#endif
#include "spsc.h"


#ifndef _RPIHUB75_H
#define _RPIHUB75_H 1

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

//////////////////////////////////////////////////////////

// LINEAR OFFSET SCALE
#ifndef RED_SCALE
    #define RED_SCALE 0.0f
#endif

#ifndef GREEN_SCALE
    #define GREEN_SCALE 0.0f
#endif

#ifndef BLUE_SCALE
    #define BLUE_SCALE 0.0f
#endif


// GAMMA OFFSET SCALE (MULTIPLICATION FACTORS FROM BASE GAMMA)
#ifndef RED_GAMMA_SCALE
    #define RED_GAMMA_SCALE 1.0f
#endif

#ifndef GREEN_GAMMA_SCALE
    #define GREEN_GAMMA_SCALE 1.0f
#endif

#ifndef BLUE_GAMMA_SCALE
    #define BLUE_GAMMA_SCALE 1.0f
#endif

// DEFAULT GAMMA, CAN BE SET IN SceneInfo
#ifndef GAMMA
    #define GAMMA 1.0f
#endif


#ifndef PANEL_WIDTH
    #define PANEL_WIDTH 64
#endif
#ifndef PANEL_HEIGHT
    #define PANEL_HEIGHT PANEL_WIDTH 
#endif
#ifndef IMG_WIDTH
    #define IMG_WIDTH 64
#endif
#ifndef IMG_HEIGHT
    #define IMG_HEIGHT 64
#endif

// ideally you should use aligned bit_depth
//  for bit depths of 16,32,48,64 use BIT_DEPTH_ALIGNMENT 16
//  for bit depths of 8,16,24,32,40,48,56,64, use BIT_DEPTH_ALIGNMENT 8
//  for bit depths of 4,8,12,16,20,24,28..., use BIT_DEPTH_ALIGNMENT 4
//  for bit depths of 2,4,6,8,10,.... use BIT_DEPTH_ALIGNMENT 2
//  for all others use BIT_DEPTH_ALIGNMENT 1
#define BIT_DEPTH_ALIGNMENT 4

#define SERVER_PORT 22222

// global OE jitter mask, should be a prime >1031 and <=4093
// we don't want to make this too large, as it will consume memory
// and decrease L1-L3 cache locality of other data
#define JITTER_SIZE 65521 

// number of bcm buffers to allocate
#define BCM_BUFFERS 3 

#define JITTER_MAX_RUN_LEN 4
#define JITTER_PASSES 3

//////////////////////////////////////////////////////////

#ifndef CONSOLE_DEBUG
    #define CONSOLE_DEBUG 1
#endif
#ifndef ENABLE_ASSERTS
    #define ENABLE_ASSERTS 0 
#endif
#define PACKET_SIZE 1450
#define PREAMBLE 0xdeadcafe

//////////////////////////////////////////////////////////

#if ENABLE_ASSERTS
    #define ASSERT(x) assert(x)
#else
    #define ASSERT(ignore) assert(ignore)
#endif

#define MAX_BITS 64

#define MAX_PANEL_TYPES 8
#define MAX_PANELS     24


    #define PERI3_BASE   0x3F000000
    #define GPIO3_OFFSET 0x200000
    #define RIO3_OFFSET  0x2E0000 / 4
    #define PAD3_OFFSET  0x2F0000 / 4

    #define PERI4_BASE   0xFE000000
    #define GPIO4_OFFSET 0x200000
    #define RIO4_OFFSET  0x2E0000 / 4
    #define PAD4_OFFSET  0x2F0000 / 4

    #define PERI5_BASE 0x1f000D0000
    //#define PERI_BASE 0x1f00000000 // for root access to /dev/mem , skip the 0xD0000 offset
    #define GPIO5_OFFSET 0x00000 / 4  // 0xD0000 is alreay added to the PERI_BASE
    #define RIO5_OFFSET  0x10000 / 4
    #define PAD5_OFFSET  0x20000 / 4




/** @brief  set all GPIO pins to absolute bit mask */
#define rio ((rioregs *)RIOBase)
/** @brief  XOR GPIO pins with bit mask */
#define rioXOR ((rioregs *)(RIOBase + 0x1000 / 4))
/** @brief  SET GPIO pins in bit mask (dont touch pins not in mask) */
#define rioSET ((rioregs *)(RIOBase + 0x2000 / 4))
/** @brief  CLEAR GPIO pins in bit mask (dont touch pins not in mask) */
#define rioCLR ((rioregs *)(RIOBase + 0x3000 / 4))

#define SLOW for (volatile int s=0;s<40;s++) { asm volatile ("" : : : "memory"); asm(""); }
#define SLOW2 for (volatile int s=0;s<8;s++) { asm volatile ("" : : : "memory"); asm(""); }
#define CLK_SETUP_DELAY() asm volatile("nop; nop;")


// helpers for timing things...
#define PRE_TIME struct timeval start, end; gettimeofday(&start, NULL);
#define POST_TIME gettimeofday(&end, NULL); long elapsed_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec); printf("microseconds (1/1000 ms): %ld\n", elapsed_time);


#ifdef ADA_HAT

    #define ADDRESS_TYPE "ADAFRUIT_HAT"

    #define ADDRESS_P0_G1 13
    #define ADDRESS_P0_G2 16
    #define ADDRESS_P0_B1 6
    #define ADDRESS_P0_B2 23 
    #define ADDRESS_P0_R1 5
    #define ADDRESS_P0_R2 12

    #define ADDRESS_P1_G1 0
    #define ADDRESS_P1_G2 0
    #define ADDRESS_P1_B1 0
    #define ADDRESS_P1_B2 0 
    #define ADDRESS_P1_R1 0
    #define ADDRESS_P1_R2 0

    #define ADDRESS_P2_G1 0
    #define ADDRESS_P2_G2 0
    #define ADDRESS_P2_B1 0
    #define ADDRESS_P2_B2 0 
    #define ADDRESS_P2_R1 0
    #define ADDRESS_P2_R2 0

    #define ADDRESS_A 23
    #define ADDRESS_B 26
    #define ADDRESS_C 27
    #define ADDRESS_D 20
    #define ADDRESS_E 24
    #define ADDRESS_STROBE 21
    #define ADDRESS_CLK 17
    #define ADDRESS_OE 4

#else

    #define ADDRESS_TYPE "HZELLER_HAT"
    /**
     * standard pin assignments
     */
    #define ADDRESS_P0_G1 27
    #define ADDRESS_P0_G2 9
    #define ADDRESS_P0_B1 7
    #define ADDRESS_P0_B2 10
    #define ADDRESS_P0_R1 11
    #define ADDRESS_P0_R2 8

    #define ADDRESS_P1_G1 5
    #define ADDRESS_P1_G2 13
    #define ADDRESS_P1_B1 6
    #define ADDRESS_P1_B2 20
    #define ADDRESS_P1_R1 12
    #define ADDRESS_P1_R2 19

    #define ADDRESS_P2_R1 14
    #define ADDRESS_P2_R2 26
    #define ADDRESS_P2_G1 2
    #define ADDRESS_P2_G2 16
    #define ADDRESS_P2_B1 3
    #define ADDRESS_P2_B2 21

    #define ADDRESS_A 22
    #define ADDRESS_B 23
    #define ADDRESS_C 24
    #define ADDRESS_D 25
    #define ADDRESS_E 15
    #define ADDRESS_STROBE 4
    #define ADDRESS_CLK 17
    #define ADDRESS_OE 18

#endif

// control pins bit masks
#define PIN_OE (1 << ADDRESS_OE)
#define PIN_LATCH (1 << ADDRESS_STROBE)
#define PIN_CLK (1 << ADDRESS_CLK)



#define MASK_DATA  (1 << ADDRESS_P0_R1 | 1 << ADDRESS_P0_R2 | 1 << ADDRESS_P0_G1 | 1 << ADDRESS_P0_G2 | 1 << ADDRESS_P0_B1 | 1 << ADDRESS_P0_B2 | \
                   1 << ADDRESS_P1_R1 | 1 << ADDRESS_P1_R2 | 1 << ADDRESS_P1_G1 | 1 << ADDRESS_P1_G2 | 1 << ADDRESS_P1_B1 | 1 << ADDRESS_P1_B2 | \
                   1 << ADDRESS_P2_R1 | 1 << ADDRESS_P2_R2 | 1 << ADDRESS_P2_G1 | 1 << ADDRESS_P2_G2 | 1 << ADDRESS_P2_B1 | 1 << ADDRESS_P2_B2)
#define MASK_ADDRESS  (1 << ADDRESS_A | 1 << ADDRESS_B | 1 << ADDRESS_C | 1 << ADDRESS_D | 1 << ADDRESS_E)
#define MASK_CTRL  (PIN_CLK | PIN_LATCH | PIN_OE)

// helpers for "boolean"
#define TRUE 1
#define FALSE 0

// add address lines as a mask useful for clearing the address lines
#define ADDRESS_LINES_MASK (0 | 1 << ADDRESS_A | 1 << ADDRESS_B | 1 << ADDRESS_C | 1 << ADDRESS_D | 1 << ADDRESS_E)
#define ADDRESS_COLOR_MASK (0 | 1 << ADDRESS_P0_B1 | 1 << ADDRESS_P0_B2 | 1 << ADDRESS_P0_G1 | 1 << ADDRESS_P0_G2 | 1 << ADDRESS_P0_R1 | 1 << ADDRESS_P0_R2)



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

/**
 * @brief network packet structure
 * 
 */
struct udp_packet {
    uint32_t preamble;
    uint16_t packet_id;
    uint16_t total_packets;
    uint16_t frame_num;
    uint8_t data[PACKET_SIZE - 10];
};

/** @brief enumeration of supported pixel order on panel */
enum pixel_order_e {
    PIXEL_ORDER_RGB,
    PIXEL_ORDER_RBG,
    PIXEL_ORDER_BGR
};

// panel order describes which logical color drives panel wires R,G,B respectively
typedef enum {
    PANEL_RGB = 0, PANEL_RBG, PANEL_GRB, PANEL_GBR, PANEL_BRG, PANEL_BGR
} panel_order_t;


typedef struct {
    spsc_frame_ring ring;
    // 4 frames of width*height*bitdepth bytes 
    uint8_t *storage;
} frame_ctx;

// self referencing function pointers need this defined first
typedef struct scene_info scene_info;



// --------- jobs handed from render -> mapper ---------
typedef enum { JOB_CPU_PIXELS = 1, JOB_PBO, JOB_QUIT = 255 } job_kind_t;

typedef struct {
    job_kind_t kind;
    scene_info *scene;
    size_t size_bytes;
    union {
        struct { uint8_t *pixels; } cpu;
#ifdef RENDER_USE_PBO
        struct { GLuint pbo; GLsync fence; } pbo;
#endif
    } u;
} map_job_t;

// mapper thread state
typedef struct {
    spsc_ring_t *q_in;
    spsc_ring_t *q_filled;
    spsc_ring_t *q_free;
    scene_info *scene;
    volatile int run;
} mapper_ctx_t;





// void map_byte_image_to_pwm(uint8_t *image, const scene_info *scene, uint8_t fps_sync) {
// typedef void (*func_bcm_mapper_t)(struct scene_info *scene, uint8_t *image);
typedef void (*func_tone_mapper_t)(const RGBF *in, RGBF *out, const float level);
typedef uint8_t *(*func_image_mapper_t)( uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene);
typedef uint8_t *(image_mapper_t)(uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene);


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

    /** @brief the order of pixels on panel */
    enum pixel_order_e pixel_order;

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
    panel_rgb_scale panel_scale[MAX_PANEL_TYPES];
    panel_rgb_offset panel_offset[MAX_PANEL_TYPES];

    /** @brief array panel types for each output */
    uint8_t panel_types[MAX_PANELS];
    uint8_t num_panel_types;


    /** @brief dithering strength. (0-10) 0 is off, improves simulated color in dark areas but reduces image sharpness */
    float dither;
    bool quant_dither;

    /** 
     * @brief number of panels connected to each chain on the port (1-8)
     * 4 64 pixel panels at 32bit pwm yields 75 FPS. 8 panels at 32bit pwm yields 37 FPS.
     */
    uint8_t num_chains;

    /**
     * @brief points to active buffer
     * in update code use this to select the pwm buffer to render to: 
     * (scene->buffer_ptr == 1)? scene->bcm_signalA : scene->bcm_signalB; 
     */
    //uint8_t buffer_ptr;

    /** * @brief see buffer_ptr for usage */
    /*
    uint32_t *restrict bcm_signalA __attribute__((aligned(16)));
    uint32_t *restrict bcm_signalB __attribute__((aligned(16)));
    */

    // storage for our spsc ring buffers
    uint32_t *restrict bcm_buffers __attribute__((aligned(16)));

    // size of a single bcm frame buffer
    size_t bcm_frame_size;

    //atomic_bool bcm_ptr;
    //atomic_uint bcm_frame;

    //atomic_bool frame_swap;
    //_Atomic(unsigned) frame_ready;

    spsc_frame_ring dst_ctx;
    mapper_ctx_t src_ctx;

    /** * @brief see buffer_ptr for usage */
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
    _Atomic bool do_render;

    /**
     * set to true to show the FPS on the screen
     */
    bool show_fps;

    /**
     * @brief current frame index, increments every frame rendered
     */
    uint32_t frame_index;
    
} scene_info;




/**
 * @brief map an image of RGB or RGBA pixels to a pwm signal
 * handles double buffering and tone mapping for you
 * 
 * @param image pointer to the image data
 * @param scene scene configuration
 * @param do_fps_sync set to true to sync to scene->fps based on current time
 */
// void map_byte_image_to_pwm(uint8_t *restrict image, const scene_info *scene, const uint8_t do_fps_sync);

/**
 * @brief map an image of RGB or RGBA pixels to a pwm signal with dithering
 * dithering is based on color error diffusion from full 32bpp down to our 
 * target bit depth (usually 15bpp for 32 pwm bits).
 * 
 * dithering 
 * 
 * @param image pointer to the image data
 * @param scene scene configuration
 * @param do_fps_sync set to true to sync to scene->fps based on current time
 */
// void map_byte_image_to_pwm_dithered(uint8_t *image, const scene_info *scene, const uint8_t do_fps_sync);
void aces_inplace(RGB *in);
Normal normalize_8(const uint8_t in);

/*
void aces_tone_mapper(const RGB *in, RGB *out);
void aces_tone_mapperF(const RGBF *__restrict__ in, RGBF *__restrict__ out);
void hable_tone_mapper(const RGB *in, RGB *out);
void hable_tone_mapperF(const RGBF *in, RGBF *out);
void copy_tone_mapperF(const RGBF *in, RGBF *out);
void reinhard_tone_mapper(const RGB *in, RGB *out);
void reinhard_tone_mapperF(const RGBF *in, RGBF *out);
*/

Normal hable_tone_map(const Normal color);
void hable_inplace(RGB *in);
void adjust_contrast_saturation(RGBF *__restrict__ in, const float contrast, const float saturation);
uint8_t* u_mapper(uint8_t *image, uint8_t *output_image, const scene_info *scene);
uint8_t *flip_mapper(uint8_t *image, uint8_t *image_out, const struct scene_info *scene);
uint8_t *mirror_mapper(uint8_t *image, uint8_t *image_out, const struct scene_info *scene);
uint8_t *mirror_flip_mapper(uint8_t *image, uint8_t *image_out, const struct scene_info *scene);


void *mapper_thread_main(void *arg);

void *render_shader(void *arg);
void dither_image(uint8_t *image, int width, int height);
void apply_noise_dithering(uint8_t *image, int width, int height);

/**
 * @brief verify that the scene configuration is valid
 * will die() if invalid configuration is found
 * @param scene 
 */
pthread_t start_scene(scene_info *scene);


uint8_t *u_mapper_impl(uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene);
uint8_t *flip_mapper_impl(const uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene);

/**
 * @brief render the PWM signal to the GPIO pins forever...
 * 
 * @param scene 
 */
void* render_forever(void *arg);

#endif
