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

#include "hub75gpu.h"

#ifndef __RPIHUB75_H__
#define __RPIHUB75_H__

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

    #ifdef ADA_3HAT
        #define ADDRESS_TYPE "ADA_3HAT"
    #else
        #define ADDRESS_TYPE "HZELLER_HAT"
    #endif
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


void *render_shader(void *arg);
void dither_image(uint8_t *image, int width, int height);
void apply_noise_dithering(uint8_t *image, int width, int height);

/**
 * @brief verify that the scene configuration is valid
 * will die() if invalid configuration is found
 * @param scene 
 */
void check_scene(const scene_info *scene);


/**
 * @brief render the PWM signal to the GPIO pins forever...
 * 
 * @param scene 
 */
void *render_forever(const scene_info *scene);

/**
 * @brief initialize the hub75gpu library
 */
void hub75gpu_init();

#endif
