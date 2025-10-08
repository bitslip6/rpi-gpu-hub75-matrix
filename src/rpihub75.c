/**
 * https://www.i-programmer.info/programming/148-hardware/16887-raspberry-pi-iot-in-c-pi-5-memory-mapped-gpio.html
 * This code was made possible by the work of Harry Fairhead to describe the RPI5 GPIO interface.
 * As Raspberry Pi5 support increases, this code will be updated to reflect the latest GPIO interface.
 * 
 * After Linux kernel 6.12 goes into Raspberry pi mainline, you should compile the kernel with
 * PREEMPT_RT patch to get the most stable performance out of the GPIO interface.
 * 
 * This code does not require root privileges and is quite stable even under system load.
 * 
 * This is about 80 hours of work to deconstruct the HUB75 protocol and the RPI5 GPIO interface
 * as well as build the PWM modulation, abstractions, GPU shader renderer and debug. 
 * 
 * You are welcome to use and adapt this code for your own projects.
 * If you use this code, please provide a back-link to the github repo, drop a star and give me a shout out.
 * 
 * Happy coding...
 * 
 * @file gpio.c
 * @author Cory A Marsh (coryamarsh@gmail.com)
 * @brief 
 * @version 0.33
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/param.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <stdatomic.h>

#include "rpihub75.h"
#include "util.h"


/**
 * @brief calculate an address line pin mask for row y
 * not used outside this file
 * @param y the panel row number to calculate the mask for
 * @return uint32_t the bitmask for the address lines at row y
 */
uint32_t row_to_address(const int y, uint16_t half_height) {

    // if they pass in image y not panel y, convert to panel y
    uint16_t row = (y-1) % half_height;
    uint32_t bitmask = 0;

    // Map each bit from the input to the corresponding bit position in the bitmask
    if (row & (1 << 0)) bitmask |= (1 << ADDRESS_A);  // Map bit 0
    if (row & (1 << 1)) bitmask |= (1 << ADDRESS_B);  // Map bit 1
    if (row & (1 << 2)) bitmask |= (1 << ADDRESS_C);  // Map bit 2
    if (row & (1 << 3)) bitmask |= (1 << ADDRESS_D);  // Map bit 3
    if (row & (1 << 4)) bitmask |= (1 << ADDRESS_E);  // Map bit 4


    return bitmask;
}




/**
 * @brief verify that the scene configuration is valid
 * will die() if invalid configuration is found
 * @param scene 
 */
void check_scene(const scene_info *scene) {
    printf("ports: %d, chains: %d, width: %d, height: %d, stride: %d, bit_depth: %d\n", 
        scene->num_ports, scene->num_chains, scene->width, scene->height, scene->stride, scene->bit_depth);
    if (CONSOLE_DEBUG) {
        printf("ports: %d, chains: %d, width: %d, height: %d, stride: %d, bit_depth: %d\n", 
            scene->num_ports, scene->num_chains, scene->width, scene->height, scene->stride, scene->bit_depth);
    }
    if (scene->num_ports > 3) {
        die("Only 3 port supported at this time [%d]\n", scene->num_ports);
    }
    if (scene->num_ports < 1) {
        die("Require at last 1 port\n");
    }
    if (scene->num_chains < 1) {
        die("Require at last 1 panel per chain: [%d]\n", scene->num_chains);
    }
    if (scene->num_chains > 16) {
        die("max 16 panels supported on each chain\n");
    }
    if (scene->bcm_mapper == NULL) {
        die("A bcm mapping function is required\n");
    }
    if (scene->stride != 3 && scene->stride != 4) { 
        die("Only 3 or 4 byte stride supported\n");
    }
    if (scene->bcm_signalA == NULL) {
        die("No bcm signal buffer A defined\n");
    }
    if (scene->bcm_signalB == NULL) {
        die("No bcm signal buffer B defined\n");
    }
    if (scene->image == NULL) {
        die("No RGB image buffer defined\n");
    }
    if (scene->bit_depth < 4 || scene->bit_depth > 64) {
        die("Only 4-64 bit depth supported\n");
    }
    if (scene->motion_blur_frames > 32) {
        die("Max motion blur frames is 32\n");
    }

    if (scene->brightness > 254) {
        die("Max brightness is 254\n");
    }
    if (scene->bit_depth % BIT_DEPTH_ALIGNMENT != 0) {
        die("requested bit_depth %d, but %d is not aligned to %d bytes\n"
            "To use this bit depth, you must #define BIT_DEPTH_ALIGNMENT to the\n"
            "least common denominator of %d\n", 
            scene->bit_depth, scene->bit_depth, BIT_DEPTH_ALIGNMENT);
    }
}

/**
 * @brief map the lower half of the image to the front of the image. this allows connecting
 * panels in a left, left, down, right pattern (or right, right, down, left) if the image is
 * mirrored.
 * 
 * NOTE: This code is un-tested. If you have the time, please send me an implementation of U and V
 * mappers
 * 
 * 
 * @param image - input buffer to map
 * @param output_image - if NULL, the output buffer will be allocated for you
 * @param scene - the scene information
 * @return uint8_t* - pointer to the output buffer
 */
uint8_t *u_mapper_impl(uint8_t *image_in, uint8_t *image_out, const struct scene_info *scene) {
    static uint8_t *output_image = NULL;
    if (output_image == NULL) {
        debug("Allocating memory for u_mapper\n"); 
        output_image = (uint8_t*)aligned_alloc(64, scene->width * scene->height * scene->stride);
        if (output_image == NULL) {
            die("Failed to allocate memory for u_mapper image\n");
        }
    }
    if (image_out == NULL) {
        debug("output image is NULL, using allocated memory\n");
        image_out = output_image;
    }


    // Split image into top and bottom halves
    const uint8_t *bottom_half = image_in + (scene->width * (scene->height / 2) * scene->stride);  // Last 64 rows
    const uint32_t row_length = scene->width * scene->stride;

    debug("width: %d, stride: %d, row_length: %d", scene->width, scene->stride, row_length);
    // Remap bottom half to the first part of the output
    for (int y = 0; y < (scene->height / 2); y++) {
        // Copy each row from bottom half
        debug ("  Y: %d, offset: %d", y, y * scene->width * scene->stride);
        memcpy(output_image + (y * scene->width * scene->stride), bottom_half + (y * scene->width * scene->stride), row_length);
    }

    // Remap top half to the second part of the output
    for (int y = 0; y < (scene->height / 2); y++) {
        // Copy each row from top half
        memcpy(output_image + ((y + (scene->width / 2)) * scene->width * scene->stride), image_in + (y * scene->width * scene->stride), row_length);
    }

    return output_image;
}




/**
 * @brief invert the image vertically
 */
__attribute__((hot, flatten))
uint8_t *flip_mapper(uint8_t *__restrict image,
                             uint8_t *__restrict image_out,
                             const scene_info *__restrict scene)
{
    const size_t row_sz   = (size_t)scene->width * (size_t)scene->stride;
    const size_t height   = (size_t)scene->height;

    if (UNLIKELY(row_sz == 0 || height == 0)) return image_out ? image_out : image;

    // Fast path, out-of-place: one memcpy per row, sequential IO, best for bandwidth
    if (image_out && image_out != image) {
        uint8_t *__restrict dst = image_out;
        const uint8_t *__restrict src = image;
        // iterate top->bottom on dst, read bottom->top on src
        for (size_t y = 0; y < height; ++y) {
            const uint8_t *src_row = src + (height - 1 - y) * row_sz;
            uint8_t       *dst_row = dst + y * row_sz;

            // prefetch next rows to hide latency on A72/A76
            __builtin_prefetch(src_row - row_sz, 0, 3);
            __builtin_prefetch(dst_row + row_sz, 1, 3);

            memcpy(dst_row, src_row, row_sz);
        }
        return image_out;
    }

    printf("nom flip\n");

    return image_out ? image_out : image;
}




#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
static inline uint8x16_t reverse16_u8(uint8x16_t v) {
    // reverse bytes within 64-bit lanes, then swap halves
    uint8x16_t r = vrev64q_u8(v);
    return vextq_u8(r, r, 8);
}
#endif


__attribute__((hot, flatten))
uint8_t *mirror_mapper(uint8_t *__restrict image,
                       uint8_t *__restrict image_out,
                       const struct scene_info *__restrict scene)
{
    const size_t w   = (size_t)scene->width;
    const size_t h   = (size_t)scene->height;
    const size_t bpp = (size_t)scene->stride;   // RGB => 3
    const size_t row_sz = w * bpp;

    if (UNLIKELY(!image || !image_out || w == 0 || h == 0 || bpp == 0)) {
        return image_out;
    }

    for (size_t y = 0; y < h; ++y) {
        const uint8_t *src_row = image     + y * row_sz;
        uint8_t       *dst_row = image_out + y * row_sz;

#if (defined(__ARM_NEON) || defined(__aarch64__))
        if (LIKELY(bpp == 3)) {
            // NEON path: process 16 RGB pixels (48 bytes) per chunk
            const size_t px_block = 16;
            const size_t blk_bytes = px_block * 3;   // 48
            const size_t blocks = w / px_block;
            const size_t tail_px = w - blocks * px_block;

            // write blocks into destination from left after reserving space for tail
            // first fill the vectorized part: dest base after tail
            uint8_t *dst_blocks_base = dst_row + tail_px * 3;

            for (size_t i = 0; i < blocks; ++i) {
                const uint8_t *s = src_row + i * blk_bytes;

                // deinterleave 16 RGB pixels
                uint8x16x3_t rgb = vld3q_u8(s);

                // reverse pixel order within each channel
                rgb.val[0] = reverse16_u8(rgb.val[0]);
                rgb.val[1] = reverse16_u8(rgb.val[1]);
                rgb.val[2] = reverse16_u8(rgb.val[2]);

                // destination block position (mirror): place from right to left
                uint8_t *d = dst_blocks_base + (blocks - 1 - i) * blk_bytes;

                // interleaved store
                vst3q_u8(d, rgb);
            }

            // tail pixels (remaining leftmost in dest): scalar copy, mirrored
            for (size_t t = 0; t < tail_px; ++t) {
                const size_t src_x  = w - 1 - t;        // rightmost going left
                const size_t dst_x  = t;                // leftmost going right
                const uint8_t *sp = src_row + src_x * 3;
                uint8_t       *dp = dst_row + dst_x * 3;
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
            }
            continue;
        }
#endif
        // Scalar fallback (any bpp), still cache-friendly
        // Copy each row reversed in pixels, preserving per-pixel byte order
        size_t left  = 0;
        size_t right = w - 1;
        while (left < right) {
            uint8_t *dl = dst_row + left  * bpp;
            uint8_t *dr = dst_row + right * bpp;
            const uint8_t *sl = src_row + (w - 1 - left)  * bpp;
            const uint8_t *sr = src_row + (w - 1 - right) * bpp;

            // write two pixels per iteration to reduce loop overhead
            memcpy(dl, sr, bpp);
            memcpy(dr, sl, bpp);

            ++left;
            --right;
        }
        if (left == right) {
            memcpy(dst_row + left * bpp, src_row + (w - 1 - left) * bpp, bpp);
        }
    }

    return image_out;
}


uint8_t *mirror_mapper_old(uint8_t *image, uint8_t *image_out, const struct scene_info *scene) {

    uint16_t row_sz = scene->width * scene->stride;

    // Iterate through each row
    for (int y = 0; y < scene->height; y++) {
        // Get a pointer to the start of the current row
        uint8_t *row = image + y * row_sz;

        // Swap pixels from left to right within the row
        for (int x = 0; x < scene->width / 2; x++) {
            int left_index = x * scene->stride;
            int right_index = (scene->width - x - 1) * scene->stride;

            // Swap the left pixel with the right pixel (3 bytes: R, G, B)
            for (int i = 0; i < 3; i++) {
                uint8_t temp = row[left_index + i];
                row[left_index + i] = row[right_index + i];
                row[right_index + i] = temp;
            }
        }
    }
    return image;
}



__attribute__((hot, flatten))
uint8_t *mirror_flip_mapper(uint8_t *__restrict image,
                            uint8_t *__restrict image_out,
                            const struct scene_info *__restrict scene)
{
    const size_t w   = (size_t)scene->width;
    const size_t h   = (size_t)scene->height;
    const size_t bpp = (size_t)scene->stride;         // RGB => 3
    const size_t row_sz = w * bpp;

    if (UNLIKELY(!image || !image_out || w == 0 || h == 0 || bpp == 0)) {
        return image_out;
    }

    for (size_t y = 0; y < h; ++y) {
        // vertical flip selects source row from bottom
        const uint8_t *src_row = image     + (h - 1 - y) * row_sz;
        uint8_t       *dst_row = image_out + y * row_sz;

#if (defined(__ARM_NEON) || defined(__aarch64__))
        if (LIKELY(bpp == 3)) {
            // process 16 RGB pixels (48 bytes) per block with deinterleave/reverse/interleave
            const size_t px_block  = 16;
            const size_t blk_bytes = px_block * 3;  // 48
            const size_t blocks    = w / px_block;
            const size_t tail_px   = w - blocks * px_block;

            // vectorized blocks: dst left→right, src right→left in block-sized chunks
            for (size_t i = 0; i < blocks; ++i) {
                // source block starts px_block pixels from the right edge, moving left
                const size_t src_px   = w - (i + 1) * px_block;
                const uint8_t *s      = src_row + src_px * 3;
                uint8_t       *d      = dst_row + i * blk_bytes;

                __builtin_prefetch(s - 96, 0, 3);
                __builtin_prefetch(d + 96, 1, 3);

                // deinterleave 16 pixels of RGB
                uint8x16x3_t rgb = vld3q_u8(s);

                // reverse pixel order within each channel to mirror horizontally
                rgb.val[0] = reverse16_u8(rgb.val[0]);
                rgb.val[1] = reverse16_u8(rgb.val[1]);
                rgb.val[2] = reverse16_u8(rgb.val[2]);

                // interleaved store to destination
                vst3q_u8(d, rgb);
            }

            // tail pixels at the left of dst, sourced from the left of src but reversed
            for (size_t t = 0; t < tail_px; ++t) {
                const size_t src_x = tail_px - 1 - t; // right-to-left within the remaining tail
                const uint8_t *sp  = src_row + src_x * 3;
                uint8_t       *dp  = dst_row + (blocks * px_block + t) * 3;
                dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
            }
            continue;
        }
#endif
        // scalar fallback: generic bpp, still sequential writes and reads
        const uint8_t *sp = src_row + (w - 1) * bpp; // start at rightmost pixel of source row
        uint8_t       *dp = dst_row;                 // start at leftmost pixel of dest row

        // copy w pixels, mirrored horizontally
        for (size_t x = 0; x < w; ++x) {
            __builtin_prefetch(sp - 3 * bpp, 0, 1);
            // copy one pixel of bpp bytes
            switch (bpp) {
                case 3:
                    dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                    break;
                case 4:
                    // common case for RGBA buffers
                    ((uint32_t *)dp)[0] = ((const uint32_t *)sp)[0];
                    break;
                default:
                    memcpy(dp, sp, bpp);
                    break;
            }
            dp += bpp;
            sp -= bpp;
        }
    }

    return image_out;
}



/**
 * internal method for rendering on pi zero, 3 and 4
 */
void render_forever_pi4(const scene_info *scene, int version) {

    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(0, version); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    if (version == 4) {
    	configure_gpio(PERIBase, 4);
    } else if (version == 3) {
    	configure_gpio(PERIBase, 3);
    }


     
    // index into the OE jitter mask
    uint32_t jitter_idx = 0;
    // pre compute some variables. let the compiler know the alignment for optimizations
    const uint8_t  half_height __attribute__((aligned(16))) = scene->panel_height / 2;
    const uint16_t width __attribute__((aligned(16))) = scene->width;
    const uint8_t  bit_depth __attribute__((aligned(BIT_DEPTH_ALIGNMENT))) = scene->bit_depth;

    // pointer to the current bcm data to be displayed
    uint32_t *bcm_signal = scene->bcm_signalA;
    ASSERT(width % 16 == 0);
    ASSERT(half_height % 16 == 0);
    ASSERT(bit_depth % BIT_DEPTH_ALIGNMENT == 0);

    bool last_pointer = scene->bcm_ptr;

    // create the OE jitter mask to control screen brightness
    // if we are using BCM brightness, then set OE to 0 (0 is display on ironically)
    uint32_t *jitter_mask = create_jitter_mask(JITTER_SIZE, scene->brightness);
    if (scene->jitter_brightness == false) {
        memset(jitter_mask, 0, JITTER_SIZE);
    }

    // store the row to address mapping in an array for faster access
    uint32_t addr_map[half_height];
    for (int i=0; i<half_height; i++) {
        addr_map[i] = row_to_address(i, half_height);
    }

    time_t last_time_s     = time(NULL);
    uint32_t frame_count   = 0;
    uint32_t last_addr     = 0;
    uint32_t color_pins    = 0;

    // uint8_t bright = scene->brightness;
    while(atomic_load(&scene->do_render)) {

        // iterate over the bit plane
        for (uint8_t pwm=0; pwm<bit_depth; pwm++) {
            frame_count++;
            // for the current bit plane, render the entire frame
            uint32_t offset = pwm;
            for (uint16_t y=0; y<half_height; y++) {
                asm volatile ("" : : : "memory");  // Prevents optimization

                PERIBase[7]  = addr_map[y] & ~last_addr;
                SLOW
                PERIBase[10] = ~addr_map[y] & last_addr;
                SLOW
                last_addr    = addr_map[y];

                for (uint16_t x=0; x<width; x++) {
                    asm volatile ("" : : : "memory");  // Prevents optimization
                    uint32_t new_mask = (bcm_signal[offset]);// | jitter_mask[jitter_idx]);
                    PERIBase[10]      = (~new_mask & color_pins) | PIN_CLK;
                    SLOW
                    PERIBase[7]       = (new_mask & ~color_pins);
                    SLOW
                    SLOW
                    SLOW
                    PERIBase[7]       = (new_mask) | PIN_CLK;

                    SLOW
                    SLOW
                    SLOW
                    color_pins        = new_mask;

                    // advance the global OE jitter mask 1 frame
                    jitter_idx = (jitter_idx + 1) % JITTER_SIZE;

                    // advance to the next pixel in the bcm signal
                    offset += bit_depth + 1;
                }
                PERIBase[7] = PIN_LATCH | PIN_OE;
                SLOW
                SLOW
                PERIBase[10] = PIN_LATCH;
                SLOW
                SLOW
                PERIBase[10] = PIN_OE;
                SLOW
            }

            // swap the buffers on vsync
            if (UNLIKELY(scene->bcm_ptr != last_pointer)) {
                last_pointer = scene->bcm_ptr;
                bcm_signal = (last_pointer) ? scene->bcm_signalB : scene->bcm_signalA;
            }
        }

        time_t current_time_s = time(NULL);
        if (UNLIKELY(current_time_s >= last_time_s + 5)) {

            if (scene->show_fps) {
                printf("Panel Refresh Rate: %dHz\n", frame_count / 5);
            }
            frame_count = 0;
            last_time_s = current_time_s;
        }
    }
}


static inline void io_write_barrier(void) {
#if defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory"); /* ARMv8 device write fence */
#endif
}

/* light store barrier for device stores */
static inline void io_store_barrier(void) {
#if defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("dmb ishst" ::: "memory"); /* much cheaper than dsb */
#endif
}


static void enable_rt_and_lock_mem(void) {
    pid_t pid = getpid();
    struct sched_param sp = { .sched_priority = 80 }; /* 1..99 */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);


    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        fprintf(stderr, " * Try running as root to enable real-time scheduling\n");
    } else {
        debug(" * Real-time scheduling enabled\n");
    }
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        die("mlockall failed\n");
    }
    if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) != 0) {
	    die("unable to set CPU affinity to 3\n");
    }
}


/**
 * @brief you can cause render_forever to exit by updating the value of do_render pointer
 * EG:
 * scene->do_render = false; // will cause render_forever to exit from another thread
 * 
 */
void render_forever(const scene_info *scene) {

        
    enable_rt_and_lock_mem();

    // check the CPU model to determine which GPIO function to use
    // note one cannot use file_get_contents as this file is zero length...
    char *line = NULL;
    size_t line_sz;
    int cpu_model = 0;
    FILE *file = fopen("/proc/cpuinfo", "rb");
    if (file == NULL) {
        die("Could not open file /proc/cpuinfo\n");
    }
    while (getline(&line, &line_sz, file)) {
        if (strstr(line, "Pi 5") != NULL) {
            cpu_model = 5;
            break;
        }
        else if (strstr(line, "Pi 4") != NULL) {
            cpu_model = 4;
            break;
        }
	      else if (strstr(line, "Pi 3") != NULL) {
            cpu_model = 3;
            break;
        } 
	      else if (strstr(line, "Pi Zero 2") != NULL) {
            cpu_model = 3;
            break;
        }
    }
    if (cpu_model == 0) {
        die("Unsupported CPU model detected %s\n", line);
    }
    free(line);
    fclose(file);

    debug("\ncpu_model: %d\n", cpu_model);

    if (cpu_model == 0) die("Only Pi5, Pi4, Pi3 and Pi Zero 2 are currently supported");

    if (cpu_model < 5 ) {
        render_forever_pi4(scene, cpu_model);
    }
    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(0, 5); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    uint32_t *RIOBase;
    RIOBase = PERIBase + RIO5_OFFSET;
    configure_gpio(PERIBase, 5);
         
    // index into the OE jitter mask
    uint32_t jitter_idx = 0;
    // pre compute some variables. let the compiler know the alignment for optimizations
    const uint16_t  half_height = (uint16_t)scene->panel_height / 2;
    const uint16_t width = scene->width;
    const uint8_t  bit_depth = scene->bit_depth;

    // pointer to the current bcm data to be displayed
    uint32_t *bcm_signal = scene->bcm_signalA;
    ASSERT(width % 16 == 0);
    ASSERT(half_height % 16 == 0);
    ASSERT(bit_depth % BIT_DEPTH_ALIGNMENT == 0);

    bool last_pointer = scene->bcm_ptr;

    // create the OE jitter mask to control screen brightness
    // if we are using BCM brightness, then set OE to 0 (0 is display on ironically)
    uint32_t *jitter_mask = create_jitter_mask(JITTER_SIZE, scene->brightness);
    if (scene->jitter_brightness == false) {
        memset(jitter_mask, 0, JITTER_SIZE * sizeof(*jitter_mask));
    }

    // store the row to address mapping in an array for faster access
    uint32_t addr_map[half_height];
    for (int i=0; i<half_height; i++) {
        addr_map[i] = row_to_address(i, half_height);
    }

    struct   timeval end_time, start_time;
    time_t   last_time_s = time(NULL);
    uint32_t frame_count = 0;
    gettimeofday(&start_time, NULL);


    /* cache local aliases to MMIO regs, keep them volatile */
    volatile uint32_t * const reg_out   = &rio->Out;
    volatile uint32_t * const reg_set   = &rioSET->Out;
    volatile uint32_t * const reg_clr   = &rioCLR->Out;
    const uint32_t stride      = (uint32_t)bit_depth + 1;  /* next-pixel offset */



    //const uint32_t guard_px = 4;   /* do not change OE in first/last N pixels of a row */


    // const uint32_t js = scene->width * 32;

    uint16_t phase = 1;
    while (scene->do_render) {
        phase++;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {
            uint32_t offset = pwm;


            frame_count++;
            jitter_idx = phase;
            for (uint16_t y = 0; y < half_height; y++) {

                /* optional: inhibit jitter on first couple of pixels to avoid latch-adjacent OE flips */
                //uint32_t inhibit = 2; /* set 0..2 as needed */
                const uint32_t addr_bits = addr_map[y];
                //jitter_idx = ((y * 1315423911u) + phase) % JITTER_SIZE; // decorrelate rows



                for (uint16_t x = 0; x < width; x++) {

                    /* hard guard: data must never drive control pins */
                    const uint32_t data = bcm_signal[offset] & MASK_DATA;

                    //const int guard = (x < guard_px) || ((width - 1 - x) < guard_px);
                    //const uint32_t oe_mask = (guard) ? 0u : jitter_mask[jitter_idx];
                    const uint32_t oe_mask = jitter_mask[jitter_idx];


                    /* absolute OUT twice, preserves order without fences */
                    uint32_t v = data | addr_bits | oe_mask;   /* CLK low state */
                    *reg_out = v;                 /* data + addr + oe, clk low */
                    CLK_SETUP_DELAY();
                    *reg_out = v | PIN_CLK;       /* clk high */
                    CLK_SETUP_DELAY();

                    /* advance after full edge */
                    jitter_idx = (jitter_idx + 1) % JITTER_SIZE;
                    offset += stride;
               }

                /* safe latch with OE asserted high and address held */
                *reg_set = PIN_OE;                 /* display off */
                io_write_barrier();

                //*reg_out = addr_bits | PIN_OE;   /* make address + OE explicit on OUT bus */
                //io_write_barrier();

                *reg_set = PIN_LATCH | PIN_OE;              /* latch high */
                io_write_barrier();
                *reg_clr = PIN_LATCH;              /* latch low */


            }
        }
	// swap the buffer when it changes
	unsigned before = atomic_load_explicit(&scene->frame_ready, memory_order_acquire);
	if (!(before & 1u)) {
		last_pointer = scene->bcm_ptr;
		bcm_signal = (last_pointer) ? scene->bcm_signalB : scene->bcm_signalA;
	}
        if (frame_count % 128 == 0) {
            time_t current_time_s = time(NULL);
            if (UNLIKELY(current_time_s >= last_time_s + 5)) {

                if (scene->show_fps) {
                    gettimeofday(&end_time, NULL);
                    double elapsed = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec) * 1e-6;
                    printf("Panel Refresh Rate (%f): %.4fHz\n", elapsed, (frame_count / elapsed));
                    gettimeofday(&start_time, NULL);
                }
                frame_count = 0;
                last_time_s = current_time_s;
            }
        }
    }



    // uint8_t bright = scene->brightness;
    while(scene->do_render) {

        // iterate over the bit plane, It takes about 6ms to iterate over 64 bits of BCM data per panel
        // so 3 chained panels at 64 bit depth is about 18ms or 55Hz complete refresh rate
        // new input frame data is updated instantly, over 3,000 times per second
        for (uint8_t pwm=0; pwm<bit_depth; pwm++) {
            frame_count++;
            // for the current bit plane, render the entire frame
            uint32_t offset = pwm;
            for (uint16_t y=0; y<half_height; y++) {
                asm volatile ("" : : : "memory");  // Prevents optimization

                // compute the bcm row start address for y

                for (uint16_t x=0; x<width; x++) {
                    asm volatile ("" : : : "memory");  // Prevents optimization
                    
                    uint32_t v = bcm_signal[offset] | addr_map[y] | jitter_mask[jitter_idx];
                    io_write_barrier();
                    rio->Out = v;                     // clk low 
                    io_write_barrier();
                    rio->Out = PIN_CLK;               // this will only set the pin clock, not clear any other pins

                    // advance the global OE jitter mask 1 frame
                    jitter_idx = (jitter_idx + 1) % JITTER_SIZE;

                    // advance to the next pixel in the bcm signal
                    offset += bit_depth + 1;
                }
                // make sure enable pin is high (display off) while we are latching data
                // latch the data for the entire row
                io_write_barrier();
                rioSET->Out = PIN_OE | PIN_LATCH;
                io_write_barrier();
                SLOW2         // 8 asm cycles
                rioCLR->Out = PIN_LATCH;
            }

            // swap the buffers on vsync
            if (UNLIKELY(scene->bcm_ptr != last_pointer)) {
                last_pointer = scene->bcm_ptr;
                bcm_signal = (last_pointer) ? scene->bcm_signalB : scene->bcm_signalA;
            }
        }
        
        if (frame_count % 128 == 0) {
            time_t current_time_s = time(NULL);
            if (UNLIKELY(current_time_s >= last_time_s + 5)) {

                if (scene->show_fps) {
                    gettimeofday(&end_time, NULL);
                    double elapsed = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec) * 1e-6;
                    printf("Panel Refresh Rate (%f): %.4fHz\n", elapsed, (frame_count / elapsed));
                    gettimeofday(&start_time, NULL);
                }
                frame_count = 0;
                last_time_s = current_time_s;
            }
        }
    }
}


