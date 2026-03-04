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
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>


//#define MEMGUARD_OVERRIDE_STDLIB
#include "memguard2.h"

#include "rpihub75.h"
#include "util.h"
#include "pixels.h"
#include "scene.h"
#include "lowlevel.h"


// delay execution for a number of cycles
void asm_delay(int cycles) {
    for (volatile int s=0;s<cycles;s++) { 
        asm volatile ("" : : : "memory"); 
        asm(""); 
    }
}


// Graceful shutdown helpers --------------------------------------------------
void hub75_display_request_shutdown(struct hub75_display *scene) {
    if (!scene) {
        printf("unable to request shutdown, no scene provided\n");
        return;
    }

    scene->do_render = false;
    //update_bcm_signal_64_rgb(scene, NULL, NULL, NULL, 0);

    hub75_display_wait(scene);
}

void hub75_display_wait(struct hub75_display *scene) {
    if (!scene) {
        printf("unable to wiat shutdown, no scene provided\n");
        return;
    } 
    debug(" [.] waiting for mapper thread to stop...\n");
    // mapper
    if (scene->mapper_thread) {
        pthread_t t = scene->mapper_thread;
        scene->mapper_thread = 0;
        pthread_join(t, NULL);
    }
    debug(" [$] mapper thread exited and joined\n");
    debug(" [.] waiting for render thread to stop...\n");
    // render
    if (scene->render_thread) {
        pthread_t t = scene->render_thread;
        scene->render_thread = 0;
        pthread_join(t, NULL);
    }
    debug(" [$] render thread exited and joined\n");
}


/**
 * @brief calculate an address line pin mask for row y
 * not used outside this file
 * @param y the panel row number to calculate the mask for
 * @return uint32_t the bitmask for the address lines at row y
 */
/*
 * Map logical row index y to address line bitmask.
 * Notes:
 * - We assume y is in [0, half_height). Apply modulo just in case.
 * - Some HATs wire A..E in reversed significance. Allow optional bit order flip.
 * - Optionally apply a small row offset for hardware that latches one row late/early.
 */
uint32_t row_to_address(int y, uint16_t half_height) {
    if (half_height == 0) return 0;

    // normalize to [0, half_height)
    uint16_t norm = (uint16_t)(y % half_height);
    // defend against negative y although callers only pass non-negative
    if (y < 0) {
        int yy = y % (int)half_height;
        if (yy < 0) yy += half_height;
        norm = (uint16_t)yy;
    }

    // Optional compile-time row offset to quickly test off-by-one issues
    #ifndef ROW_ADDRESS_Y_OFFSET
    #define ROW_ADDRESS_Y_OFFSET -1
    #endif
    norm = (uint16_t)((norm + ROW_ADDRESS_Y_OFFSET) % half_height);

    // Allow compile-time reversal of address bit significance if wiring differs
    #ifndef ADDRESS_LSB_IS_A
    #define ADDRESS_LSB_IS_A 1
    #endif

    const uint8_t order[5] =
    #if ADDRESS_LSB_IS_A
        { ADDRESS_A, ADDRESS_B, ADDRESS_C, ADDRESS_D, ADDRESS_E };
    #else
        { ADDRESS_E, ADDRESS_D, ADDRESS_C, ADDRESS_B, ADDRESS_A };
    #endif

    uint32_t bitmask = 0u;
    for (uint8_t i = 0; i < 5; ++i) {
        if (norm & (1u << i)) bitmask |= (1u << order[i]);
    }
    return bitmask;
}


extern hub75_display_t *g_scene;
/**
 * @brief verify that the scene configuration is valid
 * will die() if invalid configuration is found
 * @param scene 
 */
void hub75_display_bind_scene(hub75_display_t *scene) {
    debug("ports: %d, chains: %d, width: %d, height: %d, stride: %d, bit_depth: %d\n", 
        scene->num_ports, scene->num_chains, scene->width, scene->height, scene->stride, scene->bit_depth);

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
    if (scene->stride != 3 && scene->stride != 4) { 
        die("Only 3 or 4 byte stride supported\n");
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

    // initialize the memory allocator (this must be in it's own file)
    hub75gpu_init();

    // make sure the buffers are freed on re-start
    SAFE_FREE(scene->ring_buf_renderer);
    SAFE_FREE(scene->ring_buf_mapper);
    SAFE_FREE(scene->frame_buffer.data);
    scene->image = NULL;
    SAFE_FREE(scene->accum);
    SAFE_FREE(scene->quant_errors_lut);

    // Initialize frame_buffer structure
    size_t image_alloc = (size_t)((scene->width * scene->height) * 4);
    scene->frame_buffer.dimensions.x = scene->width;
    scene->frame_buffer.dimensions.y = scene->height;
    scene->frame_buffer.row_stride = scene->width * 4;

    // Allocate aligned memory for frame buffer (16 byte aligned for vectorization)
    scene->frame_buffer.data = (RGBA*)aligned_alloc(16, image_alloc);
    memset(scene->frame_buffer.data, 0, image_alloc);

    // Compatibility shim - point legacy image pointer to frame_buffer data
    scene->image = (uint8_t*)scene->frame_buffer.data;

    /* Z-buffer is managed by scene3d_t now (allocated lazily per frame) */

    // bcm mapper ring, always allocate for RGBA
    if (!(scene->ring_buf_mapper = spsc_create(8, (size_t)(scene->width * scene->height * 4)))) {
        die("failed to create mapper ring buffer\n");
    }

    // each BCM "frame" is 32 bits (6 bits for addressing, 18 for RGB data)
    const uint16_t half_height = (uint16_t)(scene->height / 2);
    size_t ring_buf_size = (size_t)(scene->width * half_height * scene->bit_depth * 4);
    // bcm renderer ring
    if (!(scene->ring_buf_renderer = spsc_create(8, ring_buf_size))) {
        die("failed to create renderer ring buffer\n");
    }

    // allocate memory for the quantization error accumulator (3 beacuse we only have RGB (dont need alpha))
    scene->accum = (int32_t*)calloc((size_t)(scene->width * scene->height * 4), sizeof(int32_t));

    // allocate memory for LUT for input 8 bit RGB to 16 bit quant error value, forgot what 256 entries for R, G, B.
    // we could probably use a single LUT for all 3 channels since they are all the same...
    scene->quant_errors_lut = (uint16_t*)calloc(768*4, sizeof(uint16_t));

    // create RGB -> BCM mapper thread
    if (pthread_create(&scene->mapper_thread, NULL, main_thread_mapper, scene) != 0) {
        debug("failed to create mapper thread! (this is a show stopper)\n");
        scene->do_render = false;
    }

    // assign the global scene pointer for shutdown access
    g_scene = scene;

    // flag to indicate that the scene is ready for rendering
    scene->frame_ready = true;
}





/**
 * internal method for rendering on pi zero, 3 and 4
 */
/*
void* render_forever_pi4(const hub75_display_t *scene, int version) {

    uint32_t *PERIBase = map_gpio(version);
    if (version == 4) {
    	configure_gpio(PERIBase, 4);
    } else if (version == 3) {
    	configure_gpio(PERIBase, 3);
    }


     
    // index into the OE jitter mask
    uint32_t jitter_idx = 0;
    // pre compute some variables. let the compiler know the alignment for optimizations
    const uint16_t half_height __attribute__((aligned(16))) = (uint16_t)(scene->panel_height / 2u);
    const uint16_t width __attribute__((aligned(16))) = scene->width;
    const uint8_t  bit_depth __attribute__((aligned(BIT_DEPTH_ALIGNMENT))) = scene->bit_depth;

    uint32_t *bcm_signal = NULL;

    // create the OE jitter mask to control screen brightness
    // if we are using BCM brightness, then set OE to 0 (0 is display on ironically)
    uint32_t *jitter_mask = jitter_create(JITTER_SIZE, scene->brightness, scene->jitter_brightness);

    // store the row to address mapping in an array for faster access
    uint32_t addr_map[half_height*2];
    for (int i=0; i<half_height+1; i++) {
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
                    uint32_t new_mask = ((bcm_signal[offset]) | jitter_mask[jitter_idx]);
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
                    offset += bit_depth;// + 1;
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

    return NULL;
}
    */


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

#ifndef WARMUP_FRAMES
#define WARMUP_FRAMES 8
#endif

void* hub75_display_run_pi4(const hub75_display_t *scene) {

    int cpu_model = cpu_get_pi_model();

    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(cpu_model); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    configure_gpio(PERIBase, cpu_model);
         
    // index into the OE jitter mask
    // pre compute some variables. let the compiler know the alignment for optimizations
    const uint16_t half_height = (uint16_t)scene->panel_height / 2;
    const uint16_t width = scene->width;
    const uint8_t  bit_depth = scene->bit_depth;

    // pointer to the current bcm data to be displayed
    ASSERT(width % 16 == 0);
    ASSERT(half_height % 16 == 0);
    ASSERT(bit_depth % BIT_DEPTH_ALIGNMENT == 0);

    // create the OE jitter mask to control screen brightness
    // if we are using BCM brightness, then set OE to 0 (0 is display on ironically)
    uint32_t *jitter_mask = jitter_create(JITTER_SIZE, scene->brightness, scene->jitter_brightness);

    // store the row to address mapping in an array for faster access
    uint32_t addr_map[half_height];
    for (int i=0; i<half_height; i++) {
        addr_map[i] = row_to_address(i, half_height);
    }

    struct   timeval end_time, start_time;
    time_t   last_time_s = time(NULL);
    uint32_t frame_count = 0;
    uint64_t frame_total = 0;
    uint32_t last_addr   = 0;
    uint32_t color_pins  = 0;
    gettimeofday(&start_time, NULL);

    __attribute__((unused)) uint16_t phase = 1;         // phase is where we start pulling jitter bits from
    uint16_t jitter_idx = 0;    // jitter_mask has an extra 16K of bits in it, so overrun is ok
    const uint32_t *bcm_signal; // pointer to the current bcm data to be displayed

    debug(" [.] waiting for first frame acquisition...\n");

    while (scene->do_render) {
        bcm_signal = spsc_pop_ptr_begin(scene->ring_buf_renderer, 1000);
        if (bcm_signal != NULL) {
            break; // success, keep the first frame
        }
        debug(" [.] still waiting for first frame acquisition...\n");
    }
    if (bcm_signal == NULL) {
        debug(" [!] unable to locate first rendered frame\n");
        return NULL;
    }
    debug(" [$] first frame acquired\n");

    // Phase 1: SCHED_FIFO 99 + mlockall for warmup measurement
    enable_rt_fifo_and_lock_mem();

    // Measure actual frame time over warmup frames
    struct timespec ts_a, ts_b;
    uint64_t worst_frame_ns = 0;
    debug(" [.] measuring frame time (%d warmup frames)...\n", WARMUP_FRAMES);
    for (int wf = 0; wf < WARMUP_FRAMES && scene->do_render; wf++) {
        clock_gettime(CLOCK_MONOTONIC, &ts_a);
        uint32_t woff = 0;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {
            for (uint16_t y = 0; y < half_height; y++) {
                asm volatile ("" : : : "memory");
                PERIBase[7]  = addr_map[y] & ~last_addr;
                SLOW
                PERIBase[10] = ~addr_map[y] & last_addr;
                SLOW
                last_addr = addr_map[y];
                for (uint16_t x = 0; x < width; x++) {
                    asm volatile ("" : : : "memory");
                    uint32_t new_mask = bcm_signal[woff] | jitter_mask[jitter_idx];
                    PERIBase[10] = (~new_mask & color_pins) | PIN_CLK;
                    SLOW
                    PERIBase[7]  = (new_mask & ~color_pins);
                    SLOW SLOW SLOW
                    PERIBase[7]  = new_mask | PIN_CLK;
                    SLOW SLOW SLOW
                    color_pins = new_mask;
                    jitter_idx = (jitter_idx + 1) % JITTER_SIZE;
                    woff += bit_depth;
                }
                PERIBase[7] = PIN_LATCH | PIN_OE;
                SLOW SLOW
                PERIBase[10] = PIN_LATCH;
                SLOW SLOW
                PERIBase[10] = PIN_OE;
                SLOW
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_b);
        uint64_t elapsed_ns = (uint64_t)(ts_b.tv_sec - ts_a.tv_sec) * 1000000000ULL
                             + (uint64_t)(ts_b.tv_nsec - ts_a.tv_nsec);
        if (elapsed_ns > worst_frame_ns) worst_frame_ns = elapsed_ns;
        sched_yield();
    }
    debug(" [*] measured worst-case frame time: %luμs\n", (unsigned long)(worst_frame_ns / 1000));

    // Phase 2: attempt SCHED_DEADLINE with measured timing
    bool is_deadline = enable_deadline_from_measurement(worst_frame_ns, bit_depth, 3);

    while (scene->do_render) {

        uint32_t offset = 0;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {

            // check for a new frame, reset jitter phase once we hit the end
            if ((pwm & 15u) == 0u) {   // true at i = 0,16,32,...
                // Only swap to a new frame if there is at least one additional item
                // beyond the one we currently hold. This avoids re-popping the same slot
                // and releasing it too early while still in use.
                if (spsc_count(scene->ring_buf_renderer) >= 2) {
                    // release the current frame
                    // acquire the next frame (non-blocking)
                    const uint32_t *tmp = spsc_pop_ptr_begin(scene->ring_buf_renderer, 0);
                    if (tmp) {
                        spsc_pop_ptr_commit(scene->ring_buf_renderer);
                        bcm_signal = tmp;
                    }
                }
            }


            frame_count++;

            for (uint16_t y = 0; y < half_height; y++) {
                asm volatile ("" : : : "memory");  // Prevents optimization

                PERIBase[7]  = addr_map[y] & ~last_addr;
                SLOW
                PERIBase[10] = ~addr_map[y] & last_addr;
                SLOW
                last_addr    = addr_map[y];

                for (uint16_t x=0; x<width; x++) {
                    asm volatile ("" : : : "memory");  // Prevents optimization
                    uint32_t new_mask = ((bcm_signal[offset]) | jitter_mask[jitter_idx]);
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
                    offset += bit_depth;// + 1;
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

            // if using phase, uncomment this code to advance phase each row
            // phase += 8; if (phase >= JITTER_SIZE - (width + width)) { phase = 0; }
        }

        // End of one full BCM frame — yield to the scheduler.
        // With SCHED_DEADLINE the kernel reclaims only the unused portion of the
        // period, so the panel is driven nearly continuously with no artificial gap.
        // Without deadline, fall back to the old yield-every-other-frame approach.
        if (is_deadline) {
            sched_yield();
        } else if (frame_count & 1) {
            usleep(500);
            sched_yield();
        }

        // only hit the sys call after about 4.6 seconds or so (render speed should be about 3200Hz)
        if (frame_count > 15000) {
            time_t current_time_s = time(NULL);  // syscalls are slow, so avoid them when possible...
            if (UNLIKELY(current_time_s >= last_time_s + 5)) {
                if (scene->show_fps) {
                    gettimeofday(&end_time, NULL);
                    double elapsed = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec) * 1e-6;
                    float hz = (float)(frame_count) / (float)(elapsed);
                    float percent = (float)(hz) / 3220.0f;
                    debug(" [%2.2f%%] Panel Refresh Rate: %.1fHz\n", (double)(percent * 100.0f), (frame_count / elapsed));
                    gettimeofday(&start_time, NULL);
                }
                frame_total += frame_count;
                frame_count = 0;
                last_time_s = current_time_s;
            }
        }
    }

    debug(" [-] display run render loop exiting. [%ld] total frames rendered\n", (frame_total + frame_count));

    return NULL;
}


void* hub75_display_run(const hub75_display_t *scene) {

    int cpu_model = cpu_get_pi_model();
    debug(" [+] display_run CPU model %d: pinning to CPU 3\n", cpu_model);
    cpu_affinity(3);

    if (cpu_model < 2) {
        die(" [!] Unsupported CPU model detected %d\n", cpu_model);
    }

    if (cpu_model < 5 ) {
        return hub75_display_run_pi4(scene);
    }

    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(5); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    uint32_t *RIOBase;
    RIOBase = PERIBase + RIO5_OFFSET;
    configure_gpio(PERIBase, 5);
         
    // index into the OE jitter mask
    // pre compute some variables. let the compiler know the alignment for optimizations
    const uint16_t half_height = (uint16_t)scene->panel_height / 2;
    const uint16_t width = scene->width;
    const uint8_t  bit_depth = scene->bit_depth;

    // pointer to the current bcm data to be displayed
    ASSERT(width % 16 == 0);
    ASSERT(half_height % 16 == 0);
    ASSERT(bit_depth % BIT_DEPTH_ALIGNMENT == 0);

    // create the OE jitter mask to control screen brightness
    // if we are using BCM brightness, then set OE to 0 (0 is display on ironically)
    uint32_t *jitter_mask = jitter_create(JITTER_SIZE, scene->brightness, scene->jitter_brightness);

    // store the row to address mapping in an array for faster access
    uint32_t addr_map[half_height];
    for (int i=0; i<half_height; i++) {
        addr_map[i] = row_to_address(i, half_height);
    }

    struct   timeval end_time, start_time;
    time_t   last_time_s = time(NULL);
    uint32_t frame_count = 0;
    uint64_t frame_total = 0;
    gettimeofday(&start_time, NULL);


    // reg_out allows you to turn on bits, but will not turn bits off
    volatile uint32_t *const reg_out   = &rio->Out;
    // reg_set allows you to set the full state of the bits, on or off
    volatile uint32_t *const reg_set   = &rioSET->Out;
    // reg_clr allows you to only clear bits
    volatile uint32_t *const reg_clr   = &rioCLR->Out;


    __attribute__((unused)) uint16_t phase = 1;         // phase is where we start pulling jitter bits from
    uint16_t jitter_idx = 0;    // jitter_mask has an extra 16K of bits in it, so overrun is ok
    const uint32_t *bcm_signal; // pointer to the current bcm data to be displayed

    debug(" [.] waiting for first frame acquisition...\n");

    while (scene->do_render) {
        bcm_signal = spsc_pop_ptr_begin(scene->ring_buf_renderer, 1000);
        if (bcm_signal != NULL) {
            break; // success, keep the first frame
        }
        debug(" [.] still waiting for first frame acquisition...\n");
    }
    if (bcm_signal == NULL) {
        debug(" [!] unable to locate first rendered frame\n");
        return NULL;
    }
    debug(" [$] first frame acquired\n");

    // Phase 1: SCHED_FIFO 99 + mlockall for warmup measurement
    enable_rt_fifo_and_lock_mem();

    // ---- Warmup: measure actual frame time under FIFO 99 ----
    #ifndef WARMUP_FRAMES
    #define WARMUP_FRAMES 8
    #endif
    struct timespec ts_a, ts_b;
    uint64_t worst_frame_ns = 0;

    debug(" [.] measuring frame time (%d warmup frames)...\n", WARMUP_FRAMES);
    for (int wf = 0; wf < WARMUP_FRAMES && scene->do_render; wf++) {
        clock_gettime(CLOCK_MONOTONIC, &ts_a);

        uint32_t offset = 0;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {
            for (uint16_t y = 0; y < half_height; y++) {
                const uint32_t addr_bits = addr_map[y];
                uint32_t oe_addr = PIN_OE | addr_bits;
                *reg_out = oe_addr;
                io_store_barrier();
                jitter_idx = 0;

                for (uint16_t x = 0; x < width; x++) {
                    uint32_t v = bcm_signal[offset] | addr_bits;// | jitter_mask[jitter_idx];
                    *reg_out = v;
                    *reg_out = v | PIN_CLK;
                    offset++;
                    jitter_idx++;
                }
                __asm__ __volatile__("" ::: "memory");
                *reg_set = PIN_OE;
                SLOW2
                *reg_set = PIN_LATCH | PIN_OE;
                *reg_clr = PIN_LATCH;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_b);
        uint64_t elapsed_ns = (uint64_t)(ts_b.tv_sec - ts_a.tv_sec) * 1000000000ULL
                             + (uint64_t)(ts_b.tv_nsec - ts_a.tv_nsec);
        if (elapsed_ns > worst_frame_ns) worst_frame_ns = elapsed_ns;
        sched_yield();   // brief yield between warmup frames
    }
    debug(" [*] measured worst-case frame time: %luμs\n", (unsigned long)(worst_frame_ns / 1000));

    // Phase 2: attempt SCHED_DEADLINE with measured timing
    bool is_deadline = enable_deadline_from_measurement(worst_frame_ns, bit_depth, 3);

    // ---- Main display loop ----
    while (scene->do_render) {

        uint32_t offset = 0;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {

            // check for a new frame, reset jitter phase once we hit the end
            if ((pwm & 15u) == 0u) {   // true at i = 0,16,32,...
                if (spsc_count(scene->ring_buf_renderer) >= 2) {
                    const uint32_t *tmp = spsc_pop_ptr_begin(scene->ring_buf_renderer, 0);
                    if (tmp) {
                        spsc_pop_ptr_commit(scene->ring_buf_renderer);
                        bcm_signal = tmp;
                    }
                }
            }

            frame_count++;

            for (uint16_t y = 0; y < half_height; y++) {

                const uint32_t addr_bits = addr_map[y];

                uint32_t oe_addr = PIN_OE | addr_bits;
                *reg_out = oe_addr;
                io_store_barrier();

                jitter_idx = 0;

                for (uint16_t x = 0; x < width; x++) {

                    uint32_t v = bcm_signal[offset] | addr_bits;// | jitter_mask[jitter_idx];
                    *reg_out = v;                 // set data + addr + oe, clk low
                    *reg_out = v | PIN_CLK;       // clk high

                    offset ++;
                    jitter_idx++;
                }

                __asm__ __volatile__("" ::: "memory");

                *reg_set = PIN_OE;
                SLOW2
                *reg_set = PIN_LATCH | PIN_OE;
                *reg_clr = PIN_LATCH;
            }
        }

        // End of full BCM frame — yield to scheduler.
        // Deadline: kernel reclaims only unused portion of the period.
        // FIFO on isolated CPU: yield is a near-no-op (returns immediately),
        // but prevents soft lockup warnings on long runs.
        sched_yield();

        if (frame_count > 15000) {
            time_t current_time_s = time(NULL);
            if (UNLIKELY(current_time_s >= last_time_s + 5)) {
                if (scene->show_fps) {
                    gettimeofday(&end_time, NULL);
                    double elapsed = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec) * 1e-6;
                    float hz = (float)(frame_count) / (float)(elapsed);
                    float percent = (float)(hz) / 3220.0f;
                    debug(" [%2.2f%%] Panel Refresh Rate: %.1fHz\n", (double)(percent * 100.0f), (frame_count / elapsed));
                    gettimeofday(&start_time, NULL);
                }
                frame_total += frame_count;
                frame_count = 0;
                last_time_s = current_time_s;
            }
        }
    }

    debug(" [-] display run render loop exiting. [%ld] total frames rendered\n", (frame_total + frame_count));

    return NULL;
}


