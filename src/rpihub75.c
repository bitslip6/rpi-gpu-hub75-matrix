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
#include <sys/mman.h>
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
#include "weighted_rio.h"
#include "gpio_lock.h"


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

    /* Request only. Joining here can self-join a render worker. */
}

void hub75_display_wait(struct hub75_display *scene) {
    if (!scene) {
        printf("unable to wiat shutdown, no scene provided\n");
        return;
    } 
    debug(" [.] waiting for mapper thread to stop...\n");
    // mapper
    if (scene->mapper_thread && !pthread_equal(scene->mapper_thread, pthread_self())) {
        pthread_t t = scene->mapper_thread;
        scene->mapper_thread = 0;
        pthread_join(t, NULL);
    }
    debug(" [$] mapper thread exited and joined\n");
    debug(" [.] waiting for render thread to stop...\n");
    // render
    if (scene->render_thread && !pthread_equal(scene->render_thread, pthread_self())) {
        pthread_t t = scene->render_thread;
        scene->render_thread = 0;
        pthread_join(t, NULL);
    }
    debug(" [$] render thread exited and joined\n");
}


extern hub75_display_t *g_scene;
/**
 * @brief verify that the scene configuration is valid
 * will die() if invalid configuration is found
 * @param scene 
 */
void hub75_display_bind_scene(hub75_display_t *scene) {
    size_t weighted_words = 0;
    if (scene->output_mode == H75_OUTPUT_WEIGHTED_BCM) {
        if (cpu_get_pi_model() != 5)
            die("Weighted BCM requires Pi 5 RIO; no fallback is supported\n");
        h75_weighted_lut lut;
        if (h75_weighted_word_count(scene, &weighted_words) != H75_OK ||
            h75_weighted_lut_build(scene, &lut) != H75_OK || !scene->pwm_lsb_ns ||
            scene->width % 16 || scene->panel_width % 16 ||
            weighted_words * sizeof(uint32_t) > SPSC_MAX_ITEM_SIZE ||
            (size_t)scene->width * scene->height * 4 > SPSC_MAX_ITEM_SIZE)
            die("Invalid weighted BCM configuration/topology\n");
    } else if (scene->output_mode != H75_OUTPUT_EQUAL_SLOT) {
        die("Unknown output mode\n");
    }
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
    const uint16_t half_height = (uint16_t)(scene->panel_height / 2);
    size_t ring_buf_size = scene->output_mode == H75_OUTPUT_WEIGHTED_BCM
        ? weighted_words * sizeof(uint32_t)
        : (size_t)scene->width * half_height * scene->bit_depth * sizeof(uint32_t);
    // bcm renderer ring
    if (!(scene->ring_buf_renderer = spsc_create(8, ring_buf_size))) {
        die("failed to create renderer ring buffer\n");
    }

    if (scene->output_mode == H75_OUTPUT_EQUAL_SLOT) {
        // Legacy-only quantization/dithering state.
        scene->accum = (int32_t*)calloc((size_t)(scene->width * scene->height * 4), sizeof(int32_t));
        scene->quant_errors_lut = (uint16_t*)calloc(768*4, sizeof(uint16_t));
    }

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

static void legacy_scan_release(uint32_t *map,const hub75_display_t *scene,bool pi5,
                                uint32_t *jitter,const uint32_t *held) {
    if(pi5)map[RIO5_OFFSET+0x2000/4]=PIN_OE;
    else map[7]=PIN_OE;
    io_write_barrier();
    if(held)spsc_pop_ptr_commit(scene->ring_buf_renderer);
    free(jitter);munmap(map,64*1024*1024);
}

/* Same weighted encoder, SPSC boundary handoff and prepared scanner as the
 * embedding runtime; only entry-point lifecycle/reporting differs. */
static void *legacy_weighted_run(const hub75_display_t *scene, int model) {
    if (model != 5 || !scene->ring_buf_renderer) {
        fprintf(stderr, "weighted output requires Pi 5 RIO and a weighted frame queue\n");
        return NULL;
    }
    int lock_fd = h75_gpio_lock_acquire();
    if (lock_fd < 0) {
        fprintf(stderr, "cannot acquire GPIO advisory lock for weighted output\n");
        return NULL;
    }
    uint32_t *map = map_gpio(5);
    configure_gpio(map, 5);
    /* Blank before schedule building or any allocation/validation failure. */
    map[RIO5_OFFSET + 0x2000 / 4] = PIN_OE;
    io_write_barrier();
    h75_weighted_rio scan = {0};
    uint32_t addresses[32];
    const unsigned rows = scene->panel_height / 2u;
    if (rows != 16 && rows != 32) {
        legacy_scan_release(map, scene, true, NULL, NULL);
        close(lock_fd);
        return NULL;
    }
    for (unsigned row = 0; row < rows; row++)
        addresses[row] = h75_weighted_row_to_address((int)row, (uint16_t)rows);
    h75_status status = h75_weighted_rio_init(&scan, scene, model,
        map + RIO5_OFFSET, addresses, &scene->do_render, true);
    const uint32_t *held = NULL;
    if (status != H75_OK)
        goto done;
    uint64_t completed_cycles = 0;
    struct timespec report_start;
    clock_gettime(CLOCK_MONOTONIC, &report_start);
    while (atomic_load(&scene->do_render)) {
        status = h75_weighted_rio_frame_begin(&scan, scene->ring_buf_renderer, &held, 20);
        if (status == H75_TIMEOUT)
            continue;
        if (status != H75_OK)
            break;
        status = h75_weighted_rio_cycle(&scan);
        if (status != H75_OK)
            break;
        /* A refresh statistic is one full weighted cycle, repeats included. */
        completed_cycles++;
        if (scene->show_fps && completed_cycles >= 128) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            const double seconds = (double)(now.tv_sec - report_start.tv_sec) +
                (double)(now.tv_nsec - report_start.tv_nsec) * 1e-9;
            if (seconds > 0.0)
                debug("Weighted full-cycle refresh: %.1fHz\n", (double)completed_cycles / seconds);
            completed_cycles = 0;
            report_start = now;
        }
    }
done:
    if (status != H75_OK && status != H75_CANCELLED)
        fprintf(stderr, "weighted RIO scanner failed (%d)\n", status);
    h75_weighted_rio_destroy(&scan);
    legacy_scan_release(map, scene, true, NULL, held);
    close(lock_fd);
    return NULL;
}

#ifndef WARMUP_FRAMES
#define WARMUP_FRAMES 8
#endif

void* hub75_display_run_pi4(const hub75_display_t *scene) {

    int cpu_model = cpu_get_pi_model();
    if (scene->output_mode == H75_OUTPUT_WEIGHTED_BCM)
        return legacy_weighted_run(scene, cpu_model);

    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(cpu_model); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    configure_gpio(PERIBase, cpu_model);
    PERIBase[7]=PIN_OE;
         
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
    uint32_t jitter_idx = 0;    // jitter_mask has an extra 16K of bits in it, so overrun is ok
    const uint32_t *bcm_signal = NULL; // pointer to the current bcm data to be displayed

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
        legacy_scan_release(PERIBase,scene,false,jitter_mask,NULL);
        return NULL;
    }
    debug(" [$] first frame acquired\n");

    // Legacy host tuning is explicitly opt-in, never embedding policy.
    if(getenv("HUB75_LEGACY_RT"))enable_rt_fifo_and_lock_mem();

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
                    woff++;
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
    bool is_deadline = getenv("HUB75_LEGACY_RT") &&
        enable_deadline_from_measurement(worst_frame_ns, bit_depth, 3);

    while (scene->do_render) {

        uint32_t offset = 0;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {

            /* Swap only at a full BCM boundary. The current slot remains
             * reserved for its entire scan lifetime, including repeated scans. */
            if (pwm == 0 && spsc_count(scene->ring_buf_renderer) >= 2) {
                spsc_pop_ptr_commit(scene->ring_buf_renderer);
                bcm_signal = spsc_pop_ptr_begin(scene->ring_buf_renderer, 0);
                /* Single consumer: the already published next item must exist. */
                if (!bcm_signal) break;
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
                    offset++;
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

    legacy_scan_release(PERIBase,scene,false,jitter_mask,bcm_signal);
    return NULL;
}


void* hub75_display_run(const hub75_display_t *scene) {

    int cpu_model = cpu_get_pi_model();
    debug(" [+] display_run CPU model %d: pinning to CPU 3\n", cpu_model);
    if (getenv("HUB75_LEGACY_RT")) cpu_affinity(3);

    if (scene->output_mode == H75_OUTPUT_WEIGHTED_BCM)
        return legacy_weighted_run(scene, cpu_model);

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
    PERIBase[RIO5_OFFSET+0x2000/4]=PIN_OE;
         
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
    const uint32_t *bcm_signal = NULL; // pointer to the current bcm data to be displayed

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
        legacy_scan_release(PERIBase,scene,true,jitter_mask,NULL);
        return NULL;
    }
    debug(" [$] first frame acquired\n");

    // Legacy host tuning is explicitly opt-in, never embedding policy.
    if(getenv("HUB75_LEGACY_RT"))enable_rt_fifo_and_lock_mem();

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
                    uint32_t v = bcm_signal[offset] | addr_bits | jitter_mask[jitter_idx];
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
    if (getenv("HUB75_LEGACY_RT"))
        (void)enable_deadline_from_measurement(worst_frame_ns, bit_depth, 3);

    // ---- Main display loop ----
    while (scene->do_render) {

        uint32_t offset = 0;
        for (uint8_t pwm = 0; pwm < bit_depth; pwm++) {

            /* Release the old reservation BEFORE acquiring the next tail;
             * never read a slot after returning it to the producer. */
            if (pwm == 0 && spsc_count(scene->ring_buf_renderer) >= 2) {
                spsc_pop_ptr_commit(scene->ring_buf_renderer);
                bcm_signal = spsc_pop_ptr_begin(scene->ring_buf_renderer, 0);
                if (!bcm_signal) break;
            }

            frame_count++;

            for (uint16_t y = 0; y < half_height; y++) {

                const uint32_t addr_bits = addr_map[y];

                uint32_t oe_addr = PIN_OE | addr_bits;
                *reg_out = oe_addr;
                io_store_barrier();

                jitter_idx = 0;

                for (uint16_t x = 0; x < width; x++) {

                    uint32_t v = bcm_signal[offset] | addr_bits | jitter_mask[jitter_idx];
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

    legacy_scan_release(PERIBase,scene,true,jitter_mask,bcm_signal);
    return NULL;
}


