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


#define MEMGUARD_OVERRIDE_STDLIB
#include "memguard2.h"

#include "rpihub75.h"
#include "util.h"
#include "pixels.h"


/**
 * @brief calculate an address line pin mask for row y
 * not used outside this file
 * @param y the panel row number to calculate the mask for
 * @return uint32_t the bitmask for the address lines at row y
 */
uint32_t row_to_address(int y, uint16_t half_height) {
    // normalize y into [0, half_height) safely (accept negative defensively)
    int norm = y - 1;
    if (norm < 0) norm = 0;
    // use modulo on int then cast (avoid implicit narrowing warning)
    int mod = (half_height > 0) ? (norm % (int)half_height) : 0;
    uint16_t row = (uint16_t)mod;
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
void start_scene(scene_info *scene) {
    debug("ports: %d, chains: %d, width: %d, height: %d, stride: %d, bit_depth: %d\n", 
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

    // create  buffers
    const size_t buffer_size = (size_t)(scene->width + 1) * (scene->height + 1) * 3 * scene->bit_depth;
    scene->bcm_frame_size = buffer_size;

    scene->bcm_signalA = aligned_alloc(16, buffer_size * 4);
    scene->bcm_signalB = aligned_alloc(16, buffer_size * 4);

    /*
    // force the buffers to be 16 byte aligned to improve auto vectorization
    scene->bcm_buffers = aligned_alloc(16, buffer_size * 4 * BCM_BUFFERS);
    scene->image = aligned_alloc(16, scene->width * scene->height * 4); // make sure we always have enough for RGBA

    // ------------------------------------------------------------------
    // Initialize SPSC frame ring (dst_ctx) used between mapper (producer)
    // and render_forever() (consumer). The ring capacity MUST be a power
    // of two. BCM_BUFFERS is 3 (not a power of two) so we pick 4 here.
    // We over-allocated above ( * 4 * BCM_BUFFERS ) so we have plenty.
    // ------------------------------------------------------------------
    const size_t frame_ring_capacity = 4; // power-of-two
    if (!spsc_frame_init(&scene->dst_ctx,
                         scene->bcm_buffers,           // base pointer
                         frame_ring_capacity,
                         scene->bcm_frame_size)) {
        die("Failed to init dst_ctx frame ring (capacity=%zu)\n", frame_ring_capacity);
    }
    debug("dst_ctx initialized: capacity=%zu frame_size=%zu bytes\n",
          frame_ring_capacity, scene->bcm_frame_size);

    // render thread will be spawned lazily by mapper thread (or externally) so zero IDs
    scene->render_thread = 0;
    scene->mapper_thread = 0;
    scene->gpu_thread    = 0;
    */
}





/**
 * internal method for rendering on pi zero, 3 and 4
 */
void render_forever_pi4(const scene_info *scene, int version) {

    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(version); // for root on pi5 (/dev/mem, offset is 0xD0000)
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
    const uint16_t half_height __attribute__((aligned(16))) = (uint16_t)(scene->panel_height / 2u);
    const uint16_t width __attribute__((aligned(16))) = scene->width;
    const uint8_t  bit_depth __attribute__((aligned(BIT_DEPTH_ALIGNMENT))) = scene->bit_depth;

    // pointer to the current bcm data to be displayed
    uint32_t *bcm_signal = scene->bcm_signalA;
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

/**
 * @brief request graceful render loop to shutdown
 * @param scene 
 */
void render_loop_shutdown(struct scene_info *scene) {
    if (!scene) {
        printf("unable to shutdown render loop, no scene provided\n");
        return;
    }
    printf("shutting down render loop...\n");
    scene->do_render = false;
    update_bcm_signal_64_rgb(scene, NULL, NULL, NULL, NULL, 0);
}

// Graceful shutdown helpers --------------------------------------------------
void hub75_request_shutdown(struct scene_info *scene) {
    if (!scene) {
        printf("unable to request shutdown, no scene provided\n");
        return;
    }

    scene->do_render = false;
    update_bcm_signal_64_rgb(scene, NULL, NULL, NULL, NULL, 0);

    hub75_wait_shutdown(scene);
}

void hub75_wait_shutdown(struct scene_info *scene) {
    if (!scene) {
        printf("unable to wiat shutdown, no scene provided\n");
        return;
    } 
    printf("waiting for mapper to stop...\n");
    // mapper
    if (scene->mapper_thread) {
        pthread_t t = scene->mapper_thread;
        scene->mapper_thread = 0;
        pthread_join(t, NULL);
    }
    printf("waiting for render to stop...\n");
    // render
    if (scene->render_thread) {
        pthread_t t = scene->render_thread;
        scene->render_thread = 0;
        pthread_join(t, NULL);
    }
    printf("all threads completed\n");
    // gpu/update thread join happens in example.c (main) because it owns that thread handle
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
    struct sched_param sp = { .sched_priority = 30 }; /* 1..99 */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);


    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        fprintf(stderr, " * Try running as root to enable real-time scheduling\n");
    } else {
        debug(" * Real-time scheduling enabled\n");
    }
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        fprintf(stderr, " * Try running as root to enable memory-time locking to reduce page faults\n");
    }
}

void cpu_affinity(int cpu) {
    cpu_set_t cs; 
    CPU_ZERO(&cs); 
    CPU_SET(cpu, &cs);

    int result = sched_setaffinity(0, sizeof(cs), &cs);
    if (result == -1) {
        die("sched_setaffinity failed: %s\n", strerror(errno));
    }
}


/**
 * @brief you can cause render_forever to exit by updating the value of do_render pointer
 * EG:
 * scene->do_render = false; // will cause render_forever to exit from another thread
 * 
 */
void render_forever(const scene_info *scene) {
    // enable_rt_and_lock_mem();
    cpu_affinity(3); // pin only this thread
    printf("render forever on CPU 3...\n");

    /* pin only this thread */
    //cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(3, &cs);
    //pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);

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
    // free(line);
    fclose(file);

    debug("\ncpu_model: %d\n", cpu_model);

    if (cpu_model == 0) die("Only Pi5, Pi4, Pi3 and Pi Zero 2 are currently supported");

    if (cpu_model < 5 ) {
        render_forever_pi4(scene, cpu_model);
    }
    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(5); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    uint32_t *RIOBase;
    RIOBase = PERIBase + RIO5_OFFSET;
    configure_gpio(PERIBase, 5);
         
    // index into the OE jitter mask
    uint32_t jitter_idx = 0;
    // pre compute some variables. let the compiler know the alignment for optimizations
    const uint16_t half_height = (uint16_t)scene->panel_height / 2;
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
    volatile uint32_t *const reg_out   = &rio->Out;
    volatile uint32_t *const reg_set   = &rioSET->Out;
    volatile uint32_t *const reg_clr   = &rioCLR->Out;
    const uint32_t stride      = (uint32_t)bit_depth;// + 1;  /* next-pixel offset */



    //const uint32_t guard_px = 4;   /* do not change OE in first/last N pixels of a row */

    uint16_t phase = 1;
    printf("while\n");
    while (scene->do_render) {
        sleep(1);
    }
    return;
    

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

                    // v = data + addr + oe
                    const uint32_t v = bcm_signal[offset] | addr_bits | jitter_mask[jitter_idx];
                    *reg_out = v;                 // set data + addr + oe, clk low
                    *reg_out = v | PIN_CLK;       // clk high 

                    /* advance after full edge */
                    jitter_idx = (jitter_idx + 1) % JITTER_SIZE;
                    offset += stride;
               }

                // latch the complete row into the display
                *reg_set = PIN_LATCH | PIN_OE;     // latch high
                *reg_clr = PIN_LATCH;              // latch low
            }

            // swap the buffer when it changes
            unsigned before = atomic_load_explicit(&scene->frame_ready, memory_order_acquire);
            if (!(before & 1u)) {
                last_pointer = scene->bcm_ptr;
                bcm_signal = (last_pointer) ? scene->bcm_signalB : scene->bcm_signalA;
            }
        }

        sched_yield(); // should be ~50hz for 3 panel chain, 150hz for 1 panel chain
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


