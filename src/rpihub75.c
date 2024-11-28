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
uint32_t row_to_address(const int y, uint8_t half_height) {

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
    if (CONSOLE_DEBUG) {
        printf("ports: %d, chains: %d, width: %d, height: %d, stride: %d, bit_depth: %d\n", 
            scene->num_ports, scene->num_chains, scene->width, scene->height, scene->stride, scene->bit_depth);
    }
    if (scene->num_ports > 3) {
        die("Only 3 port supported at this time\n");
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



image_mapper_t flip_mapper;
uint8_t *flip_mapper(uint8_t *image, uint8_t *image_out, const struct scene_info *scene) {

    uint16_t row_sz = scene->width * scene->stride;

    uint8_t *temp_row = malloc(row_sz);

    for (uint16_t y=0; y < scene->height / 2; y++) {
        uint8_t *top_row = image + y * row_sz;
        uint8_t *bottom_row = image + (scene->height - y - 1) * row_sz;

        // Swap the rows using the temp_row buffer
        memcpy(temp_row, top_row, row_sz);        // Copy top row to temp buffer
        memcpy(top_row, bottom_row, row_sz);      // Copy bottom row to top row
        memcpy(bottom_row, temp_row, row_sz);     // Copy temp buffer (original top row) to bottom row
    }

    return image;
}


image_mapper_t u_mirror_impl;
uint8_t *mirror_mapper(uint8_t *image, uint8_t *image_out, const struct scene_info *scene) {

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

uint8_t *mirror_flip_mapper(uint8_t *image, uint8_t *image_out, const struct scene_info *scene) {

    int row_size = scene->width * scene->stride; // Each row has 'width' pixels, 3 bytes per pixel (R, G, B)
    uint8_t temp_pixel[3];    // Temporary storage for a single pixel (3 bytes: R, G, B)

    // Iterate through the top half of the image
    for (int y = 0; y < scene->height / 2; y++) {
        uint8_t *top_row = image + y * row_size;
        uint8_t *bottom_row = image + (scene->height - y - 1) * row_size;

        // Swap pixels from left to right within both the top and bottom rows
        for (int x = 0; x < scene->width; x++) {
            int left_index = x * scene->stride;
            int right_index = (scene->width - x - 1) * scene->stride;

            // Swap top-left pixel with bottom-right pixel (3 bytes: R, G, B)
            memcpy(temp_pixel, &top_row[left_index], scene->stride);                  // Store top-left pixel
            memcpy(&top_row[left_index], &bottom_row[right_index], scene->stride);    // Move bottom-right to top-left
            memcpy(&bottom_row[right_index], temp_pixel, scene->stride);              // Move temp (top-left) to bottom-right
        }
    }
    return image;
}


/**
 * internal method for rendering on pi zero, 3 and 4
 */
void render_forever_pi4(const scene_info *scene, int version) {

    srand(time(NULL));
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
    uint16_t jitter_idx = 0;
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
    while(scene->do_render) {

        // iterate over the bit plane
        for (uint8_t pwm=0; pwm<bit_depth; pwm++) {
            time_t current_time_s = time(NULL);
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

            if (UNLIKELY(current_time_s >= last_time_s + 5)) {

                if (scene->show_fps) {
                    printf("Panel Refresh Rate: %dHz\n", frame_count / 5);
                }
                frame_count = 0;
                last_time_s = current_time_s;
            }
        }
    }
}


/**
 * @brief you can cause render_forever to exit by updating the value of do_hub65_render pointer
 * EG:
 * 
 * 
 */
void render_forever(const scene_info *scene) {

    pid_t pid = getpid();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);

    if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) != 0) {
	    die("unable to set CPU affinity to 3\n");
    }

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
        if (strstr(line, "Pi 5") == NULL) {
            cpu_model = 5;
            break;
        }
        if (strstr(line, "Pi 4") == NULL) {
            cpu_model = 4;
            break;
        }
        if (strstr(line, "Pi 3") == NULL) {
            cpu_model = 3;
            break;
        }
    }
    free(line);
    fclose(file);
    if (cpu_model == 0) die("Only Pi5, Pi4 and Pi3 are currently supported");
    if (cpu_model < 5 ) {
        render_forever_pi4(scene, cpu_model);
    }
    // This is Pi 5
    srand(time(NULL));
    // map the gpio address to we can control the GPIO pins
    uint32_t *PERIBase = map_gpio(0, 5); // for root on pi5 (/dev/mem, offset is 0xD0000)
    // offset to the RIO registers (required for #define register access. 
    // TODO: this needs to be improved and #define to RIOBase removed)
    uint32_t *RIOBase;
    RIOBase = PERIBase + RIO5_OFFSET;
    configure_gpio(PERIBase, 5);
         
    // index into the OE jitter mask
    uint16_t jitter_idx = 0;
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
    // uint32_t addr_pins     = 0;
    // uint32_t color_pins    = 0;


    // uint8_t bright = scene->brightness;
    while(scene->do_render) {

        // iterate over the bit plane
        //PRE_TIME;
        for (uint8_t pwm=0; pwm<bit_depth; pwm++) {
            time_t current_time_s = time(NULL);
            frame_count++;
            // for the current bit plane, render the entire frame
            uint32_t offset = pwm;
            for (uint16_t y=0; y<half_height; y++) {
                asm volatile ("" : : : "memory");  // Prevents optimization

                // compute the bcm row start address for y
                // uint32_t offset = ((y * scene->width ) * bit_depth) + pwm;

                for (uint16_t x=0; x<width; x++) {
                    asm volatile ("" : : : "memory");  // Prevents optimization
                    // set all bits in 1 op. RGB data, current row address and the OE jitter mask (brightness control)
                    rio->Out = bcm_signal[offset] | addr_map[y] | jitter_mask[jitter_idx];

                    // SLOW2
                    // toggle clock pin high
                    rioSET->Out = PIN_CLK;

                    // advance the global OE jitter mask 1 frame
                    jitter_idx = (jitter_idx + 1) % JITTER_SIZE;

                    // advance to the next pixel in the bcm signal
                    offset += bit_depth + 1;
                }
                // make sure enable pin is high (display off) while we are latching data
                // latch the data for the entire row
                rioSET->Out = PIN_OE | PIN_LATCH;
                SLOW2
                rioCLR->Out = PIN_LATCH;
            }

            // swap the buffers on vsync
            if (UNLIKELY(scene->bcm_ptr != last_pointer)) {
                last_pointer = scene->bcm_ptr;
                bcm_signal = (last_pointer) ? scene->bcm_signalB : scene->bcm_signalA;
            }

            if (UNLIKELY(current_time_s >= last_time_s + 5)) {
                if (scene->show_fps) {
                    printf("Panel Refresh Rate: %dHz\n", frame_count / 5);
                }
                frame_count = 0;
                last_time_s = current_time_s;
            }
        }

    }
}


