/**
 * Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "parallel_out.pio.h"

// Pi5 GPIO memory mapped registers
#define PERI5_BASE 0x1f000D0000UL
#define PAD5_OFFSET  (0x20000 / 4)

#define DATA_WORDS 196608  // 196608 words = 6,291,456 bits
#define NUM_GPIO_PINS 27   // GPIO 0-26

// Simple tight loop helper for polling - prevents compiler over-optimization
#define tight_loop_contents() __asm volatile("" ::: "memory")

// Global variables for signal handler
static volatile bool running = true;
static PIO cleanup_pio = NULL;
static int cleanup_sm = -1;
static uint cleanup_offset = 0;

// Set fast slew rate for a GPIO pin via memory-mapped PAD registers
int set_gpio_fast_slew(uint gpio) {
    int mem_fd;
    volatile uint32_t *peri_base;
    volatile uint32_t *pad_base;

    mem_fd = open("/dev/gpiomem0", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        printf("Warning: Could not open /dev/gpiomem0: %s\n", strerror(errno));
        printf("Run with sudo or add user to gpio group for slew rate control\n");
        return -1;
    }

    peri_base = (volatile uint32_t *)mmap(NULL, 0x30000, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, mem_fd, 0);
    if (peri_base == MAP_FAILED) {
        printf("Warning: mmap failed: %s\n", strerror(errno));
        close(mem_fd);
        return -1;
    }

    pad_base = peri_base + PAD5_OFFSET + 1;  // +1 offset as per working code

    // Set fast slew rate (0x15) with strong drive
    // Bit layout (based on working example):
    // - 0x01 = slow slew, low drive
    // - 0x15 = fast slew, strong drive
    pad_base[gpio] = 0x15;

    munmap((void *)peri_base, 0x30000);
    close(mem_fd);

    printf("Set GPIO %d to fast slew rate (0x15)\n", gpio);
    return 0;
}

// Signal handler for Ctrl-C
void signal_handler(int signum) {
    printf("\nReceived signal %d, cleaning up...\n", signum);
    running = false;

    if (cleanup_pio != NULL && cleanup_sm >= 0) {
        // Disable the state machine
        pio_sm_set_enabled(cleanup_pio, cleanup_sm, false);

        // Clear FIFOs
        pio_sm_clear_fifos(cleanup_pio, cleanup_sm);

        // Unclaim the state machine
        pio_sm_unclaim(cleanup_pio, cleanup_sm);

        // Remove the program
        pio_remove_program(cleanup_pio, &parallel_out_program, cleanup_offset);

        printf("PIO resources released\n");
    }

    exit(0);
}

int main(int argc, const char **argv) {
    uint32_t databuf[DATA_WORDS];
    bool use_dma = true;
    int ret = 0;
    int i, j;

    (void)argc;  // Unused
    (void)argv;  // Unused

    stdio_init_all();

    // Register signal handler for Ctrl-C
    signal(SIGINT, signal_handler);

    PIO pio = pio0;

    // Startup sequence: Stop any currently running state machines
    printf("Stopping any running PIO state machines...\n");
    for (int sm_num = 0; sm_num < 4; sm_num++) {
        // Disable the state machine
        pio_sm_set_enabled(pio, sm_num, false);

        // Clear FIFOs
        pio_sm_clear_fifos(pio, sm_num);

        // Restart the state machine (resets PC to wrap_target)
        pio_sm_restart(pio, sm_num);
    }
    printf("All state machines stopped and reset\n");

    int sm = pio_claim_unused_sm(pio, true);
    uint offset = pio_add_program(pio, &parallel_out_program);

    // Store for cleanup
    cleanup_pio = pio;
    cleanup_sm = sm;
    cleanup_offset = offset;

    printf("Loaded program at %d, using sm %d\n", offset, sm);
    printf("Configuring GPIO 0-26 for parallel output:\n");
    printf("  - Pins: 27 GPIO pins (0-26)\n");
    printf("  - Drive strength: 12mA\n");
    printf("  - Slew rate: FAST (0x15)\n");

    // 256 byte transfer size (should probably be 4K)
    pio_sm_config_xfer(pio, sm, PIO_DIR_TO_SM, 256, 1);

    // Initialize all GPIO pins 0-26 for PIO output
    for (uint gpio = 0; gpio < NUM_GPIO_PINS; gpio++) {
        pio_gpio_init(pio, gpio);

        // Set maximum drive strength for faster slew rate
        gpio_set_drive_strength(gpio, GPIO_DRIVE_STRENGTH_12MA);

        // Disable pulls for cleaner signal
        gpio_disable_pulls(gpio);

        // Set fast slew rate via direct PAD register access
        set_gpio_fast_slew(gpio);
    }

    // Set all pins as outputs
    pio_sm_set_consecutive_pindirs(pio, sm, 0, NUM_GPIO_PINS, true);

    // Configure state machine for parallel output
    pio_sm_config c = parallel_out_program_get_default_config(offset);
    sm_config_set_sideset_pins(&c, 27);  // Sideset on pin 27 (clock output)
    sm_config_set_out_pins(&c, 0, NUM_GPIO_PINS);  // Output on pins 0-26
    sm_config_set_clkdiv(&c, 16.0);

    printf("\nPIO Configuration:\n");
    printf("  OUT pins: base=0, count=%d (GPIO 0-%d)\n", NUM_GPIO_PINS, NUM_GPIO_PINS-1);
    printf("  Sideset pin: GPIO 27\n");
    printf("  Clock divider: 16.0\n");
    printf("  Expected update rate: %.2f MHz\n\n", 200.0 / 16.0 / 2.0);

    pio_sm_init(pio, sm, offset, &c);
    pio_sm_set_enabled(pio, sm, true);
    printf("State machine enabled and running\n");

    // Initialize buffer with a test pattern that's easy to see on a scope
    // First 32 words: all LOW
    for (i = 0; i < 32; i++) {
	    databuf[i] = 0x00000000;
    }
    // Next 32 words: all HIGH (bits 0-26)
    for (i = 32; i < 64; i++) {
	    databuf[i] = 0x07FFFFFF;  // Set bits 0-26 HIGH
    }
    // Rest: alternating between all HIGH and all LOW
    for (i = 64; i < DATA_WORDS; i++) {
        if (i % 2 == 0)
	        databuf[i] = 0x07FFFFFF;  // All pins HIGH
        else
	        databuf[i] = 0x00000000;  // All pins LOW
    }

    printf("\nBuffer pattern:\n");
    printf("  Words 0-31:    All pins LOW  (0x00000000)\n");
    printf("  Words 32-63:   All pins HIGH (0x07FFFFFF)\n");
    printf("  Words 64+:     Alternating HIGH/LOW\n");
    printf("  Test on GPIO 24: Should toggle HIGH/LOW\n");
    printf("  Test on GPIO 27: Sideset pin (clock), always LOW in this program\n");

    printf("\nStarting continuous parallel DMA output of %d words\n", DATA_WORDS);
    printf("Each word updates all 27 GPIO pins (0-26) simultaneously\n");
    printf("At clkdiv=16.0, update rate ~%.2f MHz\n", 200.0 / 16.0 / 2.0);

    // Debug: First test direct GPIO control (bypass PIO program)
    printf("\nDEBUG Phase 1: Testing direct pin control via pio_sm_set_pins...\n");
    printf("This bypasses the PIO program to test if GPIO control works at all.\n");
    pio_sm_set_enabled(pio, sm, false);  // Disable SM temporarily
    for (int test = 0; test < 5; test++) {
        uint32_t pin_values = test % 2 ? 0x07FFFFFF : 0x00000000;
        pio_sm_set_pins(pio, sm, pin_values);
        sleep_ms(200);
        printf("  Set pins to: 0x%08X - GPIO 24 should be %s\n",
               pin_values, test % 2 ? "HIGH" : "LOW");
    }
    printf("DEBUG Phase 1 complete. Did GPIO 24 toggle?\n\n");
    sleep_ms(1000);

    // Re-enable and test via PIO program
    pio_sm_set_enabled(pio, sm, true);
    printf("DEBUG Phase 2: Testing via PIO program (FIFO writes)...\n");
    for (int test = 0; test < 10; test++) {
        pio_sm_put_blocking(pio, sm, test % 2 ? 0x07FFFFFF : 0x00000000);
        sleep_ms(100);  // 100ms between toggles
        printf("  Sent to FIFO: 0x%08X (should %s all pins 0-26)\n",
               test % 2 ? 0x07FFFFFF : 0x00000000,
               test % 2 ? "SET" : "CLEAR");
    }
    printf("DEBUG Phase 2 complete. Check if GPIO 24 toggled.\n");
    printf("Press Ctrl-C to stop, or continuing with DMA in 2 seconds...\n");
    sleep_ms(2000);

    int iteration = 0;
    int consecutive_errors = 0;
    const int MAX_CONSECUTIVE_ERRORS = 3;
    const int CHUNK_SIZE = 64;  // Transfer 64 words at a time (256 bytes)

    while (running) {
        // Transfer the buffer in smaller chunks
        for (int chunk = 0; chunk < DATA_WORDS && running; chunk += CHUNK_SIZE) {
            int words_to_send = (chunk + CHUNK_SIZE > DATA_WORDS) ? (DATA_WORDS - chunk) : CHUNK_SIZE;
            int bytes_to_send = words_to_send * sizeof(databuf[0]);

            // Wait for PIO FIFO to have at least some space (not completely full)
            int fifo_level;
            int wait_count = 0;
            while (running && (fifo_level = pio_sm_get_tx_fifo_level(pio, sm)) >= 8) {
                tight_loop_contents();
                wait_count++;
                if (wait_count > 1000000) {  // Much higher threshold - only warn if truly stuck
                    printf("Warning: FIFO stuck at level %d for extended period\n", fifo_level);
                    wait_count = 0;
                }
            }

            if (!running) break;

            ret = pio_sm_xfer_data(pio, sm, PIO_DIR_TO_SM, bytes_to_send, &databuf[chunk]);
            if (ret) {
                consecutive_errors++;
                printf("\n=== DMA Error ===\n");
                printf("Error code: %d at iteration %d, chunk %d/%d\n", ret, iteration, chunk/CHUNK_SIZE, DATA_WORDS/CHUNK_SIZE);
                printf("TX FIFO level: %d\n", pio_sm_get_tx_fifo_level(pio, sm));
                printf("RX FIFO level: %d\n", pio_sm_get_rx_fifo_level(pio, sm));
                printf("Consecutive errors: %d\n", consecutive_errors);
                printf("Bytes to send: %d\n", bytes_to_send);

                if (consecutive_errors >= MAX_CONSECUTIVE_ERRORS) {
                    printf("Too many consecutive errors, exiting\n");
                    break;
                }

                printf("Attempting recovery...\n");

                // Disable and re-enable the state machine
                pio_sm_set_enabled(pio, sm, false);
                sleep_ms(10);

                // Clear the FIFOs
                pio_sm_clear_fifos(pio, sm);

                // Restart and re-enable
                pio_sm_restart(pio, sm);
                pio_sm_set_enabled(pio, sm, true);
                sleep_ms(10);

                printf("Recovery complete, retrying transfer...\n");

                // Retry the transfer
                ret = pio_sm_xfer_data(pio, sm, PIO_DIR_TO_SM, bytes_to_send, &databuf[chunk]);
                if (ret) {
                    printf("Retry failed: %d\n", ret);
                    printf("Will try again next iteration\n");
                    break;  // Break out of chunk loop
                } else {
                    printf("Retry successful!\n");
                    consecutive_errors = 0;  // Reset error counter on success
                }
            } else {
                // Success - reset error counter
                consecutive_errors = 0;
            }
        }

        // Completed full buffer
        iteration++;
        if (iteration % 100 == 0) {
            printf("Completed %d iterations (FIFO level: %d)\n", iteration, pio_sm_get_tx_fifo_level(pio, sm));
        }
    }

    printf("Exiting cleanly after %d iterations\n", iteration);
    signal_handler(SIGINT);

    printf("not hit\n");

    for (i = 0; i < DATA_WORDS; i++) {
        printf("Iter %d:\n", i);
        pio_sm_put_blocking(pio, sm, i);
        if (use_dma) {
            ret = pio_sm_xfer_data(pio, sm, PIO_DIR_FROM_SM, (i + 1) * sizeof(databuf[0]), databuf);
            if (ret)
               break;

            for (j = i; j >= 0; j--)
            {
                int v = databuf[i - j];
                if (v != j)
                    printf(" %d: %d\n", j, v);
            }
        } else {
            for (j = i; j >= 0; j--)
            {
                int v = pio_sm_get_blocking(pio, sm);
                if (v != j)
                    printf(" %d: %d\n", j, v);
            }
        }
        sleep_ms(10);
    }

    if (ret)
        printf("* error %d\n", ret);
    return ret;
}
