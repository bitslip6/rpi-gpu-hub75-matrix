#ifndef LOWLEVEL_H
#define LOWLEVEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/mman.h>
#include <bits/cpu-set.h>

/**
 * @brief attempt to enable real-time scheduling and lock memory to reduce page faults
 * @return true if real-time scheduling was enabled
 * @return false if real-time scheduling was not enabled
 */
bool enable_rt_and_lock_mem(void);

/**
 * @brief set the CPU affinity of the current thread to the given CPU
 * 
 * @param cpu the CPU to set the affinity to
 */
void cpu_affinity(unsigned int cpu);

/**
 * @brief pin the current thread (self) to the given CPU
 */
int cpu_pin_thread(size_t cpu_id);

/**
 * @brief get the Raspberry Pi CPU model
 * @return int - CPU model number (e.g., 3 for Pi 3, 4 for Pi 4, 5 for Pi 5) 
 * return 0 if unknown, -1 if /proc/cpuinfo cannot be read
 */
int cpu_get_pi_model(void);

/**
 * @brief Phase 1: enable SCHED_FIFO 99 + mlockall for warmup measurement frames
 */
bool enable_rt_fifo_and_lock_mem(void);

/**
 * @brief Phase 2: switch to SCHED_DEADLINE using measured frame time.
 * On isolated CPUs, stays on SCHED_FIFO 99 (no deadline overhead needed).
 * On non-isolated CPUs, tries progressively lower utilization targets.
 *
 * @param measured_runtime_ns  worst-case measured frame time in nanoseconds
 * @param bit_depth            BCM bit depth
 * @param display_cpu          CPU the display thread is pinned to
 * @return true if SCHED_DEADLINE was activated
 */
bool enable_deadline_from_measurement(uint64_t measured_runtime_ns, uint8_t bit_depth,
                                      unsigned int display_cpu);

/**
 * @brief Legacy single-call wrapper (prefers two-phase approach above)
 */
bool enable_deadline_scheduler(uint16_t width, uint16_t half_height,
                               uint8_t bit_depth, int pi_model);

/**
 * @brief install signal handlers for graceful shutdown on SIGINT/SIGTERM
 *
 */
void signal_handler_install(void);

#endif // LOWLEVEL_H