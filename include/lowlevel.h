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
 * @brief install signal handlers for graceful shutdown on SIGINT/SIGTERM
 * 
 */
void signal_handler_install(void);

#endif // LOWLEVEL_H