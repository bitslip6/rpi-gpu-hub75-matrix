#define _GNU_SOURCE
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <stddef.h>
#include <sys/param.h>
#include <sys/mman.h>
#include <bits/cpu-set.h>

#include "util.h"

/**
 * @brief attempt to enable real-time scheduling and lock memory to reduce page faults
 * @return true if real-time scheduling was enabled
 * @return false if real-time scheduling was not enabled
 */
bool enable_rt_and_lock_mem(void) {
    struct sched_param sp = { .sched_priority = 70 }; /* 1..99 */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);

    bool is_realtime = false;
    if (sched_setscheduler(0, SCHED_RR, &sp) != 0) {
        debug(" * Try running as root to enable real-time scheduling\n");
    } else {
        debug(" * Real-time scheduling enabled\n");
        is_realtime = true;
    }
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        debug(" * Try running as root to enable memory locking to reduce page faults\n");
    }

    return is_realtime;
}

/**
 * @brief set the CPU affinity of the current thread to the given CPU
 * 
 * @param cpu the CPU to set the affinity to
 */
void cpu_affinity(unsigned int cpu) {
    cpu_set_t cs; 
    CPU_ZERO(&cs); 
    CPU_SET(cpu, &cs);

    int result = sched_setaffinity(0, sizeof(cs), &cs);
    if (result == -1) {
        die("sched_setaffinity failed: %s\n", strerror(errno));
    }
}


/**
 * @brief pin the current thread (self) to the given CPU
 */
int cpu_pin_thread(size_t cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);

    pthread_t self = pthread_self();
    int rc = pthread_setaffinity_np(self, sizeof(set), &set);
    if (rc != 0) {
        fprintf(stderr, "pthread_setaffinity_np failed: %s\n", strerror(rc));
        return -1;
    }
    return 0;
}

/**
 * @brief check the CPU model by reading /proc/cpuinfo
 * 
 * @return int CPU model number (3,4,5), 0 if model could not be determined
 *  return -1 if /proc/cpuinfo could not be opened
 */
int cpu_get_pi_model(void) {
    // check the CPU model to determine which GPIO function to use
    // note one cannot use file_get_contents as this file is zero length...
    char *line = NULL;
    size_t line_sz;
    int cpu_model = 0;
    FILE *file = fopen("/proc/cpuinfo", "rb");
    if (file == NULL) {
        debug("Could not open file /proc/cpuinfo\n");
        return -1;
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
 
    fclose(file);

    return cpu_model;
}

// --------------- Graceful Shutdown Handling ---------------------

static int g_sigpipe[2] = {-1, -1};
scene_info *g_scene = NULL;

/**
 * @brief signal handler for graceful shutdown on SIGINT/SIGTERM
 * 
 * @param sig signal number 
 */
static void signal_handler(int sig) {
    debug(" [@] Signal %d received, shutting down g_scene: [%p]...\n", sig, (void *)g_scene); 

    if (g_scene != nullptr) {
        g_scene->do_render = false;
    }
    if (g_sigpipe[1] != -1) {
        uint8_t b = (uint8_t)sig;
        (void)!write(g_sigpipe[1], &b, 1); // async-signal-safe
    }
}

/**
 * @brief install the signal handler for SIGINT/SIGTERM
 */
void signal_handler_install(void) {

    if (pipe(g_sigpipe) == -1) {
        // best-effort; still usable without the pipe
        g_sigpipe[0] = g_sigpipe[1] = -1;
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}