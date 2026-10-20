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
#include <sys/syscall.h>
#include <linux/sched.h>
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
        debug(" [_] Try running as root to enable real-time scheduling\n");
    } else {
        debug(" [^] Real-time scheduling enabled\n");
        is_realtime = true;
    }
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        // debug(" * Try running as root to enable memory locking to reduce page faults\n");
    }

    return is_realtime;
}

/* forward declaration — defined below */
void cpu_affinity(unsigned int cpu);

/* ---- SCHED_DEADLINE support ---- */

#ifndef SCHED_DEADLINE
#define SCHED_DEADLINE 6
#endif

/*
 * The kernel's struct sched_attr — not always exposed by glibc headers,
 * but this toolchain does provide it.  We use our own name to avoid
 * potential redefinition errors across distros.
 */
struct hub75_sched_attr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;    /* ns — max CPU time per period   */
    uint64_t sched_deadline;   /* ns — relative deadline          */
    uint64_t sched_period;     /* ns — period of the task         */
};

/**
 * @brief check if the given CPU is in the kernel's isolated set (isolcpus=).
 */
static bool cpu_is_isolated(unsigned int cpu) {
    FILE *f = fopen("/sys/devices/system/cpu/isolated", "r");
    if (!f) return false;
    char buf[64] = {0};
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return false; }
    fclose(f);
    // parse a simple cpulist like "3" or "2-3" or "1,3"
    unsigned int a, b;
    char *p = buf;
    while (*p) {
        if (sscanf(p, "%u-%u", &a, &b) == 2) {
            if (cpu >= a && cpu <= b) return true;
        } else if (sscanf(p, "%u", &a) == 1) {
            if (cpu == a) return true;
        }
        // advance past this entry
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return false;
}

/**
 * @brief try to set the CPU frequency governor to "performance" for the given CPU.
 * Eliminates frequency scaling jitter during display output.
 */
static void cpu_set_performance_governor(unsigned int cpu) {
    char path[80];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_governor", cpu);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "performance\n");
        fclose(f);
        debug(" [^] CPU %u governor set to performance\n", cpu);
    }
}

/**
 * @brief try to raise sched_rt_runtime_us to allow higher RT/deadline utilization.
 * Setting to -1 disables the RT throttling safety net entirely.
 */
static void try_raise_rt_runtime(void) {
    FILE *f = fopen("/proc/sys/kernel/sched_rt_runtime_us", "w");
    if (f) {
        fprintf(f, "-1\n");
        fclose(f);
        debug(" [^] sched_rt_runtime_us set to -1 (no RT throttling)\n");
    } else {
        debug(" [_] could not raise sched_rt_runtime_us: %s\n", strerror(errno));
    }
}

/**
 * @brief Phase 1: enable SCHED_FIFO 99, lock memory, tune system for RT.
 *
 * Also applies system-level tuning when running as root:
 * - Sets CPU frequency governor to "performance" on the display CPU
 * - Raises sched_rt_runtime_us to -1 (disable RT throttling)
 */
bool enable_rt_fifo_and_lock_mem(void) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        debug(" [_] mlockall failed: %s\n", strerror(errno));
    }

    // System tuning (best-effort, requires root)
    cpu_set_performance_governor(3);
    try_raise_rt_runtime();

    struct sched_param sp = { .sched_priority = 99 };
    if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) {
        debug(" [^] SCHED_FIFO 99 enabled (warmup phase)\n");
        return true;
    }
    debug(" [_] SCHED_FIFO 99 failed (%s), trying SCHED_RR 70\n", strerror(errno));
    enable_rt_and_lock_mem();
    return false;
}

/**
 * @brief Phase 2: switch to SCHED_DEADLINE using measured frame time,
 * or stay on SCHED_FIFO 99 if the CPU is isolated.
 *
 * When isolcpus= includes the display CPU, SCHED_FIFO 99 is actually
 * preferable: no other user tasks can run on that CPU, so the deadline
 * scheduler's bandwidth accounting just wastes cycles enforcing idle
 * gaps against interference that doesn't exist.  We skip deadline in
 * that case and let FIFO 99 run at full utilization.
 *
 * When the CPU is NOT isolated, SCHED_DEADLINE provides protection
 * against other tasks stealing CPU time mid-frame.
 *
 * @param measured_runtime_ns  worst-case measured frame time in nanoseconds
 * @param bit_depth            BCM bit depth (for Hz estimate in log)
 * @param display_cpu          the CPU the display thread is pinned to
 * @return true   if SCHED_DEADLINE was activated
 * @return false  if we stay on SCHED_FIFO 99
 */
bool enable_deadline_from_measurement(uint64_t measured_runtime_ns, uint8_t bit_depth,
                                      unsigned int display_cpu) {

    bool isolated = cpu_is_isolated(display_cpu);
    if (isolated) {
        debug(" [^] CPU %u is isolated — staying on SCHED_FIFO 99 (no deadline overhead)\n",
              display_cpu);
        return false;
    }

    debug(" [!] CPU %u is NOT isolated — using SCHED_DEADLINE for protection\n", display_cpu);

    // Add 5% safety margin to the measured worst-case
    uint64_t runtime_ns = measured_runtime_ns + measured_runtime_ns / 20;

    // Try utilization targets from tight to loose.
    // With sched_rt_runtime_us=-1 (set in phase 1), even 95%+ may be admitted.
    static const uint8_t util_pct[] = { 95, 93, 90, 85, 80 };
    const int n_attempts = (int)(sizeof(util_pct) / sizeof(util_pct[0]));

    for (int attempt = 0; attempt < n_attempts; attempt++) {
        uint64_t period_ns   = runtime_ns * 100 / util_pct[attempt];
        uint64_t deadline_ns = period_ns;

        if (period_ns < 50000)       period_ns = 50000;
        if (period_ns > 100000000)   period_ns = 100000000;
        if (deadline_ns > period_ns) deadline_ns = period_ns;

        debug(" [>] SCHED_DEADLINE attempt %d/%d: measured=%luμs runtime=%luμs  period=%luμs  util=%d%%\n",
              attempt + 1, n_attempts,
              (unsigned long)(measured_runtime_ns / 1000),
              (unsigned long)(runtime_ns / 1000),
              (unsigned long)(period_ns / 1000),
              (int)util_pct[attempt]);

        // Must drop from SCHED_FIFO to SCHED_OTHER before setting SCHED_DEADLINE
        struct sched_param sp_other = { .sched_priority = 0 };
        sched_setscheduler(0, SCHED_OTHER, &sp_other);

        struct hub75_sched_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.size           = sizeof(attr);
        attr.sched_policy   = SCHED_DEADLINE;
        attr.sched_runtime  = runtime_ns;
        attr.sched_deadline = deadline_ns;
        attr.sched_period   = period_ns;

        long ret = syscall(SYS_sched_setattr, 0, &attr, 0u);
        if (ret == 0) {
            uint64_t est_hz = (uint64_t)bit_depth * 1000000000ULL / period_ns;
            debug(" [^] SCHED_DEADLINE enabled (runtime=%luμs, period=%luμs, ~%luHz)\n",
                  (unsigned long)(runtime_ns / 1000),
                  (unsigned long)(period_ns / 1000),
                  (unsigned long)est_hz);
            return true;
        }
        debug(" [_] attempt %d failed: %s\n", attempt + 1, strerror(errno));

        // Restore SCHED_FIFO for next attempt (and as fallback)
        struct sched_param sp_fifo = { .sched_priority = 99 };
        sched_setscheduler(0, SCHED_FIFO, &sp_fifo);
    }

    debug(" [_] SCHED_DEADLINE not admitted, staying on SCHED_FIFO 99\n");
    return false;
}

/**
 * @brief Legacy wrapper (deprecated — use two-phase API)
 */
bool enable_deadline_scheduler(uint16_t width, uint16_t half_height,
                               uint8_t bit_depth, int pi_model) {
    (void)width; (void)half_height; (void)bit_depth; (void)pi_model;
    return enable_rt_fifo_and_lock_mem() ? false : false;
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
hub75_display_t *g_scene = NULL;

/**
 * @brief signal handler for graceful shutdown on SIGINT/SIGTERM
 * 
 * @param sig signal number 
 */
static void signal_handler(int sig) {
    debug(" [@] Signal %d received, shutting down g_scene: [%p]...\n", sig, (void *)g_scene); 

    if (g_scene != NULL) {
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
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}