#ifndef H75_RUNTIME_DASHBOARD_METRICS_H
#define H75_RUNTIME_DASHBOARD_METRICS_H

#include <stdint.h>

#define RUNTIME_DEMO_REFRESH_NS UINT64_C(500000000)

/* Four half-second intervals: a small, time-weighted CPU rolling average.
 * Sum durations rather than averaging percentages so delayed/partial updates
 * have the correct weight. This smooths tickless accounting; it does not
 * replace the underlying process CPU clock or clamp multicore usage to 100%. */
#define RUNTIME_DEMO_CPU_SAMPLES 4u
typedef struct {
    uint64_t cpu_ns[RUNTIME_DEMO_CPU_SAMPLES];
    uint64_t wall_ns[RUNTIME_DEMO_CPU_SAMPLES];
    unsigned next, count;
} runtime_demo_cpu_window;

static inline float runtime_demo_cpu_average(runtime_demo_cpu_window *window,
                                             uint64_t cpu_ns, uint64_t wall_ns) {
    if (wall_ns) {
        window->cpu_ns[window->next] = cpu_ns;
        window->wall_ns[window->next] = wall_ns;
        window->next = (window->next + 1) % RUNTIME_DEMO_CPU_SAMPLES;
        if (window->count < RUNTIME_DEMO_CPU_SAMPLES) ++window->count;
    }
    uint64_t cpu = 0, wall = 0;
    for (unsigned i = 0; i < window->count; ++i) {
        cpu += window->cpu_ns[i];
        wall += window->wall_ns[i];
    }
    return wall ? 100.f * (float)cpu / (float)wall : 0;
}

#endif
