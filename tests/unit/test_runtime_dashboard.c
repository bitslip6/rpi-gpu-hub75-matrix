#include "runtime_dashboard_metrics.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void close_to(float actual, float expected) {
    assert(fabsf(actual - expected) < .001f);
}

int main(void) {
    runtime_demo_cpu_window window = {0};
    close_to(runtime_demo_cpu_average(&window, 0, 0), 0);
    /* Tickless accounting alternates between undercount and catch-up. */
    close_to(runtime_demo_cpu_average(&window, 45000000, 500000000), 9);
    close_to(runtime_demo_cpu_average(&window, 1065000000, 500000000), 111);
    runtime_demo_cpu_average(&window, 45000000, 500000000);
    close_to(runtime_demo_cpu_average(&window, 1065000000, 500000000), 111);
    close_to(runtime_demo_cpu_average(&window, 45000000, 500000000), 111);
    close_to(runtime_demo_cpu_average(&window, 1065000000, 500000000), 111);
    assert(window.count == RUNTIME_DEMO_CPU_SAMPLES);

    /* Old samples expire; genuine workload changes are not hidden forever. */
    for (unsigned i = 0; i < RUNTIME_DEMO_CPU_SAMPLES; ++i)
        runtime_demo_cpu_average(&window, 0, 500000000);
    close_to(runtime_demo_cpu_average(&window, 0, 0), 0);

    window = (runtime_demo_cpu_window){0};
    close_to(runtime_demo_cpu_average(&window, 1500000000, 1500000000), 100);
    close_to(runtime_demo_cpu_average(&window, 1000000000, 500000000), 125);
    /* A short final update must carry its actual duration, not equal weight. */
    close_to(runtime_demo_cpu_average(&window, 0, 500000000), 100);
    unsigned next = window.next;
    close_to(runtime_demo_cpu_average(&window, 123, 0), 100);
    assert(window.next == next);

    window = (runtime_demo_cpu_window){0};
    close_to(runtime_demo_cpu_average(&window, 2000000000, 500000000), 400);
    assert(RUNTIME_DEMO_REFRESH_NS == UINT64_C(500000000));
    puts("runtime dashboard metrics: OK");
    return 0;
}
