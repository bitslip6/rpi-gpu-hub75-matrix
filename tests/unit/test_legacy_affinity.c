#define _GNU_SOURCE
#include "fake_gpio_affinity.h"
#include "hub75gpu.h"
#include "util.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static hub75_display_t *display;
static atomic_uint configured;
static unsigned mapped, released;
static int model, lock_fd;
static bool stop_on_config = true;
int cpu_get_pi_model(void) { return model; }
uint32_t *map_gpio(int version) {
    (void)version;
    assert(h75_fake_cpu3());
    mapped++;
    void *p = mmap(NULL, 64*1024*1024, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    return p;
}
void configure_gpio(uint32_t *map, int version) {
    (void)map; (void)version;
    assert(h75_fake_cpu3());
    configured++;
    /* Stop-before-first-frame: exercise real dispatch/setup/cleanup without
     * executing pixel kernels or requiring a producer. */
    if (stop_on_config) atomic_store(&display->do_render, false);
}
int h75_gpio_lock_acquire(void) {
    assert(h75_fake_cpu3());
    lock_fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
    assert(lock_fd >= 0);
    return lock_fd;
}
int __real_munmap(void *address, size_t size);
int __wrap_munmap(void *address, size_t size) {
    assert(h75_fake_cpu3());
    released++;
    return __real_munmap(address, size);
}
static void reset(void) {
    h75_fake_affinity_reset();
    configured = mapped = released = 0;
    lock_fd = -1;
    stop_on_config = true;
    atomic_store(&display->do_render, true);
}
static void *run(bool direct) {
    return direct ? hub75_display_run_pi4(display) : hub75_display_run(display);
}
static void *producer(void *unused) {
    (void)unused;
    assert(h75_fake_original());
    return NULL;
}
static void *stop_later(void *unused) {
    (void)unused;
    assert(h75_fake_original()); /* worker was created before scanner pin */
    struct timespec poll = {0, 1000000};
    while (!atomic_load(&configured)) nanosleep(&poll, NULL);
    /* Wait for the actual held-frame acquisition, not elapsed setup time
     * (legacy jitter creation may take longer on loaded/sanitized hosts). */
    int available = 1;
    while (available) {
        assert(!sem_getvalue(&display->ring_buf_renderer->can_pop, &available));
        if (available) nanosleep(&poll, NULL);
    }
    struct timespec delay = {0, 20000000};
    nanosleep(&delay, NULL);
    atomic_store(&display->do_render, false);
    return NULL;
}
int main(void) {
    unsetenv("HUB75_LEGACY_RT");
    display = hub75_display_new();
    assert(display);
    display->width = 32; display->height = 32;
    display->panel_width = 32; display->panel_height = 32;
    display->bit_depth = 32;
    display->num_ports = 1; display->num_chains = 1;
    for (unsigned weighted = 0; weighted < 2; weighted++) {
        if (display->ring_buf_renderer) spsc_destroy(display->ring_buf_renderer);
        display->ring_buf_renderer = spsc_create(2, 32 * 16 *
            (weighted ? display->pwm_bits : display->bit_depth) * sizeof(uint32_t));
        assert(display->ring_buf_renderer);
        display->output_mode = weighted ? H75_OUTPUT_WEIGHTED_BCM : H75_OUTPUT_EQUAL_SLOT;
        for (unsigned direct = 0; direct < 2; direct++) {
            for (model = weighted ? 5 : 3; model <= 5; model++) {
                reset();
                pthread_t worker;
                assert(!pthread_create(&worker, NULL, producer, NULL));
                assert(!pthread_join(worker, NULL));
                assert(!run(direct != 0));
                assert(configured == 1 && mapped == 1 && released == 1);
                assert(atomic_load(&h75_fake_sets) == 2); /* pin + restore, not nested pin */
                assert(h75_fake_original());
                if (lock_fd >= 0) assert(fcntl(lock_fd, F_GETFD) == -1 && errno == EBADF);
                /* Also run real scan loops against RAM with an immutable held
                 * frame, stopping from a producer created before the pin. */
                reset(); stop_on_config = false;
                uint32_t *frame = spsc_push_ptr_begin(display->ring_buf_renderer, 0);
                assert(frame);
                memset(frame, 0, display->ring_buf_renderer->item_size);
                spsc_push_ptr_commit(display->ring_buf_renderer);
                assert(!pthread_create(&worker, NULL, stop_later, NULL));
                assert(!run(direct != 0));
                assert(!pthread_join(worker, NULL));
                assert(mapped == 1 && released == 1 && h75_fake_original());
                assert(!spsc_count(display->ring_buf_renderer));
                assert(atomic_load(&h75_fake_sets) == 2);
                if (lock_fd >= 0) assert(fcntl(lock_fd, F_GETFD) == -1 && errno == EBADF);
            }
            /* Save, set, verification-get and mismatched-mask failures. */
            for (unsigned failure = 0; failure < 4; failure++) {
                reset();
                if (failure == 0) atomic_store(&h75_fake_fail_get_at, 1);
                if (failure == 1) h75_test_affinity_fail_set(EINVAL);
                if (failure == 2) atomic_store(&h75_fake_fail_get_at, 2);
                /* Wrong mask on pin readback, not the original-mask save. */
                if (failure == 3) atomic_store(&h75_fake_wrong_at, 2);
                assert(run(direct != 0)); /* example maps this to EXIT_FAILURE */
                assert(!atomic_load(&display->do_render));
                assert(!mapped && !configured && !released);
                assert(h75_fake_original());
            }
        }
    }
    /* Restore failure must reach stdout and the void* status adapter after
     * scanner cleanup. No exit(), and scene stop is visible to producers. */
    for (unsigned weighted = 0; weighted < 2; weighted++) {
        display->output_mode = weighted ? H75_OUTPUT_WEIGHTED_BCM : H75_OUTPUT_EQUAL_SLOT;
        for (unsigned direct = 0; direct < 2; direct++) {
            for (unsigned failure = 0; failure < 3; failure++) {
                reset(); model = 5;
                if (failure == 0) atomic_store(&h75_fake_fail_set_at, 2);
                if (failure == 1) atomic_store(&h75_fake_fail_get_at, 3);
                if (failure == 2) atomic_store(&h75_fake_wrong_at, 3);
                int pipefd[2]; assert(!pipe(pipefd));
                fflush(stdout);
                int saved = dup(STDOUT_FILENO); assert(saved >= 0);
                assert(dup2(pipefd[1], STDOUT_FILENO) >= 0); close(pipefd[1]);
                assert(run(direct != 0));
                assert(dup2(saved, STDOUT_FILENO) >= 0); close(saved);
                char output[1024] = {0};
                assert(read(pipefd[0], output, sizeof(output)-1) > 0); close(pipefd[0]);
                assert(strstr(output, "mandatory GPIO CPU3 affinity") && strstr(output, "restoring"));
                assert(released == 1 && !atomic_load(&display->do_render));
                if (lock_fd >= 0) assert(fcntl(lock_fd, F_GETFD) == -1 && errno == EBADF);
            }
        }
    }
    reset(); atomic_store(&display->do_render, false);
    assert(!run(false));
    assert(!atomic_load(&h75_fake_sets) && !atomic_load(&h75_fake_gets) && !mapped);
    spsc_destroy(display->ring_buf_renderer);
    free(display);
    puts("legacy affinity lifecycle tests passed (RAM/fake syscalls only)");
    return 0;
}
