#define _GNU_SOURCE
#include "fake_gpio_affinity.h"
#include <assert.h>
#include <errno.h>

atomic_uint h75_fake_sets, h75_fake_gets, h75_fake_caller_sets;
atomic_int h75_fake_set_error, h75_fake_get_error;
atomic_uint h75_fake_fail_get_at, h75_fake_fail_set_at, h75_fake_wrong_at;
atomic_bool h75_fake_wrong_mask;
static pthread_t caller;
static _Thread_local cpu_set_t current;
static _Thread_local bool initialized;

static void initial_mask(void) {
    CPU_ZERO(&current);
    CPU_SET(0, &current); CPU_SET(1, &current); CPU_SET(2, &current);
    initialized = true;
}
void h75_fake_affinity_reset(void) {
    caller = pthread_self();
    initial_mask();
    atomic_store(&h75_fake_sets, 0);
    atomic_store(&h75_fake_gets, 0);
    atomic_store(&h75_fake_caller_sets, 0);
    atomic_store(&h75_fake_set_error, 0);
    atomic_store(&h75_fake_get_error, 0);
    atomic_store(&h75_fake_fail_get_at, 0);
    atomic_store(&h75_fake_fail_set_at, 0);
    atomic_store(&h75_fake_wrong_mask, false);
    atomic_store(&h75_fake_wrong_at, 0);
}
void h75_test_affinity_fail_set(int rc) {
    atomic_store(&h75_fake_set_error, rc);
}
bool h75_fake_cpu3(void) {
    return initialized && CPU_COUNT(&current) == 1 && CPU_ISSET(3, &current);
}
bool h75_fake_original(void) {
    if (!initialized) initial_mask();
    return CPU_COUNT(&current) == 3 && CPU_ISSET(0, &current) &&
        CPU_ISSET(1, &current) && CPU_ISSET(2, &current);
}
int h75_fake_setaffinity(pthread_t thread, size_t size, const cpu_set_t *mask) {
    assert(pthread_equal(thread, pthread_self()));
    assert(size == sizeof(*mask));
    unsigned n = atomic_fetch_add(&h75_fake_sets, 1) + 1;
    if (pthread_equal(thread, caller))
        atomic_fetch_add(&h75_fake_caller_sets, 1);
    int rc = atomic_exchange(&h75_fake_set_error, 0);
    if (rc) return rc;
    if (n == atomic_load(&h75_fake_fail_set_at)) return EPERM;
    /* Both mandatory singleton and exact saved original are legitimate. */
    assert((CPU_COUNT(mask) == 1 && CPU_ISSET(3, mask)) ||
           (CPU_COUNT(mask) == 3 && CPU_ISSET(0, mask) && CPU_ISSET(1, mask) && CPU_ISSET(2, mask)));
    current = *mask;
    initialized = true;
    return 0;
}
int h75_fake_getaffinity(pthread_t thread, size_t size, cpu_set_t *mask) {
    assert(pthread_equal(thread, pthread_self()));
    assert(size == sizeof(*mask));
    unsigned n = atomic_fetch_add(&h75_fake_gets, 1) + 1;
    int rc = atomic_exchange(&h75_fake_get_error, 0);
    if (rc) return rc;
    if (n == atomic_load(&h75_fake_fail_get_at)) return EACCES;
    if (!initialized) initial_mask();
    *mask = current;
    if (atomic_exchange(&h75_fake_wrong_mask, false) || n == atomic_load(&h75_fake_wrong_at)) {
        if (CPU_ISSET(3, mask)) CPU_SET(2, mask);
        else CPU_CLR(0, mask);
    }
    return 0;
}
