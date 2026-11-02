#define _GNU_SOURCE
#include "gpio_affinity.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef H75_AFFINITY_TESTING
#define SET_AFFINITY h75_fake_setaffinity
#define GET_AFFINITY h75_fake_getaffinity
#else
#define SET_AFFINITY pthread_setaffinity_np
#define GET_AFFINITY pthread_getaffinity_np
#endif

enum { GPIO_CPU = 3 };

static int failure(char *message, size_t size, const char *operation, int rc) {
    if (message && size)
        snprintf(message, size, "mandatory GPIO CPU3 affinity: %s: %s; "
                 "check CPU3 is online and permitted by the session cpuset/cgroup",
                 operation, strerror(rc));
    return rc;
}

int h75_gpio_affinity_save(cpu_set_t *mask, char *message, size_t size) {
    int rc = GET_AFFINITY(pthread_self(), sizeof(*mask), mask);
    return rc ? failure(message, size, "saving caller mask", rc) : 0;
}

static int set_verified(const cpu_set_t *mask, char *message, size_t size,
                        const char *operation) {
    int rc = SET_AFFINITY(pthread_self(), sizeof(*mask), mask);
    if (rc)
        return failure(message, size, operation, rc);
    cpu_set_t actual;
    CPU_ZERO(&actual);
    rc = GET_AFFINITY(pthread_self(), sizeof(actual), &actual);
    if (rc)
        return failure(message, size, operation, rc);
    if (!CPU_EQUAL(mask, &actual))
        return failure(message, size, operation, EIO);
    return 0;
}

int h75_gpio_affinity_require(char *message, size_t size) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(GPIO_CPU, &mask);
    return set_verified(&mask, message, size, "setting/verifying scanner mask exactly {3}");
}

int h75_gpio_affinity_restore(const cpu_set_t *mask, char *message, size_t size) {
    return set_verified(mask, message, size, "restoring/verifying caller mask");
}
