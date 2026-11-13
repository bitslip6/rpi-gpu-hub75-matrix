#define _GNU_SOURCE
#include "fake_gpio_affinity.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    char message[512];
    cpu_set_t original;
    h75_fake_affinity_reset();
    assert(!h75_gpio_affinity_save(&original, message, sizeof(message)));
    assert(!h75_gpio_affinity_require(message, sizeof(message)));
    assert(h75_fake_cpu3());
    assert(!h75_gpio_affinity_restore(&original, message, sizeof(message)));
    assert(h75_fake_original());
    h75_test_affinity_fail_set(EPERM);
    errno = ENOMEM; /* pthread rc, not unrelated errno, must be reported. */
    assert(h75_gpio_affinity_require(message, sizeof(message)) == EPERM);
    assert(strstr(message, strerror(EPERM)) && strstr(message, "mandatory GPIO CPU3"));
    atomic_store(&h75_fake_get_error, EINVAL);
    assert(h75_gpio_affinity_require(message, sizeof(message)) == EINVAL);
    atomic_store(&h75_fake_wrong_mask, true);
    assert(h75_gpio_affinity_require(message, sizeof(message)) == EIO);
    puts("GPIO affinity helper tests passed (fake syscalls only)");
    return 0;
}
