#ifndef H75_GPIO_AFFINITY_H
#define H75_GPIO_AFFINITY_H
/* Private lifecycle API. Include with _GNU_SOURCE; never used by pixel kernels. */
#include <pthread.h>
#include <sched.h>
#include <stddef.h>

int h75_gpio_affinity_require(char *message, size_t size);
int h75_gpio_affinity_save(cpu_set_t *mask, char *message, size_t size);
int h75_gpio_affinity_restore(const cpu_set_t *mask, char *message, size_t size);

#ifdef H75_AFFINITY_TESTING
/* Compile-time replacements only. No production callback/override mechanism. */
int h75_fake_setaffinity(pthread_t thread, size_t size, const cpu_set_t *mask);
int h75_fake_getaffinity(pthread_t thread, size_t size, cpu_set_t *mask);
#endif
#endif
