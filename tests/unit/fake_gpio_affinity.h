#ifndef H75_FAKE_GPIO_AFFINITY_H
#define H75_FAKE_GPIO_AFFINITY_H
#include "gpio_affinity.h"
#include <stdatomic.h>
#include <stdbool.h>
extern atomic_uint h75_fake_sets, h75_fake_gets, h75_fake_caller_sets;
extern atomic_int h75_fake_set_error, h75_fake_get_error;
extern atomic_uint h75_fake_fail_get_at, h75_fake_fail_set_at, h75_fake_wrong_at;
extern atomic_bool h75_fake_wrong_mask;
void h75_fake_affinity_reset(void);
void h75_test_affinity_fail_set(int rc);
bool h75_fake_cpu3(void);
bool h75_fake_original(void);
#endif
