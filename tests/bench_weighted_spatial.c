/* Pin-free encoder benchmark. No libraries, GPIO, EGL or shader activation.
 * Define BENCH_BASELINE to build against the original non-spatial encoder. */
#include "weighted_bcm.h"
#include "rpihub75.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static int compare(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
int main(void) {
    hub75_display_t d = {.width = 192, .height = 192, .panel_width = 64,
        .panel_height = 64, .num_ports = 3, .num_chains = 3, .stride = 4,
        .pwm_bits = 11, .brightness = 32, .gamma = 1, .color_curve = H75_COLOR_CIE,
        .panel_order = PANEL_RGB};
#ifdef ADA_HAT
    d.num_ports = 1; d.height = 64;
#endif
    size_t words;
    if (h75_weighted_word_count(&d, &words) != H75_OK) return 1;
    size_t pixels = (size_t)d.width * d.height, bytes = pixels * 4;
    uint8_t *image = malloc(bytes);
    uint32_t *output = malloc(words * sizeof(*output));
#ifndef BENCH_BASELINE
    int8_t *pattern = malloc(pixels * 3);
    if (!pattern) return 1;
#endif
    if (!image || !output) return 1;
    h75_weighted_lut lut;
    if (h75_weighted_lut_build(&d, &lut) != H75_OK) return 1;
    const char *names[] = {"gradient", "dark", "bright"};
    const unsigned iterations = 300, warmup = 40;
    volatile uint32_t checksum = 0;
    for (unsigned source = 0; source < 3; ++source) {
        for (size_t i = 0; i < bytes; ++i)
            image[i] = source == 0 ? (uint8_t)(i * 17u + i / 19u) : source == 1 ? 16 : 128;
        for (unsigned strength = 0; strength <= 2; strength += 2) {
#ifdef BENCH_BASELINE
            if (strength) continue;
#else
            d.dither = (float)strength;
            if (strength && h75_weighted_spatial_build(&d, pattern, pixels * 3) != H75_OK) return 1;
#endif
            uint64_t samples[300];
            for (unsigned i = 0; i < iterations + warmup; ++i) {
                uint64_t begin = now_ns();
                h75_status status;
#ifndef BENCH_BASELINE
                if (strength)
                    status = h75_weighted_encode_spatial(&d, &lut, image, bytes, output, words,
                                                         pattern, pixels * 3);
                else
#endif
                    status = h75_weighted_encode(&d, &lut, image, bytes, output, words);
                uint64_t elapsed = now_ns() - begin;
                if (status != H75_OK) return 1;
                if (i >= warmup) samples[i - warmup] = elapsed;
                checksum = checksum ^ output[i % words];
            }
            qsort(samples, iterations, sizeof(*samples), compare);
            printf("%s strength=%u median_us=%.1f p95_us=%.1f\n", names[source], strength,
                   (double)((float)samples[iterations / 2] / 1000.f),
                   (double)((float)samples[iterations * 95 / 100] / 1000.f));
        }
    }
    printf("checksum=%u\n", checksum);
#ifndef BENCH_BASELINE
    free(pattern);
#endif
    free(output); free(image);
    return 0;
}
