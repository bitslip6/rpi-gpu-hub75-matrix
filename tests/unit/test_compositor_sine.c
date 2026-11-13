#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "compositor.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static float coverage(float p, int size) {
    return fmaxf(0.0f, fminf(1.0f, fminf(p + 1.0f, (float)size - p)));
}

static void check_pixel(RGBA actual, RGBA expected) {
    CHECK(abs((int)actual.r - expected.r) <= 1);
    CHECK(abs((int)actual.g - expected.g) <= 1);
    CHECK(abs((int)actual.b - expected.b) <= 1);
    CHECK(abs((int)actual.a - expected.a) <= 1);
}

static void test_clipping_and_stride(void) {
    enum { SW = 65, SH = 12, SS = SW + 3, DW = 32, DH = 40, DS = DW + 2 };
    const RGBA ink = {200, 120, 40, 192}, background = {19, 37, 51, 255};
    RGBA source[SH * SS], original[SH * SS], pixels[DH * DS];
    for (size_t i = 0; i < SH * SS; ++i) source[i] = ink;
    memcpy(original, source, sizeof(source));
    image_buffer_t src = {.data = source, .dimensions = {SW, SH}, .row_stride = SS * 4};
    image_buffer_t dst = {.data = pixels, .dimensions = {DW, DH}, .row_stride = DS * 4};
    const float xs[] = {-100.0f, -50.25f, -3.25f, 0.0f, 17.5f, 32.0f, 100.0f};
    const float ys[] = {-12.0f, 4.75f, 28.0f};
    const float waves[][3] = {
        {2.0f, 24.5f, 0.0f}, {0.5f, 6.25f, 1.3f},
        {0.0f, 8.0f, 1.3f}, {3.0f, 0.0f, 1.3f}, {1.0f, 12.0f, -1.3f}
    };

    for (size_t xi = 0; xi < sizeof(xs) / sizeof(xs[0]); ++xi) {
        for (size_t yi = 0; yi < sizeof(ys) / sizeof(ys[0]); ++yi) {
            for (size_t wi = 0; wi < sizeof(waves) / sizeof(waves[0]); ++wi) {
                for (size_t i = 0; i < DH * DS; ++i) pixels[i] = background;
                composite_rgba_sine_over_rgba(&dst, &src, xs[xi], ys[yi],
                                              waves[wi][0], waves[wi][1], waves[wi][2]);
                for (int dx = 0; dx < DW; ++dx) {
                    float top = ys[yi] + waves[wi][1] * sinf(((float)dx + 0.5f) *
                                    (2.0f * (float)M_PI * waves[wi][0] / DW) + waves[wi][2]);
                    for (int dy = 0; dy < DH; ++dy) {
                        RGBA sample = ink, expected;
                        sample.a = (uint8_t)((float)ink.a * coverage((float)dx - xs[xi], SW) *
                                                         coverage((float)dy - top, SH));
                        composite_rgba(&expected, &sample, &background);
                        check_pixel(pixels[dy * DS + dx], expected);
                    }
                }
                for (int dy = 0; dy < DH; ++dy) {
                    for (int dx = DW; dx < DS; ++dx) {
                        CHECK(memcmp(&pixels[dy * DS + dx], &background, sizeof(RGBA)) == 0);
                    }
                }
            }
        }
    }
    CHECK(memcmp(source, original, sizeof(source)) == 0);
}

static void test_translation_and_fractional_sampling(void) {
    RGBA source[12], pixels[8 * 40];
    for (int i = 0; i < 12; ++i) source[i] = (RGBA){(uint8_t)(i * 20), 80, 40, 255};
    image_buffer_t src = {.data = source, .dimensions = {3, 4}, .row_stride = 12};
    image_buffer_t dst = {.data = pixels, .dimensions = {8, 40}, .row_stride = 32};
    const float step = 4.0f * (float)M_PI / 8.0f;

    // At this column the wave is zero. Integer translations must be exact,
    // including when the left side of the source is outside the display.
    for (int x = -1; x <= 1; x += 2) {
        int dx = x + 1;
        memset(pixels, 0, sizeof(pixels));
        composite_rgba_sine_over_rgba(&dst, &src, (float)x, 3.0f, 2.0f, 24.5f,
                                      -((float)dx + 0.5f) * step);
        for (int row = 0; row < 4; ++row) {
            check_pixel(pixels[(row + 3) * 8 + dx], source[row * 3 + 1]);
        }
    }

    RGBA gradient[] = {{0, 0, 0, 255}, {200, 100, 60, 255}};
    src = (image_buffer_t){.data = gradient, .dimensions = {1, 2}, .row_stride = 4};
    const RGBA background = {80, 80, 80, 255};
    for (size_t i = 0; i < sizeof(pixels) / sizeof(pixels[0]); ++i) pixels[i] = background;
    float y = 8.25f - 24.5f * sinf(3.5f * step);
    composite_rgba_sine_over_rgba(&dst, &src, 3.0f, y, 2.0f, 24.5f, 0.0f);
    check_pixel(pixels[9 * 8 + 3], (RGBA){150, 75, 45, 255});
    check_pixel(pixels[8 * 8 + 3], (RGBA){20, 20, 20, 255});
    check_pixel(pixels[10 * 8 + 3], (RGBA){110, 85, 75, 255});
    CHECK(memcmp(&pixels[11 * 8 + 3], &background, sizeof(RGBA)) == 0);
}

static void test_option_order(void) {
    char *args[][8] = {
        {"example", "-e", "hello", NULL},
        {"example", "-W", "2:24.5:1", "-e", "hello", NULL},
        {"example", "-e", "-", "-W", "0.5:8:-2", NULL},
        {"example", "-W", "3:4:0", "-E", "32", "-e", "hello", NULL},
        {"example", "-E", "24", "-e", "hello", "-W", "0:0:0", NULL}
    };
    const float expected[][3] = {{0, 0, 0}, {2, 24.5f, 1}, {0.5f, 8, -2}, {3, 4, 0}, {0, 0, 0}};
    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); ++i) {
        int argc = 0;
        while (args[i][argc]) ++argc;
        optind = 0;
        hub75_display_t *scene = hub75_display_parse_args(argc, args[i]);
        CHECK(scene->text_overlay != NULL);
        CHECK(scene->text_overlay->sine_scroll == (i != 0));
        CHECK(scene->text_overlay->stdin_text == (i == 2));
        CHECK(scene->text_overlay->font_size == (i == 4 ? 24.0f : i == 3 ? 32.0f : 64.0f));
        CHECK(scene->text_overlay->sine_frequency == expected[i][0]);
        CHECK(scene->text_overlay->sine_amplitude == expected[i][1]);
        CHECK(scene->text_overlay->sine_offset_speed == expected[i][2]);
        free(scene->text_overlay->text);
        free(scene->text_overlay);
        free(scene);
    }
}

static void test_invalid_options(void) {
    const char *invalid[] = {
        NULL, "", "2", "2:24.5", "2:24.5:", ":24.5:1", "2::1",
        "2:24.5:1:4", "2:24.5:1junk", "x:24.5:1", "2:24.5:nan",
        "inf:24.5:1", "2:nan:1", "2:24.5:-inf", "2:24.5:1e99"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        pid_t child = fork();
        CHECK(child >= 0);
        if (child == 0) {
            (void)freopen("/dev/null", "w", stderr);
            (void)freopen("/dev/null", "w", stdout);
            char *args[] = {"example", "-W", (char*)invalid[i], NULL};
            optind = 0;
            hub75_display_parse_args(invalid[i] ? 3 : 2, args);
            _exit(0);
        }
        int status;
        CHECK(waitpid(child, &status, 0) == child);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    }
}

int main(void) {
    test_clipping_and_stride();
    test_translation_and_fractional_sampling();
    test_option_order();
    test_invalid_options();
    puts("PASS: sine compositor clipping, strides, alpha, translation, sampling and CLI options");
    return 0;
}
