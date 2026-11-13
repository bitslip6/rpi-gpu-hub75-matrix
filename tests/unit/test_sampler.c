#include "pixels.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    RGBA data[] = {
        {0, 20, 40, 255}, {100, 120, 140, 255}, {200, 220, 240, 255}, {100, 120, 140, 255}};
    image_buffer_t image = {.dimensions = {2, 2}, .row_stride = 8, .data = data};
    RGBA out;
    sample_rgba_bilinear(&out, &image, 0, 0);
    assert(memcmp(&out, &data[0], 4) == 0);
    sample_rgba_bilinear(&out, &image, 1, 1);
    assert(memcmp(&out, &data[3], 4) == 0);
    sample_rgba_bilinear(&out, &image, -2, -2);
    assert(memcmp(&out, &data[0], 4) == 0);
    sample_rgba_bilinear(&out, &image, .5f, .5f);
    assert(out.r == 100 && out.g == 120 && out.b == 140 && out.a == 255);
    const uint8_t gray[] = {0, 100, 200, 100};
    assert(sample_gray8_bilinear_buf(gray, 2, 2, .5f, .5f) == 0);
    assert(sample_gray8_bilinear_buf(gray, 2, 2, 1.5f, .5f) == 100);
    assert(sample_gray8_bilinear_buf(gray, 2, 2, 1, 1) == 100);
    assert(sample_gray8_bilinear_buf(gray, 1, 1, 100, -10) == 0);
    assert(sample_gray8_bilinear_buf(NULL, 0, 0, 0, 0) == 0);
    puts("sampler tests passed");
    return 0;
}
