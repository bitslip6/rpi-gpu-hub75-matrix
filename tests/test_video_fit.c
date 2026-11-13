#include <assert.h>
#include <stdio.h>
#include "video_fit.h"

static void check(int sw, int sh, int sn, int sd, int dw, int dh,
                  bool preserve, int ew, int eh, int ex, int ey) {
    int w, h, x, y;
    video_fit(sw, sh, sn, sd, dw, dh, preserve, &w, &h, &x, &y);
    assert(w == ew && h == eh && x == ex && y == ey);
}
int main(void) {
    check(1920, 1080, 1, 1, 64, 64, false, 64, 64, 0, 0);
    check(1920, 1080, 1, 1, 64, 64, true, 64, 36, 0, 14);
    check(1080, 1920, 1, 1, 64, 64, true, 36, 64, 14, 0);
    check(640, 480, 1, 1, 128, 96, true, 128, 96, 0, 0);
    check(720, 576, 16, 15, 64, 64, true, 64, 48, 0, 8);
    check(640, 480, 0, 0, 64, 64, true, 64, 48, 0, 8);
    check(1920, 1080, 1, 1, 65, 64, true, 65, 37, 0, 13);
    check(8192, 1, 1, 1, 64, 64, true, 64, 1, 0, 31);
    puts("video fit tests passed");
    return 0;
}
