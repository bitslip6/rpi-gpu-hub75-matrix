#ifndef HUB75_VIDEO_FIT_H
#define HUB75_VIDEO_FIT_H

#include <stdbool.h>

/* Fit the display aspect ratio (including non-square source pixels) inside the
 * output. Round to the nearest pixel and leave any odd padding on right/bottom. */
static inline void video_fit(int src_w, int src_h, int sar_num, int sar_den,
                             int dst_w, int dst_h, bool preserve,
                             int *w, int *h, int *x, int *y) {
    *w = dst_w;
    *h = dst_h;
    if (preserve && src_w > 0 && src_h > 0) {
        if (sar_num <= 0 || sar_den <= 0)
            sar_num = sar_den = 1;
        float aspect = ((float)src_w / (float)src_h) *
                       ((float)sar_num / (float)sar_den);
        if (aspect > (float)dst_w / (float)dst_h) {
            *h = (int)((float)dst_w / aspect + 0.5f);
            if (*h < 1) *h = 1;
            if (*h > dst_h) *h = dst_h;
        } else {
            *w = (int)((float)dst_h * aspect + 0.5f);
            if (*w < 1) *w = 1;
            if (*w > dst_w) *w = dst_w;
        }
    }
    *x = (dst_w - *w) / 2;
    *y = (dst_h - *h) / 2;
}

#endif
