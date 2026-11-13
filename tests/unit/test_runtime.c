/* Pin-free native runtime/queue reference workload. No GPIO or GPU activation. */
#define _GNU_SOURCE
#include "fake_gpio_affinity.h"
#include <errno.h>
#include "hub75_runtime.h"
#include "pixels.h"
#include "rpihub75.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef H75_TESTING
extern void h75_test_fail(unsigned stage);
extern void h75_test_pause_sink(bool paused);
extern void h75_test_hold_exit(bool hold);
extern void h75_test_gpio_model(unsigned model);
static atomic_uint configured;
void __real_configure_gpio(uint32_t *map, int model);
void __wrap_configure_gpio(uint32_t *map, int model) {
    assert(h75_fake_cpu3()); /* verification must finish before configuration */
    atomic_fetch_add(&configured, 1);
    __real_configure_gpio(map, model);
}
static void affinity_failures(h75_config *c) {
    for (unsigned stage = 12; stage <= 13; stage++) {
        h75_fake_affinity_reset();
        unsigned before = atomic_load(&configured);
        h75_runtime *r = NULL;
        h75_error e = {0};
        assert(h75_runtime_create(c, &r, &e) == H75_OK);
        h75_test_fail(stage);
        assert(h75_runtime_start(r, &e) == H75_CANCELLED);
        assert(atomic_load(&configured) == before);
        assert(atomic_load(&h75_fake_sets) == (stage == 12 ? 0u : 1u));
        h75_stats stats;
        h75_runtime_stats(r, &stats);
        assert(!stats.scanned && stats.state == H75_CLOSED);
        h75_runtime_destroy(r);
    }
    for (unsigned failure = 0; failure < 3; failure++) {
        h75_fake_affinity_reset();
        unsigned before = atomic_load(&configured);
        h75_runtime *r = NULL;
        h75_error e = {0};
        if (failure == 0) h75_test_affinity_fail_set(EPERM);
        if (failure == 1) atomic_store(&h75_fake_get_error, EINVAL);
        if (failure == 2) atomic_store(&h75_fake_wrong_mask, true);
        assert(h75_runtime_create(c, &r, &e) == H75_OK);
        assert(!atomic_load(&h75_fake_sets)); /* constructor doesn't pin caller */
        assert(h75_runtime_start(r, &e) == H75_IO);
        assert(strstr(e.message, "mandatory GPIO CPU3 affinity"));
        assert(h75_runtime_error(r, &e) == H75_IO);
        h75_stats stats;
        h75_runtime_stats(r, &stats);
        assert(!stats.scanned && !stats.mapped && stats.state == H75_CLOSED);
        assert(atomic_load(&configured) == before);
        assert(atomic_load(&h75_fake_sets) == 1);
        assert(!atomic_load(&h75_fake_caller_sets) && h75_fake_original());
        h75_runtime_destroy(r);
        /* Startup failure released both process claim and all workers. */
        assert(h75_runtime_create(c, &r, &e) == H75_OK);
        assert(h75_runtime_start(r, &e) == H75_OK);
        assert(h75_runtime_close(r, 3000, &e) == H75_OK);
        h75_runtime_destroy(r);
    }
}
static void physical_ram_model(void) {
    h75_config c;
    h75_config_default(&c);
    c.width = 32;
    c.height = 64;
    c.panel_width = 32;
    c.panel_height = 32;
    c.ports = 2;
    c.queue_depth = 2;
    c.bit_depth = 32;
    h75_error e = {0};
    h75_runtime *r = NULL, *other = NULL;
    h75_test_gpio_model(5);
    for (unsigned stage = 8; stage <= 10; stage++) {
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        h75_test_fail(stage);
        assert(h75_runtime_start(r, &e) == (stage == 8 ? H75_MEMORY : H75_IO));
        h75_runtime_destroy(r);
    }
    for (unsigned model = 3; model <= 5; model++) {
        h75_test_gpio_model(model);
        affinity_failures(&c);
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        unsigned before = atomic_load(&h75_fake_sets);
        assert(h75_runtime_start(r, &e) == H75_OK);
        assert(atomic_load(&h75_fake_sets) == before + 1);
        assert(!atomic_load(&h75_fake_caller_sets) && h75_fake_original());
        assert(h75_runtime_create(&c, &other, &e) == H75_OK);
        assert(h75_runtime_start(other, &e) == H75_BUSY);
        h75_runtime_destroy(other);
        for (unsigned i = 0; i < 12; i++) {
            assert(h75_frame_acquire(r, 1000, &e) == H75_OK);
            h75_command cmd = {.op = H75_CLEAR,
                               .color = {(uint8_t)(i * 17), (uint8_t)(i * 11), 7, 255}};
            assert(h75_frame_command(r, &cmd, &e) == H75_OK);
            assert(h75_frame_submit(r, &e) == H75_OK);
        }
        assert(h75_runtime_flush(r, 3000, &e) == H75_OK);
        h75_stats stats;
        h75_runtime_stats(r, &stats);
        assert(stats.mapped > 0 && stats.scanned > 0 && stats.map_ns > 0 && stats.scan_ns > 0);
        assert(h75_runtime_close(r, 3000, &e) == H75_OK);
        h75_runtime_destroy(r);
        /* Stop-before-first-frame on each actual scanner implementation. */
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        assert(h75_runtime_start(r, &e) == H75_OK);
        assert(h75_runtime_close(r, 3000, &e) == H75_OK);
        h75_runtime_destroy(r);
    }
    h75_test_gpio_model(0);
}
static h75_stats wait_for_scans(h75_runtime *r, uint64_t target) {
    h75_stats stats;
    for (unsigned i = 0; i < 3000; i++) {
        h75_runtime_stats(r, &stats);
        if (stats.scanned >= target)
            return stats;
        struct timespec delay = {0, 1000000};
        nanosleep(&delay, NULL);
    }
    assert(!"weighted scanner failed to complete cycles");
    return stats;
}

static void weighted_physical_ram_model(void) {
    h75_config c;
    h75_config_default(&c);
    c.output_mode = H75_OUTPUT_WEIGHTED_BCM;
    c.width = 32; c.panel_width = 16; c.chains = 2;
    c.panel_height = 32; c.height = 32; c.queue_depth = 2;
    h75_error e = {0};
    h75_runtime *r = NULL;
    for (unsigned model = 3; model <= 4; model++) {
        h75_test_gpio_model(model);
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        assert(h75_runtime_start(r, &e) == H75_UNSUPPORTED);
        h75_runtime_destroy(r);
    }
    h75_test_gpio_model(5);
    affinity_failures(&c);
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    h75_test_fail(11); /* scanner allocation/setup failure must release GPIO */
    assert(h75_runtime_start(r, &e) == H75_MEMORY);
    h75_runtime_destroy(r);
#ifdef ADA_HAT
    const unsigned max_ports = 1;
#else
    const unsigned max_ports = 3;
#endif
    for (unsigned ports = 1; ports <= max_ports; ports++) {
        c.ports = ports; c.height = ports * c.panel_height;
        c.pwm_bits = ports == 2 ? 1 : 11;
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        assert(h75_runtime_start(r, &e) == H75_OK);
        h75_command cmd = {.op = H75_CLEAR, .color = {3, 17, 91, 255}};
        assert(h75_frame_acquire(r, 1000, &e) == H75_OK);
        assert(h75_frame_command(r, &cmd, &e) == H75_OK);
        assert(h75_frame_submit(r, &e) == H75_OK);
        assert(h75_runtime_flush(r, 3000, &e) == H75_OK);
        h75_stats first = wait_for_scans(r, 1);
        h75_stats repeated = wait_for_scans(r, first.scanned + 2);
        assert(repeated.mapped == 1 && repeated.submitted == 1);
        for (unsigned frame = 0; frame < 4; frame++) {
            cmd.color.r = (uint8_t)(frame + 32);
            assert(h75_frame_acquire(r, 1000, &e) == H75_OK);
            assert(h75_frame_command(r, &cmd, &e) == H75_OK);
            assert(h75_frame_submit(r, &e) == H75_OK);
        }
        assert(h75_runtime_flush(r, 3000, &e) == H75_OK);
        h75_stats finished = wait_for_scans(r, repeated.scanned + 2);
        assert(finished.submitted == 5 && finished.consumed == 5 && finished.mapped == 5);
        assert(finished.dropped == 0 && finished.scan_ns && finished.map_ns);
        assert(h75_runtime_close(r, 3000, &e) == H75_OK);
        h75_runtime_destroy(r);
        /* Stop before any weighted frame is published. */
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        assert(h75_runtime_start(r, &e) == H75_OK);
        assert(h75_runtime_close(r, 3000, &e) == H75_OK);
        h75_runtime_destroy(r);
    }
    h75_test_gpio_model(0);
}

static void failures(void) {
    h75_config c;
    h75_config_default(&c);
    c.offscreen = true;
    c.queue_depth = 1;
    h75_error e = {0};
    h75_runtime *r = NULL;
    for (unsigned stage = 1; stage <= 4; stage++) {
        h75_test_fail(stage);
        assert(h75_runtime_create(&c, &r, &e) == H75_MEMORY);
        assert(r == NULL);
    }
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    h75_test_fail(5);
    assert(h75_runtime_start(r, &e) == H75_IO);
    h75_runtime_destroy(r);
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    assert(h75_runtime_start(r, &e) == H75_OK);
    h75_resource *list = h75_resource_new(H75_DRAWLIST, &e);
    assert(list);
    h75_test_fail(6);
    assert(h75_runtime_play(r, list, &e) == H75_IO);
    h75_resource_release(list);
    h75_runtime_destroy(r);
    c.offscreen = false;
    for (unsigned i = 0; i < 2; i++) {
        assert(h75_runtime_create(&c, &r, &e) == H75_OK);
        h75_test_fail(7);
        assert(h75_runtime_start(r, &e) == H75_IO);
        h75_runtime_destroy(r);
    }
    c.offscreen = true;
    h75_test_pause_sink(true);
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    assert(h75_runtime_start(r, &e) == H75_OK);
    assert(h75_frame_acquire(r, 0, &e) == H75_OK);
    assert(h75_frame_submit(r, &e) == H75_OK);
    assert(h75_frame_acquire(r, 10, &e) == H75_TIMEOUT);
    list = h75_resource_new(H75_DRAWLIST, &e);
    assert(list);
    bool accepted = true;
    assert(h75_runtime_try_draw(r, list, 0, &accepted, &e) == H75_OK);
    assert(!accepted);
    h75_stats dropped_stats;
    h75_runtime_stats(r, &dropped_stats);
    assert(dropped_stats.dropped == 1);
    h75_resource_release(list);
    assert(h75_runtime_close(r, 1000, &e) == H75_OK);
    h75_runtime_destroy(r);
    h75_test_pause_sink(false);
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    assert(h75_runtime_start(r, &e) == H75_OK);
    h75_test_hold_exit(true);
    assert(h75_runtime_close(r, 0, &e) == H75_TIMEOUT);
    h75_stats stats;
    h75_runtime_stats(r, &stats);
    assert(stats.state == H75_STOPPING);
    h75_test_hold_exit(false);
    assert(h75_runtime_close(r, 1000, &e) == H75_OK);
    h75_runtime_destroy(r);
}
#endif

static void null_arguments(void) {
    h75_error e = {0};
    h75_stats stats;
    assert(h75_frame_acquire(NULL, 0, &e) == H75_INVALID);
    assert(h75_frame_cancel(NULL, &e) == H75_INVALID);
    assert(h75_frame_submit(NULL, &e) == H75_INVALID);
    assert(h75_frame_command(NULL, NULL, &e) == H75_INVALID);
    assert(h75_frame_draw(NULL, NULL, 0, &e) == H75_INVALID);
    assert(h75_frame_upload(NULL, NULL, 0, 0, 4, &e) == H75_INVALID);
    assert(h75_runtime_flush(NULL, 0, &e) == H75_INVALID);
    assert(h75_runtime_content_ready(NULL, 0, &e) == H75_INVALID);
    assert(h75_runtime_pause(NULL, false, &e) == H75_INVALID);
    assert(h75_runtime_stop_content(NULL, 0, &e) == H75_INVALID);
    assert(h75_runtime_error(NULL, &e) == H75_INVALID);
    assert(h75_runtime_read(NULL, NULL, 0, &e) == H75_INVALID);
    assert(h75_image_load(NULL, &e) == NULL && e.code == H75_INVALID);
    h75_runtime_stats(NULL, &stats);
    assert(stats.state == H75_CLOSED);
    h75_runtime_stats(NULL, NULL);
}

static void queues(void) {
    assert(!spsc_create(3, 4));
    spsc_semring_t *q = spsc_create(2, sizeof(unsigned));
    assert(q);
    unsigned *p = spsc_push_ptr_begin(q, 0);
    assert(p);
    *p = 42;
    spsc_push_ptr_cancel(q);
    assert(spsc_count(q) == 0);
    p = spsc_push_ptr_begin(q, 0);
    assert(p);
    *p = 11;
    spsc_push_ptr_commit(q);
    p = spsc_push_ptr_begin(q, 0);
    assert(p);
    *p = 22;
    spsc_push_ptr_commit(q);
    const unsigned *held = spsc_pop_ptr_begin(q, 0);
    assert(held && *held == 11);
    /* Producer cannot reuse the scanned slot while its consumer lease is live. */
    assert(!spsc_push_ptr_begin(q, 0));
    assert(spsc_count(q) == 2);
    assert(*held == 11);
    spsc_pop_ptr_commit(q);
    held = spsc_pop_ptr_begin(q, 0);
    assert(held && *held == 22);
    p = spsc_push_ptr_begin(q, 0);
    assert(p);
    *p = 33;
    spsc_push_ptr_commit(q);
    assert(*held == 22);
    spsc_pop_ptr_commit(q);
    held = spsc_pop_ptr_begin(q, 0);
    assert(*held == 33);
    spsc_pop_ptr_commit(q);
    assert(!spsc_pop_ptr_begin(q, 0));
    spsc_destroy(q);
}
static void session(unsigned width, unsigned height) {
    h75_config c;
    h75_config_default(&c);
    c.offscreen = true;
    c.width = width;
    c.height = height;
    c.queue_depth = 1;
    h75_error e = {0};
    h75_runtime *r = NULL;
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    assert(h75_frame_acquire(r, 0, &e) == H75_STATE);
    assert(h75_runtime_start(r, &e) == H75_OK);
    assert(h75_runtime_start(r, &e) == H75_STATE);
    h75_command cmd = {.op = H75_CLEAR, .color = {12, 34, 56, 255}};
    for (unsigned i = 0; i < 30; i++) {
        assert(h75_frame_acquire(r, 100, &e) == H75_OK);
        assert(h75_frame_acquire(r, 0, &e) == H75_BUSY);
        assert(h75_frame_command(r, &cmd, &e) == H75_OK);
        if (i % 2)
            assert(h75_frame_submit(r, &e) == H75_OK);
        else
            assert(h75_frame_cancel(r, &e) == H75_OK);
    }
    assert(h75_runtime_flush(r, 1000, &e) == H75_OK);
    size_t bytes = (size_t)width * height * 4;
    RGBA *out = malloc(bytes);
    assert(out);
    assert(h75_runtime_read(r, out, bytes, &e) == H75_OK);
    for (size_t i = 0; i < (size_t)width * height; i++)
        assert(out[i].r == 12 && out[i].g == 34 && out[i].b == 56 && out[i].a == 255);
    h75_stats st;
    h75_runtime_stats(r, &st);
    assert(st.submitted == 15 && st.consumed == 15);
    assert(st.latency_ns > 0 && st.latency_total_ns >= st.latency_ns);
    assert(h75_runtime_close(r, 1000, &e) == H75_OK);
    assert(h75_runtime_close(r, 1000, &e) == H75_OK);
    assert(h75_runtime_start(r, &e) == H75_STATE);
    h75_runtime_destroy(r);
    free(out);
}
static void weighted_config(void) {
    h75_config c;
    h75_error e = {0};
    h75_runtime *r = NULL;
    h75_config_default(&c);
    assert(hub75_runtime_abi() == 3);
    assert(c.output_mode == H75_OUTPUT_EQUAL_SLOT && c.bit_depth == 64);
    assert(c.pwm_bits == 11 && c.pwm_lsb_ns == 130 && c.color_curve == H75_COLOR_CIE);
    assert(h75_config_validate(&c, &e) == H75_OK);
    hub75_display_t *d = hub75_display_new();
    assert(d && d->output_mode == c.output_mode && d->pwm_bits == c.pwm_bits &&
           d->pwm_lsb_ns == c.pwm_lsb_ns && d->color_curve == c.color_curve);
    free(d);
    c.output_mode = H75_OUTPUT_WEIGHTED_BCM;
    for (unsigned bits = 1; bits <= 11; bits++) {
        c.pwm_bits = bits;
        c.pwm_lsb_ns = bits == 11 ? UINT32_MAX : 1;
        assert(h75_config_validate(&c, &e) == H75_OK);
        assert(h75_runtime_create(&c, &r, &e) == H75_OK && r);
        h75_runtime_destroy(r);
    }
    c.queue_depth = 1;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.offscreen = true;
    assert(h75_runtime_create(&c, &r, &e) == H75_OK && r);
    assert(h75_runtime_start(r, &e) == H75_OK);
    assert(h75_runtime_close(r, 1000, &e) == H75_OK);
    h75_runtime_destroy(r);
    c.pwm_bits = 0;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.pwm_bits = 12;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.pwm_bits = 11;
    c.pwm_lsb_ns = 0;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.pwm_lsb_ns = 130;
    c.color_curve = 2;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.color_curve = H75_COLOR_GAMMA;
    assert(h75_config_validate(&c, &e) == H75_OK);
    c.output_mode = 2;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    h75_config_default(&c);
    c.size--;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    /* 4096 * 32 * 64 words exceeds 10MiB; 11 planes fits. */
    h75_config_default(&c);
    c.width = 4096; c.panel_width = 256; c.chains = 16;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.output_mode = H75_OUTPUT_WEIGHTED_BCM;
    assert(h75_config_validate(&c, &e) == H75_OK);
}

static void config(void) {
    weighted_config();
    h75_error e = {0};
    h75_config c;
    h75_config_default(&c);
    c.offscreen = true;
    c.gamma = NAN;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.gamma = 1;
    c.width = UINT32_MAX;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.width = 64;
    c.queue_depth = 3;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
    c.queue_depth = 4;
    c.offscreen = false;
    c.ports = 2;
    assert(h75_config_validate(&c, &e) == H75_INVALID);
}
static void stop_before_frame(void) {
    h75_config c;
    h75_config_default(&c);
    c.offscreen = true;
    h75_error e = {0};
    h75_runtime *r;
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    assert(h75_runtime_start(r, &e) == H75_OK);
    h75_runtime_request_stop(r);
    assert(h75_runtime_wait(r, 1000, &e) == H75_OK);
    h75_runtime_destroy(r);
    assert(h75_runtime_create(&c, &r, &e) == H75_OK);
    h75_runtime_destroy(r); /* never started */
}
static void mapper_ports(unsigned ports) {
    /* Conversion is pin-free. Redzones/ASan catch absent-port reads and stride
     * mistakes. Consumer keeps the entire returned frame leased while checked. */
    hub75_display_t *d = hub75_display_new();
    assert(d);
    d->width = 32;
    d->height = (uint16_t)(ports * 32);
    d->panel_height = 32;
    d->panel_width = 32;
    d->num_ports = (uint8_t)ports;
    d->num_chains = 1;
    d->bit_depth = 32;
    d->gamma = 1;
    d->jitter_brightness = true;
    d->panel_order = PANEL_RGB;
    size_t bytes = (size_t)d->width * d->height * 4;
    uint8_t *rgba = calloc(bytes, 1);
    assert(rgba);
    d->accum = calloc((size_t)d->width * d->height * 4, sizeof(int32_t));
    d->quant_errors_lut = calloc(768 * 4, sizeof(uint16_t));
    d->ring_buf_renderer =
        spsc_create(2, (size_t)d->width * (d->panel_height / 2) * d->bit_depth * 4);
    assert(d->accum && d->quant_errors_lut && d->ring_buf_renderer);
    for (unsigned i = 0; i < 4; i++) {
        d->quant_dither = (i & 1) != 0;
        assert(hub75_display_map_image_to_bcm(d, rgba) == HUB75_OK);
        const uint32_t *held = spsc_pop_ptr_begin(d->ring_buf_renderer, 0);
        assert(held);
        for (size_t j = 0; j < d->ring_buf_renderer->item_size / 4; j++)
            assert(held[j] == 0);
        spsc_pop_ptr_commit(d->ring_buf_renderer);
    }
    /* Full-intensity alternating channels must be identical in every packed
     * BCM plane, including all active parallel ports. This catches a logical-
     * height plane stride or a scanner walking beyond the packed payload. */
    const unsigned pins[3][2][3] = {{{ADDRESS_P0_R1, ADDRESS_P0_G1, ADDRESS_P0_B1},
                                     {ADDRESS_P0_R2, ADDRESS_P0_G2, ADDRESS_P0_B2}},
                                    {{ADDRESS_P1_R1, ADDRESS_P1_G1, ADDRESS_P1_B1},
                                     {ADDRESS_P1_R2, ADDRESS_P1_G2, ADDRESS_P1_B2}},
                                    {{ADDRESS_P2_R1, ADDRESS_P2_G1, ADDRESS_P2_B1},
                                     {ADDRESS_P2_R2, ADDRESS_P2_G2, ADDRESS_P2_B2}}};
    for (unsigned p = 0; p < ports; p++)
        for (unsigned half = 0; half < 2; half++)
            for (unsigned y = 0; y < 16; y++)
                for (unsigned x = 0; x < 32; x++) {
                    size_t at = ((size_t)p * 32 + half * 16 + y) * 32 + x;
                    rgba[at * 4 + (p + half + ((x + y) & 1)) % 3] = 255;
                    rgba[at * 4 + 3] = 255;
                }
    d->quant_dither = false;
    assert(hub75_display_map_image_to_bcm(d, rgba) == HUB75_OK);
    const uint32_t *held = spsc_pop_ptr_begin(d->ring_buf_renderer, 0);
    assert(held);
    for (unsigned plane = 0; plane < d->bit_depth; plane++)
        for (unsigned y = 0; y < 16; y++)
            for (unsigned x = 0; x < 32; x++) {
                uint32_t expected = 0;
                for (unsigned p = 0; p < ports; p++)
                    for (unsigned half = 0; half < 2; half++)
                        expected |= 1u << pins[p][half][(p + half + ((x + y) & 1)) % 3];
                assert(held[((size_t)plane * 16 + y) * 32 + x] == expected);
            }
    spsc_pop_ptr_commit(d->ring_buf_renderer);
    hub75_mapper_cache_clear();
    spsc_destroy(d->ring_buf_renderer);
    free(d->accum);
    free(d->quant_errors_lut);
    free(rgba);
    free(d);
}
int main(void) {
#ifdef H75_TESTING
    h75_fake_affinity_reset();
    failures();
    assert(!atomic_load(&h75_fake_sets) && !atomic_load(&h75_fake_gets));
    physical_ram_model();
    weighted_physical_ram_model();
    h75_fake_affinity_reset();
#endif
    null_arguments();
    queues();
    config();
    stop_before_frame();
    for (unsigned i = 0; i < 10; i++) {
        session(8, 16);
        session(128, 64);
    }
    for (unsigned ports = 1; ports <= 3; ports++)
        mapper_ports(ports);
    assert(!atomic_load(&h75_fake_sets) && !atomic_load(&h75_fake_gets));
    puts("native runtime/queue tests passed (no GPIO, fake affinity only)");
    return 0;
}
