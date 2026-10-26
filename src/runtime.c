#define _GNU_SOURCE
#include "runtime_internal.h"
#include "rpihub75.h"
#include "pixels.h"
#include "transformers.h"
#include "lowlevel.h"
#include "weighted_rio.h"
#include "gpio_lock.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct h75_runtime {
    h75_config config;
    hub75_display_t *display;
    pthread_mutex_t mutex, join_mutex, read_mutex, error_mutex;
    atomic_bool stop, content_stop, sink_done, gpio_done, content_done, init_done, paused,
        content_started, gpio_ready;
    pthread_t sink_thread, gpio_thread, content_thread, frame_owner;
    bool sink_started, gpio_started, leased, physical_claim;
    _Atomic(h75_state) state;
    uint8_t *latest;
    size_t bytes;
    h75_resource *content;
    char *path;
    bool video, loop;
    float time_scale;
    uint32_t *gpio;
    int gpio_model, lock_fd;
    h75_error worker_error;
    atomic_uint_fast64_t submitted, consumed, mapped, scanned, dropped, queue_wait_ns, render_ns;
    atomic_uint_fast64_t map_ns, copy_ns, scan_ns, latency_ns, latency_total_ns;
    atomic_uint_fast64_t submission_times[32]; /* same ownership/index as RGBA slots */
};
static atomic_bool physical_active = false;
static void pause_ms(void);
#ifdef H75_TESTING
/* Only the pin-free test executable compiles these hooks. They are not part of
 * either installed shared library or the Python module. */
static atomic_uint failure_stage, gpio_model_mock;
void h75_test_gpio_model(unsigned model) {
    atomic_store(&gpio_model_mock, model);
}
static atomic_bool sink_paused, hold_exit;
void h75_test_fail(unsigned stage) {
    atomic_store(&failure_stage, stage);
}
void h75_test_pause_sink(bool paused) {
    atomic_store(&sink_paused, paused);
}
void h75_test_hold_exit(bool hold) {
    atomic_store(&hold_exit, hold);
}
static bool inject(unsigned stage) {
    return atomic_compare_exchange_strong(&failure_stage, &stage, 0);
}
#define INJECT(stage) inject(stage)
#else
#define INJECT(stage) false
#endif

h75_status h75_fail(h75_error *e, h75_status code, const char *fmt, ...) {
    if (e) {
        e->code = code;
        va_list a;
        va_start(a, fmt);
        vsnprintf(e->message, sizeof(e->message), fmt, a);
        va_end(a);
    }
    return code;
}
uint64_t h75_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
uint32_t hub75_runtime_abi(void) {
    return HUB75_RUNTIME_ABI;
}
uint32_t hub75_runtime_capabilities(void) {
    uint32_t caps = HUB75_CAP_CPU | HUB75_CAP_GPIO |
                    (h75_media_open ? (HUB75_CAP_SHADER | HUB75_CAP_VIDEO) : 0);
#ifdef __ARM_NEON
    caps |= HUB75_CAP_NEON;
#endif
#ifdef HAVE_LIBPNG
    caps |= HUB75_CAP_PNG;
#endif
#ifdef HAVE_LIBJPEG
    caps |= HUB75_CAP_JPEG;
#endif
    return caps;
}
const char *hub75_runtime_build(void) {
    return "hub75-runtime-2;library=0.3.0;rgba8;float32";
}
void h75_config_default(h75_config *c) {
    *c = (h75_config){.size = sizeof(*c),
                      .width = 64,
                      .height = 64,
                      .panel_width = 64,
                      .panel_height = 64,
                      .ports = 1,
                      .chains = 1,
                      .bit_depth = 64,
                      .output_mode = H75_OUTPUT_EQUAL_SLOT,
                      .pwm_bits = 11,
                      .pwm_lsb_ns = 130,
                      .color_curve = H75_COLOR_CIE,
                      .fps = 60,
                      .brightness = 200,
                      .queue_depth = 4,
                      .gamma = 1,
                      .jitter = true,
                      .gpio_cpu = -1};
}
h75_status h75_config_validate(const h75_config *c, h75_error *e) {
    if (!c || c->size != sizeof(*c))
        return h75_fail(e, H75_INVALID, "configuration ABI/size mismatch");
    if (!c->width || !c->height || c->width > 4096 || c->height > 4096 ||
        (uint64_t)c->width * c->height * 4 > SPSC_MAX_ITEM_SIZE)
        return h75_fail(e, H75_INVALID, "dimensions must be 1..4096 and RGBA storage <=10MiB");
    if (c->ports < 1 || c->ports > 3 || c->chains < 1 || c->chains > 16 || !c->panel_width ||
        !c->panel_height || c->panel_width > 4096 || c->panel_height > 4096)
        return h75_fail(e, H75_INVALID, "invalid panel topology");
    if (c->output_mode > H75_OUTPUT_WEIGHTED_BCM || c->color_curve > H75_COLOR_GAMMA ||
        c->pwm_bits < 1 || c->pwm_bits > 11 || !c->pwm_lsb_ns)
        return h75_fail(e, H75_INVALID, "invalid output mode, color curve, PWM bits or LSB duration");
    /* Widen before multiplying: even UINT32_MAX ns at 11 bits and 2048
     * scanned rows fits uint64_t. No electrical timing guarantees implied. */
    uint64_t cycle_ns = (uint64_t)c->pwm_lsb_ns * ((1u << c->pwm_bits) - 1) *
                        (c->panel_height / 2);
    if (c->output_mode == H75_OUTPUT_WEIGHTED_BCM && !cycle_ns)
        return h75_fail(e, H75_INVALID, "weighted cycle requires scanned rows");
    if (c->bit_depth < 32 || c->bit_depth > 64 || c->bit_depth % BIT_DEPTH_ALIGNMENT)
        return h75_fail(e, H75_INVALID, "native BCM mapper supports aligned bit depths 32..64");
    if (c->fps < 1 || c->fps > FPS_MAX || c->brightness > 254 || c->panel_order > 5 ||
        c->mapping > 4 || c->tone_mapping > 5)
        return h75_fail(e, H75_INVALID,
                        "invalid fps, brightness, panel order, mapper or tone mapper");
    if (!isfinite(c->gamma) || c->gamma <= 0 || c->gamma > 8 || !isfinite(c->dither) ||
        c->dither < 0 || c->dither > 10 || !isfinite(c->tone_level) || c->tone_level < 0 ||
        c->tone_level > 100 || c->gpio_cpu < -1 || c->gpio_cpu >= CPU_SETSIZE)
        return h75_fail(e, H75_INVALID, "invalid gamma, dithering, tone level or affinity");
    if (!c->queue_depth || c->queue_depth > SPSC_MAX_CAPACITY ||
        (c->queue_depth & (c->queue_depth - 1)))
        return h75_fail(e, H75_INVALID, "queue depth must be a power of two in 1..32");
    if (!c->offscreen) {
        if (c->output_mode == H75_OUTPUT_WEIGHTED_BCM && c->queue_depth < 2)
            return h75_fail(e, H75_INVALID, "physical weighted output requires queue depth >=2 for held-frame replacement");
#ifdef ADA_HAT
        if (c->output_mode == H75_OUTPUT_WEIGHTED_BCM && c->ports != 1)
            return h75_fail(e, H75_UNSUPPORTED, "this HAT wiring supports one weighted port only");
#endif
        if (c->width != c->panel_width * c->chains || c->height != c->panel_height * c->ports ||
            c->width % 16 || c->panel_width % 16 ||
            (c->panel_height != 32 && c->panel_height != 64) ||
            (uint64_t)c->width * (c->panel_height / 2) *
                (c->output_mode == H75_OUTPUT_WEIGHTED_BCM ? c->pwm_bits : c->bit_depth) *
                sizeof(uint32_t) > SPSC_MAX_ITEM_SIZE)
            return h75_fail(e, H75_INVALID,
                            "physical topology requires width=panel_width*chains, "
                            "height=panel_height*ports; 32/64-row panels and aligned width");
        if (c->mapping == 4)
            return h75_fail(e, H75_UNSUPPORTED,
                            "U-mapping is not yet supported by the structured physical topology");
    }
    return H75_OK;
}
static void worker_fail(h75_runtime *r, const h75_error *e) {
    pthread_mutex_lock(&r->error_mutex);
    if (!r->worker_error.code)
        r->worker_error = *e;
    pthread_mutex_unlock(&r->error_mutex);
}
static h75_status worker_status(h75_runtime *r, h75_error *e) {
    pthread_mutex_lock(&r->error_mutex);
    h75_status s = r->worker_error.code;
    if (s && e)
        *e = r->worker_error;
    pthread_mutex_unlock(&r->error_mutex);
    return s;
}
h75_status h75_runtime_create(const h75_config *c, h75_runtime **out, h75_error *e) {
    if (!out)
        return h75_fail(e, H75_INVALID, "null output handle");
    *out = NULL;
    h75_status s = h75_config_validate(c, e);
    if (s)
        return s;
    h75_runtime *r = INJECT(1) ? NULL : calloc(1, sizeof(*r));
    if (!r)
        return h75_fail(e, H75_MEMORY, "allocating runtime");
    r->config = *c;
    r->lock_fd = -1;
    atomic_init(&r->state, H75_CREATED);
    r->bytes = (size_t)c->width * c->height * 4;
    /* Each mutex is initialized independently for a valid partial-failure path. */
    int n = 0;
    if (pthread_mutex_init(&r->mutex, NULL))
        goto mutex_fail;
    n++;
    if (pthread_mutex_init(&r->join_mutex, NULL))
        goto mutex_fail;
    n++;
    if (pthread_mutex_init(&r->read_mutex, NULL))
        goto mutex_fail;
    n++;
    if (pthread_mutex_init(&r->error_mutex, NULL))
        goto mutex_fail;
    n++;
    atomic_init(&r->stop, false);
    atomic_init(&r->content_stop, false);
    atomic_init(&r->paused, false);
    atomic_init(&r->content_started, false);
    atomic_init(&r->submitted, 0);
    atomic_init(&r->consumed, 0);
    atomic_init(&r->mapped, 0);
    atomic_init(&r->scanned, 0);
    atomic_init(&r->dropped, 0);
    atomic_init(&r->queue_wait_ns, 0);
    atomic_init(&r->render_ns, 0);
    atomic_init(&r->map_ns, 0);
    atomic_init(&r->copy_ns, 0);
    atomic_init(&r->scan_ns, 0);
    atomic_init(&r->latency_ns, 0);
    atomic_init(&r->latency_total_ns, 0);
    for (unsigned i = 0; i < 32; i++)
        atomic_init(&r->submission_times[i], 0);
    atomic_init(&r->sink_done, true);
    atomic_init(&r->gpio_done, true);
    atomic_init(&r->gpio_ready, false);
    atomic_init(&r->content_done, true);
    atomic_init(&r->init_done, false);
    r->display = INJECT(2) ? NULL : hub75_display_new();
    r->latest = INJECT(3) ? NULL : calloc(r->bytes, 1);
    if (!r->display || !r->latest)
        goto memory_fail;
    hub75_display_t *d = r->display;
    d->width = (uint16_t)c->width;
    d->height = (uint16_t)c->height;
    d->panel_width = (uint16_t)c->panel_width;
    d->panel_height = (uint16_t)c->panel_height;
    d->num_ports = (uint8_t)c->ports;
    d->num_chains = (uint8_t)c->chains;
    d->bit_depth = (uint8_t)c->bit_depth;
    d->output_mode = c->output_mode;
    d->pwm_bits = c->pwm_bits;
    d->pwm_lsb_ns = c->pwm_lsb_ns;
    d->color_curve = c->color_curve;
    d->fps = (uint16_t)c->fps;
    d->auto_fps = false;
    d->brightness = (uint8_t)c->brightness;
    d->gamma = c->gamma;
    d->dither = c->dither;
    d->tone_level = c->tone_level;
    d->quant_dither = c->quant_dither;
    d->jitter_brightness = c->jitter;
    d->panel_order = c->panel_order;
    func_image_mapper_t mappings[] = {NULL, flip_mapper, mirror_mapper, mirror_flip_mapper,
                                      u_mapper_impl};
    func_tone_mapper_t tones[] = {NULL,
                                  aces_tone_mapperF,
                                  reinhard_tone_mapperF,
                                  hable_tone_mapperF,
                                  saturation_tone_mapperF,
                                  sigmoid_tone_mapperF};
    d->image_mapper = mappings[c->mapping];
    d->tone_mapper = tones[c->tone_mapping];
    d->frame_buffer = (image_buffer_t){.dimensions = {(int32_t)c->width, (int32_t)c->height},
                                       .row_stride = c->width * 4};
    d->ring_buf_mapper = INJECT(4) ? NULL : spsc_create(c->queue_depth, r->bytes);
    if (!d->ring_buf_mapper)
        goto memory_fail;
    if (!c->offscreen) {
        const uint32_t planes = c->output_mode == H75_OUTPUT_WEIGHTED_BCM ? c->pwm_bits : c->bit_depth;
        d->ring_buf_renderer = spsc_create(
            c->queue_depth, (size_t)c->width * (c->panel_height / 2) * planes * sizeof(uint32_t));
        if (!d->ring_buf_renderer)
            goto memory_fail;
        if (c->output_mode == H75_OUTPUT_EQUAL_SLOT) {
            d->accum = calloc((size_t)c->width * c->height * 4, sizeof(int32_t));
            d->quant_errors_lut = calloc(768 * 4, sizeof(uint16_t));
            if (!d->accum || !d->quant_errors_lut)
                goto memory_fail;
        }
    }
    *out = r;
    return H75_OK;
memory_fail:
    h75_runtime_destroy(r);
    return h75_fail(e, H75_MEMORY, "allocating runtime buffers");
mutex_fail:
    if (n >= 3)
        pthread_mutex_destroy(&r->read_mutex);
    if (n >= 2)
        pthread_mutex_destroy(&r->join_mutex);
    if (n >= 1)
        pthread_mutex_destroy(&r->mutex);
    free(r);
    return h75_fail(e, H75_IO, "initializing native mutexes");
}
static void *sink_main(void *arg) {
    h75_runtime *r = arg;
    hub75_display_t *d = r->display;
    hub75_error_t (*map_frame)(const hub75_display_t *, uint8_t *) =
        d->output_mode == H75_OUTPUT_WEIGHTED_BCM ? hub75_display_map_image_to_weighted_bcm
                                                : hub75_display_map_image_to_bcm;
    while (!atomic_load(&r->stop)) {
#ifdef H75_TESTING
        if (atomic_load(&sink_paused)) {
            struct timespec t = {0, 1000000};
            nanosleep(&t, NULL);
            continue;
        }
#endif
        uint8_t *p = spsc_pop_ptr_begin(d->ring_buf_mapper, 20);
        if (!p)
            continue;
        size_t ordinal = atomic_load(&r->consumed) & (r->config.queue_depth - 1);
        uint64_t submitted_at = atomic_load(&r->submission_times[ordinal]);
        uint64_t copy_begin = h75_now_ns();
        pthread_mutex_lock(&r->read_mutex);
        memcpy(r->latest, p, r->bytes);
        pthread_mutex_unlock(&r->read_mutex);
        atomic_fetch_add(&r->copy_ns, h75_now_ns() - copy_begin);
        if (!r->config.offscreen) {
            h75_error e = {0};
            uint64_t map_begin = h75_now_ns();
            hub75_error_t s = map_frame(d, p);
            atomic_fetch_add(&r->map_ns, h75_now_ns() - map_begin);
            if (s == HUB75_OK)
                atomic_fetch_add(&r->mapped, 1);
            else if (s == HUB75_ERR_RING_BUFFER_FULL)
                atomic_fetch_add(&r->dropped, 1);
            else {
                h75_fail(&e, s == HUB75_ERR_OUT_OF_MEMORY ? H75_MEMORY : H75_INVALID,
                         "BCM conversion failed (%d)", s);
                worker_fail(r, &e);
                h75_runtime_request_stop(r);
            }
        }
        uint64_t latency = h75_now_ns() - submitted_at;
        atomic_store(&r->latency_ns, latency);
        atomic_fetch_add(&r->latency_total_ns, latency);
        spsc_pop_ptr_commit(d->ring_buf_mapper);
        atomic_fetch_add(&r->consumed, 1);
    }
    hub75_mapper_cache_clear();
#ifdef H75_TESTING
    while (atomic_load(&hold_exit)) {
        struct timespec t = {0, 1000000};
        nanosleep(&t, NULL);
    }
#endif
    atomic_store(&r->sink_done, true);
    return NULL;
}
static inline void io_barrier(void) {
#if defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("dmb ishst" ::: "memory");
#endif
}
static void gpio_blank(h75_runtime *r) {
    if (!r->gpio)
        return;
    if (r->gpio_model == 5) {
        volatile uint32_t *base = r->gpio + RIO5_OFFSET;
        base[0x2000 / 4] = PIN_OE; /* RIO SET Out */
        base[0x3000 / 4] = PIN_CLK | PIN_LATCH;
        io_barrier();
    } else {
        r->gpio[7] = PIN_OE;
        r->gpio[10] = PIN_CLK | PIN_LATCH;
        io_barrier();
    }
}
/* Selected once, never from the pixel loop. Only complete weighted cycles
 * contribute to scanned; an interrupted partial cycle does not. */
static void gpio_weighted_loop(h75_runtime *r, const uint32_t *addresses) {
    h75_weighted_rio scan = {0};
    h75_error error = {0};
    h75_status status = INJECT(11) ? H75_MEMORY : h75_weighted_rio_init(&scan, r->display, r->gpio_model,
        r->gpio + RIO5_OFFSET, addresses, &r->stop, false);
    const uint32_t *held = NULL;
    if (status != H75_OK)
        goto failed;
    atomic_store(&r->gpio_ready, true);
    while (!atomic_load(&r->stop)) {
        status = h75_weighted_rio_frame_begin(&scan, r->display->ring_buf_renderer, &held, 20);
        if (status == H75_TIMEOUT)
            continue;
        if (status != H75_OK)
            goto failed;
        const uint64_t begin = h75_now_ns();
        status = h75_weighted_rio_cycle(&scan);
        atomic_fetch_add(&r->scan_ns, h75_now_ns() - begin);
        if (status == H75_CANCELLED)
            break;
        if (status != H75_OK)
            goto failed;
        atomic_fetch_add(&r->scanned, 1);
    }
    goto done;
failed:
    h75_fail(&error, status, "weighted RIO scanner failed (%d)", status);
    worker_fail(r, &error);
    h75_runtime_request_stop(r);
done:
    /* Blank BEFORE returning a held slot or freeing the staging buffer. */
    gpio_blank(r);
    h75_weighted_rio_destroy(&scan);
    if (held)
        spsc_pop_ptr_commit(r->display->ring_buf_renderer);
}

static void *gpio_main(void *arg) {
    h75_runtime *r = arg;
    hub75_display_t *d = r->display;
    if (atomic_load(&r->stop)) {
        atomic_store(&r->gpio_done, true);
        return NULL;
    }
    if (r->config.gpio_cpu >= 0) {
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        CPU_SET((size_t)r->config.gpio_cpu, &cpus);
        int rc = INJECT(10) ? EINVAL : pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
        if (rc) {
            h75_error e = {0};
            h75_fail(&e, H75_IO, "setting GPIO affinity: %s", strerror(rc));
            worker_fail(r, &e);
            h75_runtime_request_stop(r);
            atomic_store(&r->gpio_done, true);
            return NULL;
        }
    }
    configure_gpio(r->gpio, r->gpio_model);
    gpio_blank(r);
    uint32_t addresses[32];
    unsigned rows = d->panel_height / 2;
    /* Select the address policy once; weighted exposes newly latched data,
     * unlike the legacy equal-slot shift/exposure pipeline. */
    uint32_t (*address_for_row)(int, uint16_t) =
        d->output_mode == H75_OUTPUT_WEIGHTED_BCM ? h75_weighted_row_to_address : row_to_address;
    for (unsigned y = 0; y < rows; y++)
        addresses[y] = address_for_row((int)y, (uint16_t)rows);
    if (d->output_mode == H75_OUTPUT_WEIGHTED_BCM) {
        gpio_weighted_loop(r, addresses);
        atomic_store(&r->gpio_done, true);
        return NULL;
    }
    atomic_store(&r->gpio_ready, true);
    const uint32_t *held = NULL;
    unsigned brightness_acc = 0;
    while (!atomic_load(&r->stop)) {
        if (!held) {
            held = spsc_pop_ptr_begin(d->ring_buf_renderer, 20);
            if (!held)
                continue;
        }
        /* A held reservation counts in occupancy. At a complete scan boundary
         * release it only when a successor is already published. */
        if (spsc_count(d->ring_buf_renderer) >= 2) {
            spsc_pop_ptr_commit(d->ring_buf_renderer);
            held = spsc_pop_ptr_begin(d->ring_buf_renderer, 0);
            if (!held)
                continue;
        }
        uint64_t scan_begin = h75_now_ns();
        for (unsigned plane = 0; plane < d->bit_depth; plane++) {
            size_t off = (size_t)plane * d->width * rows;
            for (unsigned y = 0; y < rows; y++) {
                uint32_t addr = addresses[y];
                if (r->gpio_model == 5) {
                    volatile uint32_t *reg = r->gpio + RIO5_OFFSET;
                    *reg = PIN_OE | addr;
                    io_barrier();
                    for (unsigned x = 0; x < d->width; x++) {
                        uint32_t oe = 0;
                        if (d->jitter_brightness) {
                            brightness_acc += d->brightness;
                            if (brightness_acc < 255)
                                oe = PIN_OE;
                            else
                                brightness_acc -= 255;
                        }
                        uint32_t v = held[off++] | addr | oe;
                        *reg = v;
                        *reg = v | PIN_CLK;
                    }
                    reg[0x2000 / 4] = PIN_OE;
                    SLOW2
                    reg[0x2000 / 4] = PIN_LATCH | PIN_OE;
                    reg[0x3000 / 4] = PIN_LATCH;
                } else {
                    volatile uint32_t *reg = r->gpio;
                    reg[7] = PIN_OE | addr;
                    reg[10] = (~addr) & MASK_ADDRESS;
                    SLOW for (unsigned x = 0; x < d->width; x++) {
                        uint32_t oe = 0;
                        if (d->jitter_brightness) {
                            brightness_acc += d->brightness;
                            if (brightness_acc < 255)
                                oe = PIN_OE;
                            else
                                brightness_acc -= 255;
                        }
                        uint32_t v = held[off++] | addr | oe;
                        reg[10] = (~v & (MASK_DATA | PIN_OE)) | PIN_CLK;
                        reg[7] = v;
                        SLOW reg[7] = PIN_CLK;
                        SLOW
                    }
                    reg[7] = PIN_LATCH | PIN_OE;
                    SLOW reg[10] = PIN_LATCH;
                    SLOW reg[10] = PIN_OE;
                }
            }
        }
        atomic_fetch_add(&r->scan_ns, h75_now_ns() - scan_begin);
        atomic_fetch_add(&r->scanned, 1);
    }
    gpio_blank(r);
    if (held)
        spsc_pop_ptr_commit(d->ring_buf_renderer);
    atomic_store(&r->gpio_done, true);
    return NULL;
}
/* Same advisory lock as the standalone GPIO scope tool, including root/user
 * safety under sticky /tmp. Never unlink a live lock inode. */
static h75_status physical_open(h75_runtime *r, h75_error *e) {
    bool expected = false;
    if (!atomic_compare_exchange_strong(&physical_active, &expected, true))
        return h75_fail(e, H75_BUSY, "one physical session is allowed per process");
    r->physical_claim = true;
    if (INJECT(7))
        return h75_fail(e, H75_IO, "injected failure before any hardware access");
#ifdef H75_TESTING
    /* The pin-free executable can exercise the actual scanner against anonymous
     * RAM. This branch precedes ALL device/lock opens and is absent from libs. */
    unsigned model = atomic_load(&gpio_model_mock);
    if (model) {
        r->gpio_model = (int)model;
        if (r->config.output_mode == H75_OUTPUT_WEIGHTED_BCM && model != 5)
            return h75_fail(e, H75_UNSUPPORTED, "weighted BCM requires Pi 5 RIO");
        void *memory = INJECT(8) ? MAP_FAILED
                                 : mmap(NULL, 64 * 1024 * 1024, PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (memory == MAP_FAILED)
            return h75_fail(e, H75_MEMORY, "allocating RAM GPIO model");
        r->gpio = memory;
        gpio_blank(r);
        return H75_OK;
    }
#endif
    r->lock_fd = h75_gpio_lock_acquire();
    if (r->lock_fd < 0)
        return h75_fail(e, H75_BUSY, "cannot acquire GPIO advisory lock: %s", strerror(errno));
    r->gpio_model = cpu_get_pi_model();
    if (r->gpio_model < 3 || r->gpio_model > 5)
        return h75_fail(e, H75_UNSUPPORTED, "GPIO requires a supported Raspberry Pi 3/4/5");
    if (r->config.output_mode == H75_OUTPUT_WEIGHTED_BCM && r->gpio_model != 5)
        return h75_fail(e, H75_UNSUPPORTED, "weighted BCM requires Pi 5 RIO");
    int fd =
        open(r->gpio_model == 5 ? "/dev/gpiomem0" : "/dev/gpiomem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0)
        return h75_fail(e, H75_IO, "opening GPIO: %s", strerror(errno));
    off_t offset = r->gpio_model == 5 ? PERI5_BASE
                                      : (r->gpio_model == 4 ? PERI4_BASE + GPIO4_OFFSET
                                                            : PERI3_BASE + GPIO3_OFFSET);
    void *p = mmap(NULL, 64 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    int saved = errno;
    close(fd);
    if (p == MAP_FAILED)
        return h75_fail(e, H75_IO, "mapping GPIO: %s", strerror(saved));
    r->gpio = p;
    return H75_OK;
}
static void physical_close(h75_runtime *r) {
    if (r->gpio) {
        gpio_blank(r);
        munmap(r->gpio, 64 * 1024 * 1024);
        r->gpio = NULL;
    }
    if (r->lock_fd >= 0) {
        close(r->lock_fd);
        r->lock_fd = -1;
    }
    if (r->physical_claim) {
        atomic_store(&physical_active, false);
        r->physical_claim = false;
    }
}
h75_status h75_runtime_start(h75_runtime *r, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    if (pthread_mutex_trylock(&r->join_mutex))
        return h75_fail(e, H75_BUSY, "another lifecycle operation is in progress");
    pthread_mutex_lock(&r->mutex);
    h75_status s = H75_OK;
    bool partial_failure = false;
    if (r->state != H75_CREATED || atomic_load(&r->stop)) {
        s = h75_fail(e, H75_STATE, "display is one-shot and has already started or closed");
        goto done;
    }
    if (!r->config.offscreen) {
        s = physical_open(r, e);
        if (s)
            goto failed;
    }
    atomic_store(&r->sink_done, false);
    if (INJECT(5) || pthread_create(&r->sink_thread, NULL, sink_main, r)) {
        atomic_store(&r->sink_done, true);
        s = h75_fail(e, H75_IO, "starting frame consumer");
        goto failed;
    }
    r->sink_started = true;
    if (!r->config.offscreen) {
        atomic_store(&r->gpio_done, false);
        if (INJECT(9) || pthread_create(&r->gpio_thread, NULL, gpio_main, r)) {
            atomic_store(&r->gpio_done, true);
            s = h75_fail(e, H75_IO, "starting GPIO scanner");
            goto failed;
        }
        r->gpio_started = true;
        while (!atomic_load(&r->gpio_ready) && !atomic_load(&r->gpio_done) &&
               !atomic_load(&r->stop))
            pause_ms();
        if (!atomic_load(&r->gpio_ready)) {
            s = worker_status(r, e);
            if (!s)
                s = h75_fail(e, H75_CANCELLED, "GPIO startup cancelled");
            goto failed;
        }
    }
    if (atomic_load(&r->stop)) {
        s = worker_status(r, e);
        if (!s)
            s = h75_fail(e, H75_CANCELLED, "startup cancelled");
        goto failed;
    }
    h75_state expected = H75_CREATED;
    if (!atomic_compare_exchange_strong(&r->state, &expected, H75_RUNNING)) {
        s = worker_status(r, e);
        if (!s)
            s = h75_fail(e, H75_CANCELLED, "startup cancelled");
        goto failed;
    }
    goto done;
failed:
    partial_failure = true;
    atomic_store(&r->stop, true);
    r->state = H75_STOPPING;
done:
    pthread_mutex_unlock(&r->mutex);
    pthread_mutex_unlock(&r->join_mutex);
    if (partial_failure)
        h75_runtime_close(r, -1, NULL);
    return s;
}
h75_status h75_runtime_error(h75_runtime *r, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    return worker_status(r, e);
}
h75_state h75_runtime_get_state(h75_runtime *r) {
    if (!r)
        return H75_CLOSED;
    h75_state state = atomic_load(&r->state);
    return state == H75_RUNNING && atomic_load(&r->stop) ? H75_STOPPING : state;
}
void h75_runtime_request_stop(h75_runtime *r) {
    if (!r)
        return;
    atomic_store(&r->stop, true);
    h75_state state = atomic_load(&r->state);
    while ((state == H75_CREATED || state == H75_RUNNING) &&
           !atomic_compare_exchange_weak(&r->state, &state, H75_STOPPING)) {
    }
    atomic_store(&r->content_stop, true);
    if (r->display)
        atomic_store(&r->display->do_render, false);
}
static bool expired(uint64_t begin, int timeout) {
    return timeout >= 0 && h75_now_ns() - begin >= (uint64_t)timeout * 1000000u;
}
static void pause_ms(void) {
    struct timespec t = {0, 1000000};
    nanosleep(&t, NULL);
}
static h75_status lock_deadline(pthread_mutex_t *mutex, int timeout, uint64_t begin, h75_error *e) {
    for (;;) {
        int rc = pthread_mutex_trylock(mutex);
        if (rc == 0)
            return H75_OK;
        if (rc != EBUSY)
            return h75_fail(e, H75_IO, "locking runtime: %s", strerror(rc));
        if (expired(begin, timeout))
            return h75_fail(e, H75_TIMEOUT, "runtime operation is still in progress");
        pause_ms();
    }
}
static h75_status join_one(pthread_t thread, atomic_bool *done, int timeout, uint64_t begin,
                           h75_error *e) {
    if (pthread_equal(thread, pthread_self()))
        return h75_fail(e, H75_STATE, "worker cannot join itself");
    while (!atomic_load(done)) {
        if (expired(begin, timeout))
            return h75_fail(e, H75_TIMEOUT, "worker is still stopping; resources remain live");
        pause_ms();
    }
    int rc = pthread_join(thread, NULL);
    return rc ? h75_fail(e, H75_IO, "joining worker: %s", strerror(rc)) : H75_OK;
}
/* Caller holds join_mutex, not mutex (producer needs mutex to finish). */
static h75_status join_content(h75_runtime *r, int timeout, uint64_t begin, h75_error *e) {
    if (!r->content_started)
        return H75_OK;
    h75_status s = join_one(r->content_thread, &r->content_done, timeout, begin, e);
    if (s)
        return s;
    pthread_mutex_lock(&r->mutex);
    r->content_started = false;
    atomic_store(&r->paused, false);
    h75_resource_release(r->content);
    r->content = NULL;
    free(r->path);
    r->path = NULL;
    pthread_mutex_unlock(&r->mutex);
    return H75_OK;
}
h75_status h75_runtime_wait(h75_runtime *r, int timeout, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    uint64_t begin = h75_now_ns();
    h75_status s = lock_deadline(&r->join_mutex, timeout, begin, e);
    if (s)
        return s;
    s = join_content(r, timeout, begin, e);
    if (s)
        goto done;
    if (r->sink_started) {
        s = join_one(r->sink_thread, &r->sink_done, timeout, begin, e);
        if (s)
            goto done;
        r->sink_started = false;
    }
    if (r->gpio_started) {
        s = join_one(r->gpio_thread, &r->gpio_done, timeout, begin, e);
        if (s)
            goto done;
        r->gpio_started = false;
    }
    s = lock_deadline(&r->mutex, timeout, begin, e);
    if (s)
        goto done;
    if (r->state != H75_CLOSED && r->state != H75_CREATED)
        r->state = H75_STOPPED;
    pthread_mutex_unlock(&r->mutex);
    physical_close(r);
    s = worker_status(r, e);
done:
    pthread_mutex_unlock(&r->join_mutex);
    return s;
}
h75_status h75_runtime_close(h75_runtime *r, int timeout, h75_error *e) {
    if (!r)
        return H75_OK;
    uint64_t begin = h75_now_ns();
    h75_runtime_request_stop(r);
    h75_status s = h75_runtime_wait(r, timeout, e);
    if (s == H75_TIMEOUT)
        return s;
    h75_status lock_status = lock_deadline(&r->join_mutex, timeout, begin, e);
    if (lock_status)
        return lock_status;
    if (r->sink_started || r->gpio_started || r->content_started) {
        pthread_mutex_unlock(&r->join_mutex);
        return s;
    }
    lock_status = lock_deadline(&r->mutex, timeout, begin, e);
    if (lock_status) {
        pthread_mutex_unlock(&r->join_mutex);
        return lock_status;
    }
    if (r->leased) {
        spsc_push_ptr_cancel(r->display->ring_buf_mapper);
        r->leased = false;
    }
    if (r->display) {
        r->display->frame_buffer.data = NULL;
        r->display->image = NULL;
    }
    r->state = H75_CLOSED;
    pthread_mutex_unlock(&r->mutex);
    pthread_mutex_unlock(&r->join_mutex);
    return s;
}
void h75_runtime_destroy(h75_runtime *r) {
    if (!r)
        return;
    h75_runtime_close(r, -1, NULL);
    /* A join error must leak a still-live runtime rather than free a worker's
     * memory. Normal and timeout-capable close paths retain a usable handle. */
    if (r->state != H75_CLOSED)
        return;
    if (r->display) {
        spsc_destroy(r->display->ring_buf_mapper);
        spsc_destroy(r->display->ring_buf_renderer);
        free(r->display->accum);
        free(r->display->quant_errors_lut);
        free(r->display);
    }
    free(r->latest);
    pthread_mutex_destroy(&r->mutex);
    pthread_mutex_destroy(&r->join_mutex);
    pthread_mutex_destroy(&r->read_mutex);
    pthread_mutex_destroy(&r->error_mutex);
    free(r);
}
static h75_status acquire(h75_runtime *r, int timeout, bool native, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    uint64_t begin = h75_now_ns();
    h75_status s = lock_deadline(&r->mutex, timeout, begin, e);
    if (s)
        return s;
    if (r->state != H75_RUNNING || atomic_load(&r->stop)) {
        s = h75_fail(e, H75_STATE, "display is not running");
        goto done;
    }
    if (r->leased || (!native && r->content_started)) {
        s = h75_fail(e, H75_BUSY, "a frame or native content producer is already active");
        goto done;
    }
    for (;;) {
        if (atomic_load(&r->stop) || (native && atomic_load(&r->content_stop))) {
            s = h75_fail(e, H75_CANCELLED, "frame acquisition cancelled");
            break;
        }
        int slice = timeout == 0 ? 0 : 10;
        if (timeout > 0) {
            uint64_t elapsed = (h75_now_ns() - begin) / 1000000u;
            if (elapsed >= (uint64_t)timeout) {
                s = h75_fail(e, H75_TIMEOUT, "frame queue full");
                break;
            }
            if ((uint64_t)slice > (uint64_t)timeout - elapsed)
                slice = (int)((uint64_t)timeout - elapsed);
        }
        void *p = spsc_push_ptr_begin(r->display->ring_buf_mapper, slice);
        if (p) {
            r->leased = true;
            r->frame_owner = pthread_self();
            r->display->frame_buffer.data = p;
            r->display->image = p;
            memset(p, 0, r->bytes);
            break;
        }
        if (timeout == 0) {
            s = h75_fail(e, H75_TIMEOUT, "frame queue full");
            break;
        }
    }
    atomic_fetch_add(&r->queue_wait_ns, h75_now_ns() - begin);
done:
    pthread_mutex_unlock(&r->mutex);
    return s;
}
h75_status h75_frame_acquire(h75_runtime *r, int timeout, h75_error *e) {
    return acquire(r, timeout, false, e);
}
static h75_status frame_check(h75_runtime *r, h75_error *e) {
    if (!r->leased || r->state != H75_RUNNING || atomic_load(&r->stop))
        return h75_fail(e, H75_STATE, "no active frame on a running display");
    if (!pthread_equal(r->frame_owner, pthread_self()))
        return h75_fail(e, H75_STATE, "frame belongs to another thread");
    return H75_OK;
}
static h75_status finish(h75_runtime *r, bool submit, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    pthread_mutex_lock(&r->mutex);
    h75_status s = frame_check(r, e);
    if (!s) {
        if (submit) {
            size_t ordinal = atomic_load(&r->submitted) & (r->config.queue_depth - 1);
            atomic_store(&r->submission_times[ordinal], h75_now_ns());
            spsc_push_ptr_commit(r->display->ring_buf_mapper);
            atomic_fetch_add(&r->submitted, 1);
        } else
            spsc_push_ptr_cancel(r->display->ring_buf_mapper);
        r->leased = false;
        r->display->frame_buffer.data = NULL;
        r->display->image = NULL;
    }
    pthread_mutex_unlock(&r->mutex);
    return s;
}
h75_status h75_frame_submit(h75_runtime *r, h75_error *e) {
    return finish(r, true, e);
}
h75_status h75_frame_cancel(h75_runtime *r, h75_error *e) {
    return finish(r, false, e);
}
h75_status h75_frame_command(h75_runtime *r, const h75_command *c, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    pthread_mutex_lock(&r->mutex);
    h75_status s = frame_check(r, e);
    if (!s) {
        pthread_mutex_lock(&h75_resources_mutex);
        uint64_t begin = h75_now_ns();
        s = h75_command_execute(r->display, c, 0, e);
        atomic_fetch_add(&r->render_ns, h75_now_ns() - begin);
        pthread_mutex_unlock(&h75_resources_mutex);
    }
    pthread_mutex_unlock(&r->mutex);
    return s;
}
h75_status h75_frame_upload(h75_runtime *r, const void *data, size_t size, uint32_t pitch,
                            int channels, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    pthread_mutex_lock(&r->mutex);
    h75_status s = frame_check(r, e);
    if (!s) {
        uint64_t begin = h75_now_ns();
        s = h75_copy_rgba(&r->display->frame_buffer, data, size, pitch, channels, e);
        atomic_fetch_add(&r->render_ns, h75_now_ns() - begin);
    }
    pthread_mutex_unlock(&r->mutex);
    return s;
}
h75_status h75_frame_draw(h75_runtime *r, h75_resource *list, float time, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    pthread_mutex_lock(&r->mutex);
    h75_status s = frame_check(r, e);
    if (!s) {
        pthread_mutex_lock(&h75_resources_mutex);
        uint64_t begin = h75_now_ns();
        s = h75_list_execute(r->display, list, time, e);
        if (!atomic_load(&r->content_started))
            atomic_fetch_add(&r->render_ns, h75_now_ns() - begin);
        pthread_mutex_unlock(&h75_resources_mutex);
    }
    pthread_mutex_unlock(&r->mutex);
    return s;
}
h75_status h75_runtime_try_draw(h75_runtime *r, h75_resource *list, float time, bool *accepted,
                                h75_error *e) {
    if (accepted)
        *accepted = false;
    if (!r || !list || list->kind != H75_DRAWLIST || !accepted || !isfinite(time))
        return h75_fail(e, H75_INVALID,
                        "expected runtime, draw list, finite time and acceptance output");
    if (atomic_load(&r->content_started))
        return h75_fail(e, H75_BUSY, "stop native playback before manual submission");
    h75_status s = acquire(r, 0, false, e);
    if (s == H75_TIMEOUT) {
        atomic_fetch_add(&r->dropped, 1);
        if (e)
            *e = (h75_error){0};
        return H75_OK;
    }
    if (s)
        return s;
    s = h75_frame_draw(r, list, time, e);
    if (s) {
        h75_frame_cancel(r, NULL);
        return s;
    }
    s = h75_frame_submit(r, e);
    if (!s)
        *accepted = true;
    else
        h75_frame_cancel(r, NULL);
    return s;
}
h75_status h75_runtime_read(h75_runtime *r, void *rgba, size_t size, h75_error *e) {
    if (!r || !rgba || size != r->bytes)
        return h75_fail(e, H75_INVALID, "readback size mismatch");
    pthread_mutex_lock(&r->read_mutex);
    memcpy(rgba, r->latest, size);
    pthread_mutex_unlock(&r->read_mutex);
    return H75_OK;
}
h75_status h75_runtime_flush(h75_runtime *r, int timeout, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    uint64_t begin = h75_now_ns(), target = atomic_load(&r->submitted);
    while (atomic_load(&r->consumed) < target) {
        h75_status s = worker_status(r, e);
        if (s)
            return s;
        if (atomic_load(&r->stop))
            return h75_fail(e, H75_CANCELLED, "display stopped before queue drained");
        if (expired(begin, timeout))
            return h75_fail(e, H75_TIMEOUT, "queue drain timed out");
        pause_ms();
    }
    return worker_status(r, e);
}
void h75_runtime_stats(h75_runtime *r, h75_stats *out) {
    if (!out)
        return;
    if (!r) {
        *out = (h75_stats){.state = H75_CLOSED};
        return;
    }
    pthread_mutex_lock(&r->mutex);
    *out = (h75_stats){.submitted = atomic_load(&r->submitted),
                       .consumed = atomic_load(&r->consumed),
                       .mapped = atomic_load(&r->mapped),
                       .scanned = atomic_load(&r->scanned),
                       .dropped = atomic_load(&r->dropped),
                       .queue_wait_ns = atomic_load(&r->queue_wait_ns),
                       .render_ns = atomic_load(&r->render_ns),
                       .map_ns = atomic_load(&r->map_ns),
                       .copy_ns = atomic_load(&r->copy_ns),
                       .scan_ns = atomic_load(&r->scan_ns),
                       .latency_ns = atomic_load(&r->latency_ns),
                       .latency_total_ns = atomic_load(&r->latency_total_ns),
                       .queue_occupancy = (uint32_t)spsc_count(r->display->ring_buf_mapper),
                       .state = (atomic_load(&r->stop) && r->state == H75_RUNNING) ? H75_STOPPING
                                                                                   : r->state,
                       .playing = r->content_started && !atomic_load(&r->content_done),
                       .paused = atomic_load(&r->paused),
                       .ready = atomic_load(&r->init_done)};
    pthread_mutex_unlock(&r->mutex);
}
static uint64_t pace(h75_runtime *r, uint64_t deadline) {
    uint64_t paused_ns = 0;
    while (!atomic_load(&r->stop) && !atomic_load(&r->content_stop)) {
        if (atomic_load(&r->paused)) {
            uint64_t begin = h75_now_ns();
            while (atomic_load(&r->paused) && !atomic_load(&r->stop) &&
                   !atomic_load(&r->content_stop))
                pause_ms();
            uint64_t duration = h75_now_ns() - begin;
            paused_ns += duration;
            deadline += duration;
        }
        uint64_t now = h75_now_ns();
        if (now >= deadline)
            break;
        uint64_t ns = deadline - now;
        if (ns > 10000000)
            ns = 10000000;
        struct timespec t = {.tv_sec = 0, .tv_nsec = (long)ns};
        nanosleep(&t, NULL);
    }
    return paused_ns;
}
static bool presentation_deadline(uint64_t start, float pts, float scale, uint64_t *deadline,
                                  h75_error *e) {
    float seconds = pts / scale;
    if (!isfinite(seconds) || seconds < 0 || seconds > 1e9f) {
        h75_fail(e, H75_INVALID, "video timestamp exceeds supported presentation range");
        return false;
    }
    *deadline = start + (uint64_t)(seconds * 1e9f);
    return true;
}
static void *content_main(void *arg) {
    h75_runtime *r = arg;
    h75_error e = {0};
    h75_media *media = NULL;
    h75_status s = H75_OK;
    uint64_t start = h75_now_ns(), previous = start, index = 0;
    if (r->path)
        s = h75_media_open(&media, r->path, r->video, r->config.width, r->config.height,
                           r->config.video_preserve_aspect, &r->content_stop, &e);
    start = previous = h75_now_ns();
    if (s && !atomic_load(&r->stop) && !atomic_load(&r->content_stop))
        worker_fail(r, &e);
    atomic_store(&r->init_done, true);
    while (!s && !atomic_load(&r->stop) && !atomic_load(&r->content_stop)) {
        if (atomic_load(&r->paused)) {
            uint64_t duration = pace(r, h75_now_ns());
            start += duration;
            previous += duration;
        }
        s = acquire(r, 20, true, &e);
        if (s == H75_TIMEOUT) {
            s = H75_OK;
            continue;
        }
        if (s)
            break;
        uint64_t now = h75_now_ns();
        float time = (float)(now - start) * 1e-9f * r->time_scale;
        float dt = (float)(now - previous) * 1e-9f * r->time_scale, pts = -1;
        previous = now;
        if (media) {
            int result =
                h75_media_next(media, (uint8_t *)r->display->frame_buffer.data, time, dt, &pts, &e);
            if (result == 1) {
                h75_frame_cancel(r, NULL);
                if (r->loop && !atomic_load(&r->content_stop)) {
                    uint64_t deadline;
                    if (!presentation_deadline(start, pts, r->time_scale, &deadline, &e)) {
                        s = e.code;
                        break;
                    }
                    pace(r, deadline);
                    if (atomic_load(&r->stop) || atomic_load(&r->content_stop))
                        break;
                    h75_media_close(media);
                    media = NULL;
                    s = h75_media_open(&media, r->path, r->video, r->config.width, r->config.height,
                                       r->config.video_preserve_aspect, &r->content_stop, &e);
                    start = previous = h75_now_ns();
                    index = 0;
                    continue;
                }
                break;
            }
            if (result < 0)
                s = e.code ? e.code : H75_IO;
        }
        if (!s && r->content)
            s = h75_frame_draw(r, r->content, time, &e);
        atomic_fetch_add(&r->render_ns, h75_now_ns() - now);
        if (s) {
            h75_frame_cancel(r, NULL);
            break;
        }
        /* Video timestamps govern presentation, not decoder throughput. */
        if (r->video && pts >= 0) {
            uint64_t deadline;
            if (!presentation_deadline(start, pts, r->time_scale, &deadline, &e)) {
                s = e.code;
                h75_frame_cancel(r, NULL);
                break;
            }
            uint64_t duration = pace(r, deadline);
            start += duration;
            previous += duration;
        }
        s = h75_frame_submit(r, &e);
        if (s)
            break;
        index++;
        if (!r->video) {
            uint64_t duration = pace(r, start + index * 1000000000u / r->config.fps);
            start += duration;
            previous += duration;
        }
    }
    if (media)
        h75_media_close(media);
    if (s && s != H75_CANCELLED && !atomic_load(&r->stop) && !atomic_load(&r->content_stop))
        worker_fail(r, &e);
    /* Cancellation after stop is cleaned by close; otherwise finish the lease. */
    pthread_mutex_lock(&r->mutex);
    if (r->leased && pthread_equal(r->frame_owner, pthread_self())) {
        spsc_push_ptr_cancel(r->display->ring_buf_mapper);
        r->leased = false;
        r->display->frame_buffer.data = NULL;
        r->display->image = NULL;
    }
    pthread_mutex_unlock(&r->mutex);
    atomic_store(&r->content_done, true);
    return NULL;
}
static h75_status play(h75_runtime *r, h75_resource *list, const char *path, bool video, bool loop,
                       float scale, h75_error *e) {
    if (!isfinite(scale) || scale < 0.001f || scale > 100)
        return h75_fail(e, H75_INVALID, "time scale must be finite and in [0.001,100]");
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    if (pthread_mutex_trylock(&r->join_mutex))
        return h75_fail(e, H75_BUSY, "another lifecycle operation is in progress");
    pthread_mutex_lock(&r->mutex);
    h75_status s = H75_OK;
    if (r->state != H75_RUNNING || atomic_load(&r->stop)) {
        s = h75_fail(e, H75_STATE, "display is not running");
        goto done;
    }
    if (r->leased || r->content_started) {
        s = h75_fail(e, H75_BUSY, "stop the previous producer before changing content mode");
        goto done;
    }
    if (list && h75_resource_kind(list) != H75_DRAWLIST) {
        s = h75_fail(e, H75_INVALID, "expected retained draw list");
        goto done;
    }
    if (path && !h75_media_open) {
        s = h75_fail(e, H75_UNSUPPORTED,
                     "GPU/video support was not compiled into this installation");
        goto done;
    }
    if (path) {
        r->path = strdup(path);
        if (!r->path) {
            s = h75_fail(e, H75_MEMORY, "copying media path");
            goto done;
        }
    }
    r->content = list;
    h75_resource_retain(list);
    r->video = video;
    r->loop = loop;
    r->time_scale = scale;
    pthread_mutex_lock(&r->error_mutex);
    memset(&r->worker_error, 0, sizeof(r->worker_error));
    pthread_mutex_unlock(&r->error_mutex);
    atomic_store(&r->content_stop, false);
    atomic_store(&r->content_done, false);
    atomic_store(&r->init_done, false);
    atomic_store(&r->paused, false);
    if (INJECT(6) || pthread_create(&r->content_thread, NULL, content_main, r)) {
        h75_resource_release(r->content);
        r->content = NULL;
        free(r->path);
        r->path = NULL;
        atomic_store(&r->content_done, true);
        s = h75_fail(e, H75_IO, "starting content producer");
        goto done;
    }
    r->content_started = true;
done:
    pthread_mutex_unlock(&r->mutex);
    pthread_mutex_unlock(&r->join_mutex);
    return s;
}
h75_status h75_runtime_play(h75_runtime *r, h75_resource *list, h75_error *e) {
    if (!list)
        return h75_fail(e, H75_INVALID, "null retained draw list");
    return play(r, list, NULL, false, false, 1, e);
}
h75_status h75_runtime_play_file(h75_runtime *r, const char *path, bool video, bool loop,
                                 float scale, h75_resource *overlay, h75_error *e) {
    if (!path || !path[0])
        return h75_fail(e, H75_INVALID, "empty media path");
    return play(r, overlay, path, video, loop, scale, e);
}
h75_status h75_runtime_content_ready(h75_runtime *r, int timeout, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    uint64_t begin = h75_now_ns();
    while (!atomic_load(&r->init_done)) {
        if (atomic_load(&r->content_done))
            return h75_fail(e, H75_STATE, "no content producer has been started");
        if (expired(begin, timeout))
            return h75_fail(e, H75_TIMEOUT, "content is still initializing");
        pause_ms();
    }
    return worker_status(r, e);
}
h75_status h75_runtime_pause(h75_runtime *r, bool paused, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    pthread_mutex_lock(&r->mutex);
    h75_status s = H75_OK;
    if (!r->content_started || atomic_load(&r->content_done))
        s = h75_fail(e, H75_STATE, "no active content producer");
    else
        atomic_store(&r->paused, paused);
    pthread_mutex_unlock(&r->mutex);
    return s;
}
h75_status h75_runtime_stop_content(h75_runtime *r, int timeout, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    atomic_store(&r->content_stop, true);
    return h75_runtime_wait_content(r, timeout, e);
}
h75_status h75_runtime_wait_content(h75_runtime *r, int timeout, h75_error *e) {
    if (!r)
        return h75_fail(e, H75_INVALID, "null runtime");
    uint64_t begin = h75_now_ns();
    h75_status s = lock_deadline(&r->join_mutex, timeout, begin, e);
    if (s)
        return s;
    s = join_content(r, timeout, begin, e);
    if (!s)
        s = worker_status(r, e);
    pthread_mutex_unlock(&r->join_mutex);
    return s;
}
