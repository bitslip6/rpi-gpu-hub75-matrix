#ifndef H75_RUNTIME_DEMO_H
#define H75_RUNTIME_DEMO_H
/* CLI only. Shared by ./example --runtime and the native runtime demo.
 * Offscreen by default; never use --gpio before verifying wiring/topology. */
#include "hub75_runtime.h"
#include "runtime_dashboard_metrics.h"
#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

static volatile sig_atomic_t runtime_demo_stop;
static void runtime_demo_signal(int sig) {
    (void)sig;
    runtime_demo_stop = 1;
}
static uint64_t runtime_demo_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static uint64_t runtime_demo_cpu_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static int runtime_demo_compare(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
/* Diagnostic-only UI: no renderer/scanner thread or public ABI changes. */
typedef enum {
    RUNTIME_STYLE_NORMAL, RUNTIME_STYLE_TITLE, RUNTIME_STYLE_GOOD,
    RUNTIME_STYLE_ACCENT, RUNTIME_STYLE_MUTED, RUNTIME_STYLE_WARNING, RUNTIME_STYLE_ERROR
} runtime_demo_style;

typedef struct {
    bool enabled, terminal, color;
    unsigned columns, rows, row;
    uint64_t started, previous_ns, previous_cpu;
    h75_stats previous;
    runtime_demo_cpu_window cpu_window;
} runtime_demo_dashboard;

typedef struct {
    char text[416];
} runtime_demo_value;

typedef struct {
    size_t start, length;
} runtime_demo_span;

static runtime_demo_value runtime_demo_fixed(const char *format, ...) {
    runtime_demo_value value;
    va_list args;
    va_start(args, format);
    vsnprintf(value.text, sizeof(value.text), format, args);
    va_end(args);
    return value;
}

static void runtime_demo_dashboard_render(runtime_demo_dashboard *d, runtime_demo_style style,
                                          char *line, const runtime_demo_span *spans, unsigned count) {
    /* Source paths must never inject terminal controls or break snapshot rows. */
    for (char *p = line; *p; ++p)
        if ((unsigned char)*p < 32 || (unsigned char)*p == 127) *p = '?';
    if (!d->terminal) {
        fprintf(stderr, "%s\n", line);
        return;
    }
    if (++d->row >= d->rows) return;
    static const char *const labels[] = {
        "\033[0;37m", "\033[0;1;36m", "\033[0;1;32m", "\033[0;36m", "\033[0;2m",
        "\033[0;1;33m", "\033[0;1;31m"
    };
    /* Clip visible text, not ANSI bytes; never split a color escape sequence.
     * Restore label style after each value, including bold/dim attributes. */
    size_t visible = strlen(line), offset = 0;
    if (visible >= d->columns) visible = d->columns - 1;
    fputs("\033[2K", stderr);
    if (d->color) fputs(labels[style], stderr);
    for (unsigned i = 0; i < count && spans[i].start < visible; ++i) {
        size_t start = spans[i].start, length = spans[i].length;
        fwrite(line + offset, 1, start - offset, stderr);
        if (length > visible - start) length = visible - start;
        if (d->color) fputs("\033[0;94m", stderr);
        fwrite(line + start, 1, length, stderr);
        if (d->color) fputs(labels[style], stderr);
        offset = start + length;
    }
    fwrite(line + offset, 1, visible - offset, stderr);
    if (d->color) fputs("\033[0m", stderr);
    fputc('\n', stderr);
}

static void runtime_demo_dashboard_line(runtime_demo_dashboard *d, runtime_demo_style style,
                                         const char *format, ...) {
    char line[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    runtime_demo_dashboard_render(d, style, line, NULL, 0);
}

/* Internal templates use {} for individually styled values. Templates and
 * control sequences are trusted; all inserted text is sanitized by render. */
static void runtime_demo_dashboard_fields(runtime_demo_dashboard *d, runtime_demo_style style,
                                           const char *format, const runtime_demo_value *values,
                                           unsigned count) {
    char line[1024];
    runtime_demo_span spans[16];
    size_t length = 0;
    unsigned used = 0;
    for (const char *p = format; *p && length < sizeof(line) - 1; ++p) {
        if (p[0] == '{' && p[1] == '}' && used < count && used < 16) {
            size_t n = strlen(values[used].text);
            if (n > sizeof(line) - 1 - length) n = sizeof(line) - 1 - length;
            spans[used] = (runtime_demo_span){length, n};
            memcpy(line + length, values[used].text, n);
            length += n;
            ++used;
            ++p;
        } else line[length++] = *p;
    }
    line[length] = '\0';
    runtime_demo_dashboard_render(d, style, line, spans, used);
}

#define RUNTIME_FIELDS(d, style, text, ...) do { \
    const runtime_demo_value fields_[] = {__VA_ARGS__}; \
    runtime_demo_dashboard_fields((d), (style), (text), fields_, \
        (unsigned)(sizeof(fields_) / sizeof(fields_[0]))); \
} while (0)

static void runtime_demo_dashboard_begin(runtime_demo_dashboard *d, bool enabled,
                                         h75_runtime *r) {
    *d = (runtime_demo_dashboard){.enabled = enabled, .columns = 80, .rows = 24};
    if (!enabled) return;
    const char *term = getenv("TERM");
    d->terminal = isatty(STDERR_FILENO) && term && strcmp(term, "dumb");
    const char *no_color = getenv("NO_COLOR");
    d->color = d->terminal && !(no_color && *no_color);
    d->started = d->previous_ns = runtime_demo_ns();
    d->previous_cpu = runtime_demo_cpu_ns();
    h75_runtime_stats(r, &d->previous);
    if (d->terminal) {
        /* An alternate screen preserves shell scrollback; always restore on exit. */
        struct winsize size;
        if (!ioctl(STDERR_FILENO, TIOCGWINSZ, &size) && size.ws_col && size.ws_row) {
            d->columns = size.ws_col;
            d->rows = size.ws_row;
        }
        fputs("\033[?1049h\033[?25l\033[2J\033[H", stderr);
        runtime_demo_dashboard_line(d, RUNTIME_STYLE_TITLE, "HUB75 / LIVE RUNTIME");
        runtime_demo_dashboard_line(d, RUNTIME_STYLE_MUTED, "Collecting first 0.5 s sample...");
        fflush(stderr);
    }
}

static void runtime_demo_dashboard_update(runtime_demo_dashboard *d, h75_runtime *r,
                                          const h75_config *c, const char *source, float speed,
                                          bool force) {
    if (!d->enabled) return;
    uint64_t now = runtime_demo_ns(), interval = now - d->previous_ns;
    if (!force && interval < RUNTIME_DEMO_REFRESH_NS) return;
    if (!interval) return;
    h75_stats stats;
    h75_runtime_stats(r, &stats);
    uint64_t cpu = runtime_demo_cpu_ns();
    uint64_t submitted = stats.submitted - d->previous.submitted;
    uint64_t consumed = stats.consumed - d->previous.consumed;
    uint64_t mapped = stats.mapped - d->previous.mapped;
    uint64_t scanned = stats.scanned - d->previous.scanned;
    float hz = 1e9f / (float)interval;
    float cpu_percent = runtime_demo_cpu_average(&d->cpu_window, cpu - d->previous_cpu, interval);
    float latency = consumed ? (float)(stats.latency_total_ns - d->previous.latency_total_ns) /
                                   (float)consumed / 1e6f : 0;
    float render = submitted ? (float)(stats.render_ns - d->previous.render_ns) /
                                  (float)submitted / 1e6f : 0;
    float wait = submitted ? (float)(stats.queue_wait_ns - d->previous.queue_wait_ns) /
                                (float)submitted / 1e6f : 0;
    float map = mapped ? (float)(stats.map_ns - d->previous.map_ns) / (float)mapped / 1e6f : 0;
    float copy = consumed ? (float)(stats.copy_ns - d->previous.copy_ns) / (float)consumed / 1e6f : 0;
    char memory[48] = "n/a";
    float resident_mib = 0;
    FILE *statm = fopen("/proc/self/statm", "r");
    unsigned long total_pages, resident_pages;
    long page_size = sysconf(_SC_PAGESIZE);
    if (statm) {
        if (fscanf(statm, "%lu %lu", &total_pages, &resident_pages) == 2 && page_size > 0) {
            resident_mib = (float)resident_pages * (float)page_size / 1048576.f;
            snprintf(memory, sizeof(memory), "%.1f MiB", (double)resident_mib);
        }
        fclose(statm);
    }
    uint64_t uptime = (now - d->started) / 1000000000u;
    const char *state = stats.state == H75_RUNNING ? "RUNNING" :
                        stats.state == H75_STOPPING ? "STOPPING" :
                        stats.state == H75_STOPPED ? "STOPPED" :
                        stats.state == H75_CLOSED ? "CLOSED" : "CREATED";
    if (stats.paused) state = "PAUSED";
    if (d->terminal) {
        struct winsize size;
        if (!ioctl(STDERR_FILENO, TIOCGWINSZ, &size) && size.ws_col && size.ws_row) {
            d->columns = size.ws_col;
            d->rows = size.ws_row;
        }
        d->row = 0;
        fputs("\033[H", stderr);
    }
    runtime_demo_dashboard_line(d, RUNTIME_STYLE_TITLE,
        "HUB75  /  LIVE RUNTIME                              Ctrl+C to stop");
    RUNTIME_FIELDS(d, stats.paused ? RUNTIME_STYLE_WARNING :
        stats.state == H75_RUNNING ? RUNTIME_STYLE_GOOD : RUNTIME_STYLE_MUTED,
        "{}  |  up {}  |  UI {} ms / sample {} s",
        runtime_demo_fixed("%s", state),
        runtime_demo_fixed("%02" PRIu64 ":%02" PRIu64 ":%02" PRIu64,
                           uptime / 3600, uptime / 60 % 60, uptime % 60),
        runtime_demo_fixed("%" PRIu64, RUNTIME_DEMO_REFRESH_NS / UINT64_C(1000000)),
        runtime_demo_fixed("%.2f", (double)((float)interval / 1e9f)));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Canvas {}  |  panels {}  |  {} chains x {} ports",
        runtime_demo_fixed("%ux%u", c->width, c->height),
        runtime_demo_fixed("%ux%u", c->panel_width, c->panel_height),
        runtime_demo_fixed("%u", c->chains), runtime_demo_fixed("%u", c->ports));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Output {} |  {} |  curve {}  |  {} bits",
        runtime_demo_fixed("%-9s", c->offscreen ? "offscreen" : "GPIO"),
        runtime_demo_fixed("%-12s", c->output_mode == H75_OUTPUT_WEIGHTED_BCM ? "weighted-bcm" : "equal-slot"),
        runtime_demo_fixed("%s", c->color_curve == H75_COLOR_CIE ? "CIE" : "gamma"),
        runtime_demo_fixed("%u", c->output_mode == H75_OUTPUT_WEIGHTED_BCM ? c->pwm_bits : c->bit_depth));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Brightness {}/254  |  playback speed {}x",
        runtime_demo_fixed("%u", c->brightness), runtime_demo_fixed("%.3g", (double)speed));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Dither temporal {}  |  spatial {}{}",
        runtime_demo_fixed("%s", c->output_mode == H75_OUTPUT_WEIGHTED_BCM ? "n/a" :
                           c->quant_dither ? "on" : "off"),
        runtime_demo_fixed("%.0f", (double)c->dither),
        runtime_demo_fixed("%s", c->offscreen ? " (offscreen: not applied)" : ""));
    RUNTIME_FIELDS(d, (float)consumed * hz < (float)c->fps * .9f ?
        RUNTIME_STYLE_WARNING : RUNTIME_STYLE_GOOD, "Content FPS {}  / target {}  |  submitted {} fps",
        runtime_demo_fixed("%7.1f", (double)((float)consumed * hz)),
        runtime_demo_fixed("%u", c->fps),
        runtime_demo_fixed("%7.1f", (double)((float)submitted * hz)));
    if (c->offscreen) {
        RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Panel refresh {}  (offscreen: no GPIO scanning)",
            runtime_demo_fixed("    n/a"));
    } else {
        float cycle = scanned ? (float)(stats.scan_ns - d->previous.scan_ns) / (float)scanned / 1e6f : 0;
        RUNTIME_FIELDS(d, RUNTIME_STYLE_ACCENT, "Panel refresh {} Hz  |  cycle {} ms  |  map {} fps",
            runtime_demo_fixed("%7.1f", (double)((float)scanned * hz)),
            runtime_demo_fixed("%.3f", (double)cycle),
            runtime_demo_fixed("%.1f", (double)((float)mapped * hz)));
    }
    RUNTIME_FIELDS(d, RUNTIME_STYLE_ACCENT, "CPU {}% (2s avg; 100% = 1 core)  |  RSS {}",
        runtime_demo_fixed("%6.1f", (double)cpu_percent), runtime_demo_fixed("%s", memory));
    RUNTIME_FIELDS(d, stats.dropped > d->previous.dropped ? RUNTIME_STYLE_ERROR :
        stats.queue_occupancy == c->queue_depth ? RUNTIME_STYLE_WARNING : RUNTIME_STYLE_NORMAL,
        "Input queue {}/{}  |  map drops +{}  / total {}",
        runtime_demo_fixed("%u", stats.queue_occupancy), runtime_demo_fixed("%u", c->queue_depth),
        runtime_demo_fixed("%" PRIu64, stats.dropped - d->previous.dropped),
        runtime_demo_fixed("%" PRIu64, stats.dropped));
    float last_latency = (float)stats.latency_ns / 1e6f;
    RUNTIME_FIELDS(d, latency > 1000.f / (float)c->fps ? RUNTIME_STYLE_WARNING : RUNTIME_STYLE_NORMAL,
        "Latency avg {} ms  |  last {} ms (submit -> consume)",
        runtime_demo_fixed("%7.3f", (double)latency), runtime_demo_fixed("%7.3f", (double)last_latency));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Stage ms/frame: render {}  wait {}  map {}  copy {}",
        runtime_demo_fixed("%.3f", (double)render), runtime_demo_fixed("%.3f", (double)wait),
        runtime_demo_fixed("%.3f", (double)map), runtime_demo_fixed("%.3f", (double)copy));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_MUTED, "Totals: submitted {}  consumed {}  scans {}",
        runtime_demo_fixed("%" PRIu64, stats.submitted), runtime_demo_fixed("%" PRIu64, stats.consumed),
        runtime_demo_fixed("%" PRIu64, stats.scanned));
    RUNTIME_FIELDS(d, RUNTIME_STYLE_NORMAL, "Source: {}", runtime_demo_fixed("%.400s", source));
    if (d->terminal) fputs("\033[J", stderr);
    else fputc('\n', stderr);
    fflush(stderr);
    d->previous = stats;
    d->previous_ns = now;
    d->previous_cpu = cpu;
}

static void runtime_demo_dashboard_end(runtime_demo_dashboard *d) {
    if (d->terminal) {
        fputs("\033[?25h\033[?1049l", stderr);
        fflush(stderr);
        d->terminal = false;
    }
    d->enabled = false;
}

#undef RUNTIME_FIELDS

static int runtime_demo_argument_error(const char *option, const char *value) {
    if (value)
        fprintf(stderr, "runtime: invalid value '%s' for %s; use --help\n", value, option);
    else
        fprintf(stderr, "runtime: missing value for %s; use --help\n", option);
    return 2;
}

/* Inspect only the final path component, not dots in parent directories. */
static int runtime_demo_file_type(const char *path, bool *video) {
    const char *name = strrchr(path, '/');
    name = name ? name + 1 : path;
    const char *extension = strrchr(name, '.');
    static const char *const shaders[] = {".glsl", ".frag", ".fs"};
    static const char *const videos[] = {
        ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".ts"
    };
    if (extension) {
        for (size_t i = 0; i < sizeof(shaders) / sizeof(shaders[0]); ++i)
            if (!strcasecmp(extension, shaders[i])) {
                *video = false;
                return 0;
            }
        for (size_t i = 0; i < sizeof(videos) / sizeof(videos[0]); ++i)
            if (!strcasecmp(extension, videos[i])) {
                *video = true;
                return 0;
            }
    }
    fprintf(stderr, "runtime: unsupported file type for --file '%s'; expected shader "
                    "(.glsl, .frag, .fs) or video "
                    "(.mp4, .m4v, .mkv, .webm, .mov, .avi, .mpg, .mpeg, .ts)\n", path);
    return 2;
}

static int hub75_runtime_demo(int argc, char **argv) {
    h75_config c;
    h75_config_default(&c);
    c.offscreen = true;
    c.color_curve = H75_COLOR_CIE;
    c.output_mode = H75_OUTPUT_WEIGHTED_BCM;
    unsigned frames = 0, lines = 512, warmup = 50;
    int64_t seconds = -1;
    float speed = 1.f;
    bool ports_set = false, chains_set = false, verbose = false, speed_set = false;
    const char *dump = NULL, *media = NULL, *font = NULL;
    bool video = false, loop = false, retained = false, upload = false, scene = false,
         shadows = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v")) {
            verbose = true;
            continue;
        }
        if (!strcmp(argv[i], "--gpio")) {
            c.offscreen = false;
            continue;
        }
        if (!strcmp(argv[i], "--offscreen")) {
            c.offscreen = true;
            continue;
        }
        if (!strcmp(argv[i], "--quant-dither") || !strcmp(argv[i], "-q")) {
            c.quant_dither = true;
            continue;
        }
        if (!strcmp(argv[i], "--retained")) {
            retained = true;
            continue;
        }
        if (!strcmp(argv[i], "--loop")) {
            loop = true;
            continue;
        }
        if (!strcmp(argv[i], "--upload")) {
            upload = true;
            continue;
        }
        if (!strcmp(argv[i], "--shadows")) {
            shadows = true;
            continue;
        }
        if (!strcmp(argv[i], "--scene")) {
            scene = true;
            continue;
        }
        if (!strcmp(argv[i], "--file")) {
            if (++i >= argc || !strncmp(argv[i], "--", 2))
                return runtime_demo_argument_error("--file", NULL);
            media = argv[i];
            if (runtime_demo_file_type(media, &video))
                return 2;
            continue;
        }
        if (!strcmp(argv[i], "--dump")) {
            if (++i >= argc)
                return runtime_demo_argument_error("--dump", NULL);
            dump = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--text")) {
            if (++i >= argc)
                return runtime_demo_argument_error("--text", NULL);
            font = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "--help")) {
            puts("--offscreen (default) | --gpio (mandatory CPU3 scanner; launch taskset -c 0-2)\n"
                 "--width N --height N --ports N --chains N (omitted topology inferred from panel size)\n"
                 "--panel-width N --panel-height N --fps N --bit-depth N --queue-depth N\n"
                 "--brightness N (0..254; default 200; 0=off, 254=maximum)\n"
                 "--quant-dither, -q (temporal quantization dithering; default off)\n"
                 "--dither N, -l N (spatial dithering, integer 0..10; default 0/off)\n"
                 "  Spatial: fixed pattern in both modes; temporal: equal-slot only. GPIO output only.\n"
                 "--speed N (shader/video time scale 0.001..100; default 1; independent of FPS)\n"
                 "--output-mode bcm|weighted-bcm|equal-slot (default bcm) --pwm-bits N --pwm-lsb-ns N\n"
                 "--color-curve cie|gamma (default cie) --gamma N (weighted GPIO requires Pi 5 RIO)\n"
                 "--seconds N|-1|forever (default forever) --frames N --warmup N --lines N --dump rgba.bin\n"
                 "-v, --verbose (live runtime dashboard on stderr, updates every 500 ms)\n"
                 "--upload | --scene [--shadows] | --text font-directory | --retained\n"
                 "--file PATH [--loop] (GPU build only; type selected by extension, case-insensitive)\n"
                 "  Shader: .glsl .frag .fs\n"
                 "  Video: .mp4 .m4v .mkv .webm .mov .avi .mpg .mpeg .ts\n"
                 "  --loop repeats video playback");
            return 0;
        }
        const char *option = argv[i];
        if (++i >= argc)
            return runtime_demo_argument_error(option, NULL);
        if (!strcmp(option, "--output-mode")) {
            if (!strcmp(argv[i], "equal-slot")) c.output_mode = H75_OUTPUT_EQUAL_SLOT;
            else if (!strcmp(argv[i], "weighted-bcm") || !strcmp(argv[i], "bcm"))
                c.output_mode = H75_OUTPUT_WEIGHTED_BCM;
            else return runtime_demo_argument_error(option, argv[i]);
            continue;
        }
        if (!strcmp(option, "--color-curve")) {
            if (!strcmp(argv[i], "cie")) c.color_curve = H75_COLOR_CIE;
            else if (!strcmp(argv[i], "gamma")) c.color_curve = H75_COLOR_GAMMA;
            else return runtime_demo_argument_error(option, argv[i]);
            continue;
        }
        errno = 0;
        char *end = NULL;
        if (!strcmp(option, "--speed")) {
            speed = strtof(argv[i], &end);
            /* The CLI is built with -ffast-math: use bits, not isfinite(),
             * to reject NaN/Inf reliably before any output/GPIO is started. */
            uint32_t bits;
            memcpy(&bits, &speed, sizeof(bits));
            if (errno || end == argv[i] || *end || (bits & 0x7f800000u) == 0x7f800000u ||
                speed < .001f || speed > 100.f) {
                fprintf(stderr, "runtime: --speed must be finite and in 0.001..100 (got '%s')\n", argv[i]);
                return 2;
            }
            speed_set = true;
            continue;
        }
        if (!strcmp(option, "--gamma")) {
            c.gamma = strtof(argv[i], &end);
            if (errno || end == argv[i] || *end)
                return runtime_demo_argument_error(option, argv[i]);
            continue;
        }
        if (!strcmp(option, "--seconds")) {
            if (!strcmp(argv[i], "forever") || !strcmp(argv[i], "-1")) {
                seconds = -1;
                continue;
            }
            seconds = strtoll(argv[i], &end, 10);
            if (errno || end == argv[i] || *end || seconds <= 0 || seconds > 1000000)
                return runtime_demo_argument_error(option, argv[i]);
            continue;
        }
        unsigned long value = strtoul(argv[i], &end, 10);
        if (errno || !end || end == argv[i] || *end || value > UINT32_MAX || argv[i][0] == '-')
            return runtime_demo_argument_error(option, argv[i]);
        if (strcmp(option, "--pwm-lsb-ns") && value > 1000000)
            return runtime_demo_argument_error(option, argv[i]);
        if (!strcmp(option, "--width"))
            c.width = (uint32_t)value;
        else if (!strcmp(option, "--height"))
            c.height = (uint32_t)value;
        else if (!strcmp(option, "--ports")) {
            c.ports = (uint32_t)value;
            ports_set = true;
        } else if (!strcmp(option, "--chains")) {
            c.chains = (uint32_t)value;
            chains_set = true;
        }
        else if (!strcmp(option, "--panel-width"))
            c.panel_width = (uint32_t)value;
        else if (!strcmp(option, "--panel-height"))
            c.panel_height = (uint32_t)value;
        else if (!strcmp(option, "--fps"))
            c.fps = (uint32_t)value;
        else if (!strcmp(option, "--brightness")) {
            if (value > 254) {
                fprintf(stderr, "runtime: --brightness must be in 0..254 (got '%s')\n", argv[i]);
                return 2;
            }
            c.brightness = (uint32_t)value;
        }
        else if (!strcmp(option, "--dither") || !strcmp(option, "-l")) {
            if (value > 10) {
                fprintf(stderr, "runtime: %s must be an integer in 0..10 (got '%s')\n", option, argv[i]);
                return 2;
            }
            c.dither = (float)value;
        }
        else if (!strcmp(option, "--bit-depth"))
            c.bit_depth = (uint32_t)value;
        else if (!strcmp(option, "--pwm-bits"))
            c.pwm_bits = (uint32_t)value;
        else if (!strcmp(option, "--pwm-lsb-ns"))
            c.pwm_lsb_ns = (uint32_t)value;
        else if (!strcmp(option, "--queue-depth"))
            c.queue_depth = (uint32_t)value;
        else if (!strcmp(option, "--frames")) {
            if (!value) return runtime_demo_argument_error(option, argv[i]);
            frames = (unsigned)value;
        }
        else if (!strcmp(option, "--warmup"))
            warmup = (unsigned)value;
        else if (!strcmp(option, "--lines"))
            lines = (unsigned)value;
        else {
            fprintf(stderr, "runtime: unknown option '%s'; use --help\n", option);
            return 2;
        }
    }
    if (c.output_mode != H75_OUTPUT_EQUAL_SLOT && c.quant_dither) {
        fprintf(stderr, "runtime: temporal dithering requires --output-mode equal-slot; "
                        "weighted-bcm supports spatial dithering only\n");
        return 2;
    }
    if (speed_set && !media) {
        fprintf(stderr, "runtime: --speed requires --file\n");
        return 2;
    }
    if (upload && (retained || media)) {
        fprintf(stderr, "runtime: --upload cannot be combined with --retained or --file\n");
        return 2;
    }
    if ((!chains_set && (!c.panel_width || !c.width || c.width % c.panel_width)) ||
        (!ports_set && (!c.panel_height || !c.height || c.height % c.panel_height))) {
        fprintf(stderr, "runtime: dimensions must be multiples of panel size to infer chains/ports\n");
        return 2;
    }
    if (!chains_set) c.chains = c.width / c.panel_width;
    if (!ports_set) c.ports = c.height / c.panel_height;
    h75_error e = {0};
    h75_runtime *r = NULL;
    h75_resource *list = NULL, *owned[5] = {0};
    uint8_t *pixels = NULL;
    uint64_t *samples = NULL;
    unsigned measured = 0;
    runtime_demo_dashboard dashboard = {0};
    const char *source = media ? media : font ? font : scene ? "CPU scene" :
                         upload ? "RGBA upload" : "CPU drawlist";
    h75_status s = h75_runtime_create(&c, &r, &e);
    if (s)
        goto done;
    s = h75_runtime_start(r, &e);
    if (s)
        goto done;
    list = h75_resource_new(H75_DRAWLIST, &e);
    if (!list) {
        s = e.code;
        goto done;
    }
    h75_command cmd = {.op = H75_CLEAR, .color = {0, 0, 0, 255}};
    s = h75_drawlist_append(list, &cmd, &e);
    if (s)
        goto done;
    if (scene) {
        owned[0] = h75_mesh_primitive("cube", 0, 0, 1, true, &e);
        owned[1] = h75_resource_new(H75_TRANSFORM, &e);
        owned[2] = h75_resource_new(H75_SCENE, &e);
        owned[3] = h75_resource_new(H75_CAMERA, &e);
        for (unsigned i = 0; i < 4; i++)
            if (!owned[i]) {
                s = H75_MEMORY;
                goto done;
            }
        float color[] = {255, 120, 30}, aspect = (float)c.width / (float)c.height;
        s = h75_resource_set(owned[0], "color", color, 3, &e);
        if (s)
            goto done;
        s = h75_resource_set(owned[3], "aspect", &aspect, 1, &e);
        if (s)
            goto done;
        uint16_t id;
        s = h75_scene_add(owned[2], owned[0], owned[1], retained ? (vec3){.3f, .7f, 0} : (vec3){0},
                          &id, &e);
        if (s)
            goto done;
        s = h75_scene_light(owned[2], (vec3){-1, -1, -1}, (RGBF){1, 1, 1}, 1, shadows, &id, &e);
        if (s)
            goto done;
        cmd = (h75_command){.op = H75_RENDER_SCENE, .a = owned[2], .b = owned[3]};
        s = h75_drawlist_append(list, &cmd, &e);
        if (s)
            goto done;
    } else if (font) {
        owned[0] = h75_font_load(font, 16, &e);
        if (!owned[0]) {
            s = e.code;
            goto done;
        }
        owned[1] = h75_text_new(owned[0], "NATIVE HUB75", &e);
        if (!owned[1]) {
            s = e.code;
            goto done;
        }
        float position[] = {2, 2}, speed = 20;
        s = h75_resource_set(owned[1], "position", position, 2, &e);
        if (s)
            goto done;
        s = h75_resource_set(owned[1], "speed", &speed, 1, &e);
        if (s)
            goto done;
        cmd = (h75_command){.op = H75_RENDER_TEXT, .a = owned[1]};
        s = h75_drawlist_append(list, &cmd, &e);
        if (s)
            goto done;
    } else if (upload) {
        size_t bytes = (size_t)c.width * c.height * 4;
        pixels = malloc(bytes);
        if (!pixels) {
            s = H75_MEMORY;
            goto done;
        }
        for (size_t i = 0; i < bytes / 4; i++) {
            pixels[4 * i] = (uint8_t)(i * 17);
            pixels[4 * i + 1] = (uint8_t)(i * 31);
            pixels[4 * i + 2] = (uint8_t)(i * 47);
            pixels[4 * i + 3] = 255;
        }
    } else
        for (unsigned i = 0; i < lines; i++) {
            cmd = (h75_command){
                .op = H75_LINE,
                .color = {(uint8_t)(i * 17u), (uint8_t)(i * 31u), (uint8_t)(i * 47u), 255},
                .p = {(float)(i % c.width), (float)((i * 3u) % c.height),
                      (float)((i * 7u) % c.width), (float)((i * 11u) % c.height)}};
            s = h75_drawlist_append(list, &cmd, &e);
            if (s)
                goto done;
        }
    /* Bound latency storage even during an indefinite run. */
    unsigned sample_capacity = frames ? frames : 65536;
    samples = calloc(sample_capacity, sizeof(*samples));
    if (!samples) {
        s = H75_MEMORY;
        goto done;
    }
    runtime_demo_stop = 0;
    struct sigaction old_int, old_term, action = {0};
    action.sa_handler = runtime_demo_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, &old_int);
    sigaction(SIGTERM, &action, &old_term);
    uint64_t start = runtime_demo_ns(), cpu_start = runtime_demo_cpu_ns();
    uint64_t deadline = seconds < 0 ? UINT64_MAX : start + (uint64_t)seconds * 1000000000u;
    h75_stats before = {0}, after = {0};
    runtime_demo_dashboard_begin(&dashboard, verbose, r);
    if (media || retained) {
        s = media ? h75_runtime_play_file(r, media, video, loop, speed, NULL, &e)
                  : h75_runtime_play(r, list, &e);
        if (!s)
            s = h75_runtime_content_ready(r, 10000, &e);
        bool warmed = warmup == 0;
        if (warmed) {
            h75_runtime_stats(r, &before);
            start = runtime_demo_ns();
            cpu_start = runtime_demo_cpu_ns();
        }
        while (!s && !runtime_demo_stop && runtime_demo_ns() < deadline) {
            runtime_demo_dashboard_update(&dashboard, r, &c, source, speed, false);
            h75_stats stats;
            h75_runtime_stats(r, &stats);
            if (!warmed && stats.consumed >= warmup) {
                before = stats;
                start = runtime_demo_ns();
                cpu_start = runtime_demo_cpu_ns();
                warmed = true;
            }
            if ((frames && warmed && stats.consumed >= before.consumed + frames) || !stats.playing)
                break;
            struct timespec delay = {0, 1000000};
            nanosleep(&delay, NULL);
        }
        if (!s)
            s = h75_runtime_stop_content(r, 5000, &e);
    } else
        for (uint64_t i = 0; (!frames || i < (uint64_t)frames + warmup) &&
                             !runtime_demo_stop && runtime_demo_ns() < deadline; i++) {
            runtime_demo_dashboard_update(&dashboard, r, &c, source, speed, false);
            if (i == warmup) {
                s = h75_runtime_flush(r, 5000, &e);
                if (s)
                    break;
                h75_runtime_stats(r, &before);
                start = runtime_demo_ns();
                cpu_start = runtime_demo_cpu_ns();
            }
            uint64_t begin = runtime_demo_ns();
            s = h75_frame_acquire(r, 1000, &e);
            if (s)
                break;
            s = upload ? h75_frame_upload(r, pixels, (size_t)c.width * c.height * 4, c.width * 4, 4,
                                          &e)
                       : h75_frame_draw(r, list, (float)i / (float)c.fps, &e);
            if (s) {
                h75_frame_cancel(r, NULL);
                break;
            }
            s = h75_frame_submit(r, &e);
            if (s)
                break;
            if (i >= warmup && measured < sample_capacity)
                samples[measured++] = runtime_demo_ns() - begin;
        }
    if (!s)
        s = h75_runtime_flush(r, 5000, &e);
    uint64_t elapsed = runtime_demo_ns() - start, cpu_elapsed = runtime_demo_cpu_ns() - cpu_start;
    h75_runtime_stats(r, &after);
    runtime_demo_dashboard_update(&dashboard, r, &c, source, speed, true);
    runtime_demo_dashboard_end(&dashboard);
    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGTERM, &old_term, NULL);
    if (!s && dump) {
        size_t bytes = (size_t)c.width * c.height * 4;
        void *rgba = malloc(bytes);
        if (!rgba)
            s = H75_MEMORY;
        else {
            s = h75_runtime_read(r, rgba, bytes, &e);
            FILE *file = fopen(dump, "wb");
            if (!file)
                s = H75_IO;
            else {
                if (fwrite(rgba, 1, bytes, file) != bytes)
                    s = H75_IO;
                if (fclose(file))
                    s = H75_IO;
            }
            free(rgba);
        }
    }
    uint64_t shutdown_start = runtime_demo_ns();
    if (!s)
        s = h75_runtime_close(r, 5000, &e);
    uint64_t shutdown_ns = runtime_demo_ns() - shutdown_start;
    if (!s) {
        qsort(samples, measured, sizeof(*samples), runtime_demo_compare);
        printf("{\"frames\":%" PRIu64 ",\"elapsed_ns\":%" PRIu64
               ",\"width\":%u,\"height\":%u,\"lines\":%u,"
               "\"p50_ns\":%" PRIu64 ",\"p95_ns\":%" PRIu64 ",\"p99_ns\":%" PRIu64
               ",\"render_ns\":%" PRIu64 ",\"queue_wait_ns\":%" PRIu64 ",\"map_ns\":%" PRIu64
               ",\"copy_ns\":%" PRIu64 ",\"scan_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64
               ",\"shutdown_ns\":%" PRIu64 ",\"latency_ns\":%" PRIu64
               ",\"mean_latency_ns\":%" PRIu64 "}\n",
               after.consumed - before.consumed, elapsed, c.width, c.height, lines,
               measured ? samples[(measured - 1) * 50 / 100] : 0,
               measured ? samples[(measured - 1) * 95 / 100] : 0,
               measured ? samples[(measured - 1) * 99 / 100] : 0,
               after.render_ns - before.render_ns, after.queue_wait_ns - before.queue_wait_ns,
               after.map_ns - before.map_ns, after.copy_ns - before.copy_ns,
               after.scan_ns - before.scan_ns, cpu_elapsed, shutdown_ns, after.latency_ns,
               (after.latency_total_ns - before.latency_total_ns) /
                   (after.consumed > before.consumed ? after.consumed - before.consumed : 1));
    }
done:
    runtime_demo_dashboard_end(&dashboard);
    if (s)
        fprintf(stderr, "runtime: %s (status %d)\n", e.message, s);
    free(samples);
    free(pixels);
    h75_resource_release(list);
    for (unsigned i = 0; i < 5; i++)
        h75_resource_release(owned[i]);
    h75_runtime_destroy(r);
    return s ? 1 : 0;
}
#endif
