#define _GNU_SOURCE
#include "rio_scan.h"
#include "rio_scope_io.h"
#include "rio_scope_lock.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <linux/gpio.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define SAMPLE_CAPACITY 65536u

typedef struct {
    uint64_t cycle, start_ns, emit_ns;
} sample;

typedef struct {
    rio_scan_pool *pool;
    sigset_t signals;
    uint64_t requested_ns; /* Written once before release publication of stop. */
} signal_context;

typedef struct {
    void *map;
    int lock_fd;
    uint32_t saved_ctrl[28], saved_pad[28], saved_out, saved_oe;
    uint32_t pads_to_enable, pads_changed;
    bool changed;
} hardware;

static uint64_t now_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) {
        perror("clock_gettime");
        abort();
    }
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static volatile uint32_t *reg_at(void *map, size_t offset)
{
    return (volatile uint32_t *)((unsigned char *)map + offset);
}

static void *wait_signal(void *arg)
{
    signal_context *context = arg;
    int sig;
    int error = sigwait(&context->signals, &sig);
    if (error || sig == SIGINT || sig == SIGTERM) {
        context->requested_ns = now_ns();
        rio_scan_request_stop(context->pool);
    }
    return NULL;
}

static bool number(const char *text, uint64_t min, uint64_t max, uint64_t *out)
{
    if (!text[0])
        return false;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9')
            return false;
    errno = 0;
    char *end;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value < min || value > max)
        return false;
    *out = (uint64_t)value;
    return true;
}

static void usage(FILE *out)
{
    fputs("Standalone RP1 RIO scope benchmark. Default: volatile RAM ONLY, no GPIO.\n"
          "Options:\n"
          "  --mode clock|alternating|sequence|protocol  (default alternating)\n"
          "  --store xor|out             (default xor, owned-bit writes)\n"
          "  --seconds 1..3600            (default 1; stops at a cycle boundary)\n"
          "  --cycles N                  optional cycle cap for smoke tests\n"
          "  --width 1..4096 --rows 1..32 --planes 1..64  (default 64/32/8)\n"
          "  --delay-nops 0..1024         protocol setup/latch delay loops (default 8)\n"
          "  --cpu N --controller-cpu N   explicit emitter/sigwait affinity\n"
          "  --scheduler other|fifo --priority 1..90     (default other/80)\n"
          "  --mlock                     require locked/prefaulted process memory\n"
          "  --sample-every N --samples PATH   sampled cycle CSV, written AFTER output\n"
          "  --output --panels-disconnected   explicit GPIO opt-in (Pi 5 only)\n"
          "  --exclusive-bank            additionally REQUIRED for hardware --store out\n"
          "  --enable-pad-output         explicitly clear owned pad OD bits; restore on exit\n"
          "OE stays HIGH in ALL modes. No visible image mode. Do not run other GPIO/PIO\n"
          "applications. Out writes the entire bank; xor touches only owned pins.\n"
          "Pad drive/slew/pulls stay unchanged. OD clearing needs explicit opt-in.\n"
          "No governor/IRQ/boot/RT-throttle changes.\n"
          "JSON stdout reports software-issued work, NOT scoped MHz or pixel correctness.\n", out);
}

#if defined(__aarch64__)
static bool pi5(void)
{
    char model[256] = {0};
    FILE *file = fopen("/proc/device-tree/model", "r");
    if (!file)
        return false;
    size_t n = fread(model, 1, sizeof(model) - 1, file);
    fclose(file);
    return n && strstr(model, "Raspberry Pi 5") != NULL;
}

static bool check_gpio_users(uint32_t owned)
{
    int fd = open("/dev/gpiochip0", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return false;
    struct gpiochip_info chip = {0};
    bool ok = ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, &chip) == 0 &&
              strstr(chip.label, "rp1") != NULL && chip.lines >= 28;
    for (unsigned pin = 0; ok && pin < 28; ++pin) {
        if (!(owned & (UINT32_C(1) << pin)))
            continue;
        struct gpioline_info line = {.line_offset = pin};
        if (ioctl(fd, GPIO_GET_LINEINFO_IOCTL, &line) ||
            (line.flags & GPIOLINE_FLAG_KERNEL)) {
            fprintf(stderr, "GPIO%u is kernel-owned or line info unavailable\n", pin);
            ok = false;
        }
    }
    close(fd);
    return ok;
}

#endif

/* Shared with the anonymous-RAM tests. No GPIO writes, including on failure.
 * CTRL/PAD contain reserved bits: all-ones is not a valid pin configuration.
 * Diagnose that before interpreting an all-ones RIO OE as an ownership conflict.
 */
static inline bool hardware_preflight(hardware *hw, rio_scope_io *io,
                                      bool use_xor, bool enable_pad_output)
{
    hw->saved_out = RIO_LOAD(io->out);
    hw->saved_oe = RIO_LOAD(reg_at(hw->map, RIO_REG_OE));
    hw->pads_to_enable = 0;
    fprintf(stderr, "RP1 preflight: Out=0x%08x RIO_OE=0x%08x owned=0x%08x\n",
            hw->saved_out, hw->saved_oe, io->owned);
    for (unsigned pin = 0; pin < 28; ++pin) {
        if (!(io->owned & (UINT32_C(1) << pin)))
            continue;
        hw->saved_ctrl[pin] = RIO_LOAD(reg_at(hw->map, pin * 8u + 4u));
        hw->saved_pad[pin] = RIO_LOAD(reg_at(hw->map, RIO_PAD_BASE + pin * 4u));
        fprintf(stderr, "GPIO%u ctrl=0x%08x pad=0x%08x\n",
                pin, hw->saved_ctrl[pin], hw->saved_pad[pin]);
    }
    for (unsigned pin = 0; pin < 28; ++pin) {
        if ((io->owned & (UINT32_C(1) << pin)) &&
            (hw->saved_ctrl[pin] == UINT32_MAX || hw->saved_pad[pin] == UINT32_MAX)) {
            fputs("Invalid RP1 MMIO readback (all-ones CTRL/PAD); refusing GPIO writes. "
                  "This is not a pin-ownership diagnosis. Check mapping/access, PCIe state "
                  "and power/undervoltage logs; do not force a takeover.\n", stderr);
            return false;
        }
    }
    uint32_t conflict = hw->saved_oe & (use_xor ? io->owned : UINT32_MAX);
    if (conflict) {
        fprintf(stderr, "RIO outputs already enabled: RIO_OE=0x%08x conflicts=0x%08x "
                "owned=0x%08x (%s); refusing takeover\n", hw->saved_oe, conflict,
                io->owned, use_xor ? "owned XOR" : "full-bank Out");
        return false;
    }
    for (unsigned pin = 0; pin < 28; ++pin) {
        if (!(io->owned & (UINT32_C(1) << pin)))
            continue;
        uint32_t ctrl = hw->saved_ctrl[pin], pad = hw->saved_pad[pin];
        if ((ctrl & 31u) != 5u && (ctrl & 31u) != 31u) {
            fprintf(stderr, "GPIO%u has an alternate function; refusing takeover\n", pin);
            return false;
        }
        if (pad & RIO_PAD_OD) {
            if (!enable_pad_output) {
                fprintf(stderr, "GPIO%u pad output-disable is set; refusing to change it without "
                        "--enable-pad-output (clears OD only; panels must be disconnected)\n", pin);
                return false;
            }
            fprintf(stderr, "GPIO%u OD clear requested; all other pad bits preserved\n", pin);
            hw->pads_to_enable |= UINT32_C(1) << pin;
        }
    }
    return true;
}

/* Advisory ownership checks do not replace exclusive operator control:
 * unrelated PIO programs do not use our lock. */
static bool hardware_open(hardware *hw, rio_scope_io *io, bool use_xor, bool enable_pad_output)
{
#if !defined(__aarch64__)
    (void)hw; (void)io; (void)use_xor; (void)enable_pad_output;
    fputs("Hardware output is only supported on aarch64 Pi 5\n", stderr);
    return false;
#else
    if (!pi5() || !check_gpio_users(io->owned)) {
        fputs("Need Pi 5 RP1 gpiochip0 with unclaimed selected pins\n", stderr);
        return false;
    }
    hw->lock_fd = rio_scope_lock_open("/tmp/rpi-hub75-rio-scope.lock");
    if (hw->lock_fd < 0) {
        perror("RIO benchmark lock (never run PIO/RIO concurrently)");
        return false;
    }
    int fd = open("/dev/gpiomem0", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) {
        perror("open /dev/gpiomem0");
        return false;
    }
    hw->map = mmap(NULL, RIO_MAP_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (hw->map == MAP_FAILED) {
        hw->map = NULL;
        perror("mmap RP1 IO/RIO/PADS");
        return false;
    }
    io->out = reg_at(hw->map, RIO_REG_OUT);
    io->pixel = reg_at(hw->map, use_xor ? RIO_REG_XOR : RIO_REG_OUT);
    io->set = reg_at(hw->map, RIO_REG_SET);
    io->clr = reg_at(hw->map, RIO_REG_CLR);
    return hardware_preflight(hw, io, use_xor, enable_pad_output);
#endif
}

static bool hardware_enable(hardware *hw, rio_scope_io *io)
{
    hw->changed = true;
    /* Load blanked values before enabling RIO outputs or selecting the mux. */
    if (!rio_scope_blank(io))
        return false;
    for (unsigned pin = 0; pin < 28; ++pin) {
        uint32_t bit = UINT32_C(1) << pin;
        if (!(io->owned & bit))
            continue;
        RIO_STORE(reg_at(hw->map, pin * 8u + 4u), 5);
        if (hw->pads_to_enable & bit) {
            volatile uint32_t *pad = reg_at(hw->map, RIO_PAD_BASE + pin * 4u);
            if (RIO_LOAD(pad) != hw->saved_pad[pin]) {
                fprintf(stderr, "GPIO%u pad changed after preflight; refusing output\n", pin);
                return false;
            }
            hw->pads_changed |= bit; /* Restore even if the following readback fails. */
            uint32_t enabled = rio_scope_pad_output_enabled(hw->saved_pad[pin]);
            RIO_STORE(pad, enabled);
            rio_scope_barrier();
            if (RIO_LOAD(pad) != enabled) {
                fprintf(stderr, "GPIO%u pad enable readback failed\n", pin);
                return false;
            }
        }
    }
    RIO_STORE(reg_at(hw->map, RIO_REG_SET + 4u), io->owned);
    rio_scope_barrier();
    return (RIO_LOAD(reg_at(hw->map, RIO_REG_OE)) & io->owned) == io->owned;
}

static bool hardware_close(hardware *hw, rio_scope_io *io)
{
    bool ok = true;
    if (hw->changed) {
        if (!rio_scope_blank(io)) {
            fputs("WARNING: final blanking readback mismatch\n", stderr);
            ok = false;
        }
        RIO_STORE(reg_at(hw->map, RIO_REG_CLR + 4u), io->owned);
        RIO_STORE(io->clr, io->owned & ~hw->saved_out);
        RIO_STORE(io->set, io->owned & hw->saved_out);
        for (unsigned pin = 0; pin < 28; ++pin) {
            uint32_t bit = UINT32_C(1) << pin;
            if (!(io->owned & bit))
                continue;
            if (hw->pads_changed & bit) {
                volatile uint32_t *pad = reg_at(hw->map, RIO_PAD_BASE + pin * 4u);
                RIO_STORE(pad, rio_scope_pad_restore_od(RIO_LOAD(pad), hw->saved_pad[pin]));
            }
            RIO_STORE(reg_at(hw->map, pin * 8u + 4u), hw->saved_ctrl[pin]);
        }
        RIO_STORE(reg_at(hw->map, RIO_REG_SET + 4u), hw->saved_oe & io->owned);
        rio_scope_barrier();
        if (((RIO_LOAD(io->out) ^ hw->saved_out) & io->owned) ||
            ((RIO_LOAD(reg_at(hw->map, RIO_REG_OE)) ^ hw->saved_oe) & io->owned))
            ok = false;
        for (unsigned pin = 0; pin < 28; ++pin) {
            uint32_t bit = UINT32_C(1) << pin;
            if ((io->owned & bit) && RIO_LOAD(reg_at(hw->map, pin * 8u + 4u)) != hw->saved_ctrl[pin])
                ok = false;
            if ((hw->pads_changed & bit) &&
                RIO_LOAD(reg_at(hw->map, RIO_PAD_BASE + pin * 4u)) != hw->saved_pad[pin])
                ok = false;
        }
        if (!ok) fputs("WARNING: GPIO restoration/readback failed; inspect pins before reuse\n", stderr);
    }
    if (hw->map)
        munmap(hw->map, RIO_MAP_BYTES);
    if (hw->lock_fd >= 0)
        close(hw->lock_fd);
    return ok;
}

static void prefault(void *memory, size_t bytes)
{
    volatile unsigned char *p = memory;
    for (size_t i = 0; i < bytes; i += 4096)
        p[i] = 0;
    p[bytes - 1] = 0;
}

static uint64_t timeval_ns(struct timeval t)
{
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_usec * 1000u;
}

static int compare_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    rio_scope_mode mode = RIO_ALTERNATING;
    const char *mode_name = "alternating", *sample_path = NULL;
    bool output = false, disconnected = false, exclusive = false, use_xor = true;
    bool lock_memory = false, fifo = false, enable_pad_output = false;
    int cpu = -1, controller_cpu = -1;
    uint64_t seconds = 1, cycle_limit = UINT64_MAX, every = 64, priority = 80, nops = 8;
    rio_scan_geometry geometry = {64, 32, 8};
    enum { OPT_MODE = 1000, OPT_STORE, OPT_SECONDS, OPT_CYCLES, OPT_WIDTH, OPT_ROWS,
           OPT_PLANES, OPT_DELAY, OPT_CPU, OPT_CONTROLLER, OPT_SCHEDULER, OPT_PRIORITY,
           OPT_MLOCK, OPT_EVERY, OPT_SAMPLES, OPT_OUTPUT, OPT_DISCONNECTED, OPT_EXCLUSIVE,
           OPT_ENABLE_PAD };
    const struct option options[] = {
        {"help", no_argument, NULL, 'h'}, {"mode", required_argument, NULL, OPT_MODE},
        {"store", required_argument, NULL, OPT_STORE}, {"seconds", required_argument, NULL, OPT_SECONDS},
        {"cycles", required_argument, NULL, OPT_CYCLES}, {"width", required_argument, NULL, OPT_WIDTH},
        {"rows", required_argument, NULL, OPT_ROWS}, {"planes", required_argument, NULL, OPT_PLANES},
        {"delay-nops", required_argument, NULL, OPT_DELAY}, {"cpu", required_argument, NULL, OPT_CPU},
        {"controller-cpu", required_argument, NULL, OPT_CONTROLLER},
        {"scheduler", required_argument, NULL, OPT_SCHEDULER}, {"priority", required_argument, NULL, OPT_PRIORITY},
        {"mlock", no_argument, NULL, OPT_MLOCK}, {"sample-every", required_argument, NULL, OPT_EVERY},
        {"samples", required_argument, NULL, OPT_SAMPLES}, {"output", no_argument, NULL, OPT_OUTPUT},
        {"panels-disconnected", no_argument, NULL, OPT_DISCONNECTED},
        {"exclusive-bank", no_argument, NULL, OPT_EXCLUSIVE},
        {"enable-pad-output", no_argument, NULL, OPT_ENABLE_PAD}, {NULL, 0, NULL, 0}
    };
    int option;
    uint64_t value;
    while ((option = getopt_long(argc, argv, "h", options, NULL)) != -1) {
        switch (option) {
        case 'h': usage(stdout); return 0;
        case OPT_MODE:
            if (!strcmp(optarg, "clock")) mode = RIO_CLOCK;
            else if (!strcmp(optarg, "alternating")) mode = RIO_ALTERNATING;
            else if (!strcmp(optarg, "sequence")) mode = RIO_SEQUENCE;
            else if (!strcmp(optarg, "protocol")) mode = RIO_PROTOCOL;
            else goto invalid;
            mode_name = optarg; break;
        case OPT_STORE:
            if (strcmp(optarg, "xor") && strcmp(optarg, "out")) goto invalid;
            use_xor = !strcmp(optarg, "xor"); break;
        case OPT_SECONDS: if (!number(optarg, 1, 3600, &seconds)) goto invalid; break;
        case OPT_CYCLES: if (!number(optarg, 1, UINT64_C(1000000000), &cycle_limit)) goto invalid; break;
        case OPT_EVERY: if (!number(optarg, 1, UINT64_C(1000000000), &every)) goto invalid; break;
        case OPT_PRIORITY: if (!number(optarg, 1, 90, &priority)) goto invalid; break;
        case OPT_DELAY: if (!number(optarg, 0, 1024, &nops)) goto invalid; break;
        case OPT_WIDTH:
            if (!number(optarg, 1, 4096, &value)) goto invalid;
            geometry.width = (uint32_t)value; break;
        case OPT_ROWS:
            if (!number(optarg, 1, 32, &value)) goto invalid;
            geometry.scanned_rows = (uint32_t)value; break;
        case OPT_PLANES:
            if (!number(optarg, 1, 64, &value)) goto invalid;
            geometry.planes = (uint32_t)value; break;
        case OPT_CPU: case OPT_CONTROLLER:
            if (!number(optarg, 0, CPU_SETSIZE - 1, &value)) goto invalid;
            if (option == OPT_CPU) cpu = (int)value; else controller_cpu = (int)value;
            break;
        case OPT_SCHEDULER:
            if (strcmp(optarg, "other") && strcmp(optarg, "fifo")) goto invalid;
            fifo = !strcmp(optarg, "fifo"); break;
        case OPT_MLOCK: lock_memory = true; break;
        case OPT_SAMPLES: sample_path = optarg; break;
        case OPT_OUTPUT: output = true; break;
        case OPT_DISCONNECTED: disconnected = true; break;
        case OPT_EXCLUSIVE: exclusive = true; break;
        case OPT_ENABLE_PAD: enable_pad_output = true; break;
        default: goto invalid;
        }
    }
    if (optind != argc || (enable_pad_output && !output) ||
        (output && (!disconnected || (!use_xor && !exclusive))) ||
        (mode == RIO_PROTOCOL && (geometry.scanned_rows & (geometry.scanned_rows - 1))))
        goto invalid;

    int status = 1;
    hardware hw = {.lock_fd = -1};
    rio_scope_io io = {.owned = rio_scope_owned(mode)};
    rio_scan_pool pool;
    size_t storage_words;
    uint32_t *storage = NULL;
    sample *samples = NULL;
    FILE *sample_file = NULL;
    pthread_t signal_thread;
    bool thread_started = false, memory_locked = false, scheduler_changed = false;
    cpu_set_t original_affinity, emitter_affinity, controller_affinity;
    int original_policy = sched_getscheduler(0);
    struct sched_param original_param;
    if (original_policy < 0 || sched_getparam(0, &original_param) ||
        sched_getaffinity(0, sizeof(original_affinity), &original_affinity)) {
        perror("read scheduling/affinity"); return 1;
    }
    if (controller_cpu < 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET((size_t)i, &original_affinity) && i != cpu) { controller_cpu = i; break; }
    }
    if (controller_cpu < 0 || (fifo && (cpu < 0 || controller_cpu == cpu))) {
        fputs("Need a housekeeping CPU; FIFO requires distinct explicit emitter CPU\n", stderr); return 1;
    }
    CPU_ZERO(&controller_affinity);
    CPU_SET((size_t)controller_cpu, &controller_affinity);
    if (cpu >= 0) {
        CPU_ZERO(&emitter_affinity);
        CPU_SET((size_t)cpu, &emitter_affinity);
        if (sched_setaffinity(0, sizeof(emitter_affinity), &emitter_affinity)) {
            perror("emitter affinity"); return 1;
        }
    }
    if (sched_getaffinity(0, sizeof(emitter_affinity), &emitter_affinity)) {
        perror("verify emitter affinity"); return 1;
    }
    if (fifo && CPU_COUNT(&emitter_affinity) != 1) {
        fputs("FIFO emitter must be pinned to one CPU\n", stderr); return 1;
    }

    signal_context signals = {.pool = &pool};
    sigemptyset(&signals.signals);
    sigaddset(&signals.signals, SIGINT);
    sigaddset(&signals.signals, SIGTERM);
    sigaddset(&signals.signals, SIGUSR1); /* Private sigwait shutdown wakeup. */
    if (pthread_sigmask(SIG_BLOCK, &signals.signals, NULL)) {
        fputs("Cannot block control signals\n", stderr); return 1;
    }
    if (!rio_scan_storage_words(geometry, 2, &storage_words))
        goto cleanup;
    storage = aligned_alloc(RIO_SCAN_ALIGNMENT, storage_words * sizeof(*storage));
    samples = calloc(SAMPLE_CAPACITY, sizeof(*samples));
    if (!storage || !samples) { perror("allocate prepared data/samples"); goto cleanup; }
    prefault(storage, storage_words * sizeof(*storage));
    prefault(samples, SAMPLE_CAPACITY * sizeof(*samples));
    volatile unsigned char stack[65536];
    for (size_t i = 0; i < sizeof(stack); i += 4096) stack[i] = 0;
    stack[sizeof(stack) - 1] = 0;
    if (!rio_scan_init(&pool, geometry, 2, storage, storage_words))
        goto cleanup;

    volatile uint32_t ram[4] = {RIO_OE, 0, 0, 0};
    io.out = &ram[0]; io.pixel = use_xor ? &ram[1] : &ram[0];
    io.set = &ram[2]; io.clr = &ram[3];
    if (sample_path) {
        sample_file = fopen(sample_path, "wx");
        if (!sample_file) { perror("create sample CSV (new file only)"); goto cleanup; }
    }
    if (output && !hardware_open(&hw, &io, use_xor, enable_pad_output))
        goto cleanup;
    uint32_t unowned = output ? hw.saved_out & ~io.owned : UINT32_C(3);
    uint32_t *words = rio_scan_build_begin(&pool);
    if (!words) goto cleanup;
    rio_scope_prepare(words, geometry.width, geometry.scanned_rows, geometry.planes,
                      mode, use_xor, unowned);
    if (!rio_scan_build_publish(&pool, 1)) goto cleanup;

    pthread_attr_t attr;
    int error = pthread_attr_init(&attr);
    if (error) goto cleanup;
    error = pthread_attr_setaffinity_np(&attr, sizeof(controller_affinity), &controller_affinity);
    /* Do not inherit an externally selected FIFO policy on the signal worker. */
    struct sched_param other = {.sched_priority = 0};
    if (!error) error = pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    if (!error) error = pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
    if (!error) error = pthread_attr_setschedparam(&attr, &other);
    if (!error) error = pthread_create(&signal_thread, &attr, wait_signal, &signals);
    pthread_attr_destroy(&attr);
    if (error) { fprintf(stderr, "sigwait worker: %s\n", strerror(error)); goto cleanup; }
    thread_started = true;
    if (lock_memory) {
        if (mlockall(MCL_CURRENT | MCL_FUTURE)) { perror("mlockall (required)"); goto cleanup; }
        memory_locked = true;
    }
    fprintf(stderr, "%s: %s, %s stores, mask=0x%08x; OE always HIGH; %" PRIu64 " seconds\n",
            output ? "GPIO OUTPUT: PANELS MUST BE DISCONNECTED" : "RAM DRY RUN (not MMIO throughput)",
            mode_name, use_xor ? "owned XOR" : "full-latch Out", io.owned, seconds);
    fflush(stderr);
    struct sched_param requested = {.sched_priority = fifo ? (int)priority : 0};
    if (sched_setscheduler(0, fifo ? SCHED_FIFO : SCHED_OTHER, &requested)) {
        perror("requested scheduler (no fallback)"); goto cleanup;
    }
    scheduler_changed = true;
    if (output && !hardware_enable(&hw, &io)) {
        fputs("GPIO enable/blank readback failed\n", stderr); goto cleanup;
    }

    struct rusage before, after;
    (void)now_ns(); /* Resolve clock path before measurement. */
    if (getrusage(RUSAGE_THREAD, &before)) { perror("getrusage before"); goto cleanup; }
    int cpu_start = sched_getcpu();
    uint64_t start = now_ns(), end = start, emit_total = 0, min_emit = UINT64_MAX, max_emit = 0;
    uint64_t cycles = 0, shutdown_ns = 0, sample_dropped = 0;
    size_t sample_count = 0;
    bool interrupted = false, readback_ok = true;
    while (cycles < cycle_limit) {
        uint64_t boundary = now_ns();
        if (boundary - start >= seconds * UINT64_C(1000000000))
            break;
        rio_scan_view view;
        rio_scan_result result = rio_scan_cycle_begin(&pool, &view);
        if (result == RIO_SCAN_STOPPED) {
            uint64_t acknowledged = now_ns();
            shutdown_ns = acknowledged - signals.requested_ns;
            interrupted = true;
            break;
        }
        if (result != RIO_SCAN_NEW && result != RIO_SCAN_REPEAT) {
            readback_ok = false;
            break;
        }
        uint64_t emit_start = now_ns();
        if (use_xor)
            rio_scope_emit_xor(&io, view.words, geometry.width, geometry.scanned_rows,
                               geometry.planes, mode == RIO_PROTOCOL, (unsigned)nops);
        else
            rio_scope_emit_out(&io, view.words, geometry.width, geometry.scanned_rows,
                               geometry.planes, mode == RIO_PROTOCOL, (unsigned)nops);
        bool blank_ok = rio_scope_blank(&io);
        uint64_t emit_end = now_ns(); /* Only after blanking and same-block readback. */
        (void)rio_scan_cycle_end(&pool);
        uint64_t elapsed = emit_end - emit_start;
        emit_total += elapsed;
        if (elapsed < min_emit) min_emit = elapsed;
        if (elapsed > max_emit) max_emit = elapsed;
        if (cycles % every == 0) {
            if (sample_count < SAMPLE_CAPACITY)
                samples[sample_count++] = (sample){cycles + 1, emit_start - start, elapsed};
            else
                ++sample_dropped;
        }
        ++cycles;
        if (output && !blank_ok) { readback_ok = false; break; }
    }
    end = now_ns();
    int cpu_end = sched_getcpu();
    if (getrusage(RUSAGE_THREAD, &after)) { perror("getrusage after"); goto cleanup; }
    /* Restore pins and lower scheduling BEFORE file I/O, allocation, sorting, logs. */
    uint32_t pads_changed = hw.pads_changed;
    if (!hardware_close(&hw, &io)) readback_ok = false;
    hw = (hardware){.lock_fd = -1};
    if (sched_setscheduler(0, original_policy, &original_param)) {
        perror("restore scheduler"); goto cleanup;
    }
    scheduler_changed = false;
    (void)pthread_kill(signal_thread, SIGUSR1);
    pthread_join(signal_thread, NULL);
    thread_started = false;
    if (memory_locked) { munlockall(); memory_locked = false; }

    if (sample_file) {
        fputs("workload_cycle,start_offset_ns,emit_including_blank_ns\n", sample_file);
        for (size_t i = 0; i < sample_count; ++i)
            fprintf(sample_file, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                    samples[i].cycle, samples[i].start_ns, samples[i].emit_ns);
        int close_error = fclose(sample_file);
        sample_file = NULL;
        if (close_error) { perror("write sample CSV"); goto cleanup; }
    }
    uint64_t *sorted = malloc(sample_count * sizeof(*sorted));
    if (sample_count && !sorted) goto cleanup;
    for (size_t i = 0; i < sample_count; ++i) sorted[i] = samples[i].emit_ns;
    if (sample_count) qsort(sorted, sample_count, sizeof(*sorted), compare_u64);
    uint64_t p50 = sample_count ? sorted[(sample_count - 1) / 2] : 0;
    uint64_t p99 = sample_count ? sorted[(sample_count * 99 + 99) / 100 - 1] : 0;
    free(sorted);
    uint64_t elapsed = end - start;
    uint64_t pixels = cycles * pool.word_count;
    double rate = elapsed ? (double)pixels * 1000.0 / (double)elapsed : 0;
    printf("{\"schema\":1,\"backend\":\"%s\",\"mode\":\"%s\",\"store\":\"%s\","
           "\"scheduler\":\"%s\",\"priority\":%d,\"cpu_requested\":%d,\"controller_cpu\":%d,"
           "\"cpu_start\":%d,\"cpu_end\":%d,\"emitter_affinity\":[",
           output ? "rio-mmio" : "ram-dry-run", mode_name, use_xor ? "xor" : "out",
           fifo ? "fifo" : "other", requested.sched_priority, cpu, controller_cpu, cpu_start, cpu_end);
    bool comma = false;
    for (int i = 0; i < CPU_SETSIZE; ++i)
        if (CPU_ISSET((size_t)i, &emitter_affinity)) { printf("%s%d", comma ? "," : "", i); comma = true; }
    printf("],\"width\":%u,\"scanned_rows\":%u,\"planes\":%u,\"delay_nops\":%" PRIu64 ","
           "\"owned_mask\":%u,\"words_per_cycle\":%zu,\"seconds_requested\":%" PRIu64 ","
           "\"elapsed_ns\":%" PRIu64 ",\"workload_cycles\":%" PRIu64 ",\"issued_pixel_edges\":%" PRIu64 ","
           "\"software_issued_mword_s\":%.6f,\"full_cycles\":%" PRIu64 ",\"plane_sweeps\":%" PRIu64 ","
           "\"prepared_generations\":%" PRIu64 ",\"renderer_content_frames\":0,"
           "\"repeats\":%" PRIu64 ",\"underruns\":%" PRIu64 ",\"service_boundaries\":%" PRIu64 ","
           "\"emit_total_ns\":%" PRIu64 ",\"service_total_ns\":%" PRIu64 ","
           "\"emit_min_ns\":%" PRIu64 ",\"emit_max_observed_ns\":%" PRIu64 ","
           "\"sample_every\":%" PRIu64 ",\"sample_count\":%zu,\"samples_dropped\":%" PRIu64 ","
           "\"sample_emit_p50_ns\":%" PRIu64 ",\"sample_emit_p99_ns\":%" PRIu64 ","
           "\"minor_faults\":%ld,\"major_faults\":%ld,"
           "\"emitter_user_ns\":%" PRIu64 ",\"emitter_system_ns\":%" PRIu64 ","
           "\"voluntary_switches\":%ld,\"involuntary_switches\":%ld,\"mlocked\":%s,\"interrupted\":%s,"
           "\"blank_readback_ok\":%s,\"pad_output_enable_requested\":%s,"
           "\"pad_output_enable_applied_mask\":%u,\"deadline_misses\":null,\"shutdown_latency_ns\":",
           geometry.width, geometry.scanned_rows, geometry.planes, nops, io.owned, pool.word_count,
           seconds, elapsed, cycles, pixels, rate,
           mode == RIO_PROTOCOL ? cycles : 0, mode == RIO_PROTOCOL ? cycles * geometry.planes : 0,
           pool.consumer.counters.content_generations, pool.consumer.counters.repeats,
           pool.consumer.counters.underruns, pool.consumer.counters.service_boundaries,
           emit_total, elapsed - emit_total, cycles ? min_emit : 0, max_emit,
           every, sample_count, sample_dropped, p50, p99,
           after.ru_minflt - before.ru_minflt, after.ru_majflt - before.ru_majflt,
           timeval_ns(after.ru_utime) - timeval_ns(before.ru_utime),
           timeval_ns(after.ru_stime) - timeval_ns(before.ru_stime),
           after.ru_nvcsw - before.ru_nvcsw, after.ru_nivcsw - before.ru_nivcsw,
           lock_memory ? "true" : "false", interrupted ? "true" : "false",
           output ? (readback_ok ? "true" : "false") : "null",
           enable_pad_output ? "true" : "false", pads_changed);
    if (interrupted) printf("%" PRIu64, shutdown_ns); else fputs("null", stdout);
    fputs(",\"scope_verified\":false,\"rgb_timing_verified\":false}\n", stdout);
    status = readback_ok ? (interrupted ? 130 : 0) : 1;

cleanup:
    if (!hardware_close(&hw, &io)) status = 1;
    if (sample_file) fclose(sample_file);
    if (scheduler_changed && sched_setscheduler(0, original_policy, &original_param))
        perror("restore scheduler");
    if (thread_started) {
        (void)pthread_kill(signal_thread, SIGUSR1);
        pthread_join(signal_thread, NULL);
    }
    if (memory_locked) munlockall();
    free(samples);
    free(storage);
    return status;

invalid:
    fputs("Invalid options or missing hardware safety acknowledgement\n", stderr);
    usage(stderr);
    return 2;
}
