// RP1 PIO TX throughput test. GPIO output is opt-in via --gpio.
// Build: cc -O2 -Wall -Wextra -o tests/pio_dma_bench tests/pio_dma_bench.c
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <misc/rp1_pio_if.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

static volatile sig_atomic_t stop;
static void on_signal(int sig) { (void)sig; stop = 1; }

static double seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int operation(int fd, unsigned long cmd, void *arg, const char *name)
{
    if (ioctl(fd, cmd, arg) >= 0) return 0;
    fprintf(stderr, "%s: %s (%d)\n", name, strerror(errno), errno);
    return -1;
}

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [seconds=5] [clock_divider=4] [chunk_KiB=64]\n"
                    "RP1 PIO TX benchmark: no GPIO changes unless --gpio or --project-pins.\n"
                    "Optional: %s [seconds] [divider] [chunk_KiB] [--gpio BASE|--project-pins] [--autopull|--pack4]\n"
                    "--gpio BASE: drive GPIO BASE..BASE+7 (BASE 0..20).\n"
                    "--project-pins: drive all 26 default project pins, GPIO2..27; implies autopull.\n"
                    "--autopull: one OUT instruction/word; OUT NULL,8 without GPIO.\n"
                    "--pack4: four 8-pin updates/word; NOT available with --project-pins.\n"
                    "CAUTION: project mode toggles OE/LAT/CLK as DATA; disconnect panels!\n"
                    "Divider is relative to RP1 PIO clock.\n", prog, prog);
}

int main(int argc, char **argv)
{
    unsigned duration = 5, divider = 4, kib = 64;
    int gpio_base = -1, pin_count = 8, autopull = 0, pack4 = 0, project_pins = 0;
    int positional = argc;
    // Options are suffixes; --pack4 implies --autopull.
    while (positional > 1) {
        if (strcmp(argv[positional-1], "--pack4") == 0 && !pack4) {
            pack4 = 1; autopull = 1; --positional;
        } else if (strcmp(argv[positional-1], "--autopull") == 0 && !autopull) {
            autopull = 1; --positional;
        } else if (strcmp(argv[positional-1], "--project-pins") == 0 && !project_pins) {
            project_pins = 1; --positional;
        } else {
            break;
        }
    }
    if (positional >= 3 && strcmp(argv[positional-2], "--gpio") == 0) {
        char *end;
        errno = 0;
        unsigned long v = strtoul(argv[positional-1], &end, 10);
        if (errno || end == argv[positional-1] || *end || v > RP1_PIO_GPIO_COUNT - 8) {
            usage(argv[0]); return 2;
        }
        gpio_base = (int)v;
        positional -= 2;
    }
    if (project_pins && (gpio_base >= 0 || pack4)) { usage(argv[0]); return 2; }
    if (project_pins) {
        // Default HZELLER_HAT pinout in include/rpihub75.h: 18 RGB + A..E + LAT/CLK/OE.
        // These happen to cover every GPIO from 2 to 27, NOT GPIO0 or GPIO1.
        gpio_base = 2;
        pin_count = 26;
        autopull = 1;
    }
    if (positional > 4) { usage(argv[0]); return 2; }
    const uint32_t pin_bits = (1u << pin_count) - 1u;
    const uint32_t pin_mask = pin_bits << (gpio_base >= 0 ? gpio_base : 0);
    unsigned *opts[] = { &duration, &divider, &kib };
    const unsigned max[] = { 3600, 65535, 1024 };
    for (int i = 1; i < positional; ++i) {
        char *end;
        errno = 0;
        unsigned long v = strtoul(argv[i], &end, 10);
        if (errno || end == argv[i] || *end || v < 1 || v > max[i-1]) {
            usage(argv[0]); return 2;
        }
        *opts[i-1] = (unsigned)v;
    }
    size_t bytes = (size_t)kib * 1024;
    uint32_t *data = aligned_alloc(64, bytes);
    if (!data) { perror("aligned_alloc"); return 1; }
    for (size_t i = 0; i < bytes / 4; ++i)
        data[i] = pack4 ? 0xff00ff00u :
                  (gpio_base >= 0 ? (i & 1 ? pin_bits : 0u) :
                                    (uint32_t)(i * 0x9e3779b9u));

    struct sigaction sa = { .sa_handler = on_signal };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    int fd = open("/dev/pio0", O_RDWR);
    if (fd < 0) { perror("open /dev/pio0"); free(data); return 1; }

    int result = 1, claimed = 0, loaded = 0, configured = 0, pins_muxed = 0, pins_output = 0;
    int sm = -1;
    struct rp1_pio_sm_claim_args claim = { .mask = 0 };
    // Try one SM at a time: never reset another process's SM.
    for (int i = 0; i < RP1_PIO_SM_COUNT; ++i) {
        claim.mask = 1u << i;
        if (ioctl(fd, PIO_IOC_SM_CLAIM, &claim) == 0) { sm = i; claimed = 1; break; }
    }
    if (sm < 0) { fprintf(stderr, "No free RP1 PIO state machine\n"); goto done; }

    // OUT PINS,n sends the low n bits to n consecutive GPIOs. With
    // autopull/n, one 32-bit DMA word is consumed per pin update (n=26
    // in project mode). pack4 instead uses n=8 and four updates per word.
    // With no pins, use OUT NULL,8 to exercise DMA/PIO safely.
    const uint16_t out_instr = gpio_base >= 0 ? (uint16_t)(0x6000u | pin_count) : 0x6068u;
    struct rp1_pio_add_program_args program = {
        .num_instrs = gpio_base >= 0 && !autopull ? 2 : 1,
        .origin = RP1_PIO_ORIGIN_ANY,
        .instrs = { autopull ? out_instr : 0x80a0, out_instr }
    };
    int origin = ioctl(fd, PIO_IOC_ADD_PROGRAM, &program);
    if (origin < 0) { perror("ADD_PROGRAM"); goto done; }
    program.origin = (uint16_t)origin; // ioctl return value, NOT an updated argument
    loaded = 1;

    struct rp1_pio_sm_init_args init = {
        .sm = (uint16_t)sm, .initial_pc = program.origin,
        .config = {
            .clkdiv = divider << 16,
            .execctrl = (uint32_t)(program.origin + program.num_instrs - 1) << 12 |
                        (uint32_t)program.origin << 7,
            .shiftctrl = (1u << 30) | (1u << 19) |
                         (autopull ? (1u << 17) |
                          (pack4 ? 0u : ((unsigned)pin_count << 25)) : 0u),
            // TX join, shift right; autopull after pin_count bits or 32 (encoded 0)
            .pinctrl = gpio_base >= 0 ? ((unsigned)pin_count << 20) | (unsigned)gpio_base : 0
        }
    };
    if (operation(fd, PIO_IOC_SM_INIT, &init, "SM_INIT")) goto done;
    configured = 1;
    struct rp1_pio_sm_config_xfer32_args xfer_cfg = {
        .sm = (uint16_t)sm, .dir = RP1_PIO_DIR_TO_SM,
        .buf_size = (uint32_t)bytes, .buf_count = 4
    };
    if (operation(fd, PIO_IOC_SM_CONFIG_XFER32, &xfer_cfg, "CONFIG_XFER32 (DMA)")) goto done;
    if (gpio_base >= 0) {
        fprintf(stderr, "WARNING: GPIO %d..%d will be driven at 3.3V. Ensure they are free and safely wired.\n",
                gpio_base, gpio_base + pin_count - 1);
        if (project_pins)
            fprintf(stderr, "WARNING: CLK, LAT and OE all toggle together. Disconnect panels! Not HUB75 protocol.\n");
        // Set selected pins low before enabling output.
        struct rp1_pio_sm_set_pins_args low = {
            .sm = (uint16_t)sm, .values = 0, .mask = pin_mask
        };
        if (operation(fd, PIO_IOC_SM_SET_PINS, &low, "SM_SET_PINS")) goto done;
        for (int pin = gpio_base; pin < gpio_base + pin_count; ++pin) {
            struct rp1_gpio_init_args gpio = { .gpio = (uint16_t)pin };
            if (operation(fd, PIO_IOC_GPIO_INIT, &gpio, "GPIO_INIT")) goto done;
            ++pins_muxed;
            // GPIO_INIT alone does not select the PIO pin function on RP1.
            struct rp1_gpio_set_function_args function = {
                .gpio = (uint16_t)pin, .fn = RP1_GPIO_FUNC_PIO
            };
            if (operation(fd, PIO_IOC_GPIO_SET_FUNCTION, &function,
                          "GPIO_SET_FUNCTION PIO")) goto done;
        }
        struct rp1_pio_sm_set_pindirs_args dirs = {
            .sm = (uint16_t)sm, .dirs = pin_mask, .mask = pin_mask
        };
        if (operation(fd, PIO_IOC_SM_SET_PINDIRS, &dirs, "SM_SET_PINDIRS")) goto done;
        pins_output = 1;
    }
    struct rp1_pio_sm_set_enabled_args enable = { .mask = (uint16_t)(1u << sm), .enable = 1 };
    if (operation(fd, PIO_IOC_SM_SET_ENABLED, &enable, "SM_SET_ENABLED")) goto done;

    printf("RP1 PIO SM%d, program @%u: %s; kernel hardware DMA TX\n"
           "clock divider %u, %zu KiB/transfer, %u s; starting...\n",
           sm, program.origin, gpio_base < 0 ?
           (pack4 ? "OUT NULL,8 + autopull/32, no GPIO" :
            (autopull ? "OUT NULL,8 + autopull, no GPIO" : "PULL block, no GPIO")) :
           (project_pins ? "OUT PINS,26 + autopull/26 (GPIO2..27)" :
            (pack4 ? "OUT PINS,8 + autopull/32 (four updates/word)" :
             (autopull ? "OUT PINS,8 + autopull (one instruction/word)" :
                         "PULL block / OUT PINS,8 (two instructions/word)"))),
           divider, bytes / 1024, duration);
    fflush(stdout);
    double start = seconds(), last = start;
    const unsigned updates_per_word = pack4 ? 4 : 1;
    uint64_t count = 0, last_count = 0;
    while (!stop && seconds() - start < duration) {
        struct rp1_pio_sm_xfer_data32_args xfer = {
            .sm = (uint16_t)sm, .dir = RP1_PIO_DIR_TO_SM,
            .data_bytes = (uint32_t)bytes, .data = data
        };
        if (ioctl(fd, PIO_IOC_SM_XFER_DATA32, &xfer) < 0) {
            if (errno == EINTR && stop) break;
            fprintf(stderr, "XFER_DATA32: %s (%d)\n", strerror(errno), errno);
            goto done;
        }
        ++count;
        double now = seconds();
        if (now - last >= 1.0) {
            double mwords = (count - last_count) * (bytes / 4) / (now - last) / 1e6;
            printf("%.2f s: %.2f MiB/s, %.2f Mword/s, %.2f Mupdates/s (~%.2f MHz alternating-pin estimate) (%" PRIu64 " transfers)\n",
                   now - start, (count - last_count) * bytes / (now - last) / 1048576.,
                   mwords, mwords * updates_per_word,
                   mwords * updates_per_word / 2, count);
            fflush(stdout);
            last = now; last_count = count;
        }
    }
    double elapsed = seconds() - start;
    double mwords = count * (bytes / 4) / elapsed / 1e6;
    printf("Total: %.3f s, %" PRIu64 " submissions, %.2f MiB/s, %.2f Mword/s, %.2f Mupdates/s (~%.2f MHz alternating-pin estimate)\n",
           elapsed, count, count * bytes / elapsed / 1048576., mwords,
           mwords * updates_per_word, mwords * updates_per_word / 2);
    puts("Note: estimates from DMA submissions, not measured GPIO timing."
         " Up to four DMA buffers may be queued. Verify actual output with a scope.");
    if (divider <= 16 && mwords < 1.0) {
        fputs("DMA did not make progress; check dmesg for a stalled DMA channel.\n", stderr);
        goto done;
    }
    // Keep the SM running until the four in-flight DMA buffers should have
    // drained. Abruptly stopping an active channel can poison subsequent runs.
    const double cycles_per_word = gpio_base >= 0 && !autopull ? 2.0 :
                                   pack4 ? 4.0 : 1.0;
    double drain = (double)bytes * divider * cycles_per_word / 200e6 * 2 + 0.25;
    double deadline = seconds() + drain;
    while (seconds() < deadline) {
        struct timespec wait = { .tv_sec = 0, .tv_nsec = 10000000 };
        nanosleep(&wait, NULL);
    }
    result = 0;

done:
    if (configured) {
        struct rp1_pio_sm_set_enabled_args off = { .mask = (uint16_t)(1u << sm), .enable = 0 };
        operation(fd, PIO_IOC_SM_SET_ENABLED, &off, "disable SM");
    }
    if (gpio_base >= 0 && pins_muxed) {
        // Stop driving before changing the mux. We cannot restore a previous
        // mux owner: leave only the selected pins in high-impedance NULL mode.
        if (pins_output) {
            struct rp1_pio_sm_set_pindirs_args dirs = {
                .sm = (uint16_t)sm, .dirs = 0, .mask = pin_mask
            };
            operation(fd, PIO_IOC_SM_SET_PINDIRS, &dirs, "release pin directions");
        }
        for (int pin = gpio_base; pin < gpio_base + pins_muxed; ++pin) {
            struct rp1_gpio_set_function_args gpio = {
                .gpio = (uint16_t)pin, .fn = 0x1f // RP1 GPIO_FUNC_NULL
            };
            operation(fd, PIO_IOC_GPIO_SET_FUNCTION, &gpio, "release GPIO mux");
        }
    }
    if (loaded) {
        struct rp1_pio_remove_program_args remove = {
            .num_instrs = program.num_instrs, .origin = program.origin
        };
        operation(fd, PIO_IOC_REMOVE_PROGRAM, &remove, "REMOVE_PROGRAM");
    }
    if (claimed) {
        claim.mask = 1u << sm;
        operation(fd, PIO_IOC_SM_UNCLAIM, &claim, "SM_UNCLAIM");
    }
    close(fd);
    free(data);
    return result;
}
