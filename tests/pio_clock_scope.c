// RP1 PIO scope test: free-running square wave on the project's HUB75 CLK (GPIO17).
// No DMA and no other GPIO outputs. Disconnect the panel; this is NOT HUB75 timing.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <misc/rp1_pio_if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define CLOCK_GPIO 17u // ADDRESS_CLK in include/rpihub75.h (default pinout)
#define CLOCK_MASK (1u << CLOCK_GPIO)

static volatile sig_atomic_t stop;
static void on_signal(int sig) { (void)sig; stop = 1; }

static int command(int fd, unsigned long code, void *arg, const char *what)
{
    if (ioctl(fd, code, arg) >= 0) return 0;
    fprintf(stderr, "%s: %s (%d)\n", what, strerror(errno), errno);
    return -1;
}

static int parse_uint(const char *s, unsigned max, unsigned *out)
{
    char *end;
    errno = 0;
    unsigned long n = strtoul(s, &end, 10);
    if (errno || end == s || *end || n == 0 || n > max) return -1;
    *out = (unsigned)n;
    return 0;
}

static double monotonic_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    unsigned duration = 10, divider = 5;
    if (argc > 3 || (argc > 1 && parse_uint(argv[1], 3600, &duration)) ||
        (argc > 2 && parse_uint(argv[2], 65535, &divider))) {
        fprintf(stderr, "Usage: %s [seconds=10] [PIO_divider=5]\n"
                        "GPIO17 clock-only scope test; DISCONNECT PANEL first.\n", argv[0]);
        return 2;
    }

    struct sigaction sa = { .sa_handler = on_signal };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    int fd = open("/dev/pio0", O_RDWR);
    if (fd < 0) { perror("open /dev/pio0"); return 1; }
    int result = 1, claimed = 0, loaded = 0, initialized = 0, muxed = 0;
    int sm = -1;
    struct rp1_pio_sm_claim_args claim = {0};
    for (int i = 0; i < RP1_PIO_SM_COUNT; ++i) {
        claim.mask = 1u << i;
        if (ioctl(fd, PIO_IOC_SM_CLAIM, &claim) == 0) {
            sm = i; claimed = 1; break;
        }
    }
    if (sm < 0) { fputs("No free PIO state machine\n", stderr); goto done; }

    // Two one-cycle NOPs (MOV Y,Y); mandatory one-bit side-set is low, high.
    // The instruction wrap repeats the pair without any FIFO/DMA access.
    struct rp1_pio_add_program_args program = {
        .num_instrs = 2, .origin = RP1_PIO_ORIGIN_ANY,
        .instrs = { 0xa042, 0xb042 } // NOP side 0; NOP side 1
    };
    int origin = ioctl(fd, PIO_IOC_ADD_PROGRAM, &program);
    if (origin < 0) { perror("ADD_PROGRAM"); goto done; }
    program.origin = (uint16_t)origin; // allocated offset is ioctl's return value
    loaded = 1;

    struct rp1_pio_sm_init_args init = {
        .sm = (uint16_t)sm, .initial_pc = program.origin,
        .config = {
            .clkdiv = divider << 16,
            .execctrl = ((uint32_t)(program.origin + 1) << 12) |
                        ((uint32_t)program.origin << 7),
            .shiftctrl = 0,
            .pinctrl = (1u << 29) | (CLOCK_GPIO << 10) // 1 side-set pin at GPIO17
        }
    };
    if (command(fd, PIO_IOC_SM_INIT, &init, "SM_INIT")) goto done;
    initialized = 1;

    // Drive a known LOW before enabling output; no other GPIOs are touched.
    struct rp1_pio_sm_set_pins_args low = {
        .sm = (uint16_t)sm, .values = 0, .mask = CLOCK_MASK
    };
    if (command(fd, PIO_IOC_SM_SET_PINS, &low, "SET_PINS low")) goto done;
    struct rp1_gpio_init_args pin = { .gpio = CLOCK_GPIO };
    if (command(fd, PIO_IOC_GPIO_INIT, &pin, "GPIO_INIT")) goto done;
    // Unlike the SDK's pio_gpio_init(), the raw GPIO_INIT ioctl does not
    // switch the GPIO mux. Function 7 connects the pad to RP1 PIO.
    muxed = 1;
    struct rp1_gpio_set_function_args function = {
        .gpio = CLOCK_GPIO, .fn = RP1_GPIO_FUNC_PIO
    };
    if (command(fd, PIO_IOC_GPIO_SET_FUNCTION, &function, "GPIO_SET_FUNCTION PIO")) goto done;
    struct rp1_pio_sm_set_pindirs_args dir = {
        .sm = (uint16_t)sm, .dirs = CLOCK_MASK, .mask = CLOCK_MASK
    };
    if (command(fd, PIO_IOC_SM_SET_PINDIRS, &dir, "SET_PINDIRS output")) goto done;
    struct rp1_pio_sm_set_enabled_args enable = {
        .mask = (uint16_t)(1u << sm), .enable = 1
    };
    if (command(fd, PIO_IOC_SM_SET_ENABLED, &enable, "SET_ENABLED")) goto done;

    printf("RP1 PIO SM%d -> GPIO%u (project CLK), side-set square wave\n"
           "Assuming 200 MHz PIO clock: divider %u, 2 instructions/cycle => nominal %.3f MHz\n"
           "Running for %u s (Ctrl-C to stop). NO DMA; measure physical pin with a scope.\n",
           sm, CLOCK_GPIO, divider, 200.0 / divider / 2, duration);
    fflush(stdout);
    double deadline = monotonic_seconds() + duration;
    while (!stop && monotonic_seconds() < deadline) {
        struct timespec wait = { .tv_sec = 0, .tv_nsec = 100000000 };
        nanosleep(&wait, NULL);
    }
    result = 0;

done:
    if (initialized) {
        struct rp1_pio_sm_set_enabled_args off = {
            .mask = (uint16_t)(1u << sm), .enable = 0
        };
        command(fd, PIO_IOC_SM_SET_ENABLED, &off, "disable SM");
    }
    if (muxed) {
        struct rp1_pio_sm_set_pins_args low = {
            .sm = (uint16_t)sm, .values = 0, .mask = CLOCK_MASK
        };
        command(fd, PIO_IOC_SM_SET_PINS, &low, "SET_PINS low");
        struct rp1_pio_sm_set_pindirs_args input = {
            .sm = (uint16_t)sm, .dirs = 0, .mask = CLOCK_MASK
        };
        command(fd, PIO_IOC_SM_SET_PINDIRS, &input, "release pin direction");
        struct rp1_gpio_set_function_args release = {
            .gpio = CLOCK_GPIO, .fn = 0x1f // GPIO_FUNC_NULL: high-impedance
        };
        command(fd, PIO_IOC_GPIO_SET_FUNCTION, &release, "release GPIO mux");
    }
    if (loaded) {
        struct rp1_pio_remove_program_args remove = {
            .num_instrs = program.num_instrs, .origin = program.origin
        };
        command(fd, PIO_IOC_REMOVE_PROGRAM, &remove, "REMOVE_PROGRAM");
    }
    if (claimed) {
        claim.mask = 1u << sm;
        command(fd, PIO_IOC_SM_UNCLAIM, &claim, "SM_UNCLAIM");
    }
    close(fd);
    return result;
}
