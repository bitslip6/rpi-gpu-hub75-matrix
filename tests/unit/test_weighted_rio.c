/* Pin-free scanner control/packing tests. No electrical pulse assertions. */
#include "weighted_rio.h"
#include "rpihub75.h"
#include <limits.h>

static uint32_t regs[0x3000 / 4 + 1];
static uint32_t state;
static uint64_t clock_value, frequency = 1000000000u, write_ticks;
static size_t stall_write;
static uint64_t stall_ticks;
static size_t writes, cancel_write, reads, cancel_read;
static atomic_bool flag;
static bool run_value;
static spsc_semring_t *publish_queue;
static size_t publish_write;
typedef struct { size_t reg; uint32_t value, state; uint64_t time; } event;
static event events[160000];

void h75_weighted_rio_test_write(volatile uint32_t *reg, uint32_t value) {
    size_t offset = (size_t)(reg - regs);
    assert(writes < sizeof(events) / sizeof(*events));
    if (offset == 0) state = value;
    else if (offset == 0x2000 / 4) state |= value;
    else if (offset == 0x3000 / 4) state &= ~value;
    else assert(!"unknown RIO register");
    clock_value += write_ticks;
    if (stall_write && writes + 1 == stall_write)
        clock_value += stall_ticks;
    events[writes++] = (event){offset, value, state, clock_value};
    regs[0] = state;
    if (cancel_write && writes == cancel_write)
        atomic_store(&flag, !run_value);
    if (publish_queue && writes == publish_write) {
        uint32_t *next = spsc_push_ptr_begin(publish_queue, 0);
        assert(next);
        for (size_t i = 0; i < publish_queue->item_size / sizeof(*next); i++)
            next[i] = 1u << ADDRESS_P0_G1;
        spsc_push_ptr_commit(publish_queue);
    }
}
uint64_t h75_weighted_rio_test_counter(void) {
    clock_value += 100;
    reads++;
    if (cancel_read && reads == cancel_read)
        atomic_store(&flag, !run_value);
    return clock_value;
}
uint64_t h75_weighted_rio_test_frequency(void) { return frequency; }

static void reset(bool value) {
    memset(regs, 0, sizeof(regs));
    state = 1u << 31; /* unowned pin must be preserved by full Out writes */
    regs[0] = state;
    writes = reads = cancel_write = cancel_read = 0;
    clock_value = 0; write_ticks = 10;
    stall_write = 0; stall_ticks = 0;
    publish_queue = NULL; publish_write = 0;
    frequency = 1000000000u;
    run_value = value;
    atomic_store(&flag, value);
}
static void assert_blank(void) {
    assert((state & (PIN_OE | PIN_CLK | PIN_LATCH)) == PIN_OE);
    assert(state & (1u << 31));
}
static hub75_display_t display(unsigned rows, unsigned bits) {
    return (hub75_display_t){.width = 32, .height = (uint16_t)(rows * 2),
        .panel_width = 16, .panel_height = (uint16_t)(rows * 2),
        .num_chains = 2, .num_ports = 1, .stride = 4,
        .pwm_bits = bits, .pwm_lsb_ns = 130, .panel_order = PANEL_RGB};
}
static void row_addresses(uint32_t *address, unsigned rows) {
    for (unsigned y = 0; y < rows; y++)
        address[y] = h75_weighted_row_to_address((int)y, (uint16_t)rows);
}
static void duration_math(void) {
    uint64_t ticks[11];
    assert(h75_weighted_rio_ticks(11, 130, 1000000000u, ticks) == H75_OK);
    for (unsigned p = 0; p < 11; p++) assert(ticks[p] == (130u << p));
    assert(h75_weighted_rio_ticks(11, 130, 54000000u, ticks) == H75_OK);
    assert(ticks[0] == 8 && ticks[1] == 15 && ticks[2] == 29);
    assert(h75_weighted_rio_ticks(11, 1, 1, ticks) == H75_OK);
    for (unsigned p = 0; p < 11; p++) assert(ticks[p] == 1);
    assert(h75_weighted_rio_ticks(11, UINT32_MAX, UINT32_MAX, ticks) == H75_OK);
    /* Independently checked maximum plane: ceil((2^32-1)^2 * 1024 / 1e9). */
    assert(ticks[10] == 18889465922683ULL);
    assert(h75_weighted_rio_ticks(0, 130, 1, ticks) == H75_INVALID);
    assert(h75_weighted_rio_ticks(12, 130, 1, ticks) == H75_INVALID);
    assert(h75_weighted_rio_ticks(11, 0, 1, ticks) == H75_INVALID);
    assert(h75_weighted_rio_ticks(11, 130, 0, ticks) == H75_INVALID);
    assert(h75_weighted_rio_ticks(11, 130, (uint64_t)UINT32_MAX + 1, ticks) == H75_INVALID);
    assert(h75_weighted_rio_ticks(11, 130, 1, NULL) == H75_INVALID);
}
/* Decode actual CLK/LATCH/OE transitions, not a fixed instruction trace. This
 * checks every plane's serial stream, including partially overlapped transfers
 * and row transitions, against immutable staging. The fake clock models only
 * RAM control flow; none of these durations are electrical assertions. */
static size_t check_cycle(const h75_weighted_rio *scan) {
    uint32_t previous = PIN_OE | (1u << 31);
    unsigned latched = 0, exposures = 0, completed = 0, x = 0;
    uint64_t exposed_at = 0;
    size_t illuminated_pixels = 0;
    for (size_t i = 0; i < writes; i++) {
        const event e = events[i];
        assert(e.state & (1u << 31));
        if (!(previous & PIN_OE)) {
            /* Including the final burst and blank transition, changing address
             * while the OLD output latch is visible is forbidden. */
            assert(!((previous ^ e.state) & MASK_ADDRESS));
            assert(latched && latched <= scan->steps);
            assert((e.state & MASK_ADDRESS) ==
                   (scan->schedule[latched - 1].blank_address & MASK_ADDRESS));
        }
        if (!(previous & PIN_CLK) && (e.state & PIN_CLK)) {
            assert(latched < scan->steps && x < scan->width);
            assert((e.state & MASK_DATA) ==
                   (scan->words[(size_t)latched * scan->width + x] & MASK_DATA));
            x++;
            if (!(e.state & PIN_OE)) {
                assert(latched > 0); /* never preload across frame boundaries */
                illuminated_pixels++;
            }
        }
        if (!(previous & PIN_LATCH) && (e.state & PIN_LATCH)) {
            assert(e.state & PIN_OE);
            assert(!(e.state & PIN_CLK));
            assert(latched < scan->steps && x == scan->width);
            assert((e.state & MASK_ADDRESS) ==
                   (scan->schedule[latched].blank_address & MASK_ADDRESS));
            latched++; x = 0;
        }
        if ((previous & PIN_OE) && !(e.state & PIN_OE)) {
            assert(latched == exposures + 1 && completed == exposures);
            assert(!(e.state & (PIN_CLK | PIN_LATCH)));
            exposures++; exposed_at = e.time;
        }
        if (!(previous & PIN_OE) && (e.state & PIN_OE)) {
            assert(exposures == completed + 1);
            const uint64_t duration = e.time - exposed_at;
            assert(duration >= scan->schedule[completed].exposure_ticks);
            if (!stall_write)
                assert(duration <= scan->schedule[completed].exposure_ticks + 200 + write_ticks);
            completed++;
        }
        previous = e.state;
    }
    assert(latched == scan->steps && exposures == scan->steps);
    assert(completed == scan->steps && x == 0);
    assert_blank();
    return illuminated_pixels;
}

static void schedule_and_cycle(void) {
    for (unsigned rows = 16; rows <= 32; rows += 16)
    for (unsigned bits = 1; bits <= 11; bits += 10) {
        reset(true);
        hub75_display_t d = display(rows, bits);
        uint32_t address[32]; row_addresses(address, rows);
        h75_weighted_rio scan = {0};
        assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_OK);
        assert(!writes && !scan.ready && !((uintptr_t)scan.words % 64));
        assert(scan.steps == rows * bits && scan.word_count == (size_t)rows * bits * d.width);
        uint32_t *planes = malloc(scan.word_count * sizeof(*planes));
        assert(planes);
        const uint32_t masks[] = {1u << ADDRESS_P0_R1, 1u << ADDRESS_P0_G2,
                                  1u << ADDRESS_P0_B1, 1u << ADDRESS_P0_R2};
        for (size_t i = 0; i < scan.word_count; i++)
            planes[i] = masks[(i ^ (i >> 5) ^ (i >> 9)) % 4];
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count) == H75_OK);
        assert_blank();
        for (unsigned y = 0; y < rows; y++)
        for (unsigned p = 0; p < bits; p++) {
            size_t step = y * bits + p;
            assert(scan.schedule[step].exposure_ticks == (130u << p));
            uint32_t prefix = address[y] | PIN_OE | (1u << 31);
            assert(scan.schedule[step].blank_address == prefix);
            for (unsigned x = 0; x < d.width; x++)
                assert(scan.words[step * d.width + x] ==
                    (planes[((size_t)p * rows + y) * d.width + x] | prefix));
        }
        /* Immutable staging can repeat without rereading/repacking the input. */
        memset(planes, 0, scan.word_count * sizeof(*planes));
        for (unsigned repeat = 0; repeat < 2; repeat++) {
            writes = 0;
            assert(h75_weighted_rio_cycle(&scan) == H75_OK);
            assert_blank();
            const size_t overlapped = check_cycle(&scan);
            if (bits == 1) assert(!overlapped); /* short exposures stay blank */
            else assert(overlapped);
        }
        /* Replacement only when caller explicitly prepares at a cycle boundary. */
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count) == H75_OK);
        for (size_t i = 0; i < scan.word_count; i++) assert(!(scan.words[i] & MASK_DATA));
        h75_weighted_rio_destroy(&scan);
        assert_blank();
        assert(!scan.words && !scan.ready);
        free(planes);
    }
}
/* RGB -> actual production row policy -> emitted GPIO. Distinct row markers
 * cover both halves, all ports, chain seams and first/last scan rows. */
static void production_row_delivery(void) {
#ifdef ADA_HAT
    const unsigned ports = 1;
#else
    const unsigned ports = 3;
#endif
    const unsigned red[3][2] = {{ADDRESS_P0_R1, ADDRESS_P0_R2},
        {ADDRESS_P1_R1, ADDRESS_P1_R2}, {ADDRESS_P2_R1, ADDRESS_P2_R2}};
    const unsigned address_pins[] = {ADDRESS_A, ADDRESS_B, ADDRESS_C, ADDRESS_D, ADDRESS_E};
    for (unsigned rows = 16; rows <= 32; rows += 16) {
        reset(true);
        hub75_display_t d = display(rows, 1);
        d.width = 192; d.panel_width = 64; d.num_chains = 3;
        d.num_ports = (uint8_t)ports; d.height = (uint16_t)(rows * 2 * ports);
        d.brightness = 254; d.gamma = 1.0f; d.color_curve = H75_COLOR_CIE;
        h75_weighted_lut lut;
        assert(h75_weighted_lut_build(&d, &lut) == H75_OK);
        const size_t bytes = (size_t)d.width * d.height * d.stride;
        uint8_t *image = calloc(bytes, 1);
        uint32_t *planes = calloc((size_t)rows * d.width, sizeof(*planes));
        assert(image && planes);
        for (unsigned port = 0; port < ports; port++)
        for (unsigned half = 0; half < 2; half++)
        for (unsigned y = 0; y < rows; y++) {
            size_t image_y = port * rows * 2 + half * rows + y;
            /* Unique per row and half; chain-start markers check x=0/64/128. */
            unsigned x = port * 64 + half * rows + y;
            image[(image_y * d.width + x) * d.stride] = 255;
        }
        assert(h75_weighted_encode(&d, &lut, image, bytes, planes,
                                    (size_t)rows * d.width) == H75_OK);
        uint32_t address[32]; row_addresses(address, rows);
        h75_weighted_rio scan = {0};
        assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_OK);
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count) == H75_OK);
        writes = 0;
        assert(h75_weighted_rio_cycle(&scan) == H75_OK);
        assert(!check_cycle(&scan)); /* one short plane: no overlap */
        for (unsigned y = 0; y < rows; y++) {
            uint32_t expected_address = 0;
            for (unsigned b = 0; b < 5; b++)
                expected_address |= ((y >> b) & 1u) << address_pins[b];
            assert((scan.schedule[y].blank_address & MASK_ADDRESS) == expected_address);
            for (unsigned x = 0; x < d.width; x++) {
                uint32_t expected_data = 0;
                for (unsigned port = 0; port < ports; port++)
                for (unsigned half = 0; half < 2; half++)
                    if (x == port * 64 + half * rows + y)
                        expected_data |= 1u << red[port][half];
                assert((scan.words[(size_t)y * d.width + x] & MASK_DATA) == expected_data);
            }
        }
        h75_weighted_rio_destroy(&scan);
        free(planes); free(image);
    }
}
static void cancellation_and_errors(void) {
    for (unsigned polarity = 0; polarity < 2; polarity++) {
        reset(polarity != 0);
        hub75_display_t d = display(16, 11);
        uint32_t address[32]; row_addresses(address, 16);
        h75_weighted_rio scan = {0};
        assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, run_value) == H75_OK);
        uint32_t *planes = calloc(scan.word_count, sizeof(*planes));
        assert(planes);
        assert(h75_weighted_rio_cycle(&scan) == H75_INVALID); assert_blank();
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count) == H75_OK);
        atomic_store(&flag, !run_value);
        assert(h75_weighted_rio_cycle(&scan) == H75_CANCELLED); assert_blank();
        atomic_store(&flag, run_value);
        writes = 0; cancel_write = 12; /* stop while shifting: never unblank */
        assert(h75_weighted_rio_cycle(&scan) == H75_CANCELLED); assert_blank();
        for (size_t i = 0; i < writes; i++)
            assert(!(events[i].reg == 0x3000 / 4 && events[i].value == PIN_OE));
        cancel_write = 0; reads = 0; cancel_read = 2;
        atomic_store(&flag, run_value);
        assert(h75_weighted_rio_cycle(&scan) == H75_CANCELLED); assert_blank();
        cancel_read = 0;
        planes[0] = PIN_OE; /* queue contains data words only */
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count) == H75_INVALID);
        assert(!scan.ready); assert_blank();
        assert(h75_weighted_rio_prepare(&scan, NULL, scan.word_count) == H75_INVALID);
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count - 1) == H75_INVALID);
        h75_weighted_rio_destroy(&scan); free(planes);
    }
    reset(true);
    hub75_display_t d = display(16, 11);
    uint32_t address[32]; row_addresses(address, 16);
    h75_weighted_rio scan = {0};
    assert(h75_weighted_rio_init(&scan, &d, 4, NULL, NULL, NULL, true) == H75_UNSUPPORTED);
    assert(!writes); /* unsupported hardware never accessed */
    d.width = d.panel_width = 17; d.num_chains = 1;
    assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_INVALID);
    d = display(16, 11); address[0] = PIN_CLK;
    assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_INVALID);
    row_addresses(address, 16); frequency = 0;
    assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_INVALID);
    assert(!writes && !scan.words);
}
static void overlap_deadlines_and_stops(void) {
    for (unsigned scenario = 0; scenario < 4; scenario++) {
        reset(true);
        if (scenario == 1) write_ticks = 200; /* slower simulated device */
        if (scenario == 2) frequency = 54000000u;
        if (scenario == 3) clock_value = UINT64_MAX - 1000; /* counter wrap */
        hub75_display_t d = display(32, 11);
        d.width = 192; d.panel_width = 64; d.num_chains = 3;
#ifndef ADA_HAT
        d.num_ports = 3; d.height = 192;
#endif
        uint32_t address[32]; row_addresses(address, 32);
        h75_weighted_rio scan = {0};
        assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_OK);
        uint32_t *planes = malloc(scan.word_count * sizeof(*planes));
        assert(planes);
        for (size_t i = 0; i < scan.word_count; i++)
            planes[i] = ((uint32_t)i * 2654435761u) & MASK_DATA;
        assert(h75_weighted_rio_prepare(&scan, planes, scan.word_count) == H75_OK);
        writes = 0;
        assert(h75_weighted_rio_cycle(&scan) == H75_OK);
        assert(check_cycle(&scan));
        size_t active_write = 0, final_exposure = 0;
        unsigned active = 0, full = 0, partial = 0, short_planes = 0;
        bool row_crossing = false;
        unsigned latched = 0;
        uint32_t previous = PIN_OE;
        for (size_t i = 0; i < writes; i++) {
            event e = events[i];
            if (!(previous & PIN_LATCH) && (e.state & PIN_LATCH)) latched++;
            if (!(e.state & PIN_OE) && !(previous & PIN_CLK) && (e.state & PIN_CLK)) {
                if (!active_write) active_write = i + 1;
                active++;
                if (!(latched % scan.bits)) row_crossing = true;
            }
            if ((previous & PIN_OE) && !(e.state & PIN_OE)) final_exposure = i + 1;
            if (!(previous & PIN_OE) && (e.state & PIN_OE)) {
                assert(!(active % 8));
                if (active == scan.width) full++;
                else if (active) partial++;
                else short_planes++;
                active = 0;
            }
            previous = e.state;
        }
        assert(partial && short_planes && row_crossing);
        if (scenario != 2) assert(full); /* 54 MHz model may not fit a full row */
        /* Simulate a stall in an illuminated burst. It may lengthen that
         * exposure (Linux is not hard real time), but must not keep shifting
         * illuminated after the next deadline check or restart the exposure. */
        writes = 0; stall_write = active_write; stall_ticks = 1000000;
        assert(h75_weighted_rio_cycle(&scan) == H75_OK);
        check_cycle(&scan);
        bool saw_blank = false;
        for (size_t i = active_write; i < writes; i++) {
            if (events[i].state & PIN_OE) { saw_blank = true; break; }
            assert(i - active_write < 16); /* at most remainder of one burst */
        }
        assert(saw_blank);
        stall_write = 0;
        /* Stop mid-active burst and during final exposure. Both polarities
         * must blank, without a later unblank or latch transition. */
        for (unsigned polarity = 0; polarity < 2; polarity++)
        for (unsigned where = 0; where < 2; where++) {
            scan.run_value = polarity != 0;
            run_value = scan.run_value;
            atomic_store(&flag, run_value);
            writes = 0; cancel_write = where ? final_exposure : active_write;
            assert(h75_weighted_rio_cycle(&scan) == H75_CANCELLED);
            assert_blank();
            bool blanked = false;
            for (size_t i = cancel_write; i < writes; i++) {
                if (events[i].state & PIN_OE) blanked = true;
                if (blanked) {
                    assert(events[i].state & PIN_OE);
                    assert(!(events[i].state & PIN_LATCH));
                }
            }
            assert(blanked);
        }
        cancel_write = 0;
        h75_weighted_rio_destroy(&scan);
        free(planes);
    }
}

static void queue_boundary_case(unsigned rows, unsigned publication) {
    reset(true);
    hub75_display_t d = display(rows, 11);
    d.width = 192; d.panel_width = 64; d.num_chains = 3;
#ifndef ADA_HAT
    d.num_ports = 3; d.height = (uint16_t)(rows * 2 * d.num_ports);
#endif
    uint32_t address[32]; row_addresses(address, rows);
    h75_weighted_rio scan = {0};
    assert(h75_weighted_rio_init(&scan, &d, 5, regs, address, &flag, true) == H75_OK);
    spsc_semring_t *queue = spsc_create(2, scan.word_count * sizeof(uint32_t));
    assert(queue);
    const uint32_t *held = NULL;
    assert(h75_weighted_rio_frame_begin(&scan, queue, &held, 0) == H75_TIMEOUT && !held);
    uint32_t *first = spsc_push_ptr_begin(queue, 0);
    assert(first);
    for (size_t i = 0; i < scan.word_count; i++) first[i] = 1u << ADDRESS_P0_R1;
    spsc_push_ptr_commit(queue);
    assert(h75_weighted_rio_frame_begin(&scan, queue, &held, 0) == H75_OK);
    assert(held == first && spsc_count(queue) == 1);
    writes = 0;
    /* Find actual first/last exposure and an illuminated pixel burst in a
     * baseline trace; overlap adds no producer reads inside the cycle. */
    assert(h75_weighted_rio_cycle(&scan) == H75_OK);
    assert(check_cycle(&scan));
    size_t first_exposure = 0, last_exposure = 0, active_burst = 0;
    for (size_t i = 0; i < writes; i++) {
        if (events[i].reg == 0x3000 / 4 && events[i].value == PIN_OE) {
            if (!first_exposure) first_exposure = i + 1;
            last_exposure = i + 1;
        }
        if (!active_burst && events[i].reg == 0 && !(events[i].state & PIN_OE))
            active_burst = i + 1;
    }
    assert(first_exposure && last_exposure && active_burst);
    writes = 0;
    publish_queue = queue;
    publish_write = publication == 2 ? last_exposure :
                    publication == 1 ? first_exposure : active_burst;
    assert(h75_weighted_rio_cycle(&scan) == H75_OK);
    publish_queue = NULL;
    assert_blank();
    assert(!(events[publish_write - 1].state & PIN_OE));
    assert(check_cycle(&scan));
    assert(spsc_count(queue) == 2 && held == first);
    assert(!spsc_push_ptr_begin(queue, 0)); /* held slot cannot be overwritten */
    for (size_t i = 0; i < writes; i++)
        if (events[i].reg == 0 && (events[i].state & MASK_DATA))
            assert((events[i].state & MASK_DATA) == (1u << ADDRESS_P0_R1));
    /* One complete cycle ended with OE blank. Only now adopt and stage the
     * successor; every GPIO write made during adoption must remain blank. */
    const size_t boundary = writes;
    assert(h75_weighted_rio_frame_begin(&scan, queue, &held, 0) == H75_OK);
    assert_blank();
    for (size_t i = boundary; i < writes; i++)
        assert(events[i].state & PIN_OE);
    assert(held != first && spsc_count(queue) == 1);
    uint32_t *reused = spsc_push_ptr_begin(queue, 0);
    assert(reused == first); /* old reservation was returned exactly once */
    memset(reused, 0, queue->item_size);
    spsc_push_ptr_cancel(queue);
    const uint32_t *second = held;
    for (unsigned repeat = 0; repeat < 2; repeat++) {
        size_t before = writes;
        assert(h75_weighted_rio_frame_begin(&scan, queue, &held, 0) == H75_OK);
        assert(held == second && writes == before); /* no repeat preparation */
        writes = 0;
        assert(h75_weighted_rio_cycle(&scan) == H75_OK);
        assert(check_cycle(&scan));
        for (size_t i = 0; i < writes; i++)
            if (events[i].reg == 0 && (events[i].state & MASK_DATA))
                assert((events[i].state & MASK_DATA) == (1u << ADDRESS_P0_G1));
    }
    /* Model the real producer's head-release/sem_post publication gap. Count
     * can show a successor before pop_begin can consume its semaphore token. */
    uint32_t *pending = spsc_push_ptr_begin(queue, 0);
    assert(pending);
    for (size_t i = 0; i < scan.word_count; i++) pending[i] = 1u << ADDRESS_P0_B1;
    unsigned head = atomic_load_explicit(&queue->head, memory_order_relaxed);
    atomic_store_explicit(&queue->head, head + 1u, memory_order_release);
    assert(h75_weighted_rio_frame_begin(&scan, queue, &held, 0) == H75_TIMEOUT);
    assert(!held && scan.ready); assert_blank();
    assert(sem_post(&queue->can_pop) == 0);
    assert(h75_weighted_rio_frame_begin(&scan, queue, &held, 0) == H75_OK && held == pending);
    for (size_t i = 0; i < scan.word_count; i++)
        assert((scan.words[i] & MASK_DATA) == (1u << ADDRESS_P0_B1));
    h75_weighted_rio_destroy(&scan); assert_blank();
    spsc_pop_ptr_commit(queue);
    assert(spsc_count(queue) == 0);
    spsc_destroy(queue);
}

static void queue_boundaries(void) {
    for (unsigned rows = 16; rows <= 32; rows += 16)
        for (unsigned publication = 0; publication < 3; publication++)
            queue_boundary_case(rows, publication);
}

int main(void) {
    duration_math(); schedule_and_cycle(); production_row_delivery();
    cancellation_and_errors(); overlap_deadlines_and_stops(); queue_boundaries();
    puts("Weighted RIO scanner tests passed (RAM only; no pulse timing assertions).");
    return 0;
}
