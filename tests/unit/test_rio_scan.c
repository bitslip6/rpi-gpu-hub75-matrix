#define _POSIX_C_SOURCE 200809L
#include "rio_scan.h"

#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Deliberately not assert(): release builds define NDEBUG. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); \
        abort(); \
    } \
} while (0)

typedef struct {
    rio_scan_pool pool;
    uint32_t *storage;
    size_t storage_words;
} fixture;

static void setup(fixture *f, rio_scan_geometry geometry, size_t capacity)
{
    CHECK(rio_scan_storage_words(geometry, capacity, &f->storage_words));
    /* Extra cache line serves as an allocation overrun canary. */
    f->storage = aligned_alloc(RIO_SCAN_ALIGNMENT,
                              f->storage_words * sizeof(uint32_t) + RIO_SCAN_ALIGNMENT);
    CHECK(f->storage != NULL);
    for (size_t i = 0; i < f->storage_words + RIO_SCAN_ALIGNMENT / sizeof(uint32_t); ++i)
        f->storage[i] = UINT32_C(0xdeadbeef);
    CHECK(rio_scan_init(&f->pool, geometry, capacity, f->storage, f->storage_words));
    for (size_t i = 0; i < capacity; ++i) {
        CHECK((uintptr_t)&f->pool.slots[i] % RIO_SCAN_ALIGNMENT == 0);
        CHECK((uintptr_t)f->pool.slots[i].words % RIO_SCAN_ALIGNMENT == 0);
    }
    CHECK((uintptr_t)&f->pool.producer % RIO_SCAN_ALIGNMENT == 0);
    CHECK((uintptr_t)&f->pool.consumer % RIO_SCAN_ALIGNMENT == 0);
}

static void teardown(fixture *f)
{
    rio_scan_request_stop(&f->pool);
    CHECK(rio_scan_reclaim(&f->pool));
    for (size_t i = 0; i < f->pool.capacity; ++i) {
        CHECK(atomic_load(&f->pool.slots[i].state) == RIO_SCAN_FREE);
        for (size_t j = f->pool.word_count; j < f->pool.stride_words; ++j)
            CHECK(f->pool.slots[i].words[j] == UINT32_C(0xdeadbeef));
    }
    for (size_t i = f->storage_words;
         i < f->storage_words + RIO_SCAN_ALIGNMENT / sizeof(uint32_t); ++i)
        CHECK(f->storage[i] == UINT32_C(0xdeadbeef));
    free(f->storage);
}

static uint32_t pattern(uint64_t generation, size_t offset)
{
    /* Deterministic, non-constant data, including high generation bits. */
    uint32_t x = (uint32_t)generation ^ (uint32_t)(generation >> 32);
    x ^= (uint32_t)offset * UINT32_C(0x9e3779b9);
    x ^= x >> 16;
    return x * UINT32_C(0x85ebca6b) + UINT32_C(0xc2b2ae35);
}

static void fill(uint32_t *words, size_t count, uint64_t generation)
{
    for (size_t i = 0; i < count; ++i)
        words[i] = pattern(generation, i);
}

static void verify(rio_scan_view view)
{
    for (size_t i = 0; i < view.word_count; ++i)
        CHECK(view.words[i] == pattern(view.generation, i));
}

static uint32_t *publish(rio_scan_pool *pool, uint64_t generation)
{
    uint32_t *words = rio_scan_build_begin(pool);
    CHECK(words != NULL);
    fill(words, pool->word_count, generation);
    CHECK(rio_scan_build_publish(pool, generation));
    return words; /* Identity only; caller must not read/write after publish. */
}

static void test_layout_and_validation(void)
{
    const rio_scan_geometry examples[] = {
        {64, 32, 8}, {64, 32, 16}, {64, 32, 32}, {128, 32, 32}
    };
    const size_t expected[] = {16384, 32768, 65536, 131072};
    for (size_t i = 0; i < sizeof(examples) / sizeof(examples[0]); ++i) {
        fixture f;
        setup(&f, examples[i], 3);
        CHECK(f.pool.word_count == expected[i]);
        /* Three RGB ports do not multiply the number of pixel clocks. */
        CHECK(f.storage_words == expected[i] * 3);
        teardown(&f);
    }
    size_t unchanged = 123;
    CHECK(!rio_scan_storage_words((rio_scan_geometry){0, 32, 8}, 3, &unchanged));
    CHECK(!rio_scan_storage_words((rio_scan_geometry){64, 0, 8}, 3, &unchanged));
    CHECK(!rio_scan_storage_words((rio_scan_geometry){64, 32, 0}, 3, &unchanged));
    CHECK(!rio_scan_storage_words((rio_scan_geometry){1, 1, 1}, 1, &unchanged));
    CHECK(!rio_scan_storage_words((rio_scan_geometry){1, 1, 1}, 5, &unchanged));
    CHECK(!rio_scan_storage_words((rio_scan_geometry){UINT32_MAX, UINT32_MAX, UINT32_MAX},
                                  3, &unchanged));
    /* Fits size_t on 64-bit machines, but not its padded byte allocation. */
    CHECK(!rio_scan_storage_words((rio_scan_geometry){UINT32_MAX, UINT32_MAX, 1},
                                  3, &unchanged));
    CHECK(unchanged == 123);
    CHECK(!rio_scan_storage_words((rio_scan_geometry){1, 1, 1}, 3, NULL));

    fixture f;
    setup(&f, (rio_scan_geometry){1, 1, 1}, 2);
    CHECK(f.pool.word_count == 1);
    CHECK(f.pool.stride_words == RIO_SCAN_ALIGNMENT / sizeof(uint32_t));
    CHECK(!rio_scan_init(NULL, f.pool.geometry, 2, f.storage, f.storage_words));
    CHECK(!rio_scan_init(&f.pool, f.pool.geometry, 2, NULL, f.storage_words));
    CHECK(!rio_scan_init(&f.pool, f.pool.geometry, 2, f.storage + 1, f.storage_words));
    CHECK(!rio_scan_init(&f.pool, f.pool.geometry, 2, f.storage, f.storage_words - 1));
    CHECK(f.pool.word_count == 1); /* Failed init did not overwrite the pool. */
    teardown(&f);
}

static void test_ownership_and_boundaries(void)
{
    fixture f;
    setup(&f, (rio_scan_geometry){7, 3, 8}, 3);
    rio_scan_pool *p = &f.pool;
    rio_scan_view view = {NULL, 99, 99};
    CHECK(!rio_scan_cycle_end(p));
    CHECK(!rio_scan_build_publish(p, 1));
    CHECK(!rio_scan_build_cancel(p));
    CHECK(!rio_scan_reclaim(p));
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_EMPTY);
    CHECK(view.word_count == 99 && view.generation == 99 && view.words == NULL);

    uint32_t *first = rio_scan_build_begin(p);
    CHECK(first != NULL);
    fill(first, p->word_count / 2, 1); /* Half-built data must remain invisible. */
    CHECK(rio_scan_build_begin(p) == NULL);
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_EMPTY);
    CHECK(!rio_scan_build_publish(p, 0));
    fill(first, p->word_count, 1);
    CHECK(rio_scan_build_publish(p, 1));
    CHECK(!rio_scan_build_publish(p, 2));
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_NEW);
    CHECK(view.words == first && view.generation == 1);
    verify(view);

    uint32_t *second = publish(p, 2);
    uint32_t *third = publish(p, 3);
    CHECK(first != second && first != third && second != third);
    CHECK(rio_scan_build_begin(p) == NULL); /* ACTIVE cannot be recycled. */
    CHECK(p->producer.counters.no_free_slots == 1);
    for (uint32_t plane = 0; plane < p->geometry.planes; ++plane) {
        rio_scan_view unchanged = view;
        CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_BUSY);
        CHECK(view.words == unchanged.words && view.generation == 1);
        verify(view); /* All rows/planes still use generation 1. */
    }
    CHECK(p->consumer.counters.full_cycles == 0);
    CHECK(p->consumer.counters.plane_sweeps == 0);
    CHECK(rio_scan_cycle_end(p));
    CHECK(!rio_scan_cycle_end(p));
    CHECK(rio_scan_build_begin(p) == NULL); /* end retains ACTIVE for repeats. */
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_NEW);
    CHECK(view.words == third && view.generation == 3); /* Latest complete wins. */
    CHECK(p->consumer.counters.dropped_generations == 1);
    verify(view);
    CHECK(rio_scan_cycle_end(p));

    /* Both superseded READY and previous ACTIVE are reusable, not current ACTIVE. */
    uint32_t *writable = rio_scan_build_begin(p);
    CHECK(writable == first || writable == second);
    CHECK(!rio_scan_build_publish(p, 3)); /* IDs cannot repeat or regress. */
    CHECK(!rio_scan_build_publish(p, 2));
    fill(writable, p->word_count, 4); /* Keep BUILDING across another cycle. */
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_REPEAT);
    CHECK(view.generation == 3);
    verify(view);
    CHECK(rio_scan_cycle_end(p));
    CHECK(rio_scan_build_cancel(p));
    CHECK(!rio_scan_build_cancel(p));
    CHECK(p->producer.counters.cancelled_builds == 1);
    CHECK(p->producer.counters.published_generations == 3);
    CHECK(p->consumer.counters.content_generations == 2);
    CHECK(p->consumer.counters.full_cycles == 3);
    CHECK(p->consumer.counters.plane_sweeps == 24);
    CHECK(p->consumer.counters.repeats == 1);
    CHECK(p->consumer.counters.underruns == 2);
    CHECK(p->consumer.counters.service_boundaries == 5);
    teardown(&f);
}

static void test_overtaken_publication(void)
{
    fixture f;
    setup(&f, (rio_scan_geometry){5, 2, 3}, 3);
    rio_scan_pool *p = &f.pool;
    rio_scan_view view;
    publish(p, 1);
    publish(p, 2);
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_NEW);
    CHECK(view.generation == 2);
    CHECK(rio_scan_cycle_end(p));

    /* White-box fixture for a reachable concurrent boundary state:
     * consumer sampled slot 0 while BUILDING; producer then published gen 1
     * there and gen 2 in slot 1 before consumer sampled slot 1. Gen 1 is READY
     * but older than ACTIVE. Sequential API calls cannot force this interleave.
     */
    fill(p->slots[0].words, p->word_count, 1);
    p->slots[0].generation = 1;
    atomic_store_explicit(&p->slots[0].state, RIO_SCAN_READY, memory_order_release);
    p->consumer.counters.dropped_generations = 0;
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_REPEAT);
    CHECK(view.generation == 2);
    CHECK(p->consumer.counters.dropped_generations == 1);
    CHECK(atomic_load(&p->slots[0].state) == RIO_SCAN_FREE);
    verify(view);
    CHECK(rio_scan_cycle_end(p));
    teardown(&f);
}

static void test_generation_limit(void)
{
    fixture f;
    setup(&f, (rio_scan_geometry){3, 2, 1}, 2);
    publish(&f.pool, UINT64_MAX);
    rio_scan_view view;
    CHECK(rio_scan_cycle_begin(&f.pool, &view) == RIO_SCAN_NEW);
    CHECK(view.generation == UINT64_MAX);
    verify(view);
    CHECK(rio_scan_cycle_end(&f.pool));
    CHECK(rio_scan_build_begin(&f.pool) != NULL);
    CHECK(!rio_scan_build_publish(&f.pool, 1));
    CHECK(!rio_scan_build_publish(&f.pool, UINT64_MAX));
    CHECK(rio_scan_build_cancel(&f.pool));
    teardown(&f);
}

static void test_stop(void)
{
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
        fixture f;
        setup(&f, (rio_scan_geometry){9, 2, 4}, 3);
        rio_scan_pool *p = &f.pool;
        rio_scan_view view = {0};
        if (scenario >= 1)
            publish(p, 1); /* READY at stop in scenario 1. */
        if (scenario >= 2) {
            CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_NEW);
            publish(p, 2); /* READY remains at stop, alongside ACTIVE. */
        }
        if (scenario == 3)
            CHECK(rio_scan_build_begin(p) != NULL); /* Unfinished BUILDING. */

        rio_scan_request_stop(p);
        rio_scan_request_stop(p); /* Idempotent; never an in-cycle release. */
        CHECK(rio_scan_build_begin(p) == NULL);
        CHECK(!rio_scan_build_publish(p, 3));
        if (scenario >= 2) {
            CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_BUSY);
            CHECK(!rio_scan_reclaim(p));
            verify(view);
            CHECK(view.generation == 1);
            CHECK(rio_scan_cycle_end(p)); /* Finish full cycle despite stop. */
        }
        CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_STOPPED);
        uint64_t boundaries = p->consumer.counters.service_boundaries;
        CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_STOPPED);
        CHECK(p->consumer.counters.service_boundaries == boundaries);
        CHECK(p->consumer.active == RIO_SCAN_NONE);
        CHECK(!rio_scan_cycle_end(p));
        if (scenario == 3)
            CHECK(rio_scan_build_cancel(p)); /* Producer still owns its build. */
        CHECK(p->consumer.counters.full_cycles == (scenario >= 2 ? 1u : 0u));
        teardown(&f); /* Quiescent reclaim handles READY, not just ACTIVE. */
    }
    fixture f;
    setup(&f, (rio_scan_geometry){1, 1, 1}, 2);
    CHECK(rio_scan_build_begin(&f.pool) != NULL);
    rio_scan_request_stop(&f.pool);
    /* Also support reclaiming an abandoned build after its worker has joined. */
    teardown(&f);
}

typedef struct {
    fixture f;
    atomic_bool cycle_started;
    atomic_bool build_started;
} stop_context;

static void *stop_test_producer(void *arg)
{
    stop_context *context = arg;
    rio_scan_pool *p = &context->f.pool;
    while (!atomic_load_explicit(&context->cycle_started, memory_order_acquire))
        sched_yield();
    publish(p, 2);
    uint32_t *words = rio_scan_build_begin(p);
    CHECK(words != NULL);
    fill(words, p->word_count / 2, 3);
    atomic_store_explicit(&context->build_started, true, memory_order_release);
    while (!atomic_load_explicit(&p->stop_requested, memory_order_acquire))
        sched_yield();
    CHECK(!rio_scan_build_publish(p, 3));
    CHECK(rio_scan_build_cancel(p));
    return NULL;
}

static void *stop_test_consumer(void *arg)
{
    stop_context *context = arg;
    rio_scan_pool *p = &context->f.pool;
    rio_scan_view view;
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_NEW);
    CHECK(view.generation == 1);
    atomic_store_explicit(&context->cycle_started, true, memory_order_release);
    while (!atomic_load_explicit(&p->stop_requested, memory_order_acquire))
        sched_yield(); /* Deliberate test interleave, not an emitter hot loop. */
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_BUSY);
    verify(view);
    CHECK(rio_scan_cycle_end(p));
    CHECK(rio_scan_cycle_begin(p, &view) == RIO_SCAN_STOPPED);
    return NULL;
}

static void test_threaded_stop(void)
{
    stop_context context;
    setup(&context.f, (rio_scan_geometry){11, 3, 4}, 3);
    atomic_init(&context.cycle_started, false);
    atomic_init(&context.build_started, false);
    publish(&context.f.pool, 1); /* Initial publication before worker handoff. */
    pthread_t producer, consumer;
    CHECK(pthread_create(&consumer, NULL, stop_test_consumer, &context) == 0);
    CHECK(pthread_create(&producer, NULL, stop_test_producer, &context) == 0);
    while (!atomic_load_explicit(&context.build_started, memory_order_acquire))
        sched_yield();
    /* Controller stops the workers with ACTIVE, READY and BUILDING all live. */
    rio_scan_request_stop(&context.f.pool);
    CHECK(pthread_join(producer, NULL) == 0);
    CHECK(pthread_join(consumer, NULL) == 0);
    CHECK(context.f.pool.producer.counters.published_generations == 2);
    CHECK(context.f.pool.producer.counters.cancelled_builds == 1);
    CHECK(context.f.pool.consumer.counters.last_generation == 1);
    CHECK(context.f.pool.consumer.counters.full_cycles == 1);
    CHECK(context.f.pool.consumer.counters.plane_sweeps == 4);
    teardown(&context.f);
}

static uint32_t random_step(uint32_t *seed)
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

/* Independent sequential model: pending generations form a small FIFO;
 * acquisition consumes all pending entries, keeping only the last. */
static void test_randomized_model(size_t capacity)
{
    fixture f;
    setup(&f, (rio_scan_geometry){13, 3, 5}, capacity);
    rio_scan_pool *p = &f.pool;
    uint32_t seed = UINT32_C(0x12345678) + (uint32_t)capacity;
    uint64_t next = 1, active = 0, pending[RIO_SCAN_MAX_BUFFERS] = {0};
    size_t ready = 0;
    uint32_t *building = NULL;
    bool in_cycle = false, repeat = false;
    uint64_t cycles = 0, repeats = 0, underruns = 0, acquired = 0, dropped = 0;
    rio_scan_view view = {0};
    for (size_t step = 0; step < 30000; ++step) {
        switch (random_step(&seed) % 5) {
        case 0: {
            uint32_t *words = rio_scan_build_begin(p);
            bool available = !building && ready + (active != 0) < capacity;
            CHECK((words != NULL) == available);
            if (words) {
                building = words;
                fill(building, p->word_count, next);
            }
            break;
        }
        case 1:
            CHECK(rio_scan_build_publish(p, next) == (building != NULL));
            if (building) {
                pending[ready++] = next++;
                building = NULL;
            }
            break;
        case 2:
            CHECK(rio_scan_build_cancel(p) == (building != NULL));
            building = NULL;
            break;
        case 3: {
            rio_scan_result result = rio_scan_cycle_begin(p, &view);
            if (in_cycle) {
                CHECK(result == RIO_SCAN_BUSY);
            } else if (ready) {
                CHECK(result == RIO_SCAN_NEW);
                active = pending[ready - 1];
                dropped += ready - 1;
                ready = 0;
                ++acquired;
                in_cycle = true;
                repeat = false;
            } else if (active) {
                CHECK(result == RIO_SCAN_REPEAT);
                in_cycle = repeat = true;
            } else {
                CHECK(result == RIO_SCAN_EMPTY);
                ++underruns;
            }
            break;
        }
        case 4:
            CHECK(rio_scan_cycle_end(p) == in_cycle);
            if (in_cycle) {
                ++cycles;
                repeats += repeat;
                in_cycle = false;
            }
            break;
        }
        if (in_cycle) {
            CHECK(view.generation == active);
            verify(view); /* Includes every step with concurrent BUILDING/READY. */
        }
        CHECK(p->consumer.counters.full_cycles == cycles);
        CHECK(p->consumer.counters.plane_sweeps == cycles * p->geometry.planes);
        CHECK(p->consumer.counters.repeats == repeats);
        CHECK(p->consumer.counters.underruns == underruns);
        CHECK(p->consumer.counters.content_generations == acquired);
        CHECK(p->consumer.counters.dropped_generations == dropped);
    }
    if (in_cycle)
        CHECK(rio_scan_cycle_end(p));
    teardown(&f);
}

#define STRESS_GENERATIONS UINT64_C(30000)
typedef struct {
    fixture f;
    atomic_bool producer_done;
} stress_context;

static void *stress_producer(void *arg)
{
    stress_context *context = arg;
    rio_scan_pool *p = &context->f.pool;
    uint32_t seed = UINT32_C(0xa341316c);
    for (uint64_t generation = 1; generation <= STRESS_GENERATIONS; ++generation) {
        uint32_t *words;
        while ((words = rio_scan_build_begin(p)) == NULL)
            sched_yield(); /* Test harness only; never in the schedule primitive. */
        /* Increase opportunities to read a partially built frame if broken. */
        size_t half = p->word_count / 2;
        fill(words, half, generation);
        if ((random_step(&seed) & 31u) == 0)
            sched_yield();
        for (size_t i = half; i < p->word_count; ++i)
            words[i] = pattern(generation, i);
        CHECK(rio_scan_build_publish(p, generation));
    }
    atomic_store_explicit(&context->producer_done, true, memory_order_release);
    return NULL;
}

static void *stress_consumer(void *arg)
{
    stress_context *context = arg;
    rio_scan_pool *p = &context->f.pool;
    uint64_t previous = 0;
    uint32_t seed = UINT32_C(0xc8013ea4);
    for (;;) {
        rio_scan_view view;
        rio_scan_result result = rio_scan_cycle_begin(p, &view);
        if (result == RIO_SCAN_EMPTY) {
            sched_yield();
            continue;
        }
        CHECK(result == RIO_SCAN_NEW || result == RIO_SCAN_REPEAT);
        CHECK(result == RIO_SCAN_NEW ? view.generation > previous :
                                      view.generation == previous);
        previous = view.generation;
        verify(view);
        if ((random_step(&seed) & 7u) == 0)
            sched_yield();
        verify(view); /* Producer has had a chance to reuse released slots. */
        CHECK(rio_scan_cycle_end(p));
        if (previous == STRESS_GENERATIONS &&
            atomic_load_explicit(&context->producer_done, memory_order_acquire))
            break;
    }
    /* Producer is now quiescent. One more sample drops any older READY
     * publication overtaken during the final concurrent sample. */
    rio_scan_view last;
    CHECK(rio_scan_cycle_begin(p, &last) == RIO_SCAN_REPEAT);
    CHECK(last.generation == STRESS_GENERATIONS);
    verify(last);
    CHECK(rio_scan_cycle_end(p));
    rio_scan_request_stop(p);
    rio_scan_view unused;
    CHECK(rio_scan_cycle_begin(p, &unused) == RIO_SCAN_STOPPED);
    return NULL;
}

static void test_threaded_stress(size_t capacity)
{
    stress_context context;
    setup(&context.f, (rio_scan_geometry){17, 3, 5}, capacity);
    atomic_init(&context.producer_done, false);
    pthread_t producer, consumer;
    CHECK(pthread_create(&consumer, NULL, stress_consumer, &context) == 0);
    CHECK(pthread_create(&producer, NULL, stress_producer, &context) == 0);
    CHECK(pthread_join(producer, NULL) == 0);
    CHECK(pthread_join(consumer, NULL) == 0);
    const rio_scan_pool *p = &context.f.pool;
    CHECK(p->producer.counters.published_generations == STRESS_GENERATIONS);
    CHECK(p->consumer.counters.last_generation == STRESS_GENERATIONS);
    CHECK(p->consumer.counters.content_generations +
          p->consumer.counters.dropped_generations == STRESS_GENERATIONS);
    CHECK(p->consumer.counters.full_cycles ==
          p->consumer.counters.content_generations + p->consumer.counters.repeats);
    CHECK(p->consumer.counters.plane_sweeps ==
          p->consumer.counters.full_cycles * p->geometry.planes);
    printf("  %zu buffers: %" PRIu64 " published, %" PRIu64 " acquired, "
           "%" PRIu64 " dropped, %" PRIu64 " repeated cycles\n", capacity,
           p->producer.counters.published_generations,
           p->consumer.counters.content_generations,
           p->consumer.counters.dropped_generations, p->consumer.counters.repeats);
    teardown(&context.f);
}

int main(void)
{
    alarm(60); /* Fail a hung test instead of spinning forever; no custom handler. */
    test_layout_and_validation();
    test_ownership_and_boundaries();
    test_overtaken_publication();
    test_generation_limit();
    test_stop();
    test_threaded_stop();
    for (size_t capacity = 2; capacity <= RIO_SCAN_MAX_BUFFERS; ++capacity) {
        test_randomized_model(capacity);
        test_threaded_stress(capacity);
    }
    puts("RIO scan ownership tests passed (pin-free; no hardware timing claims).");
    return 0;
}
