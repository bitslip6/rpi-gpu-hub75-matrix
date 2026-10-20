#include "rio_scan.h"

static bool layout(rio_scan_geometry geometry, size_t capacity,
                   size_t *words, size_t *stride, size_t *total)
{
    if (capacity < 2 || capacity > RIO_SCAN_MAX_BUFFERS ||
        !geometry.width || !geometry.scanned_rows || !geometry.planes)
        return false;

    size_t n = geometry.width;
    if (n > SIZE_MAX / geometry.scanned_rows)
        return false;
    n *= geometry.scanned_rows;
    if (n > SIZE_MAX / geometry.planes)
        return false;
    n *= geometry.planes;

    const size_t alignment_words = RIO_SCAN_ALIGNMENT / sizeof(uint32_t);
    if (n > SIZE_MAX - (alignment_words - 1))
        return false;
    size_t padded = (n + alignment_words - 1) / alignment_words * alignment_words;
    if (padded > SIZE_MAX / sizeof(uint32_t) / capacity)
        return false;

    *words = n;
    *stride = padded;
    *total = padded * capacity;
    return true;
}

bool rio_scan_storage_words(rio_scan_geometry geometry, size_t capacity,
                            size_t *storage_words)
{
    size_t words, stride, total;
    if (!storage_words || !layout(geometry, capacity, &words, &stride, &total))
        return false;
    *storage_words = total;
    return true;
}

bool rio_scan_init(rio_scan_pool *pool, rio_scan_geometry geometry,
                   size_t capacity, uint32_t *storage, size_t storage_words)
{
    size_t words, stride, total;
    atomic_uint state_probe = RIO_SCAN_FREE;
    atomic_bool stop_probe = false;
    if (!pool || !storage || (uintptr_t)storage % RIO_SCAN_ALIGNMENT ||
        !layout(geometry, capacity, &words, &stride, &total) ||
        storage_words < total || !atomic_is_lock_free(&state_probe) ||
        !atomic_is_lock_free(&stop_probe))
        return false;

    pool->geometry = geometry;
    pool->capacity = capacity;
    pool->word_count = words;
    pool->stride_words = stride;
    for (size_t i = 0; i < RIO_SCAN_MAX_BUFFERS; ++i) {
        atomic_init(&pool->slots[i].state, RIO_SCAN_FREE);
        pool->slots[i].generation = 0;
        pool->slots[i].words = i < capacity ? storage + i * stride : NULL;
    }
    pool->producer.building = RIO_SCAN_NONE;
    pool->producer.last_published = 0;
    pool->producer.counters = (rio_scan_producer_counters){0};
    pool->consumer.active = RIO_SCAN_NONE;
    pool->consumer.in_cycle = false;
    pool->consumer.repeat = false;
    pool->consumer.stopped = false;
    pool->consumer.counters = (rio_scan_consumer_counters){0};
    atomic_init(&pool->stop_requested, false);
    return true;
}

uint32_t *rio_scan_build_begin(rio_scan_pool *pool)
{
    if (pool->producer.building != RIO_SCAN_NONE ||
        atomic_load_explicit(&pool->stop_requested, memory_order_acquire))
        return NULL;

    for (size_t i = 0; i < pool->capacity; ++i) {
        /* Only the producer takes FREE slots. Acquire pairs with consumer's
         * release after its LAST read, not merely its first acquisition. */
        if (atomic_load_explicit(&pool->slots[i].state, memory_order_acquire) ==
            RIO_SCAN_FREE) {
            atomic_store_explicit(&pool->slots[i].state, RIO_SCAN_BUILDING,
                                  memory_order_relaxed);
            pool->producer.building = i;
            return pool->slots[i].words;
        }
    }
    ++pool->producer.counters.no_free_slots;
    return NULL;
}

bool rio_scan_build_publish(rio_scan_pool *pool, uint64_t generation)
{
    size_t building = pool->producer.building;
    if (building == RIO_SCAN_NONE || !generation ||
        generation <= pool->producer.last_published ||
        atomic_load_explicit(&pool->stop_requested, memory_order_acquire))
        return false;

    pool->slots[building].generation = generation;
    atomic_store_explicit(&pool->slots[building].state, RIO_SCAN_READY,
                          memory_order_release);
    pool->producer.last_published = generation;
    pool->producer.building = RIO_SCAN_NONE;
    ++pool->producer.counters.published_generations;
    return true;
}

bool rio_scan_build_cancel(rio_scan_pool *pool)
{
    size_t building = pool->producer.building;
    if (building == RIO_SCAN_NONE)
        return false;
    atomic_store_explicit(&pool->slots[building].state, RIO_SCAN_FREE,
                          memory_order_release);
    pool->producer.building = RIO_SCAN_NONE;
    ++pool->producer.counters.cancelled_builds;
    return true;
}

rio_scan_result rio_scan_cycle_begin(rio_scan_pool *pool, rio_scan_view *view)
{
    if (pool->consumer.in_cycle)
        return RIO_SCAN_BUSY;
    if (pool->consumer.stopped)
        return RIO_SCAN_STOPPED;

    ++pool->consumer.counters.service_boundaries;
    if (atomic_load_explicit(&pool->stop_requested, memory_order_acquire)) {
        if (pool->consumer.active != RIO_SCAN_NONE) {
            atomic_store_explicit(&pool->slots[pool->consumer.active].state,
                                  RIO_SCAN_FREE, memory_order_release);
            pool->consumer.active = RIO_SCAN_NONE;
        }
        pool->consumer.stopped = true;
        return RIO_SCAN_STOPPED;
    }

    size_t newest = RIO_SCAN_NONE;
    uint64_t generation = pool->consumer.counters.last_generation;
    unsigned ready_mask = 0;
    for (size_t i = 0; i < pool->capacity; ++i) {
        /* Producer cannot reclaim READY or ACTIVE. Once observed READY, both
         * payload and generation remain immutable until WE release this slot. */
        if (atomic_load_explicit(&pool->slots[i].state, memory_order_acquire) ==
            RIO_SCAN_READY) {
            ready_mask |= 1u << i;
            if (pool->slots[i].generation > generation) {
                newest = i;
                generation = pool->slots[i].generation;
            }
        }
    }

    pool->consumer.repeat = newest == RIO_SCAN_NONE;
    if (newest != RIO_SCAN_NONE) {
        atomic_store_explicit(&pool->slots[newest].state, RIO_SCAN_ACTIVE,
                              memory_order_relaxed);
        if (pool->consumer.active != RIO_SCAN_NONE)
            atomic_store_explicit(&pool->slots[pool->consumer.active].state,
                                  RIO_SCAN_FREE, memory_order_release);
        pool->consumer.active = newest;
        ++pool->consumer.counters.content_generations;
        pool->consumer.counters.last_generation = generation;
    }
    /* Free only slots in the sample; never read them after their FREE release.
     * A generation published into an already-sampled slot can be overtaken by
     * a newer publication later in that sample. Drop it NEXT time rather than
     * regress to it, even if there is no new generation this time. */
    for (size_t i = 0; i < pool->capacity; ++i) {
        if ((ready_mask & (1u << i)) && i != newest) {
            atomic_store_explicit(&pool->slots[i].state, RIO_SCAN_FREE,
                                  memory_order_release);
            ++pool->consumer.counters.dropped_generations;
        }
    }
    if (pool->consumer.active == RIO_SCAN_NONE) {
        ++pool->consumer.counters.underruns;
        return RIO_SCAN_EMPTY;
    }

    const rio_scan_slot *active = &pool->slots[pool->consumer.active];
    *view = (rio_scan_view){active->words, pool->word_count, active->generation};
    pool->consumer.in_cycle = true;
    return pool->consumer.repeat ? RIO_SCAN_REPEAT : RIO_SCAN_NEW;
}

bool rio_scan_cycle_end(rio_scan_pool *pool)
{
    if (!pool->consumer.in_cycle)
        return false;
    ++pool->consumer.counters.full_cycles;
    pool->consumer.counters.plane_sweeps += pool->geometry.planes;
    if (pool->consumer.repeat)
        ++pool->consumer.counters.repeats;
    pool->consumer.in_cycle = false;
    return true;
}

void rio_scan_request_stop(rio_scan_pool *pool)
{
    atomic_store_explicit(&pool->stop_requested, true, memory_order_release);
}

bool rio_scan_reclaim(rio_scan_pool *pool)
{
    /* This does not establish quiescence. The caller MUST join workers first. */
    if (!atomic_load_explicit(&pool->stop_requested, memory_order_acquire) ||
        pool->consumer.in_cycle)
        return false;
    for (size_t i = 0; i < pool->capacity; ++i)
        atomic_store_explicit(&pool->slots[i].state, RIO_SCAN_FREE,
                              memory_order_relaxed);
    pool->producer.building = RIO_SCAN_NONE;
    pool->consumer.active = RIO_SCAN_NONE;
    pool->consumer.stopped = true;
    return true;
}
