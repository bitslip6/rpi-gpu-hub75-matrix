#ifndef RIO_SCAN_H
#define RIO_SCAN_H

/* Internal, pin-free schedule ownership primitive. Not yet wired into the
 * display library or installed as public API. See tests/README-rio-scan.md.
 *
 * Exactly one producer and one consumer; a controller may request stop.
 * Caller supplies all memory before either worker starts. No allocation,
 * blocking, clocks, GPIO access or system tuning occurs in these functions.
 * Structures are exposed for caller allocation, not direct mutation.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

#define RIO_SCAN_ALIGNMENT 64u
#define RIO_SCAN_MAX_BUFFERS 4u
#define RIO_SCAN_NONE SIZE_MAX

typedef struct {
    uint32_t width;          /* Pixels shifted per row, shared by parallel ports. */
    uint32_t scanned_rows;   /* Usually panel_height / 2. */
    uint32_t planes;         /* Number of pwm values in a complete scan cycle. */
} rio_scan_geometry;

typedef enum {
    RIO_SCAN_FREE,
    RIO_SCAN_BUILDING,
    RIO_SCAN_READY,
    RIO_SCAN_ACTIVE
} rio_scan_state;

typedef struct {
    _Alignas(RIO_SCAN_ALIGNMENT) atomic_uint state;
    uint64_t generation;
    uint32_t *words;
} rio_scan_slot;

typedef struct {
    uint64_t published_generations;
    uint64_t no_free_slots;       /* Backpressure, not a consumer underrun. */
    uint64_t cancelled_builds;
} rio_scan_producer_counters;

typedef struct {
    uint64_t content_generations; /* Distinct generations acquired, not rendered. */
    uint64_t last_generation;     /* Zero until the first acquisition. */
    uint64_t plane_sweeps;        /* All scanned rows for one pwm value. */
    uint64_t full_cycles;         /* All rows for all planes. */
    uint64_t repeats;             /* Completed cycles reusing the active schedule. */
    uint64_t underruns;           /* Boundary attempts with no READY or ACTIVE. */
    uint64_t dropped_generations; /* READY generations superseded at a boundary. */
    uint64_t service_boundaries;  /* Non-busy begin attempts, including startup/stop. */
} rio_scan_consumer_counters;

typedef struct {
    rio_scan_geometry geometry;
    size_t capacity;
    size_t word_count;            /* Compact opaque words: width * rows * planes. */
    size_t stride_words;          /* Cache-line padded allocation per slot. */
    rio_scan_slot slots[RIO_SCAN_MAX_BUFFERS];
    struct {
        _Alignas(RIO_SCAN_ALIGNMENT) size_t building;
        uint64_t last_published;
        rio_scan_producer_counters counters;
    } producer;
    struct {
        _Alignas(RIO_SCAN_ALIGNMENT) size_t active;
        bool in_cycle;
        bool repeat;
        bool stopped;
        rio_scan_consumer_counters counters;
    } consumer;
    _Alignas(RIO_SCAN_ALIGNMENT) atomic_bool stop_requested;
} rio_scan_pool;

typedef struct {
    const uint32_t *words;
    size_t word_count;
    uint64_t generation;
} rio_scan_view;

typedef enum {
    RIO_SCAN_NEW,
    RIO_SCAN_REPEAT,
    RIO_SCAN_EMPTY,
    RIO_SCAN_STOPPED,
    RIO_SCAN_BUSY
} rio_scan_result;

/* Validate geometry/capacity (2..4), including byte-size overflow, and return
 * required storage in uint32_t words, including padding between buffers.
 * Invalid input leaves *storage_words untouched.
 */
bool rio_scan_storage_words(rio_scan_geometry geometry, size_t capacity,
                            size_t *storage_words);
/* pool and storage must be distinct, suitably aligned caller-owned objects.
 * storage must be RIO_SCAN_ALIGNMENT-aligned and live until workers have joined.
 * Initialization/reinitialization is only legal with all participants quiescent.
 * Requires lock-free atomic state/stop operations; false leaves pool untouched.
 */
bool rio_scan_init(rio_scan_pool *pool, rio_scan_geometry geometry,
                   size_t capacity, uint32_t *storage, size_t storage_words);

/* Producer only: one outstanding build. NULL means busy, full, or stopping.
 * Never retain/use the writable pointer after publish/cancel. Generation IDs
 * must be nonzero and strictly increasing (no wraparound). A failed publish
 * retains BUILDING ownership so the producer can correct it or cancel.
 */
uint32_t *rio_scan_build_begin(rio_scan_pool *pool);
bool rio_scan_build_publish(rio_scan_pool *pool, uint64_t generation);
bool rio_scan_build_cancel(rio_scan_pool *pool);

/* Consumer only, at a safe boundary: sample READY slots once, choose the newest
 * observed complete generation, and drop older observed READY slots. Publication
 * concurrent with that bounded sample may wait until the next boundary.
 *
 * NEW/REPEAT opens a cycle and sets view; other results leave view untouched.
 * A nested begin returns BUSY, even if stop has been requested. The view stays
 * immutable for the entire cycle. End declares ALL rows/planes emitted and only
 * updates local counters; it does not release ACTIVE (needed for repeats).
 * The view expires at the next successful begin or STOPPED, never at end.
 */
rio_scan_result rio_scan_cycle_begin(rio_scan_pool *pool, rio_scan_view *view);
bool rio_scan_cycle_end(rio_scan_pool *pool);

/* Controller: cooperative stop, not an emergency GPIO blanking guarantee.
 * Consumer observes it at the next begin, releases ACTIVE, and returns STOPPED.
 * Producer must cancel unfinished builds; a publish racing stop may succeed.
 */
void rio_scan_request_stop(rio_scan_pool *pool);
/* After BOTH workers have joined and stop was requested: discard any remaining
 * BUILDING/READY/ACTIVE slots, invalidate all pointers, preserve counters. This
 * is deliberately NOT a concurrent shutdown operation. Reinit to restart.
 */
bool rio_scan_reclaim(rio_scan_pool *pool);

/* Counters are plain single-owner data: read on their owner, or after joining
 * workers. Live off-core telemetry needs a separate boundary snapshot transport.
 * Service timing, deadlines and shutdown latency require external timestamps;
 * they are not fabricated by this pin-free primitive.
 */
#endif
