#include <stdatomic.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifndef _SPSC_H
#define _SPSC_H 1

#ifndef CACHELINE
#define CACHELINE 64
#endif

// --------- simple SPSC ring for pointers ---------
typedef struct {
    atomic_uint head;
    atomic_uint tail;
    unsigned size;      // power of two
    void **items;
} spsc_ring_t;

static inline void spsc_init(spsc_ring_t *q, void **storage, unsigned size_pow2) {
    atomic_store_explicit(&q->head, 0u, memory_order_relaxed);
    atomic_store_explicit(&q->tail, 0u, memory_order_relaxed);
    q->size = size_pow2;
    q->items = storage;
}

static inline int spsc_push(spsc_ring_t *q, void *p) {
    unsigned h = atomic_load_explicit(&q->head, memory_order_relaxed);
    unsigned t = atomic_load_explicit(&q->tail, memory_order_acquire);
    if (((h + 1u) & (q->size - 1u)) == (t & (q->size - 1u))) return 0;
    q->items[h & (q->size - 1u)] = p;
    atomic_store_explicit(&q->head, h + 1u, memory_order_release);
    return 1;
}
static inline void* spsc_pop(spsc_ring_t *q) {
    unsigned t = atomic_load_explicit(&q->tail, memory_order_relaxed);
    unsigned h = atomic_load_explicit(&q->head, memory_order_acquire);
    if ((t & (q->size - 1u)) == (h & (q->size - 1u))) return NULL;
    void *p = q->items[t & (q->size - 1u)];
    atomic_store_explicit(&q->tail, t + 1u, memory_order_release);
    return p;
}
static inline int spsc_try_push(spsc_ring_t *q, void *p) {
    if (spsc_push(q, p)) return 1;
    // pop one, drop it, then push
    (void)spsc_pop(q);
    return spsc_push(q, p);
}

// --------- simple SPSC ring for fixed size frames ---------


typedef struct spsc_frame_ring {
    size_t                  mask;        // capacity must be power of two
    size_t                  frame_size;  // bytes per frame
    uint8_t                *base;        // contiguous memory for all frames
    _Alignas(CACHELINE) _Atomic unsigned head; // producer owned
    _Alignas(CACHELINE) _Atomic unsigned tail; // consumer owned
} spsc_frame_ring;

// initialize ring over an existing buffer of size capacity * frame_size
static inline bool spsc_frame_init(spsc_frame_ring *r,
                                   void *base,
                                   size_t capacity,
                                   size_t single_frame_size) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) return false;
    r->mask       = capacity - 1;
    r->frame_size = single_frame_size;
    r->base       = (uint8_t *)base;
    atomic_init(&r->head, 0u);
    atomic_init(&r->tail, 0u);
    return true;
}

// producer: try to reserve a writable slot, returns pointer to frame memory
static inline bool spsc_frame_try_acquire(spsc_frame_ring *r, void **out_ptr) {
    unsigned head = atomic_load_explicit(&r->head, memory_order_relaxed);
    unsigned tail = atomic_load_explicit(&r->tail, memory_order_acquire);
    unsigned next = (head + 1u) & (unsigned)r->mask;
    if (next == tail) return false; // full
    *out_ptr = r->base + ((size_t)head * r->frame_size); // not yet published
    return true;
}

// producer: publish the frame you just wrote into the reserved slot
static inline void spsc_frame_produce(spsc_frame_ring *r) {
    unsigned head = atomic_load_explicit(&r->head, memory_order_relaxed);
    unsigned next = (head + 1u) & (unsigned)r->mask;
    atomic_store_explicit(&r->head, next, memory_order_release);
}

// consumer: peek a readable frame, returns pointer, does not advance tail
static inline bool spsc_frame_peek(spsc_frame_ring *r, const void **out_ptr) {
    unsigned tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
    unsigned head = atomic_load_explicit(&r->head, memory_order_acquire);
    if (head == tail) return false; // empty
    *out_ptr = r->base + ((size_t)tail * r->frame_size);
    return true;
}

// consumer: after processing, free the slot for reuse
static inline void spsc_frame_consume(spsc_frame_ring *r) {
    unsigned tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
    unsigned next = (tail + 1u) & (unsigned)r->mask;
    atomic_store_explicit(&r->tail, next, memory_order_release);
}

// helpers if you prefer offsets instead of pointers
static inline bool spsc_frame_try_acquire_offset(spsc_frame_ring *r, size_t *out_off) {
    void *p;
    if (!spsc_frame_try_acquire(r, &p)) return false;
    *out_off = (size_t)((uint8_t *)p - r->base);
    return true;
}

static inline bool spsc_frame_peek_offset(spsc_frame_ring *r, size_t *out_off) {
    const void *p;
    if (!spsc_frame_peek(r, &p)) return false;
    *out_off = (size_t)((const uint8_t *)p - r->base);
    return true;
}

// helpers to query pending depth, and whether a newer frame exists
static inline unsigned spsc_frame_pending(spsc_frame_ring *r) {
    unsigned head = atomic_load_explicit(&r->head, memory_order_acquire);
    unsigned tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
    return (head - tail) & (unsigned)r->mask;   // 0..mask
}

static inline bool spsc_frame_has_newer(spsc_frame_ring *r) {
    // >= 2 means one frame is active at tail, and at least one newer frame after it
    return spsc_frame_pending(r) >= 2u;
}



#endif