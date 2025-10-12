#include <stdlib.h>
#include <stdint.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>

#include "util.h"
#include "debug.h"
#include "spsc.h"

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME			0
#endif




static inline int sem_timedwait_ms(sem_t *sem, long timeout_ms) {
    struct timespec abs;
    clock_gettime(CLOCK_REALTIME, &abs);
    timespec_add_ms(&abs, timeout_ms);
    return sem_timedwait(sem, &abs);
}

// --------- simple SPSC ring for pointers ---------


/**
 * create a new spsc structure. the memory for the items is allocated at the end of the struct.
 * with a single allocation, we don't have to laod multiple sections of memeory at the same time.
 * to access the data
 */
spsc_semring_t *spsc_create(size_t capacity, size_t item_size_bytes)
{
    // has odd bit set?
    if ((capacity & (capacity - 1)) != 0)
    {
        printf("capacity %u not power of two\n", (unsigned)capacity);
        return NULL; // capacity must be a power of two
    }

    // is capacity reasonable?
    if ((capacity > SPSC_MAX_CAPACITY) || (capacity == 0))
    {
        printf("capacity %u exceeds max capacity (or is 0): %u\n", (unsigned)capacity, SPSC_MAX_CAPACITY);
        return NULL; // capacity must be a power of two
    }

    // item size is reasonable?
    if (item_size_bytes == 0 || item_size_bytes > SPSC_MAX_ITEM_SIZE)
    {
        printf("item size %zu invalid, max is %d\n", item_size_bytes, SPSC_MAX_ITEM_SIZE);
        return NULL;
    }

    size_t total_size = (capacity * item_size_bytes) + sizeof(spsc_semring_t);
    spsc_semring_t *q = (spsc_semring_t *)malloc(total_size);
    if (!q) {
        printf("unable to allocate ring buffer of size %zu\n", total_size);
        return NULL;
    }

    printf("allocated %d bytes at %p\n", (int)total_size, q);

    memset(q, 0, total_size);
    atomic_store_explicit(&q->head, 0u, memory_order_relaxed);
    atomic_store_explicit(&q->tail, 0u, memory_order_relaxed);
    q->capacity = capacity;
    q->item_size = item_size_bytes;
    q->buff_size = capacity * item_size_bytes;

    // point the buffer to the memory after the struct
    //q->items = (uint8_t *)(q + sizeof(spsc_semring_t));
    q->items = (uint8_t *)(q + 1);

    // sanity: the item region must exactly fill the allocation
    uint8_t *alloc_end = ((uint8_t *)q) + total_size;
    uint8_t *items_end = q->items + capacity * item_size_bytes;
    if (items_end != alloc_end) {
        printf("internal error: item region does not match allocation\n");
        free(q);
        return NULL;
    }

    // initialize the semaphores
    if (sem_init(&q->can_push, 0, (unsigned)capacity) != 0 ||
        sem_init(&q->can_pop,  0, 0) != 0) {
        free(q);
        return NULL;
    }

    return q;
}

/**
 * safe destruction of the spsc ring
 */
void spsc_destroy(spsc_semring_t *q)
{
    if (!q) { return; }
    sem_destroy(&q->can_push);
    sem_destroy(&q->can_pop);
    SAFE_FREE(q);
}

/* begin returns a writable pointer to the next slot, or NULL on timeout/nonblocking failure */
void *spsc_push_ptr_begin(spsc_semring_t *q, long timeout_ms)
{
    if (!q) return NULL;
    int r = (timeout_ms > 0)
          ? sem_timedwait_ms(&q->can_push, timeout_ms)
          : sem_trywait(&q->can_push);
    if (r != 0) {
        printf("[%p] sem push failed\n", q);
        return NULL; /* queue full or timeout */
    }

    /* compute slot from current head snapshot */
    unsigned head = atomic_load_explicit(&q->head, memory_order_relaxed);
    size_t mask = q->capacity - 1u;
    size_t slot = (size_t)(head & mask);
    return q->items + (slot * q->item_size);
}


/* publish the item you just wrote */
void spsc_push_ptr_commit(spsc_semring_t *q)
{
    if (!q) return;
    unsigned head = atomic_load_explicit(&q->head, memory_order_relaxed);
    atomic_store_explicit(&q->head, head + 1u, memory_order_release);
    sem_post(&q->can_pop);
}


// --------- simple SPSC ring for pointers ---------

/**
 * returns a readable pointer to the next item, or NULL on timeout/empty.
 * timeout_ms <= 0 means "do not wait".
 */
const void *spsc_pop_ptr_begin(spsc_semring_t *q, long timeout_ms)
{
    if (!q) return NULL;
    int r = (timeout_ms > 0) ? sem_timedwait_ms(&q->can_pop, timeout_ms)
                             : sem_trywait(&q->can_pop);
    if (r != 0) {
        return NULL; /* empty or timed out */
    }

    unsigned tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

    /* acquire pairs with producer's release on head advance */
    atomic_thread_fence(memory_order_acquire);

    size_t mask = q->capacity - 1u;
    size_t slot = (size_t)(tail & mask);
    return q->items + slot * q->item_size;
}

/* release the slot after finishing reading it */
void spsc_pop_ptr_commit(spsc_semring_t *q)
{
    if (!q) return;
    unsigned tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
    atomic_store_explicit(&q->tail, tail + 1u, memory_order_release);
    sem_post(&q->can_push);
}

size_t spsc_count(const spsc_semring_t *q)
{
    if (!q) return 0;
    unsigned h = atomic_load_explicit(&q->head, memory_order_acquire);
    unsigned t = atomic_load_explicit(&q->tail, memory_order_acquire);
    return (size_t)(h - t);
}

