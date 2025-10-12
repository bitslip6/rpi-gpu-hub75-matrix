#include <stdint.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdatomic.h>

#include "util.h"
#include "debug.h"

#ifndef __SPSC_H__
#define __SPSC_H__

#define SPSC_MAX_ITEM_SIZE 10485760 // 10MB max item size
#define SPSC_MAX_CAPACITY 32        // max 32 items in the ring




/*
 * Struct: spsc_semring_t
 * Description: Implements a Single Producer Single Consumer (SPSC) ring buffer for pointers.
 * Members:
 *   - head: Index where the next item will be inserted (producer side).
 *   - tail: Index where the next item will be removed (consumer side).
 *   - size: The size of the ring buffer (must be a power of two).
 *   - items: Array of pointers stored in the ring buffer.
 */
typedef struct
{
    atomic_uint head;
    atomic_uint tail;
    size_t capacity;  // power of two
    size_t buff_size; // power of two
    size_t item_size;
    sem_t can_push;
    sem_t can_pop;
    bool running;
    uint8_t *items;
} spsc_semring_t;


spsc_semring_t *spsc_create(size_t capacity, size_t item_size_bytes);
void spsc_destroy(spsc_semring_t *q);
void *spsc_push_ptr_begin(spsc_semring_t *q, long timeout_ms);
void spsc_push_ptr_commit(spsc_semring_t *q);
const void *spsc_pop_ptr_begin(spsc_semring_t *q, long timeout_ms);
void spsc_pop_ptr_commit(spsc_semring_t *q);

// Return the approximate number of items in the queue (producer head - consumer tail).
// For SPSC usage this is safe and useful to decide when a new frame is available.
size_t spsc_count(const spsc_semring_t *q);

#endif