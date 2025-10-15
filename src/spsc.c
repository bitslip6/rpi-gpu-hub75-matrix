/*
 * File: spsc.c
 * Description: Implementation of a Single Producer Single Consumer (SPSC) ring buffer with semaphore-based
 * blocking support. This provides a thread-safe, lock-free communication channel between exactly one
 * producer thread and one consumer thread. The ring buffer uses atomic operations for the head/tail
 * pointers and semaphores for blocking when the buffer is full or empty.
 * 
 * Key Features:
 * - Lock-free operation using atomic head/tail pointers
 * - Semaphore-based blocking with configurable timeouts
 * - Single allocation for both structure and buffer data
 * - Power-of-two capacity requirement for efficient modulo operations
 * - Supports arbitrary item sizes (not just pointers)
 */


#define _GNU_SOURCE

#include <stdlib.h>
#include <stdint.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>

#include "util.h"
#include "debug.h"
#include "spsc.h"

/*
 * Platform compatibility:
 * Define CLOCK_REALTIME if not available on the current platform.
 */
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME			0
#endif


/*
 * Function: sem_timedwait_ms
 * Description: Helper function that wraps sem_timedwait() with a timeout specified in milliseconds.
 *              Converts the millisecond timeout to an absolute timespec and calls sem_timedwait().
 * Parameters:
 *   - sem: Pointer to the semaphore to wait on
 *   - timeout_ms: Timeout in milliseconds
 * Returns: 0 on success, -1 on timeout or error (errno set appropriately)
 */
static inline int sem_timedwait_ms(sem_t *sem, long timeout_ms) {
#ifdef _GNU_SOURCE
    struct timespec abs;
    clock_gettime(CLOCK_MONOTONIC, &abs);
    timespec_add_ms(&abs, timeout_ms);
    return sem_clockwait(sem, CLOCK_MONOTONIC, &abs);
#else
    struct timespec abs;
    clock_gettime(CLOCK_REALTIME, &abs);
    timespec_add_ms(&abs, timeout_ms);
    return sem_timedwait(sem, &abs);
#endif
}

/*
 * ========== SPSC Ring Buffer Implementation ==========
 * 
 * The following functions implement a Single Producer Single Consumer ring buffer
 * that supports blocking operations with timeouts. The buffer is allocated in a
 * single memory block with the structure header followed immediately by the data buffer.
 */


/*
 * Function: spsc_create
 * Description: Creates and initializes a new SPSC ring buffer with semaphore-based blocking.
 *              The buffer data is allocated immediately after the structure in a single
 *              memory allocation for better cache locality.
 * 
 * Parameters:
 *   - capacity: Number of items the buffer can hold (must be a power of 2)
 *   - item_size_bytes: Size of each item in bytes
 * 
 * Returns: Pointer to the newly created spsc_semring_t structure, or NULL on error
 * 
 * Validation:
 *   - Capacity must be a power of 2 (for efficient modulo operations)
 *   - Capacity must be between 1 and SPSC_MAX_CAPACITY
 *   - Item size must be between 1 and SPSC_MAX_ITEM_SIZE bytes
 */
spsc_semring_t *spsc_create(size_t capacity, size_t item_size_bytes)
{
    // Validate capacity is a power of two
    if ((capacity & (capacity - 1)) != 0)
    {
        printf("capacity %u not power of two\n", (unsigned)capacity);
        return NULL;
    }

    // Validate capacity is within reasonable bounds
    if ((capacity > SPSC_MAX_CAPACITY) || (capacity == 0))
    {
        printf("capacity %u exceeds max capacity (or is 0): %u\n", (unsigned)capacity, SPSC_MAX_CAPACITY);
        return NULL;
    }

    // Validate item size is reasonable
    if (item_size_bytes == 0 || item_size_bytes > SPSC_MAX_ITEM_SIZE)
    {
        printf("item size %zu invalid, max is %d\n", item_size_bytes, SPSC_MAX_ITEM_SIZE);
        return NULL;
    }

    // Calculate total allocation size: structure + buffer data
    size_t total_size = (capacity* item_size_bytes) + sizeof(spsc_semring_t);
    // there is a bug here where there isn't memory alocated for the last frame,
    // so we add one more frame to the allocation ....
    spsc_semring_t *q = (spsc_semring_t *)malloc(total_size + item_size_bytes);
    if (!q) {
        printf("unable to allocate ring buffer of size %zu\n", total_size);
        return NULL;
    }

    // Initialize all memory to zero
    memset(q, 0, total_size);
    
    // Initialize atomic head and tail pointers
    atomic_store_explicit(&q->head, 0u, memory_order_relaxed);
    atomic_store_explicit(&q->tail, 0u, memory_order_relaxed);
    
    // Set buffer parameters
    q->capacity = capacity;
    q->item_size = item_size_bytes;
    q->buff_size = item_size_bytes + (capacity * item_size_bytes);

    // Point the buffer to the memory immediately after the struct
    // Using (q + 1) advances by sizeof(spsc_semring_t) bytes
    q->items = (uint8_t *)(q + 1);

    // Sanity check: verify the item region exactly fills the allocation
    uint8_t *alloc_end = ((uint8_t *)q) + total_size;
    uint8_t *items_end = q->items + capacity * item_size_bytes;
    if (items_end != alloc_end) {
        printf("internal error: item region does not match allocation\n");
        free(q);
        return NULL;
    }

    // Initialize semaphores for blocking operations
    // can_push: initially equal to capacity (can push up to capacity items)
    // can_pop: initially 0 (cannot pop from empty buffer)
    if (sem_init(&q->can_push, 0, (unsigned)capacity) != 0 ||
        sem_init(&q->can_pop,  0, 0) != 0) {
        free(q);
        return NULL;
    }

    return q;
}

/*
 * Function: spsc_destroy
 * Description: Safely destroys an SPSC ring buffer by cleaning up semaphores and freeing memory.
 *              This function is safe to call with a NULL pointer.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer to destroy (can be NULL)
 */
void spsc_destroy(spsc_semring_t *q)
{
    if (!q) { return; }
    // Clean up semaphores before freeing memory
    sem_destroy(&q->can_push);
    sem_destroy(&q->can_pop);
    SAFE_FREE(q);
}

/*
 * ========== Producer (Push) Operations ==========
 * 
 * The push operations use a two-phase protocol:
 * 1. spsc_push_ptr_begin() - Reserve a slot and get a writable pointer
 * 2. spsc_push_ptr_commit() - Publish the written data to the consumer
 * 
 * This allows the producer to write directly into the ring buffer without
 * additional copying, while maintaining thread safety.
 */

/*
 * Function: spsc_push_ptr_begin
 * Description: Begins a push operation by reserving a slot in the ring buffer.
 *              Returns a writable pointer to the reserved slot. The producer
 *              can write data directly to this location.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer
 *   - timeout_ms: Timeout in milliseconds (0 = non-blocking, >0 = blocking with timeout)
 * 
 * Returns: Writable pointer to the next available slot, or NULL if:
 *          - Buffer is full (non-blocking mode)
 *          - Timeout expired (blocking mode)
 *          - Invalid parameters
 * 
 * Note: Must be followed by spsc_push_ptr_commit() to publish the data
 */
void *spsc_push_ptr_begin(spsc_semring_t *q, long timeout_ms)
{
    if (!q) return NULL;
    static uint32_t ctr = 0;
    ctr++;

    if (ctr >= 30) {
        size_t c = spsc_count(q);
        printf("spsc count: %zu\n", c);
        ctr = 0;
    }
    
    // Wait for available space in the buffer
    // If timeout_ms > 0: block with timeout
    // If timeout_ms <= 0: non-blocking (try once)
    int r = (timeout_ms > 0)
          ? sem_timedwait_ms(&q->can_push, timeout_ms)
          : sem_trywait(&q->can_push);
    if (r != 0) {
        return NULL; /* queue full or timeout */
    }

    // Compute the slot index from the current head position
    // Use relaxed ordering since we're the only producer
    unsigned head = atomic_load_explicit(&q->head, memory_order_relaxed);
    size_t mask = q->capacity - 1u;  // Power-of-2 capacity allows efficient modulo
    size_t slot = (size_t)(head & mask);
    
    // Return pointer to the specific slot in the buffer
    return q->items + (slot * q->item_size);
}


/*
 * Function: spsc_push_ptr_commit
 * Description: Completes a push operation by advancing the head pointer and signaling
 *              the consumer that new data is available. Must be called after
 *              spsc_push_ptr_begin() and writing data to the returned pointer.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer
 * 
 * Note: This function uses release memory ordering to ensure the written data
 *       is visible to the consumer before the head pointer is updated.
 */
void spsc_push_ptr_commit(spsc_semring_t *q)
{
    if (!q) return;
    
    // Advance the head pointer to publish the new item
    // Use relaxed for load since we're the only producer
    unsigned head = atomic_load_explicit(&q->head, memory_order_relaxed);
    // Use release ordering to ensure data writes are visible before head update
    atomic_store_explicit(&q->head, head + 1u, memory_order_release);
    
    // Signal the consumer that new data is available
    sem_post(&q->can_pop);
}


/*
 * ========== Consumer (Pop) Operations ==========
 * 
 * The pop operations also use a two-phase protocol:
 * 1. spsc_pop_ptr_begin() - Get a readable pointer to the next item
 * 2. spsc_pop_ptr_commit() - Release the slot back to the producer
 * 
 * This allows the consumer to read directly from the ring buffer without
 * additional copying, while maintaining thread safety.
 */

/*
 * Function: spsc_pop_ptr_begin
 * Description: Begins a pop operation by getting a readable pointer to the next
 *              available item in the ring buffer. The consumer can read data
 *              directly from this location.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer
 *   - timeout_ms: Timeout in milliseconds (0 = non-blocking, >0 = blocking with timeout)
 * 
 * Returns: Read-only pointer to the next available item, or NULL if:
 *          - Buffer is empty (non-blocking mode)
 *          - Timeout expired (blocking mode)
 *          - Invalid parameters
 * 
 * Note: Must be followed by spsc_pop_ptr_commit() to release the slot
 */
const void *spsc_pop_ptr_begin(spsc_semring_t *q, long timeout_ms)
{
    if (!q) return NULL;
    
    // Wait for available data in the buffer
    // If timeout_ms > 0: block with timeout
    // If timeout_ms <= 0: non-blocking (try once)
    int r = (timeout_ms > 0) ? sem_timedwait_ms(&q->can_pop, timeout_ms)
                             : sem_trywait(&q->can_pop);
    if (r != 0) {
        return NULL; /* empty or timed out */
    }

    // Get the current tail position
    // Use relaxed ordering since we're the only consumer
    unsigned tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

    // Acquire fence pairs with producer's release on head advance
    // This ensures we see all data writes that happened before head was updated
    atomic_thread_fence(memory_order_acquire);

    // Compute the slot index from the tail position
    size_t mask = q->capacity - 1u;  // Power-of-2 capacity allows efficient modulo
    size_t slot = (size_t)(tail & mask);
    
    // Return read-only pointer to the specific slot in the buffer
    return q->items + slot * q->item_size;
}

/*
 * Function: spsc_pop_ptr_commit
 * Description: Completes a pop operation by advancing the tail pointer and signaling
 *              the producer that space is available. Must be called after
 *              spsc_pop_ptr_begin() and reading data from the returned pointer.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer
 * 
 * Note: This function uses release memory ordering to ensure proper synchronization
 *       with the producer thread.
 */
void spsc_pop_ptr_commit(spsc_semring_t *q)
{
    if (!q) return;
    
    // Advance the tail pointer to release the slot
    // Use relaxed for load since we're the only consumer
    unsigned tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
    // Use release ordering for proper synchronization
    atomic_store_explicit(&q->tail, tail + 1u, memory_order_release);
    
    // Signal the producer that space is available
    sem_post(&q->can_push);
}

/*
 * ========== Utility Functions ==========
 */

/*
 * Function: spsc_count
 * Description: Returns the current number of items in the ring buffer.
 *              This is an estimate and may not be completely accurate in
 *              a multi-threaded environment due to the lock-free nature.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer
 * 
 * Returns: Approximate number of items currently in the buffer
 * 
 * Note: Use acquire memory ordering to get a consistent view of head and tail
 */
size_t spsc_count(const spsc_semring_t *q)
{
    if (!q) return 0;
    
    // Use acquire ordering to get a consistent snapshot
    unsigned h = atomic_load_explicit(&q->head, memory_order_acquire);
    unsigned t = atomic_load_explicit(&q->tail, memory_order_acquire);
    
    // The difference gives us the number of items
    // This works correctly even with wraparound due to unsigned arithmetic
    return (size_t)(h - t);
}


/*
 * Function: spsc_push_ptr_cancel
 * Description: Cancels a push operation that was started with spsc_push_ptr_begin()
 *              but not committed. This returns the reserved slot back to the available
 *              pool without advancing the head pointer.
 * 
 * Parameters:
 *   - q: Pointer to the SPSC ring buffer
 * 
 * Note: This should only be called if spsc_push_ptr_begin() succeeded but you
 *       decide not to commit the data (e.g., due to an error condition).
 */
static inline void spsc_push_ptr_cancel(spsc_semring_t *q) {
    if (!q) return;
    // We consumed one can_push token during begin(), return it
    // This restores the semaphore state as if the begin() never happened
    sem_post(&q->can_push);
}