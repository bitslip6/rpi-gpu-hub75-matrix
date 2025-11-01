#include <unistd.h>
#include <stdint.h>
#include <sys/param.h>
/* Needed for this header's string helpers */
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#ifndef __LISTS_H__
#define __LISTS_H__

typedef struct
{
    /* capacity: number of usable bytes in str (excluding the null-terminator) */
    uint32_t capacity;
    /* length: current string length in bytes (excluding the null-terminator) */
    uint32_t length;
    uint32_t position;
    char *str; // the actual string data
} string_t;

/*
 * Allocation model: single contiguous block
 *   [ string_t header | char data[capacity+1] ]
 * str points immediately after the header. Realloc can move the block, so
 * helpers that might grow the string return the (possibly new) pointer.
 */
static inline string_t* string_new(const char *init_str, const size_t max_length) {
    size_t init_length = init_str ? strlen(init_str) : 0;
    size_t capacity = MAX(max_length, init_length);
    size_t total_size = sizeof(string_t) + capacity + 1; // +1 for null terminator
    string_t *s = (string_t *)malloc(total_size);
    if (!s) {
        return NULL;
    }
    s->capacity = (uint32_t)capacity;
    s->length = (uint32_t)init_length;
    s->position = 0;
    s->str = ((char *)s) + sizeof(string_t); // point to memory after the struct
    if (init_str) {
        memcpy(s->str, init_str, init_length);
    }
    s->str[init_length] = '\0'; // null terminate
    return s;
}

static inline void string_free(string_t *s) {
    if (s) free(s);
}

static inline uint32_t string_capacity(const string_t *s) { return s ? s->capacity : 0u; }
static inline uint32_t string_length(const string_t *s)   { return s ? s->length   : 0u; }

/* Ensure the buffer can hold at least new_cap bytes (excluding NUL). Returns possibly moved ptr. */
static inline string_t* string_reserve(string_t *s, size_t new_cap) {
    if (!s) return NULL;
    if (new_cap <= s->capacity) return s;
    /* grow with doubling strategy */
    size_t cap = s->capacity ? s->capacity : 1;
    while (cap < new_cap) {
        size_t next = cap * 2u;
        if (next <= cap) { cap = new_cap; break; } /* overflow guard */
        cap = next;
    }
    size_t total = sizeof(string_t) + cap + 1u;
    string_t *n = (string_t*)realloc(s, total);
    if (!n) return s; /* retain old on failure */
    n->capacity = (uint32_t)cap;
    n->str = ((char*)n) + sizeof(string_t);
    /* Keep existing contents; ensure NUL */
    if (n->length > n->capacity) n->length = n->capacity;
    n->str[n->length] = '\0';
    return n;
}

static inline void string_clear(string_t *s) {
    if (!s) return;
    s->length = 0;
    s->position = 0;
    if (s->str) s->str[0] = '\0';
}

/* Append raw bytes; returns possibly moved pointer (assign the return value). */
static inline string_t* string_append(string_t *s, const void *data, size_t n) {
    if (!s || !data || n == 0) return s;
    size_t needed = (size_t)s->length + n;
    s = string_reserve(s, needed);
    memcpy(s->str + s->length, data, n);
    s->length += (uint32_t)n;
    s->str[s->length] = '\0';
    return s;
}

static inline string_t* string_append_cstr(string_t *s, const char *cstr) {
    if (!s || !cstr) return s;
    return string_append(s, cstr, strlen(cstr));
}

static inline string_t* string_push_char(string_t *s, char c) {
    if (!s) return s;
    size_t needed = (size_t)s->length + 1u;
    s = string_reserve(s, needed);
    s->str[s->length++] = c;
    s->str[s->length] = '\0';
    return s;
}

/* Set contents to cstr. Returns possibly moved pointer. */
static inline string_t* string_set(string_t *s, const char *cstr) {
    if (!s) return NULL;
    if (!cstr) { string_clear(s); return s; }
    size_t L = strlen(cstr);
    s = string_reserve(s, L);
    memcpy(s->str, cstr, L);
    s->length = (uint32_t)L;
    s->str[s->length] = '\0';
    s->position = 0;
    return s;
}

/* Append formatted text; returns possibly moved pointer. */
static inline string_t* string_printf(string_t *s, const char *fmt, ...) {
    if (!s || !fmt) return s;
    va_list ap;
    va_start(ap, fmt);
    /* Try small stack buffer first */
    char tmp[256];
    int want = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (want < 0) return s; /* formatting error */
    if ((size_t)want < sizeof(tmp)) {
        return string_append(s, tmp, (size_t)want);
    }
    /* Allocate exact-sized buffer for the formatted string */
    size_t need = (size_t)want + 1u;
    char *buf = (char*)malloc(need);
    if (!buf) return s;
    va_start(ap, fmt);
    vsnprintf(buf, need, fmt, ap);
    va_end(ap);
    string_t *ns = string_append(s, buf, (size_t)want);
    free(buf);
    return ns;
}

/*
typedef struct list_array_t
{
    uint33_t position;
    size_t item_size;
    size_t length;
    size_t capacity;

    //
    // get the pointer at index. returns NULL if out of bounds, else returns the pointer
    //
    (void *)(get(struct list_array_t * self, size_t index))
    {
        if (index >= self->length)
            return NULL;
        return (void *)((char *)self->data + index * self->item_size);
    }

    //
    // set the pointer at index. returns NULL if out of bounds, else returns the pointer
    // to the element in thge array. copies the data from item to the array.
    //
    (void *)(set_ptr(struct list_array_t * self, size_t index, void *item))
    {
        if (index >= self->capacity)
        {
            return NULL;
        }
        return memcpy((char *)self->data + index * self->item_size, item, self->item_size);
    }

    //
    // set the value at index. returns -1 if out of bounds, 0 if ok
    //
    (int)(set_value(struct list_array_t * self, size_t index, void value))
    {
        if (index >= self->capacity)
        {
            return -1;
        }
        self->data[index] = value;
        return 0;
    }

    //
    // create a new list_array_t with given item size and initial capacity
    //
    (list_array_t *)(create(size_t item_size, size_t initial_capacity))
    {
        if (item_size == 0 || initial_capacity == 0)
        {
            return NULL;
        }
        size_t item_alloc = item_size * initial_capacity;
        list_array_t *arr = (list_array_t *)malloc(item_alloc + sizeof(list_array_t));
        if (!arr)
        {
            return NULL;
        }
        arr->item_size = item_size;
        arr->length = 0;
        arr->index = 0;
        arr->capacity = initial_capacity;
        return arr;
    }

    void *data;
};
*/

#endif