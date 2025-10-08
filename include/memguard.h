/* memguard.h
 *
 * Header-only allocation guard and stats for Linux.
 * Features
 *  - Front and back guard regions prefilled with patterns
 *  - Corruption checks per block and global scan
 *  - Allocation list for leak reporting
 *  - Thread safe with pthread mutex
 *  - Optional overrides for malloc family
 *
 * Usage
 *  #define MEMGUARD_IMPLEMENTATION
 *  #include "memguard.h"
 *
 *  // Optional, to override malloc/free in this TU:
 *  #define MEMGUARD_OVERRIDE_STDLIB
 *  #include "memguard.h"
 */

#ifndef MEMGUARD_H
#define MEMGUARD_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <execinfo.h>


#ifdef __cplusplus
extern "C" {
#endif


/* Configuration defaults */
#ifndef MEMGUARD_FRONT_BYTES
#define MEMGUARD_FRONT_BYTES 4096u
#endif

#ifndef MEMGUARD_BACK_BYTES
#define MEMGUARD_BACK_BYTES 4096u
#endif

#ifndef MEMGUARD_PATTERN_FRONT
#define MEMGUARD_PATTERN_FRONT 0xCAu
#endif

#ifndef MEMGUARD_PATTERN_BACK
#define MEMGUARD_PATTERN_BACK  0xFEu
#endif

#ifndef MEMGUARD_HEADER_MAGIC
#define MEMGUARD_HEADER_MAGIC 0xB173F1A5u
#endif

#ifndef MEMGUARD_TAIL_MAGIC
#define MEMGUARD_TAIL_MAGIC   0xA11C0FFEu
#endif

#ifndef MEMGUARD_BT_FRAMES
#define MEMGUARD_BT_FRAMES 8
#endif


/* Stats structure */
struct memguard_stats {
    size_t total_allocs;
    size_t total_frees;
    size_t failed_allocs;
    size_t bytes_current;      /* sum of requested sizes currently live */
    size_t bytes_peak;         /* high water mark of bytes_current */
    size_t bytes_overhead;     /* bytes consumed in guards + headers currently live */
    size_t blocks_current;     /* number of live blocks */
};

/* API */
void   memguard_init(size_t front_guard_bytes, size_t back_guard_bytes);
void   memguard_set_enabled(int enabled); /* enable or disable guarding at runtime */
void*  memguard_malloc(size_t n, const char *file, int line);
void*  memguard_calloc(size_t nmemb, size_t size, const char *file, int line);
void*  memguard_realloc(void *ptr, size_t n, const char *file, int line);
char*  memguard_strdup(const char *s, const char *file, int line);
void   memguard_free(void *ptr, const char *file, int line);

int    memguard_check_block(const void *user_ptr);     /* returns 0 on OK, nonzero on corruption */
size_t memguard_check_all(FILE *report);               /* returns number of corrupt blocks found */
void   memguard_get_stats(struct memguard_stats *out); /* snapshot stats */
size_t memguard_report_leaks(FILE *out);               /* prints any live blocks, returns count */
void   memguard_teardown(void);                        /* frees internal state, no leak report */

/* Convenience macros to capture file and line */
#define mg_malloc(n)        memguard_malloc((n), __FILE__, __LINE__)
#define mg_calloc(c, s)     memguard_calloc((c), (s), __FILE__, __LINE__)
#define mg_realloc(p, n)    memguard_realloc((p), (n), __FILE__, __LINE__)
#define mg_strdup(s)        memguard_strdup((s), __FILE__, __LINE__)
#define mg_free(p)          memguard_free((p), __FILE__, __LINE__)

/* Optional overrides for the standard functions in this translation unit */
#ifdef MEMGUARD_OVERRIDE_STDLIB
#  undef malloc
#  undef calloc
#  undef realloc
#  undef free
#  undef strdup
#  define malloc(n)        mg_malloc(n)
#  define calloc(c,s)      mg_calloc((c),(s))
#  define realloc(p,n)     mg_realloc((p),(n))
#  define free(p)          mg_free(p)
#  define strdup(s)        mg_strdup(s)
#endif

#ifdef __cplusplus
}
#endif

/* ===================== Implementation ===================== */
#ifdef MEMGUARD_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <stdalign.h>

#ifndef MEMGUARD_ALIGN
#define MEMGUARD_ALIGN alignof(max_align_t)
#endif

#define MG_ALIGN_UP(v,a)   (((v) + ((a) - 1)) & ~((a) - 1))

typedef struct mg_header mg_header;

struct mg_header {
    uint32_t magic;             /* header integrity */
    uint32_t tail_magic;        /* static constant to help detect underruns in header area */
    size_t   requested_size;    /* bytes requested by the caller */
    size_t   alloc_size;        /* bytes actually reserved for user payload */
    size_t   front_guard;       /* bytes before user payload */
    size_t   back_guard;        /* bytes after user payload */
    const char *file;           /* allocation site */
    int      line;              /* allocation site */
    int      is_active;         /* set to 0 on free */
    uint32_t header_crc;        // checksum of the header, excluding this field and magic


    void   *bt[MEMGUARD_BT_FRAMES];
    int     bt_count;

    mg_header *prev;
    mg_header *next;
    /* flexible area follows:
       [front_guard bytes of MEMGUARD_PATTERN_FRONT]
       [user payload of alloc_size bytes]
       [back_guard bytes of MEMGUARD_PATTERN_BACK]
       [uint32_t tail_magic duplicate] */
};

/* Globals */
static struct {
    pthread_mutex_t lock;
    mg_header *head;
    struct memguard_stats stats;
    size_t front_bytes;
    size_t back_bytes;
    int enabled;
} mg_g = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .head = NULL,
    .stats = {0},
    .front_bytes = MEMGUARD_FRONT_BYTES,
    .back_bytes  = MEMGUARD_BACK_BYTES,
    .enabled = 1
};

static void mg_list_insert(mg_header *h) {
    h->prev = NULL;
    h->next = mg_g.head;
    if (mg_g.head) mg_g.head->prev = h;
    mg_g.head = h;
}

static void mg_list_remove(mg_header *h) {
    if (h->prev) h->prev->next = h->next;
    else mg_g.head = h->next;
    if (h->next) h->next->prev = h->prev;
}

static uint8_t* mg_front_ptr(mg_header *h) {
    return (uint8_t*)(h + 1);
}

static void* mg_user_ptr(mg_header *h) {
    return mg_front_ptr(h) + h->front_guard;
}

static uint8_t* mg_back_ptr(mg_header *h) {
    return (uint8_t*)mg_user_ptr(h) + h->alloc_size;
}

static int mg_is_in_active_list(const mg_header *h) {
    for (mg_header *it = mg_g.head; it; it = it->next) if (it == h) return 1;
    return 0;
}

/* Tiny CRC32 or Fowler–Noll–Vo 32 */
static uint32_t mg_crc32(const void *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    const uint8_t *b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) {
        c ^= b[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (-(int)(c & 1)));
    }
    return ~c;
}

static mg_header* mg_from_user(const void *user_ptr) {
    if (!user_ptr) return NULL;
    const uint8_t *p = (const uint8_t*)user_ptr;
    mg_header *h = (mg_header*)(p - sizeof(mg_header) - 0); /* header is immediately before front guard */
    /* But we do not know front_guard yet, so compute using stored field */
    h = (mg_header*)((const uint8_t*)user_ptr - sizeof(mg_header) - ((mg_header*)((const uint8_t*)user_ptr - sizeof(mg_header)))->front_guard);
    return h;
}

/* Public API impl */

void memguard_init(size_t front_guard_bytes, size_t back_guard_bytes) {
    pthread_mutex_lock(&mg_g.lock);
    mg_g.front_bytes = front_guard_bytes;
    mg_g.back_bytes  = back_guard_bytes;
    pthread_mutex_unlock(&mg_g.lock);
}

static pthread_once_t mg_once = PTHREAD_ONCE_INIT;
static void mg_init_once(void) { atexit(mg_at_exit); }

void memguard_set_enabled(int enabled) {
    pthread_mutex_lock(&mg_g.lock);
    mg_g.enabled = enabled ? 1 : 0;
    pthread_mutex_unlock(&mg_g.lock);
    pthread_once(&mg_once, mg_init_once);

}

static void mg_print_bt(FILE *out, mg_header *h) {
    char **syms = backtrace_symbols(h->bt, h->bt_count);
    if (syms) {
        for (int i = 0; i < h->bt_count; i++)
            fprintf(out, "    %s\n", syms[i]);
        free(syms);
    }
}

static void mg_fill(uint8_t *p, size_t n, uint8_t byte) {
    memset(p, (int)byte, n);
}

static void mg_update_stats_on_alloc(size_t req, size_t overhead) {
    mg_g.stats.total_allocs++;
    mg_g.stats.blocks_current++;
    mg_g.stats.bytes_current += req;
    mg_g.stats.bytes_overhead += overhead;
    if (mg_g.stats.bytes_current > mg_g.stats.bytes_peak)
        mg_g.stats.bytes_peak = mg_g.stats.bytes_current;
}

static void mg_update_stats_on_free(size_t req, size_t overhead) {
    mg_g.stats.total_frees++;
    if (mg_g.stats.blocks_current) mg_g.stats.blocks_current--;
    if (mg_g.stats.bytes_current >= req) mg_g.stats.bytes_current -= req;
    if (mg_g.stats.bytes_overhead >= overhead) mg_g.stats.bytes_overhead -= overhead;
}

/* Allocate raw block:
   [mg_header][front_guard][user][back_guard][tail_magic] */
static void* mg_alloc_block(size_t req, const char *file, int line) {
    const size_t front = MG_ALIGN_UP(mg_g.front_bytes, MEMGUARD_ALIGN);
    const size_t back  = MG_ALIGN_UP(mg_g.back_bytes, MEMGUARD_ALIGN);
    const size_t header_sz = sizeof(mg_header);
    const size_t tail_sz = sizeof(uint32_t);
    const size_t total = header_sz + front + req + back + tail_sz;

    uint8_t *raw = (uint8_t*)malloc(total);
    if (!raw) {
        mg_g.stats.failed_allocs++;
        return NULL;
    }

    mg_header *h = (mg_header*)raw;
    h->magic = MEMGUARD_HEADER_MAGIC;
    h->tail_magic = MEMGUARD_TAIL_MAGIC;
    h->requested_size = req;
    h->alloc_size = req;
    h->front_guard = front;
    h->back_guard = back;
    h->file = file;
    h->line = line;
    h->prev = h->next = NULL;
    h->bt_count = backtrace(h->bt, MEMGUARD_BT_FRAMES);
    h->header_crc = mg_crc32(h, offsetof(mg_header, prev));

    uint8_t *front_p = mg_front_ptr(h);
    uint8_t *user_p  = front_p + front;
    uint8_t *back_p  = user_p + req;
    uint32_t *tail_p = (uint32_t*)(back_p + back);

    mg_fill(front_p, front, MEMGUARD_PATTERN_FRONT);
    mg_fill(back_p,  back,  MEMGUARD_PATTERN_BACK);
    *tail_p = MEMGUARD_TAIL_MAGIC;

    mg_list_insert(h);
    mg_update_stats_on_alloc(req, front + back + header_sz + tail_sz);

    return user_p;
}

void* memguard_malloc(size_t n, const char *file, int line) {
    if (!mg_g.enabled) return malloc(n);
    void *p = NULL;
    pthread_mutex_lock(&mg_g.lock);
    p = mg_alloc_block(n, file, line);
    pthread_mutex_unlock(&mg_g.lock);
    return p;
}

void* memguard_calloc(size_t nmemb, size_t size, const char *file, int line) {
    size_t n = nmemb * size;
    void *p = memguard_malloc(n, file, line);
    if (p) memset(p, 0, n);
    return p;
}

static size_t mg_first_bad_offset(uint8_t *p, size_t n, uint8_t expect) {
    for (size_t i = 0; i < n; i++) if (p[i] != expect) return i;
    return (size_t)-1;
}

static int mg_check_ranges(mg_header *h, FILE *report, const char *what) {
    int bad = 0;
    if (h->magic != MEMGUARD_HEADER_MAGIC || h->tail_magic != MEMGUARD_TAIL_MAGIC) {
        if (report) {
            fprintf(report, "[memguard] header corrupted at %p, from %s:%d, label=%s\n",
                            (void*)h, h->file, h->line, what);
            mg_print_bt(report, h);
        }
        return 1;
    }
    uint8_t *front_p = mg_front_ptr(h);
    uint8_t *user_p  = (uint8_t*)mg_user_ptr(h);
    uint8_t *back_p  = mg_back_ptr(h);
    uint32_t *tail_p = (uint32_t*)(back_p + h->back_guard);

    for (size_t i = 0; i < h->front_guard; i++) {
        if (front_p[i] != MEMGUARD_PATTERN_FRONT) { bad = 1; break; }
    }

    
    if (bad && report) {
        fprintf(report, "[memguard] front guard overwrite detected at %p from %s:%d\n",
                (void*)h, h->file, h->line);
        mg_print_bt(report, h);
    }

    bad = bad || (*tail_p != MEMGUARD_TAIL_MAGIC);
    if (*tail_p != MEMGUARD_TAIL_MAGIC && report) {
        fprintf(report, "[memguard] tail magic mismatch at %p from %s:%d\n",
                (void*)h, h->file, h->line);
        mg_print_bt(report, h);
    }

    if (!bad) {
        for (size_t i = 0; i < h->back_guard; i++) {
            if (back_p[i] != MEMGUARD_PATTERN_BACK) { bad = 1; break; }
        }
        if (bad && report) {
            fprintf(report, "[memguard] back guard overwrite detected at %p from %s:%d\n",
                    (void*)h, h->file, h->line);
            mg_print_bt(report, h);
        }
    }

    /* Additional light sanity on user pointer alignment */
    if ((((uintptr_t)user_p) % MEMGUARD_ALIGN) != 0 && report) {
        fprintf(report, "[memguard] user pointer %p not aligned to %zu for %s:%d\n",
                (void*)user_p, (size_t)MEMGUARD_ALIGN, h->file, h->line);
        mg_print_bt(report, h);
    }

    // verify the checksum
    uint32_t crc = mg_crc32(h, offsetof(mg_header, prev));
    if (crc != h->header_crc) {
        if (report) fprintf(report, "[memguard] header crc mismatch at %p\n", (void*)h);
        bad = 1;
    }

    if (bad && report) {
        size_t off = mg_first_bad_offset(front_p, h->front_guard, MEMGUARD_PATTERN_FRONT);
        if (off != (size_t)-1) fprintf(report, "[memguard] front corrupt at +%zu bytes\n", off);

        off = mg_first_bad_offset(back_p, h->back_guard, MEMGUARD_PATTERN_BACK);
        if (off != (size_t)-1) fprintf(report, "[memguard] back corrupt at +%zu bytes\n", off);
    }

    return bad ? 1 : 0;
}

int memguard_check_block(const void *user_ptr) {
    if (!user_ptr) return 0;
    if (!mg_g.enabled) return 0;
    int rc;
    pthread_mutex_lock(&mg_g.lock);
    mg_header *h = mg_from_user(user_ptr);
    rc = mg_check_ranges(h, stderr, "single_check");
    pthread_mutex_unlock(&mg_g.lock);
    return rc;
}

size_t memguard_check_all(FILE *report) {
    if (!mg_g.enabled) return 0;
    if (!report) report = stderr;
    size_t bad = 0;
    pthread_mutex_lock(&mg_g.lock);
    for (mg_header *it = mg_g.head; it; it = it->next) {
        bad += mg_check_ranges(it, report, "global_scan");
    }
    pthread_mutex_unlock(&mg_g.lock);
    return bad;
}

void* memguard_realloc(void *ptr, size_t n, const char *file, int line) {
    if (!mg_g.enabled) {
        return realloc(ptr, n);
    }
    if (!ptr) {
        return memguard_malloc(n, file, line);
    }

    pthread_mutex_lock(&mg_g.lock);
    mg_header *old_h = mg_from_user(ptr);

    /* Check current block before resizing */
    (void)mg_check_ranges(old_h, stderr, "realloc_pre");

    /* allocate new block and copy */
    void *new_user = mg_alloc_block(n, file, line);
    if (!new_user) {
        pthread_mutex_unlock(&mg_g.lock);
        return NULL;
    }

    size_t copy_n = old_h->requested_size < n ? old_h->requested_size : n;
    memcpy(new_user, ptr, copy_n);

    /* free old block internals */
    uint8_t *back_p = mg_back_ptr(old_h);
    uint32_t *tail_p = (uint32_t*)(back_p + old_h->back_guard);
    *tail_p = 0; /* poison */
    old_h->magic = 0;

    const size_t overhead = old_h->front_guard + old_h->back_guard + sizeof(mg_header) + sizeof(uint32_t);
    mg_list_remove(old_h);
    mg_update_stats_on_free(old_h->requested_size, overhead);
    free(old_h);

    pthread_mutex_unlock(&mg_g.lock);
    return new_user;
}

char* memguard_strdup(const char *s, const char *file, int line) {
    size_t n = strlen(s) + 1;
    char *p = (char*)memguard_malloc(n, file, line);
    if (p) memcpy(p, s, n);
    return p;
}

void memguard_free(void *ptr, const char *file, int line) {
    (void)file; (void)line; /* unused, retained for symmetry and logging if desired */
    if (!ptr) return;

    if (!mg_g.enabled) {
        free(ptr);
        return;
    }

    pthread_mutex_lock(&mg_g.lock);
    mg_header *h = mg_from_user(ptr);

    /* Validate before free */
    (void)mg_check_ranges(h, stderr, "free");

    // double free check
    if (h->is_active == 0 || !mg_is_in_active_list(h)) {
        fprintf(stderr, "[memguard] invalid or double free at %p\n", ptr);
        pthread_mutex_unlock(&mg_g.lock);
        return; /* keep running for diagnostics */
    }
    h->is_active = 0;

    /* Poison and unlink */
    uint8_t *front_p = mg_front_ptr(h);
    uint8_t *back_p  = mg_back_ptr(h);
    uint32_t *tail_p = (uint32_t*)(back_p + h->back_guard);
    memset(front_p, 0xDD, h->front_guard + h->alloc_size + h->back_guard);
    *tail_p = 0;
    h->tail_magic = 0;
    h->magic = 0;

    const size_t overhead = h->front_guard + h->back_guard + sizeof(mg_header) + sizeof(uint32_t);
    mg_list_remove(h);
    mg_update_stats_on_free(h->requested_size, overhead);

    free(h);
    pthread_mutex_unlock(&mg_g.lock);
}

void memguard_get_stats(struct memguard_stats *out) {
    if (!out) return;
    pthread_mutex_lock(&mg_g.lock);
    *out = mg_g.stats;
    pthread_mutex_unlock(&mg_g.lock);
}

size_t memguard_report_leaks(FILE *out) {
    if (!out) out = stderr;
    size_t count = 0;
    pthread_mutex_lock(&mg_g.lock);
    for (mg_header *it = mg_g.head; it; it = it->next) {
        fprintf(out, "[memguard] leak: %zu bytes from %s:%d at user=%p\n",
                it->requested_size, it->file, it->line, mg_user_ptr(it));
        mg_print_bt(out, it);
        count++;
    }
    pthread_mutex_unlock(&mg_g.lock);
    return count;
}

void memguard_teardown(void) {
    pthread_mutex_lock(&mg_g.lock);
    /* free any remaining without reporting */
    mg_header *it = mg_g.head;
    while (it) {
        mg_header *next = it->next;
        free(it);
        it = next;
    }
    mg_g.head = NULL;
    pthread_mutex_unlock(&mg_g.lock);
}

// test that the guard is working
int memguard_self_test(FILE *out) {
    char *p = mg_malloc(16);
    if (!p) return -1;
    p[20] = 0x42; /* overflow */
    size_t bad = memguard_check_all(out ? out : stderr);
    mg_free(p);
    return bad ? 0 : -1;
}

static void mg_at_exit(void) {
    size_t leaks = memguard_report_leaks(stderr);
    if (leaks) fprintf(stderr, "[memguard] %zu leaks detected at exit\n", leaks);
}

#endif /* MEMGUARD_IMPLEMENTATION */

#endif /* MEMGUARD_H */

