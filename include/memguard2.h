/* memguard.h  —  allocation guards, stats, and optional mprotect redzones
 *
 * Features:
 *  - Front and back byte guards with canaries
 *  - Optional page guards via mmap + mprotect to catch underruns and overruns
 *  - Double free and invalid free detection
 *  - Small quarantine to help catch use-after-free
 *  - Poison on free, scribble on alloc (debug fills)
 *  - Optional backtrace capture for leaks and corruption
 *  - Optional header CRC
 *  - Global stats and leak report, thread safe
 *
 * Layout (non-mprotect and mprotect inner mapping are identical):
 *    [front_bytes canary][mg_header][user bytes][back_bytes canary][tail_magic]
 *  mprotect mode adds PROT_NONE guard pages before and/or after the inner mapping.
 *
 * Usage:
 *    #define MEMGUARD_IMPLEMENTATION
 *    #include "memguard.h"
 *
 *    // Optional override in other TUs:
 *    #define MEMGUARD_OVERRIDE_STDLIB
 *    #include "memguard.h"
 */

#ifndef MEMGUARD_H
#define MEMGUARD_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- Configuration ---------------- */

#ifndef MEMGUARD_FRONT_BYTES
#define MEMGUARD_FRONT_BYTES 32u
#endif
#ifndef MEMGUARD_BACK_BYTES
#define MEMGUARD_BACK_BYTES 32u
#endif

#ifndef MEMGUARD_PATTERN_FRONT
#define MEMGUARD_PATTERN_FRONT 0xCAu
#endif
#ifndef MEMGUARD_PATTERN_BACK
#define MEMGUARD_PATTERN_BACK  0xFEu
#endif
#ifndef MEMGUARD_ALLOC_SCRIBBLE
#define MEMGUARD_ALLOC_SCRIBBLE 0xCD
#endif
#ifndef MEMGUARD_FREE_POISON
#define MEMGUARD_FREE_POISON   0xDD
#endif

#ifndef MEMGUARD_HEADER_MAGIC
#define MEMGUARD_HEADER_MAGIC 0xB173F1A5u
#endif
#ifndef MEMGUARD_TAIL_MAGIC
#define MEMGUARD_TAIL_MAGIC   0xA11C0FFEu
#endif

/* Enable page guards for hard faults on underrun and overrun */
#ifndef MEMGUARD_USE_MPROTECT
#define MEMGUARD_USE_MPROTECT 1
#endif
#ifndef MEMGUARD_GUARD_FRONT_PAGES
#define MEMGUARD_GUARD_FRONT_PAGES 1  /* catch underruns */
#endif
#ifndef MEMGUARD_GUARD_BACK_PAGES
#define MEMGUARD_GUARD_BACK_PAGES 1   /* catch overruns */
#endif
#ifndef MEMGUARD_GUARDPAGE_THRESHOLD
#define MEMGUARD_GUARDPAGE_THRESHOLD 0 /* 0 = use mprotect for all allocations */
#endif

/* Quarantine to help catch UAF. 0 disables. */
#ifndef MEMGUARD_QUARANTINE_MAX
#define MEMGUARD_QUARANTINE_MAX 16
#endif

/* Optional header CRC and backtraces */
#ifndef MEMGUARD_ENABLE_CRC
#define MEMGUARD_ENABLE_CRC 1
#endif
#ifndef MEMGUARD_BT_FRAMES
#define MEMGUARD_BT_FRAMES 8  /* 0 disables backtrace capture */
#endif

#ifndef MEMGUARD_ALIGN
#include <stdalign.h>
#define MEMGUARD_ALIGN alignof(max_align_t)
#endif

/* ---------------- Public API ---------------- */

struct memguard_stats {
    size_t total_allocs;
    size_t total_frees;
    size_t failed_allocs;
    size_t bytes_current;
    size_t bytes_peak;
    size_t bytes_overhead;
    size_t blocks_current;
};

void   memguard_init(size_t front_guard_bytes, size_t back_guard_bytes);
void   memguard_set_enabled(int enabled);

void*  memguard_malloc(size_t n, const char *file, int line);
void*  memguard_calloc(size_t nmemb, size_t size, const char *file, int line);
void*  memguard_realloc(void *ptr, size_t n, const char *file, int line);
char*  memguard_strdup(const char *s, const char *file, int line);
void   memguard_free(void *ptr, const char *file, int line);

int    memguard_check_block(const void *user_ptr);     /* 0 OK, nonzero corrupted */
size_t memguard_check_all(FILE *report);               /* returns corrupt blocks */
void   memguard_get_stats(struct memguard_stats *out);
size_t memguard_report_leaks(FILE *out);
void   memguard_teardown(void);

/* Optional user region protections when mprotect is active */
int    memguard_protect_readonly(void *user_ptr);      /* returns 0 on success */
int    memguard_protect_readwrite(void *user_ptr);

/* Convenience capture site macros */
#define mg_malloc(n)        memguard_malloc((n), __FILE__, __LINE__)
#define mg_calloc(c, s)     memguard_calloc((c), (s), __FILE__, __LINE__)
#define mg_realloc(p, n)    memguard_realloc((p), (n), __FILE__, __LINE__)
#define mg_strdup(s)        memguard_strdup((s), __FILE__, __LINE__)
#define mg_free(p)          memguard_free((p), __FILE__, __LINE__)

/* Optional overrides for standard functions in this TU */
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

/* ---------------- Implementation ---------------- */
#ifdef MEMGUARD_IMPLEMENTATION

#if defined(MEMGUARD_IMPLEMENTATION) && defined(MEMGUARD_OVERRIDE_STDLIB)
#error "Do not define MEMGUARD_OVERRIDE_STDLIB in the TU that defines MEMGUARD_IMPLEMENTATION"
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>

#if MEMGUARD_USE_MPROTECT
#include <sys/mman.h>
#include <unistd.h>
#endif

#if MEMGUARD_BT_FRAMES > 0
#include <execinfo.h>
#endif

#define MG_ALIGN_UP(v,a)   (((v) + ((a) - 1)) & ~((a) - 1))

/* Forward declaration */
typedef struct mg_header mg_header;

/* Internal header lives immediately BEFORE the user region.
 * Layout: [front_canary bytes][mg_header][user][back_canary bytes][tail_magic] */
struct mg_header {
    uint32_t magic;
    uint32_t tail_magic_copy;   /* extra sentinel close to header */
    size_t   requested_size;
    size_t   alloc_size;
    size_t   front_guard;       /* bytes of front canary */
    size_t   back_guard;        /* bytes of back canary */
    const char *file;
    int      line;
    mg_header *prev;
    mg_header *next;
    int      is_active;         /* for double free detection */

#if MEMGUARD_USE_MPROTECT
    int      was_mmap;          /* allocated via mmap path */
    size_t   map_len_total;     /* total mapping including guard pages */
    size_t   page_size;         /* cached page size */
#endif

#if MEMGUARD_ENABLE_CRC
    uint32_t header_crc;        /* CRC over stable part of header */
#endif

#if MEMGUARD_BT_FRAMES > 0
    void    *bt[MEMGUARD_BT_FRAMES];
    int      bt_count;
#endif
};

/* Global state */
static struct {
    pthread_mutex_t lock;
    mg_header *head;                  /* intrusive list of live blocks */
    struct memguard_stats stats;
    size_t front_bytes;
    size_t back_bytes;
    int enabled;

#if MEMGUARD_QUARANTINE_MAX > 0
    mg_header *quarantine[MEMGUARD_QUARANTINE_MAX];
    size_t q_count;
#endif
} mg_g = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .head = NULL,
    .stats = {0},
    .front_bytes = MEMGUARD_FRONT_BYTES,
    .back_bytes  = MEMGUARD_BACK_BYTES,
    .enabled = 1,
#if MEMGUARD_QUARANTINE_MAX > 0
    .quarantine = {0},
    .q_count = 0
#endif
};

static inline uint8_t* mg_front_ptr(mg_header *h) { return (uint8_t*)h - h->front_guard; }
static inline void*    mg_user_ptr (mg_header *h) { return (void*)((uint8_t*)h + sizeof(mg_header)); }
static inline uint8_t* mg_back_ptr (mg_header *h) { return (uint8_t*)mg_user_ptr(h) + h->alloc_size; }
static inline mg_header* mg_from_user(const void *user) {
    return user ? (mg_header*)((const uint8_t*)user - sizeof(mg_header)) : NULL;
}

#if MEMGUARD_ENABLE_CRC
static uint32_t mg_crc32(const void *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    const uint8_t *b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) {
        c ^= b[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (-(int)(c & 1)));
    }
    return ~c;
}
static inline uint32_t mg_header_crc_stable(const mg_header *h) {
    /* exclude prev, next, and changing counters; compute up to prev */
    size_t n = offsetof(mg_header, prev);
    return mg_crc32(h, n);
}
#endif

/* intrusive list ops */
static void mg_list_insert(mg_header *h) {
    h->prev = NULL;
    h->next = mg_g.head;
    if (mg_g.head) mg_g.head->prev = h;
    mg_g.head = h;
}
static void mg_list_remove(mg_header *h) {
    if (h->prev) h->prev->next = h->next; else mg_g.head = h->next;
    if (h->next) h->next->prev = h->prev;
}

/* helpers */
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

void memguard_init(size_t front_guard_bytes, size_t back_guard_bytes) {
    pthread_mutex_lock(&mg_g.lock);
    mg_g.front_bytes = MG_ALIGN_UP(front_guard_bytes, MEMGUARD_ALIGN);
    mg_g.back_bytes  = MG_ALIGN_UP(back_guard_bytes, MEMGUARD_ALIGN);
    pthread_mutex_unlock(&mg_g.lock);
}

/* at-exit leak report */
static void mg_at_exit(void) {
    size_t leaks = memguard_report_leaks(stderr);
    if (leaks) fprintf(stderr, "[memguard] %zu leaks detected at exit\n", leaks);
}
static pthread_once_t mg_once = PTHREAD_ONCE_INIT;
static void mg_install_atexit(void) { atexit(mg_at_exit); }

void memguard_set_enabled(int enabled) {
    pthread_mutex_lock(&mg_g.lock);
    mg_g.enabled = enabled ? 1 : 0;
    pthread_mutex_unlock(&mg_g.lock);
    pthread_once(&mg_once, mg_install_atexit);
}

static int mg_is_in_active_list(const mg_header *h) {
    for (const mg_header *it = mg_g.head; it; it = it->next) if (it == h) return 1;
    return 0;
}

#if MEMGUARD_QUARANTINE_MAX > 0
static void mg_quarantine_push(mg_header *h) {
    if (mg_g.q_count < MEMGUARD_QUARANTINE_MAX) {
        mg_g.quarantine[mg_g.q_count++] = h;
        return;
    }
    /* FIFO: free the oldest, shift */
    mg_header *old = mg_g.quarantine[0];
    memmove(&mg_g.quarantine[0], &mg_g.quarantine[1], (MEMGUARD_QUARANTINE_MAX - 1) * sizeof(old));
    mg_g.quarantine[MEMGUARD_QUARANTINE_MAX - 1] = h;

#if MEMGUARD_USE_MPROTECT
    if (old->was_mmap) {
        uint8_t *base = (uint8_t*)old - ((size_t)MEMGUARD_GUARD_FRONT_PAGES * old->page_size);
        munmap(base, old->map_len_total);
    } else
#endif
    {
        free(mg_front_ptr(old)); /* free from the start of front canary allocation */
    }
}
#endif /* quarantine */

static size_t mg_first_bad_offset(uint8_t *p, size_t n, uint8_t expect) {
    for (size_t i = 0; i < n; i++) if (p[i] != expect) return i;
    return (size_t)-1;
}

/* -------- Allocation paths -------- */

#if MEMGUARD_USE_MPROTECT
static inline size_t mg_page_round_up(size_t n, size_t pagesz) {
    return (n + pagesz - 1) & ~(pagesz - 1);
}

static void* mg_alloc_block_mmap(size_t req, const char *file, int line) {
    const size_t pagesz = (size_t)sysconf(_SC_PAGESIZE);

    const size_t front = MG_ALIGN_UP(mg_g.front_bytes, MEMGUARD_ALIGN);
    const size_t back  = MG_ALIGN_UP(mg_g.back_bytes,  MEMGUARD_ALIGN);
    const size_t header_sz = sizeof(mg_header);
    const size_t tail_sz   = sizeof(uint32_t);

    /* inner mapping includes: front canary + header + user + back canary + tail */
    const size_t inner_span = front + header_sz + req + back + tail_sz;
    const size_t inner_len  = mg_page_round_up(inner_span, pagesz);

    const size_t guard_front = (size_t)MEMGUARD_GUARD_FRONT_PAGES * pagesz;
    const size_t guard_back  = (size_t)MEMGUARD_GUARD_BACK_PAGES  * pagesz;
    const size_t total_len   = guard_front + inner_len + guard_back;

    uint8_t *base = mmap(NULL, total_len, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) { mg_g.stats.failed_allocs++; return NULL; }

    /* inner RW region is right after the front guard page(s) */
    uint8_t *inner = base + guard_front;
    if (guard_front) mprotect(base, guard_front, PROT_NONE);
    if (guard_back)  mprotect(inner + inner_len, guard_back, PROT_NONE);

    /* lay out objects inside inner */
    uint8_t *front_p = inner;
    mg_header *h     = (mg_header*)(front_p + front);
    uint8_t *user_p  = (uint8_t*)h + sizeof(mg_header);
    uint8_t *back_p  = user_p + req;
    uint32_t *tail_p = (uint32_t*)(back_p + back);

    /* init header and guards */
    memset(front_p, MEMGUARD_PATTERN_FRONT, front);
    memset(back_p,  MEMGUARD_PATTERN_BACK,  back);

    h->magic          = MEMGUARD_HEADER_MAGIC;
    h->tail_magic_copy= MEMGUARD_TAIL_MAGIC;
    h->requested_size = req;
    h->alloc_size     = req;
    h->front_guard    = front;
    h->back_guard     = back;
    h->file           = file;
    h->line           = line;
    h->prev = h->next = NULL;
    h->is_active      = 1;
#if MEMGUARD_USE_MPROTECT
    h->was_mmap       = 1;
    h->map_len_total  = total_len;
    h->page_size      = pagesz;
#endif
#if MEMGUARD_BT_FRAMES > 0
    h->bt_count = backtrace(h->bt, MEMGUARD_BT_FRAMES);
#endif
#if MEMGUARD_ENABLE_CRC
    h->header_crc = mg_header_crc_stable(h);
#endif
    *tail_p = MEMGUARD_TAIL_MAGIC;

#if defined(MEMGUARD_ALLOC_SCRIBBLE)
    memset(user_p, MEMGUARD_ALLOC_SCRIBBLE, req);
#endif

    mg_list_insert(h);
    mg_update_stats_on_alloc(req, front + back + header_sz + tail_sz
                                 + guard_front + guard_back);
    return user_p;
}
#endif /* MEMGUARD_USE_MPROTECT */

static void* mg_alloc_block_malloc(size_t req, const char *file, int line) {
    const size_t front = MG_ALIGN_UP(mg_g.front_bytes, MEMGUARD_ALIGN);
    const size_t back  = MG_ALIGN_UP(mg_g.back_bytes, MEMGUARD_ALIGN);
    const size_t header_sz = sizeof(mg_header);
    const size_t tail_sz   = sizeof(uint32_t);

    const size_t total = front + header_sz + req + back + tail_sz;

    uint8_t *raw = (uint8_t*)malloc(total);
    if (!raw) { mg_g.stats.failed_allocs++; return NULL; }

    uint8_t *front_p = raw;
    mg_header *h     = (mg_header*)(front_p + front);
    uint8_t *user_p  = (uint8_t*)h + header_sz;
    uint8_t *back_p  = user_p + req;
    uint32_t *tail_p = (uint32_t*)(back_p + back);

    memset(front_p, MEMGUARD_PATTERN_FRONT, front);
    memset(back_p,  MEMGUARD_PATTERN_BACK,  back);

    h->magic          = MEMGUARD_HEADER_MAGIC;
    h->tail_magic_copy= MEMGUARD_TAIL_MAGIC;
    h->requested_size = req;
    h->alloc_size     = req;
    h->front_guard    = front;
    h->back_guard     = back;
    h->file           = file;
    h->line           = line;
    h->prev = h->next = NULL;
    h->is_active      = 1;
#if MEMGUARD_USE_MPROTECT
    h->was_mmap       = 0;
    h->map_len_total  = 0;
    h->page_size      = 0;
#endif
#if MEMGUARD_BT_FRAMES > 0
    h->bt_count = backtrace(h->bt, MEMGUARD_BT_FRAMES);
#endif
#if MEMGUARD_ENABLE_CRC
    h->header_crc = mg_header_crc_stable(h);
#endif
    *tail_p = MEMGUARD_TAIL_MAGIC;

#if defined(MEMGUARD_ALLOC_SCRIBBLE)
    memset(user_p, MEMGUARD_ALLOC_SCRIBBLE, req);
#endif

    mg_list_insert(h);
    mg_update_stats_on_alloc(req, front + back + header_sz + tail_sz);
    return user_p;
}

static void* mg_alloc_block(size_t req, const char *file, int line) {
#if MEMGUARD_USE_MPROTECT
    if (req >= MEMGUARD_GUARDPAGE_THRESHOLD) return mg_alloc_block_mmap(req, file, line);
#endif
    return mg_alloc_block_malloc(req, file, line);
}

/* -------- Public allocation API -------- */

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

static int mg_check_ranges(mg_header *h, FILE *report, const char *tag) {
    int bad = 0;
    if (!h) return 1;

    if (h->magic != MEMGUARD_HEADER_MAGIC || h->tail_magic_copy != MEMGUARD_TAIL_MAGIC) {
        if (report) fprintf(report, "[memguard] header magic corrupt tag=%s at %p from %s:%d\n",
                            tag, (void*)h, h->file, h->line);
        return 1;
    }

#if MEMGUARD_ENABLE_CRC
    uint32_t crc = mg_header_crc_stable(h);
    if (crc != h->header_crc) {
        if (report) fprintf(report, "[memguard] header crc mismatch tag=%s at %p from %s:%d\n",
                            tag, (void*)h, h->file, h->line);
        bad = 1;
    }
#endif

    uint8_t *front_p = mg_front_ptr(h);
    uint8_t *back_p  = mg_back_ptr(h);
    uint32_t *tail_p = (uint32_t*)(back_p + h->back_guard);

    size_t off;
    if ((off = mg_first_bad_offset(front_p, h->front_guard, MEMGUARD_PATTERN_FRONT)) != (size_t)-1) {
        if (report) fprintf(report, "[memguard] front guard overwrite +%zu bytes, from %s:%d\n",
                            off, h->file, h->line);
        bad = 1;
    }
    if (*tail_p != MEMGUARD_TAIL_MAGIC) {
        if (report) fprintf(report, "[memguard] tail magic mismatch, from %s:%d\n", h->file, h->line);
        bad = 1;
    }
    if ((off = mg_first_bad_offset(back_p, h->back_guard, MEMGUARD_PATTERN_BACK)) != (size_t)-1) {
        if (report) fprintf(report, "[memguard] back guard overwrite +%zu bytes, from %s:%d\n",
                            off, h->file, h->line);
        bad = 1;
    }
    return bad;
}

int memguard_check_block(const void *user_ptr) {
    if (!user_ptr || !mg_g.enabled) return 0;
    pthread_mutex_lock(&mg_g.lock);
    mg_header *h = mg_from_user(user_ptr);
    int rc = mg_check_ranges(h, stderr, "single");
    pthread_mutex_unlock(&mg_g.lock);
    return rc;
}

size_t memguard_check_all(FILE *report) {
    if (!mg_g.enabled) return 0;
    if (!report) report = stderr;
    size_t bad = 0;
    pthread_mutex_lock(&mg_g.lock);
    for (mg_header *it = mg_g.head; it; it = it->next) bad += mg_check_ranges(it, report, "scan");
    pthread_mutex_unlock(&mg_g.lock);
    return bad;
}

void* memguard_realloc(void *ptr, size_t n, const char *file, int line) {
    if (!mg_g.enabled) return realloc(ptr, n);
    if (!ptr) return memguard_malloc(n, file, line);

    pthread_mutex_lock(&mg_g.lock);
    mg_header *old_h = mg_from_user(ptr);
    (void)mg_check_ranges(old_h, stderr, "realloc_pre");

    void *new_user = mg_alloc_block(n, file, line);
    if (!new_user) { pthread_mutex_unlock(&mg_g.lock); return NULL; }

    size_t copy_n = old_h->requested_size < n ? old_h->requested_size : n;
    memcpy(new_user, ptr, copy_n);

    /* retire old block */
    uint8_t *front_p = mg_front_ptr(old_h);
    uint8_t *back_p  = mg_back_ptr(old_h);
    uint32_t *tail_p = (uint32_t*)(back_p + old_h->back_guard);

#if defined(MEMGUARD_FREE_POISON)
    memset(mg_user_ptr(old_h), MEMGUARD_FREE_POISON, old_h->alloc_size);
#endif
    memset(front_p, 0xDD, old_h->front_guard + old_h->alloc_size + old_h->back_guard);
    *tail_p = 0;
    old_h->magic = 0;
    old_h->is_active = 0;

    const size_t overhead = old_h->front_guard + old_h->back_guard + sizeof(mg_header) + sizeof(uint32_t)
#if MEMGUARD_USE_MPROTECT
                          + ((size_t)MEMGUARD_GUARD_FRONT_PAGES + (size_t)MEMGUARD_GUARD_BACK_PAGES) * (old_h->was_mmap ? old_h->page_size : 0)
#endif
                          ;
    mg_list_remove(old_h);
    mg_update_stats_on_free(old_h->requested_size, overhead);

#if MEMGUARD_USE_MPROTECT
    if (old_h->was_mmap) {
        uint8_t *base = (uint8_t*)old_h - ((size_t)MEMGUARD_GUARD_FRONT_PAGES * old_h->page_size);
        munmap(base, old_h->map_len_total);
    } else
#endif
    {
        free(front_p);
    }

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
    (void)file; (void)line;
    if (!ptr) return;
    if (!mg_g.enabled) { free(ptr); return; }

    pthread_mutex_lock(&mg_g.lock);
    mg_header *h = mg_from_user(ptr);

    if (!h || !h->is_active || !mg_is_in_active_list(h)) {
        fprintf(stderr, "[memguard] invalid or double free at %p\n", ptr);
        pthread_mutex_unlock(&mg_g.lock);
        return;
    }

    (void)mg_check_ranges(h, stderr, "free");

    uint8_t *front_p = mg_front_ptr(h);
    uint8_t *back_p  = mg_back_ptr(h);
    uint32_t *tail_p = (uint32_t*)(back_p + h->back_guard);

#if defined(MEMGUARD_FREE_POISON)
    memset(mg_user_ptr(h), MEMGUARD_FREE_POISON, h->alloc_size);
#endif
    memset(front_p, 0xDD, h->front_guard + h->alloc_size + h->back_guard);
    *tail_p = 0;
    h->tail_magic_copy = 0;
    h->magic = 0;
    h->is_active = 0;

    const size_t overhead = h->front_guard + h->back_guard + sizeof(mg_header) + sizeof(uint32_t)
#if MEMGUARD_USE_MPROTECT
                          + ((size_t)MEMGUARD_GUARD_FRONT_PAGES + (size_t)MEMGUARD_GUARD_BACK_PAGES) * (h->was_mmap ? h->page_size : 0)
#endif
                          ;

    mg_list_remove(h);
    mg_update_stats_on_free(h->requested_size, overhead);

#if MEMGUARD_QUARANTINE_MAX > 0
    /* For mmap blocks, flip inner RW area to PROT_NONE so UAF faults immediately */
#   if MEMGUARD_USE_MPROTECT
    if (h->was_mmap) {
        size_t inner_len = mg_page_round_up(h->front_guard + sizeof(mg_header) + h->alloc_size + h->back_guard + sizeof(uint32_t), h->page_size);
        mprotect((void*)((uint8_t*)h - h->front_guard), inner_len, PROT_NONE);
    }
#   endif
    mg_quarantine_push(h);
#else
#   if MEMGUARD_USE_MPROTECT
    if (h->was_mmap) {
        uint8_t *base = (uint8_t*)h - ((size_t)MEMGUARD_GUARD_FRONT_PAGES * h->page_size);
        munmap(base, h->map_len_total);
    } else
#   endif
    {
        free(front_p);
    }
#endif

    pthread_mutex_unlock(&mg_g.lock);
}

/* -------- Info and diagnostics -------- */

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
        fprintf(out, "[memguard] leak: %zu bytes from %s:%d user=%p\n",
                it->requested_size, it->file, it->line, mg_user_ptr(it));
#if MEMGUARD_BT_FRAMES > 0
        char **syms = backtrace_symbols(it->bt, it->bt_count);
        if (syms) {
            for (int i = 0; i < it->bt_count; i++) fprintf(out, "    %s\n", syms[i]);
            free(syms);
        }
#endif
        count++;
    }
    pthread_mutex_unlock(&mg_g.lock);
    return count;
}

size_t memguard_check_all(FILE *report);

void memguard_teardown(void) {
    pthread_mutex_lock(&mg_g.lock);
    mg_header *it = mg_g.head;
    while (it) {
        mg_header *next = it->next;
#if MEMGUARD_USE_MPROTECT
        if (it->was_mmap) {
            uint8_t *base = (uint8_t*)it - ((size_t)MEMGUARD_GUARD_FRONT_PAGES * it->page_size);
            munmap(base, it->map_len_total);
        } else
#endif
        {
            free(mg_front_ptr(it));
        }
        it = next;
    }
    mg_g.head = NULL;
#if MEMGUARD_QUARANTINE_MAX > 0
    for (size_t i = 0; i < mg_g.q_count; i++) {
        mg_header *h = mg_g.quarantine[i];
        if (!h) continue;
#if MEMGUARD_USE_MPROTECT
        if (h->was_mmap) {
            uint8_t *base = (uint8_t*)h - ((size_t)MEMGUARD_GUARD_FRONT_PAGES * h->page_size);
            munmap(base, h->map_len_total);
        } else
#endif
        {
            free(mg_front_ptr(h));
        }
    }
    mg_g.q_count = 0;
#endif
    pthread_mutex_unlock(&mg_g.lock);
}

/* Optional region protections for user memory, only valid for mmap path */
int memguard_protect_readonly(void *user_ptr) {
#if MEMGUARD_USE_MPROTECT
    mg_header *h = mg_from_user(user_ptr);
    if (!h || !h->was_mmap) return -1;
    size_t pagesz = h->page_size;
    uintptr_t begin = (uintptr_t)mg_user_ptr(h);
    uintptr_t end   = begin + h->alloc_size;
    uintptr_t pg_lo = begin & ~(pagesz - 1);
    size_t    span  = ((end - pg_lo) + pagesz - 1) & ~(pagesz - 1);
    return mprotect((void*)pg_lo, span, PROT_READ);
#else
    (void)user_ptr; return -1;
#endif
}
int memguard_protect_readwrite(void *user_ptr) {
#if MEMGUARD_USE_MPROTECT
    mg_header *h = mg_from_user(user_ptr);
    if (!h || !h->was_mmap) return -1;
    size_t pagesz = h->page_size;
    uintptr_t begin = (uintptr_t)mg_user_ptr(h);
    uintptr_t end   = begin + h->alloc_size;
    uintptr_t pg_lo = begin & ~(pagesz - 1);
    size_t    span  = ((end - pg_lo) + pagesz - 1) & ~(pagesz - 1);
    return mprotect((void*)pg_lo, span, PROT_READ | PROT_WRITE);
#else
    (void)user_ptr; return -1;
#endif
}

#endif /* MEMGUARD_IMPLEMENTATION */

#endif /* MEMGUARD_H */

