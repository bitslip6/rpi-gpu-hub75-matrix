#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// GLOBAL MEMGUARD TOGGLE IS THIS DEFINE, ony define for debug builds
#define MEMGUARD_IMPLEMENTATION

#ifdef DEBUG
#define MEMGUARD_USE_MPROTECT         1
#define MEMGUARD_GUARD_FRONT_PAGES    1
#define MEMGUARD_GUARD_BACK_PAGES     1
#define MEMGUARD_GUARDPAGE_THRESHOLD  0     // guard all allocations 

#define MEMGUARD_BT_FRAMES            8
#define MEMGUARD_ENABLE_CRC           1
#define MEMGUARD_QUARANTINE_MAX       16
#endif


#include "memguard2.h"

#ifdef MEMGUARD_IMPLEMENTATION
#ifdef MEMGUARD_OVERRIDE_STDLIB
#error "Do not define MEMGUARD_OVERRIDE_STDLIB in the TU that defines MEMGUARD_IMPLEMENTATION"
#endif

static void* mg_watchdog_thread(void *arg) {
    (void)arg;
    printf("~~ memguard watchdog thread started\n");
    for (;;) {
        sleep(2); /* configurable */
        memguard_check_all(stderr);
    }
    return NULL;
}

static inline void mg_start_watchdog(void) {
    pthread_t t; pthread_create(&t, NULL, mg_watchdog_thread, NULL);
    pthread_detach(t);
}
#endif

void hub75gpu_init() {
    printf("~~ library bringup\n");
    #ifdef DEBUG
    
    memguard_init(64, 64);
    void *p = mg_malloc(32);
    mg_free(p);

    mg_start_watchdog();
    printf("~~ memguard running\n");
    #endif
}



void mem_info() {
    #ifdef MEMGUARD_IMPLEMENTATION
        memguard_check_all(stderr);
    #endif
}

