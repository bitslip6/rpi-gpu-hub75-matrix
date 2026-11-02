#define _GNU_SOURCE
#include "gpio_affinity.h"
#include "hub75gpu.h"
#include <stdio.h>

/* Unchecked dispatch is private and reached only after mandatory affinity.
 * Keeping the guard outside dispatch also covers direct Pi4 and weighted entry. */
__attribute__((visibility("hidden"))) void *h75_legacy_run(const hub75_display_t *scene);
__attribute__((visibility("hidden"))) void *h75_legacy_run_pi4(const hub75_display_t *scene);
static const char scan_failed;

static void *checked_run(const hub75_display_t *scene, bool pi4) {
    cpu_set_t original;
    char message[512];
    void *result = NULL;
    if (!scene)
        return (void *)&scan_failed;
    if (!atomic_load(&scene->do_render))
        return NULL;
    if (h75_gpio_affinity_save(&original, message, sizeof(message))) {
        result = (void *)&scan_failed;
        goto failed;
    }
    if (h75_gpio_affinity_require(message, sizeof(message))) {
        /* A successful set followed by failed verification may have changed
         * affinity. Always restore the mask saved BEFORE attempting to pin. */
        puts(message);
        result = (void *)&scan_failed;
        atomic_store(&((hub75_display_t *)scene)->do_render, false);
    } else if (atomic_load(&scene->do_render)) {
        result = pi4 ? h75_legacy_run_pi4(scene) : h75_legacy_run(scene);
    }
    /* Scanner has blanked, released held frames, mapping and ownership first. */
    if (h75_gpio_affinity_restore(&original, message, sizeof(message))) {
        result = (void *)&scan_failed;
        goto failed;
    }
    if (result)
        atomic_store(&((hub75_display_t *)scene)->do_render, false);
    fflush(stdout);
    return result;
failed:
    puts(message); /* Deliberately stdout: visible to example/embedding callers. */
    fflush(stdout);
    atomic_store(&((hub75_display_t *)scene)->do_render, false);
    return result;
}

void *hub75_display_run(const hub75_display_t *scene) {
    return checked_run(scene, false);
}
void *hub75_display_run_pi4(const hub75_display_t *scene) {
    return checked_run(scene, true);
}
