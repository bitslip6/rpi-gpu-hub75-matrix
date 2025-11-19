/*
 * rpi-gpu-hub75 Example Application
 * ---------------------------------
 * This program demonstrates the minimal steps needed to:
 *   1. Parse command line options into a `scene_info` structure (parse_scene)
 *   2. Start the rendering system (start_scene)
 *   3. Provide a custom CPU drawing loop OR render a GPU shader / video
 *   4. Gracefully shut down on Ctrl+C (SIGINT) / SIGTERM
 *
 * Build (auto‑detects libs via pkg-config, prefer using the provided Makefile):
 *   make example            # release build (default)
 *   make BUILD=debug example
 *   ./example -h            # show command line options supplied by parse_scene()
 *
 * Quick Run (4x 64x64 -> 128x128, 2 ports, 2 chains each, 48 bpp, gamma 2.2):
 *   ./example -w 128 -h 128 -p 2 -c 2 -d 48 -b 192 -g 2.2 -f 120 -s shaders/cartoon.glsl
 *
 * To try the CPU demo (random triangles) just omit -s:
 *   ./example -w 128 -h 128 -p 2 -c 2 -d 48 -b 192 -g 2.2 -f 120
 *
 * NOTE: parse_scene() supplies many more options (see -h output).
 */
#include <stddef.h>
#include <bits/types/sig_atomic_t.h>
#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>

// #define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hub75gpu.h"
#include "util.h"
#include "pixels.h"
#include "text_sdf.h"


// --------------- Utility Helpers (example only) -----------------

// Return a random integer in [0, max-1]. If max==0 returns 0.
static inline unsigned ri(unsigned max) {
    return (max == 0) ? 0u : (unsigned)rand() % max;
}

// Return a random uint16_t in [0, max-1]
static inline uint16_t rnd16(uint16_t max) {
    return (uint16_t)ri(max);
}

// Return a random 8 bit value in [0,255]
static inline uint8_t rnd8(void) {
    return (uint8_t)ri(256u);
}

// Trivial helper to test if a filename ends with a given extension (case sensitive)
static bool has_extension(const char *filename, const char *extension) {
    const char *dot = strrchr(filename, '.');
    return (dot && dot != filename) ? (strcmp(dot + 1, extension) == 0) : false;
}



/* ---------------- Text SDF Demo (Task 5.1) ---------------- */
static void *render_text_sdf(void *arg) {
    hub75_display_t *scene = (hub75_display_t*)arg;
    printf("[TEXT] SDF text scroller demo (Ctrl+C to exit)\n");

    // Configure font: load and scale to 64px line height 
    sdf_font_t *font = sdf_font_load_scaled("assets/robots", 64.0f);
    if (!font) {
        fprintf(stderr, "[TEXT] Failed to load SDF font from 'assets/robots' (expect metrics.csv + PNGs).\n");
        scene->do_render = false;
        return NULL;
    }

    // Create text object
    const char msg[] = "Hello, HUB75 SDF Scroller!  ";
    sdf_text_t *txt = sdf_text_create(font, msg, scene->width, scene->height);
    txt->size_px = 128.0f;
    txt->color = (RGBA){0, 255, 128, 255};
    txt->alpha = 200;
    txt->y = 90.0f;
    txt->x = 100.0f;
    txt->dir_x = -1.0f;
    txt->dir_y = 0.0f;
    txt->speed = 255.0f;
    txt->softness = 0.08f;
    txt->effects.weight         = 1.2f;
    txt->effects.glow_color     = (RGBA){192, 232, 245, 255};
    txt->effects.glow_radius    = Normal_clamp(1.0f);
    txt->effects.outline_width  = Normal_clamp(0.1f); // set >0 to enable outline
    txt->effects.outline_color  = (RGBA){0, 64, 128, 255};
    txt->effects.outline_smooth = Normal_clamp(0.1f);
    
    sdf_text_update(txt);


    
    /* Render loop */
    hub75gpu_t api = hub75_api(scene);
    //const bool rgba = (scene->stride == 4);
    //const int row_stride = (int)scene->width * (int)scene->stride;
    //float dt = 0.0f;
    const bool dump_once = true;//(dump_env && *dump_env && dump_env[0] != '0');
    string_t *dump_path = string_new("text_debug.png", 128);
    uint32_t frame = 0;

    //float last_time = 0.0f;
    float this_time = 0.0f;

    // Random velocities (pixels/frame), small magnitudes
    float vx[3], vy[3];

    Polygonf_t tri;
    tri.num_points = 3;

    for (int i = 0; i < 3; ++i) {
        float ang = (float)rand() / (float)RAND_MAX * 6.28318530718f;
        float speed = .01f * ((float)rand() / (float)RAND_MAX); // allow >1 px/frame
        vx[i] = cosf(ang) * speed;
        vy[i] = sinf(ang) * speed;
        tri.points[i].x = ((float)rand() / (float)RAND_MAX);
        tri.points[i].y = ((float)rand() / (float)RAND_MAX);

        printf("  v[%d] = (%3.3f / %3.3f) %.2f, %.2f\n", i, (double)tri.points[i].x, (double)tri.points[i].y, (double)vx[i], (double)vy[i]);
    }

    // Triangle base color
    // RGB color = { 255, 160, 32 };
    RGBA color1 = { 32, 255, 160, 255 };
 
    size_t text_mem = (size_t)(ceilf(txt->dimensions.x) * ceilf(txt->dimensions.y));

    printf("Allocating text memory: %zu bytes\n", text_mem);
    uint8_t *tmem = (uint8_t*)calloc(text_mem, sizeof(RGBA));

    this_time = 1.0f;
    printf("text rendered to: [%dx%d]", txt->dimensions.x, txt->dimensions.y);

    sdf_text_render(txt, tmem, (int)scene->width, (int)scene->height, scene->stride, this_time);
    while (scene->do_render) {
        frame++;
        api.frame_begin();
        api.clear(); /* black background */

        for (int i = 0; i < 3; ++i) {
            tri.points[i].x += vx[i];
            tri.points[i].y += vy[i];
            if (tri.points[i].x < 0.01f)  { tri.points[i].x = 0.01f;  vx[i] = -vx[i]; }
            if (tri.points[i].x >= 0.99f)  { tri.points[i].x = 0.99f; vx[i] = -vx[i]; }
            if (tri.points[i].y < 0.01f)  { tri.points[i].y = 0.01f;  vy[i] = -vy[i]; }
            if (tri.points[i].y >= 0.99f)  { tri.points[i].y = 0.99f; vy[i] = -vy[i]; }
        }

        color1.r = (uint8_t)(128 + 127 * sinf((float)frame * 0.02f));
        color1.g = (uint8_t)(128 + 127 * sinf((float)frame * 0.03f));
        color1.b = (uint8_t)(128 + 127 * sinf((float)frame * 0.04f));


        api.poly(&tri, color1);

	    //composite_rgba((RGBA*)scene->image, (RGBA*)tmem, (RGBA*)scene->image);
        //blit_composite_rgba_over_rgba((uint8_t*)scene->image, scene->stride * scene->width,
        //blit_composite_rgba_over_rgba((uint8_t*)scene->image, (uint8_t*)tmem, scene->stride * (int)ceilf(txt->dimensions.x), (RGBA){255,255,255,255});

        vec2u ddim = {scene->width, scene->height};
        vec4u dspec = {0, 10, scene->width, 138};
        vec4u sspec = {0, 0, MIN(txt->dimensions.x, scene->width), txt->dimensions.y};

    //sdf_text_render(txt, scene->image, (int)scene->width, (int)scene->height, scene->stride, this_time);
    // compositer does not seem to handle alpha blending correctly, RGB vs RGBA
        blit_composite_rgba_over_rgba(
            scene->image, ddim,
            dspec,
            (uint8_t*)tmem, txt->dimensions, (vec4u){0, 0, (int)ceilf(txt->dimensions.x), (int)ceilf(txt->dimensions.y)});
 

        if (dump_once && frame == 60) {
            write_png_file(dump_path, scene);
            fprintf(stderr, "[TEXT] Wrote debug frame to %s\n", dump_path->str);
        }

        api.frame_end();
        this_time = calculate_fps(scene->fps, scene->show_fps);
    }

    /* Cleanup */
    if (dump_path) string_free(dump_path);
    sdf_text_destroy(txt);
    sdf_font_free(font);
    return NULL;
}


// --------------- Main Entry Point --------------------------------
int main(int argc, char **argv) {
    printf("rpi-gpu-hub75 example PI Hardware \"hat\": (%s)\n", ADDRESS_TYPE);
    srand((unsigned)time(NULL));

    // Parse command line into a new scene. Use -h to see available options.
    hub75_display_t *scene = hub75_display_parse_args(argc, argv);

    //scene->stride = 4;
    // Validate configuration, allocate internal buffers, etc.
    hub75_display_start(scene);

    signal_handler_install();

    /* To try the text SDF demo, replace render_3d with render_text_sdf */
    // pthread_create(&scene->render_thread, NULL, render_text_sdf, scene);
    pthread_create(&scene->render_thread, NULL, render_text_sdf, scene);

    hub75_display_run(scene);

    hub75_display_wait(scene);
    return 0;



    // Decide what to render:
    //  * No -s : run CPU demo
    //  * -s path/to/file.glsl : GPU shader
    //  * -s path/to/file.(mp4|mov|...) : Video playback
    if (scene->shader_file == NULL) {
        pthread_create(&scene->render_thread, NULL, render_cpu, scene);
    } else if (access(scene->shader_file, R_OK) == 0) {
        if (has_extension(scene->shader_file, "glsl")) {
            printf("[GPU] Shader: %s\n", scene->shader_file);
            // If your shader needs RGBA (alpha), adjust stride if desired:
            pthread_create(&scene->render_thread, NULL, render_shader, scene);
        } else {
            printf("[GPU] Video: %s\n", scene->shader_file);
            pthread_create(&scene->render_thread, NULL, render_video_fn, scene);
        }
    } else {
        fprintf(stderr, "[WARN] Unable to open '%s'; falling back to CPU renderer.\n", scene->shader_file);
        pthread_create(&scene->render_thread, NULL, render_cpu, scene);
    }

    while(scene->do_render) {
        sleep(1);
    }

    // This call never returns; it drives the BCM output loop.
    // render_forever(scene);

    // wait on threads...
    hub75_display_wait(scene);
    return 0; // not reached
}
