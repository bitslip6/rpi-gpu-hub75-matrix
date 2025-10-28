/**
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

#include <rpihub75/hub75gpu.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "stb_image_write.h"
#pragma GCC diagnostic pop

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



static void *render_3d(void *arg) {

    scene_info *scene = (scene_info*)arg;
    hub75gpu_t api = hub75gpu(scene);

    // camera setup 
    camera_t *cam = api.geo_camera();
    cam->aspect = (float)scene->width / (float)scene->height;  /* Use actual image aspect ratio */
    cam->fov_y = (float)M_PI / 4.0f;  /* 45 degrees - balanced FOV */
    cam->position.z = 5.0f;   /* Move camera back to see the cube */
    cam->target = (vec3){0, 0, 0};  /* Look at origin */

    // object transform 
    transform_t *cube_xform = api.geo_transform();
    cube_xform->scale.x = 1.5f;  /* Scale down the cube even more to fit in view frustum */
    cube_xform->scale.y = 1.25;
    cube_xform->scale.z = 1.25f;

    //cube_xform->rotation.x = 2.5f;  /* Initial rotation for better view angle */

    // create a cube object
    object_t *cube = api.geo_cube();
    if (!cube) {
        printf("Failed to create cube!\n");
        return NULL;
    }

    cube->cull_backface = true;
    scene->stride = 3;

    cam->position.y = -2.0f;
    /* animate few steps, replace with your app loop */
    for (int frame = 0; frame < 1; ++frame) {
        float t = (float)frame * 0.016f;

        // rotate cube and move it a bit 
        cube_xform->rotation.x = t * 0.7f;
        cube_xform->rotation.y = 1.14f;//t * 1.1f;
        //cube_xform->position.x = 0.75f * sinf(t * 0.5f);
        //cube_xform->position.y = 0.50f * cosf(t * 0.4f);

        // orbit camera around origin while looking at cube - closer distance with wider FOV
        cam->position.x = 5.0f * cosf(t * 0.3f);  /* Reduced distance with wider FOV for better fit */
        cam->position.z = 5.0f * sinf(t * 0.3f);  /* Reduced distance with wider FOV for better fit */
        cam->target     = cube_xform->position;

        api.geo_render_wire(cam, cube, cube_xform);
    }

    stbi_write_png("out2.png", scene->width, scene->height, 3, scene->image, scene->width * scene->stride);

    return NULL;
}


static void *render_cpu(void *arg) {
    scene_info *scene = (scene_info*)arg;
    printf("[CPU] CPU Single moving triangle (Ctrl+C to exit)\n");

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
    RGB color1 = { 32, 255, 160 };
    // RGB color2 = { 160, 232, 32 };

    hub75gpu_t api = hub75gpu(scene);

    uint32_t frame = 0;
    while (scene->do_render) {
        frame++;
        api.begin_frame(); 
        api.clear();

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

        /*
        color2.r = (uint8_t)(128 + 127 * sinf((float)frame * 0.04f + 2.0f));
        color2.g = (uint8_t)(128 + 127 * sinf((float)frame * 0.02f + 2.0f));
        color2.b = (uint8_t)(128 + 127 * sinf((float)frame * 0.03f + 2.0f));
        */

        api.poly(&tri, color1);
        api.end_frame();

        // FPS calculation
        calculate_fps(scene->fps, scene->show_fps);
    }

    //printf(" * CPU render thread calling shutdown ...\n");
    hub75_request_shutdown(scene);
    printf(" ## CPU render thread exiting...\n");
    return NULL;
}
// ...existing code...


// --------------- Main Entry Point --------------------------------
int main(int argc, char **argv) {
    printf("rpi-gpu-hub75 example PI Hardware \"hat\": (%s)\n", ADDRESS_TYPE);
    srand((unsigned)time(NULL));

    // Parse command line into a new scene. Use -h to see available options.
    scene_info *scene = parse_scene(argc, argv);

    scene->stride = 4;

    // Validate configuration, allocate internal buffers, etc.
    start_scene(scene);

    render_3d(scene);
    exit(1);

    signal_handler_install();

    // create RGB -> BCM mapper thread
    if (pthread_create(&scene->mapper_thread, NULL, mapper_thread_main, scene) != 0) {
        scene->do_render = false;
    }


    pthread_create(&scene->render_thread, NULL, render_shader, scene);

    printf("waiting forever...\n");
    render_forever(scene);

    hub75_wait_shutdown(scene);
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
    hub75_wait_shutdown(scene);
    return 0; // not reached
}
