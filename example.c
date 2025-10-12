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

#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <rpihub75/hub75gpu.h>

// Global scene pointer for signal handler access
extern scene_info *g_scene;

static volatile sig_atomic_t g_stop = 0;
static int g_sigpipe[2] = {-1, -1};

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

// --------------- Graceful Shutdown Handling ---------------------

static void hub75_signal_handler(int sig) {
    printf("\nSignal %d received, shutting down [%x]...\n", sig, g_scene); 

    g_scene->do_render = false;
    if (g_sigpipe[1] != -1) {
        uint8_t b = (uint8_t)sig;
        (void)!write(g_sigpipe[1], &b, 1); // async-signal-safe
    }


    /*
    (void)sig; // unused parameter
    if (g_scene) {
        fprintf(stderr, "\nSignal received, requesting shutdown...\n");
        hub75_request_shutdown(g_scene);
    }
    */
}

static void install_signal_handlers(void) {

    if (pipe(g_sigpipe) == -1) {
        // best-effort; still usable without the pipe
        g_sigpipe[0] = g_sigpipe[1] = -1;
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = hub75_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
}


static void *render_cpu(void *arg) {
    scene_info *scene = (scene_info*)arg;
    printf("[CPU] CPU Single moving triangle (Ctrl+C to exit)\n");

    const uint16_t W = scene->width;
    const uint16_t H = scene->height;
    const float Wf = (float)W-2;
    const float Hf = (float)H-2;
    const size_t image_sz = (size_t)(W * H * scene->stride);
    printf("W: %f, H: %f\n", Wf, Hf);
    // memset(scene->image, 0, image_sz);

    // Initialize triangle vertices at random positions
    float px[3] = { (float)ri(W), (float)ri(W), (float)ri(W) };
    float py[3] = { (float)ri(H), (float)ri(H), (float)ri(H) };
    // Keep previous positions for motion interpolation
    float ppx[3] = { px[0], px[1], px[2] };
    float ppy[3] = { py[0], py[1], py[2] };

    // Random velocities (pixels/frame), small magnitudes
    float vx[3], vy[3];
    for (int i = 0; i < 3; ++i) {
        float ang = (float)rand() / (float)RAND_MAX * 6.28318530718f;
        float speed = -2.5f + 5.0f * ((float)rand() / (float)RAND_MAX); // allow >1 px/frame
        //vx[i] = cosf(ang) * speed;
        //vy[i] = sinf(ang) * speed;
        vx[i] = speed;
        vy[i] = speed;
        printf("  v[%d] = %.2f, %.2f\n", i, vx[i], vy[i]);
    }

    // Triangle base color
    RGB color = { 255, 160, 32 };
    RGB color1 = { 32, 255, 160 };
    RGB color2 = { 160, 232, 255 };

    while (scene->do_render) {
        uint8_t *image = spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
        scene->image = image;
        memset(image, 0, image_sz);
        // Fade previous frame (trail)
        for (size_t i = 0; i < image_sz; ++i) {
            image[i] = (uint8_t)(image[i] - (image[i] ? 1 : 0));
        }

        // Update positions with bounce on edges
        for (int i = 0; i < 3; ++i) {
            px[i] += vx[i];
            py[i] += vy[i];
            if (px[i] < 2.0f)  { px[i] = 2.0f;  vx[i] = -vx[i]; }
            if (px[i] >= Wf)  { px[i] = Wf; vx[i] = -vx[i]; }
            if (py[i] < 2.0f)  { py[i] = 2.0f;  vy[i] = -vy[i]; }
            if (py[i] >= Hf)  { py[i] = Hf; vy[i] = -vy[i]; }
        }

        uint16_t x1 = (uint16_t)(px[0]);
        uint16_t y1 = (uint16_t)(py[0]);
        uint16_t x2 = (uint16_t)(px[1]);
        uint16_t y2 = (uint16_t)(py[1]);
        uint16_t x3 = (uint16_t)(px[2]);
        uint16_t y3 = (uint16_t)(py[2]);

        hub_line_aa(scene, x1, y1, x2, y2, color);
        hub_line_aa(scene, x2, y2, x3, y3, color1);
        hub_line_aa(scene, x3, y3, x1, y1, color2);

        //map_byte_image_to_bcm(scene, NULL);
        spsc_push_ptr_commit(scene->ring_buf_mapper);

        // Small sleep to tame CPU
        usleep(2000);

        // FPS calculation
        calculate_fps(scene->fps, scene->show_fps);
    }

    //printf(" * CPU render thread calling shutdown ...\n");
    //render_loop_shutdown(scene);
    printf(" ## CPU render thread exiting...\n");
    return NULL;
}
// ...existing code...



static void *render_cpu3(void *arg) {
    scene_info *scene = (scene_info*)arg;
    printf("[CPU] cpu3 Single moving triangle (Ctrl+C to exit)\n");

    const uint16_t W = scene->width;
    const uint16_t H = scene->height;
    const size_t image_sz = (size_t)(W * H * scene->stride);

    // Initialize triangle vertices at random positions
    float px[3] = { (float)ri(W), (float)ri(W), (float)ri(W) };
    float py[3] = { (float)ri(H), (float)ri(H), (float)ri(H) };

    // Random velocities (pixels/frame), small magnitudes
    float vx[3], vy[3];
    for (int i = 0; i < 3; ++i) {
        // random direction unit vector
        float ang = (float)rand() / (float)RAND_MAX * 6.28318530718f;
        // random speed in [0.5, 2.0]
        float speed = 0.5f + 1.5f * ((float)rand() / (float)RAND_MAX);
        vx[i] = cosf(ang) * speed;
        vy[i] = sinf(ang) * speed;
    }

    // Triangle color (constant); tweak if you want cycling
    RGB color1 = { 255, 160, 32 };
    RGB color2 = { 32, 160, 255 };
    RGB color3 = { 160, 32, 160 };

    while (scene->do_render) {
        // 1) Clear frame (single triangle, no trails)
        //memset(scene->image, 0, image_sz);

        // 1. Fade previous frame (simple exponential decay)
        //for (size_t i = 0; i < image_sz; ++i) {
        //    scene->image[i] = scene->image[i] - 1;//(uint8_t)((float)scene->image[i] * 0.94f);
        //}
        uint8_t *image = spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
        scene->image = image;
        memset(image, 0, image_sz);


        // 2) Update positions with bounce on edges
        for (int i = 0; i < 3; ++i) {
            float x = px[i] + vx[i];
            float y = py[i] + vy[i];

            // Bounce on X edges
            if (x < 0.0f) {
                x = 0.0f;
                vx[i] = -vx[i];
            } else if (x > (float)(W - 1)) {
                x = (float)(W - 1);
                vx[i] = -vx[i];
            }

            // Bounce on Y edges
            if (y < 0.0f) {
                y = 0.0f;
                vy[i] = -vy[i];
            } else if (y > (float)(H - 1)) {
                y = (float)(H - 1);
                vy[i] = -vy[i];
            }

            px[i] = x;
            py[i] = y;
        }

        // 3) Draw the triangle (anti-aliased)
        const uint16_t x1 = (uint16_t)(px[0] + 0.5f);
        const uint16_t y1 = (uint16_t)(py[0] + 0.5f);
        const uint16_t x2 = (uint16_t)(px[1] + 0.5f);
        const uint16_t y2 = (uint16_t)(py[1] + 0.5f);
        const uint16_t x3 = (uint16_t)(px[2] + 0.5f);
        const uint16_t y3 = (uint16_t)(py[2] + 0.5f);
        //hub_triangle_aa(scene, x1, y1, x2, y2, x3, y3, color);


        hub_line(scene, x1, y1, x2, y2, color1);
        /*
        hub_line(scene, x2, y2, x3, y3, color2);
        hub_line(scene, x3, y3, x1, y1, color3);
        */

        // 4) Map the CPU image to BCM output buffers
        map_byte_image_to_bcm(scene, NULL);

        spsc_push_ptr_commit(scene->ring_buf_mapper);
        // 5) Small sleep to tame CPU; adjust as desired
        //usleep(5000);

        // 6) FPS calculation (optional verbose controlled by scene->show_fps)
        calculate_fps(scene->fps, scene->show_fps);
    }

    printf("CPU render thread exiting...\n");
    render_loop_shutdown(scene);
    return NULL;
}
// ...existing code...



// --------------- Example CPU Renderer Thread --------------------
/**
 * A very small demo CPU renderer that:
 *   * Fades the previous frame slightly
 *   * Draws a random anti‑aliased triangle each frame
 *   * Maps the image to BCM output
 *   * Sleeps to maintain target FPS
 *
 * You can replace this entire function with your own drawing logic. The
 * framebuffer for CPU rendering is available at scene->image (size
 * width*height*stride bytes). Stride is 3 (RGB) or 4 (RGBA).
 */
static void *render_cpu2(void *arg) {
    scene_info *scene = (scene_info*)arg;
    printf("[CPU] cpu2 Rendering random triangles (Ctrl+C to exit)\n");

    const size_t image_sz = (size_t)(scene->width * scene->height * scene->stride);

    //const hub75_api *api = hub75_get_api(scene);

    RGB color1 = { rnd8(), rnd8(), rnd8() };
    RGB color2 = { rnd8(), rnd8(), rnd8() };
    RGB color3 = { rnd8(), rnd8(), rnd8() };
    while (scene->do_render) {
        // 1. Fade previous frame (simple exponential decay)
        for (size_t i = 0; i < image_sz; ++i) {
            scene->image[i] = (uint8_t)((float)scene->image[i] * 0.94f);
        }

        // 2. Draw a random anti‑aliased triangle
        const uint16_t x0 = rnd16(scene->width);
        const uint16_t x1 = rnd16(scene->width);
        const uint16_t x2 = rnd16(scene->width);
        const uint16_t y0 = rnd16(scene->height);
        const uint16_t y1 = rnd16(scene->height);
        const uint16_t y2 = rnd16(scene->height);
        //hub_triangle_aa(scene, x1, y1, x2, y2, x3, y3, color);

        hub_line_aa(scene, x0, y0, x1, y1, color1);
        hub_line_aa(scene, x1, y1, x2, y2, color2);
        hub_line_aa(scene, x2, y2, x0, y0, color3);

        // 3. Map the CPU image to BCM output buffers
        map_byte_image_to_bcm(scene, NULL);
        usleep(5000);

        // 4. Regulate FPS & optionally print frame rate (-v to enable dispaly in parse_scene)
        calculate_fps(scene->fps, scene->show_fps);
    }

    printf("CPU render thread exiting...\n");

    // Free all allocated memory
    // TODO: create renderer shutdown function
    render_loop_shutdown(scene);

    return NULL;
}

// --------------- Main Entry Point --------------------------------
int main(int argc, char **argv) {
    printf("rpi-gpu-hub75 example PI Hardware \"hat\": (%s)\n", ADDRESS_TYPE);
    srand((unsigned)time(NULL));

    // Parse command line into a new scene. Use -h to see available options.
    scene_info *scene = parse_scene(argc, argv);
    g_scene = scene;

    // Validate configuration, allocate internal buffers, etc.
    start_scene(scene);

    // Install Ctrl+C handler after scene is ready
    install_signal_handlers();

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
            scene->stride = 4; // uncomment if needed for specific shaders
            pthread_create(&scene->render_thread, NULL, render_shader, scene);
        } else {
            printf("[GPU] Video: %s\n", scene->shader_file);
            pthread_create(&scene->render_thread, NULL, render_video_fn, scene);
        }
    } else {
        fprintf(stderr, "[WARN] Unable to open '%s'; falling back to CPU renderer.\n", scene->shader_file);
        pthread_create(&scene->render_thread, NULL, render_cpu, scene);
    }

    printf("\ng_scene: [%x]...\n", g_scene); 
    // This call never returns; it drives the BCM output loop.
    render_forever(scene);
    /*
    while(scene->do_render) {
        usleep(100000);
    }
    */
    //printf("\nrender forever quit, g_scene: [%x]...\n", g_scene); 
    //pthread_join(scene->render_thread, NULL);
    //pthread_join(scene->mapper_thread, NULL);
    //printf("\nall threads complete, g_scene: [%x]...\n", g_scene);
    // Wait for signal (non-busy)
    /*
    if (g_sigpipe[0] != -1) {
        uint8_t b;
        (void)read(g_sigpipe[0], &b, 1); // blocks until signal arrives
    } else {
        while (!g_stop) { usleep(10000); }
    }
    */

    // wait on threads...
    hub75_wait_shutdown(scene);


    return 0; // not reached
}
