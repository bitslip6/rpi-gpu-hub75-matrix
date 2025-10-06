#include <stdio.h>
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <sys/param.h>
#include <gbm.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <string.h>
#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "rpihub75.h"
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Test Shader source code
const char *test_shader_source =
    "#version 310 es\n"
    "precision mediump float;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    color = vec4(1.0, 0.0, 1.0, 1.0);\n"  // Red color
    "}\n";

/**
 * @brief add inputs for shaderToy glsl shaders
 * usage: fragment_shader = sprintf(shadertoy_header, shader_source);
 */
const char *shadertoy_header =
    "#version 310 es\n"
    "precision mediump float;\n"
    "uniform vec3 iResolution;\n"
    "uniform float iGlobalTime;\n"
    "uniform vec4 iMouse;\n"
    "uniform vec4 iDate;\n"
    "uniform int iFrame;\n"
    "uniform float iSampleRate;\n"
    "uniform vec3 iChannelResolution[4];\n"
    "uniform float iChannelTime[4];\n"
    "uniform sampler2D iChannel0;\n"
    "uniform sampler2D iChannel1;\n"
    "uniform float iTime;\n"
    "uniform float iTimeDelta;\n"

    "out vec4 fragColor;\n"
    "%s\n"

    "void main() {\n"
    "    mainImage(fragColor, gl_FragCoord.xy);\n"
    "}\n";


/**
 * @brief trivial vertex shader. pass vertex directly to the GPU
 * 
 */
const char *vertex_shader_source =
    "#version 310 es\n"
    "in vec4 position;\n"
    "void main() {\n"
    "    gl_Position = position;\n"
    "}\n";


// Load texture from a PNG file using stb_image
GLuint load_texture(const char* filePath) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Load the texture data from a PNG file using stb_image
    int width, height, nrChannels;
    unsigned char *data = stbi_load(filePath, &width, &height, &nrChannels, 0);
    if (data) {
        // Determine the format based on the number of channels in the PNG file
        GLenum format;
        if (nrChannels == 1)
            format = GL_RED;
        else if (nrChannels == 3)
            format = GL_RGB;
        else if (nrChannels == 4)
            format = GL_RGBA;
        else {
            printf("Unsupported number of channels in PNG: %d\n", nrChannels);
            stbi_image_free(data);
            return 0;
        }

        // Upload texture to GPU with mipmaps
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);  // Generate mipmaps for texture

        // Set texture parameters for wrapping and filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);  // Wrap horizontally
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);  // Wrap vertically
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);  // Minify filter with mipmaps
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);               // Magnification filter

    } else {
        die("Failed to load texture: %s\n", filePath);
    }
    
    // Free image memory after loading into OpenGL
    stbi_image_free(data);

    return textureID;
}

/**
 * @brief helper method for compiling GLSL shaders
 * 
 * @param source the source code for the shader
 * @param shader_type one of GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
 * @return GLuint reference to the created shader id
 */
static GLuint compile_shader(const char *source, const GLenum shader_type) {
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        die("Shader compilation error: %s\n", info_log);
    }

    return shader;
}

/**
 * @brief Create a complete OpenGL program for a shadertoy shader
 * 
 * @param file name of the shadertoy file to load
 * @return GLuint OpenGL id of the new program
 */
static GLuint create_shadertoy_program(char *file) {
    long filesize;
    char *src = file_get_contents(file, &filesize);
    if (filesize == 0) {
        die( "Failed to read shader source\n");
    }

    char *src_with_header = (char *)malloc(filesize + 8192);
    if (src_with_header == NULL) {
        die("unable to allocate %d bytes memory for shader program\n", filesize + 8192);
    }
    snprintf(src_with_header, filesize + 8192, shadertoy_header, src);

    GLuint vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER);
    GLuint fragment_shader = compile_shader(src_with_header, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program, 512, NULL, info_log);
        die("Program linking error: %s\n", info_log);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    free(src_with_header);
    return program;
}


/**
 * @brief return a new string with the extension changed to new_extension
 * 
 * @param filename 
 * @param new_extension 
 * @return char* 
 */
char *change_file_extension(const char *filename, const char *new_extension) {
    // Find the last dot in the filename
    const char *dot = strrchr(filename, '.');
    size_t new_filename_length;

    // If there is no dot, simply append the new extension
    if (dot == NULL) {
        new_filename_length = strlen(filename) + strlen(new_extension) + 2; // +2 for dot and null terminator
    } else {
        new_filename_length = (dot - filename) + strlen(new_extension) + 2; // +2 for dot and null terminator
    }

    // Allocate memory for the new filename
    char *new_filename = (char *)malloc(new_filename_length);
    if (new_filename == NULL) {
        perror("Unable to allocate memory");
        return NULL;
    }

    // Copy the original filename up to the dot, if it exists
    if (dot == NULL) {
        strcpy(new_filename, filename);
    } else {
        strncpy(new_filename, filename, dot - filename);
        new_filename[dot - filename] = '\0'; // Null-terminate the string
    }

    // Append the new extension
    strcat(new_filename, ".");
    strcat(new_filename, new_extension);

    return new_filename;
}




#ifndef RENDER_USE_PBO
// enable when GLES 3 is available, else falls back to CPU pointer queueing
#define RENDER_USE_PBO 1
#endif

// --------- simple SPSC ring for pointers ---------
typedef struct {
    atomic_uint head;
    atomic_uint tail;
    unsigned size;      // power of two
    void **items;
} spsc_ring_t;

static inline void spsc_init(spsc_ring_t *q, void **storage, unsigned size_pow2) {
    atomic_store_explicit(&q->head, 0u, memory_order_relaxed);
    atomic_store_explicit(&q->tail, 0u, memory_order_relaxed);
    q->size = size_pow2;
    q->items = storage;
}

static inline int spsc_push(spsc_ring_t *q, void *p) {
    unsigned h = atomic_load_explicit(&q->head, memory_order_relaxed);
    unsigned t = atomic_load_explicit(&q->tail, memory_order_acquire);
    if (((h + 1u) & (q->size - 1u)) == (t & (q->size - 1u))) return 0;
    q->items[h & (q->size - 1u)] = p;
    atomic_store_explicit(&q->head, h + 1u, memory_order_release);
    return 1;
}
static inline void* spsc_pop(spsc_ring_t *q) {
    unsigned t = atomic_load_explicit(&q->tail, memory_order_relaxed);
    unsigned h = atomic_load_explicit(&q->head, memory_order_acquire);
    if ((t & (q->size - 1u)) == (h & (q->size - 1u))) return NULL;
    void *p = q->items[t & (q->size - 1u)];
    atomic_store_explicit(&q->tail, t + 1u, memory_order_release);
    return p;
}
static inline int spsc_try_push(spsc_ring_t *q, void *p) {
    if (spsc_push(q, p)) return 1;
    // pop one, drop it, then push
    (void)spsc_pop(q);
    return spsc_push(q, p);
}

// --------- jobs handed from render -> mapper ---------
typedef enum { JOB_CPU_PIXELS = 1, JOB_PBO, JOB_QUIT = 255 } job_kind_t;

typedef struct {
    job_kind_t kind;
    scene_info *scene;
    size_t size_bytes;
    union {
        struct { uint8_t *pixels; } cpu;
#if RENDER_USE_PBO
        struct { GLuint pbo; GLsync fence; } pbo;
#endif
    } u;
} map_job_t;

// mapper thread state
typedef struct {
    spsc_ring_t *q_in;
    spsc_ring_t *q_filled;
    spsc_ring_t *q_free;
    scene_info *scene;
    volatile int run;
} mapper_ctx_t;

static void *mapper_thread_main(void *arg) {
    printf("BCM mapper thread started\n");
    mapper_ctx_t *ctx = (mapper_ctx_t*)arg;
    while(ctx->run) {
        uint8_t *pixels = (uint8_t*)spsc_pop(ctx->q_filled);
        if (!pixels) {
            sched_yield();
            continue;
        }
        ctx->scene->bcm_mapper(ctx->scene, pixels);
        if (ctx->q_free) {
            (void)spsc_try_push(ctx->q_free, pixels);
        }
    }
    return NULL;
}


// ---------- full renderer ----------
void *render_shader(void *arg) {
    scene_info *scene = (scene_info*)arg;
    debug("render shader %s\n", scene->shader_file);

    // DRM / GBM
    int fd = open("/dev/dri/card0", O_RDWR);
    if (fd < 0) die("Failed to open DRM device /dev/dri/card0\n");
    struct gbm_device  *gbm     = gbm_create_device(fd);
    struct gbm_surface *surface = gbm_surface_create(
        gbm, scene->width, scene->height,
        GBM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING);

    // EGL / GLES
    EGLDisplay display = eglGetDisplay(gbm);
    eglInitialize(display, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLConfig config; EGLint num_configs;
    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_ALPHA_SIZE,      8,
        EGL_NONE
    };
    eglChooseConfig(display, attribs, &config, 1, &num_configs);
    static const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs);
    EGLSurface egl_surface = eglCreateWindowSurface(display, config, (EGLNativeWindowType)surface, NULL);
    eglMakeCurrent(display, egl_surface, egl_surface, context);
    eglSwapInterval(display, 0); // uncapped

    // program and quad
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);

    static const GLfloat verts[] = {
        -1.f,  1.f, 0.f,   -1.f, -1.f, 0.f,
         1.f,  1.f, 0.f,    1.f, -1.f, 0.f
    };
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    GLint pos_attrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(pos_attrib);
    glVertexAttribPointer(pos_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);

    // IMPORTANT: ensure tight unpack before any texture uploads
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Optional textures
    GLuint texture0 = 0, texture1 = 0;
    char *chan0 = change_file_extension(scene->shader_file, "channel0");
    if (access(chan0, R_OK) == 0) {
        texture0 = load_texture(chan0);
        if (!texture0) die("unable to load texture '%s'\n", chan0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture0);
        // force no-mipmap sampling and NPOT-safe wrap
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        
    }
    char *chan1 = change_file_extension(scene->shader_file, "channel1");
    if (access(chan1, R_OK) == 0) {
        texture1 = load_texture(chan1);
        if (!texture1) die("unable to load texture '%s'\n", chan1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    // uniforms
    GLint time_loc  = glGetUniformLocation(program, "iTime");
    GLint dtym_loc  = glGetUniformLocation(program, "iTimeDelta");
    GLint frame_loc = glGetUniformLocation(program, "iFrame");
    GLint res_loc   = glGetUniformLocation(program, "iResolution");
    GLint c0_loc    = glGetUniformLocation(program, "iChannel0");
    GLint c1_loc    = glGetUniformLocation(program, "iChannel1");
    glUniform1i(c0_loc, 0);
    glUniform1i(c1_loc, 1);
    glUniform3f(res_loc, scene->width, scene->height, 0);

    // GL state for readbacks
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, scene->width, scene->height);

    // timing
    struct timespec start_time, end_time, orig_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    clock_gettime(CLOCK_MONOTONIC, &orig_time);
    unsigned long frame = 0;

    const size_t image_sz = (size_t)scene->width * scene->height * 4u;

    // queues and mapper
    enum { RING_SIZE = 8 };
    void *ring_filled_storage[RING_SIZE];
    spsc_ring_t ring_filled;
    spsc_init(&ring_filled, ring_filled_storage, RING_SIZE);

#if RENDER_USE_PBO
    // triple PBOs
    enum { PBO_COUNT = 3 };
    typedef struct { GLuint pbo; GLsync fence; } pbo_item_t;
    pbo_item_t pboq[PBO_COUNT];
    GLuint pbos[PBO_COUNT];
    glGenBuffers(PBO_COUNT, pbos);
    for (int i = 0; i < PBO_COUNT; ++i) {
        pboq[i].pbo   = pbos[i];
        pboq[i].fence = 0;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboq[i].pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, image_sz, NULL, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // CPU buffer pool and free ring
    enum { CPU_POOL = 8 };
    uint8_t *cpu_pool[CPU_POOL];
    for (int i = 0; i < CPU_POOL; ++i) {
        // aligned_alloc requires size % alignment == 0, so round up
        size_t sz = (image_sz + 63) & ~((size_t)63);
        cpu_pool[i] = (uint8_t*)aligned_alloc(64, sz);
        if (!cpu_pool[i]) die("failed to alloc CPU staging buffer\n");
    }
    void *ring_free_storage[CPU_POOL];
    spsc_ring_t ring_free;
    spsc_init(&ring_free, ring_free_storage, CPU_POOL);
    for (int i = 0; i < CPU_POOL; ++i) (void)spsc_push(&ring_free, cpu_pool[i]);

    mapper_ctx_t mctx = { .q_filled = &ring_filled, .q_free = &ring_free, .scene = scene, .run = 1 };
#else
    mapper_ctx_t mctx = { .q_filled = &ring_filled, .q_free = NULL, .scene = scene, .run = 1 };
#endif

    pthread_t mapper_th;
    if (pthread_create(&mapper_th, NULL, mapper_thread_main, &mctx) != 0) {
        die("failed to start mapper thread\n");
    }

    int slot = 0;

    // main loop
    while (scene->do_render) {
        frame++;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        float t  = (end_time.tv_sec - orig_time.tv_sec) + (end_time.tv_nsec - orig_time.tv_nsec) / 1e9f;
        float dt = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9f;

        glUseProgram(program);
        glUniform1f(time_loc,  t);
        glUniform1f(dtym_loc,  dt);
        glUniform1f(frame_loc, (float)frame);

        // keep textures bound to expected units, harmless if absent
        if (texture0) { 
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture0);
        }
        if (texture1) { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texture1); }

        // draw
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

#if RENDER_USE_PBO
        // queue async readback into current PBO, then fence
        pbo_item_t *cur = &pboq[slot];
        glBindBuffer(GL_PIXEL_PACK_BUFFER, cur->pbo);
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        cur->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // present after queuing readback
        eglSwapBuffers(display, egl_surface);

        // harvest previous PBO if ready
        int prev_idx = (slot + PBO_COUNT - 1) % PBO_COUNT;
        pbo_item_t *prev = &pboq[prev_idx];
        if (prev->fence) {
            GLenum r = glClientWaitSync(prev->fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
            if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED) {
                glDeleteSync(prev->fence); prev->fence = 0;

                uint8_t *dst = (uint8_t*)spsc_pop(&ring_free);
                if (dst) {
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, prev->pbo);
                    uint8_t *gpu_ptr = (uint8_t*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, image_sz, GL_MAP_READ_BIT);
                    if (gpu_ptr) {
                        memcpy(dst, gpu_ptr, image_sz);
                        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                        (void)spsc_try_push(&ring_filled, dst);
                    }
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                }
            } else if (r == GL_WAIT_FAILED) {
                glFinish();
                glDeleteSync(prev->fence); prev->fence = 0;
            }
        }
        slot = (slot + 1) % PBO_COUNT;
#else
        // CPU path
        uint8_t *dst = (uint8_t*)aligned_alloc(64, (image_sz + 63) & ~((size_t)63));
        if (LIKELY(dst)) {
            glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, dst);
            eglSwapBuffers(display, egl_surface);
            (void)spsc_try_push(&ring_filled, dst);
        } else {
            eglSwapBuffers(display, egl_surface);
        }
#endif

        long slept = calculate_fps(scene->fps, scene->show_fps);
        if (scene->auto_fps) {
            long single_time = 1000000 / scene->fps;
            float percent = 100.0f - (float)slept / (float)single_time * 100.0f;
            if (percent < 95.0f) scene->fps++;
            else if (percent > 97.0f) scene->fps--;
        }
    }

    // stop mapper and join
    mctx.run = 0;
    pthread_join(mapper_th, NULL);

    // cleanup
    glDeleteBuffers(1, &vbo);
#if RENDER_USE_PBO
    glDeleteBuffers(3, (GLuint[]){ pboq[0].pbo, pboq[1].pbo, pboq[2].pbo });
    // free CPU buffer pool
    // (we cannot drain the free ring safely here, we kept our own array)
    for (int i = 0; i < 8; ++i) ; // no-op if you keep cpu_pool in a wider scope
#endif
    eglDestroySurface(display, egl_surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    gbm_surface_destroy(surface);
    gbm_device_destroy(gbm);
    close(fd);
    return NULL;
}


// ---------- main renderer ----------
void *render_shader2(void *arg) {
    scene_info *scene = (scene_info*)arg;
    debug("render shader %s\n", scene->shader_file);

    // DRM / GBM
    int fd = open("/dev/dri/card0", O_RDWR);
    if (fd < 0) die("Failed to open DRM device /dev/dri/card0\n");
    struct gbm_device  *gbm     = gbm_create_device(fd);
    struct gbm_surface *surface = gbm_surface_create(
        gbm, scene->width, scene->height,
        GBM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING);

    // EGL / GLES
    EGLDisplay display = eglGetDisplay(gbm);
    eglInitialize(display, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLConfig config; EGLint num_configs;
    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,   // ES3 preferred
        EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_ALPHA_SIZE,      8,
        EGL_NONE
    };
    eglChooseConfig(display, attribs, &config, 1, &num_configs);

    static const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs);
    EGLSurface egl_surface = eglCreateWindowSurface(display, config, (EGLNativeWindowType)surface, NULL);
    eglMakeCurrent(display, egl_surface, egl_surface, context);
    eglSwapInterval(display, 0);  // no vsync

    // program and quad
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);

    static const GLfloat verts[] = {
        -1.f,  1.f, 0.f,   -1.f, -1.f, 0.f,
         1.f,  1.f, 0.f,    1.f, -1.f, 0.f
    };
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    GLint pos_attrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(pos_attrib);
    glVertexAttribPointer(pos_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);

    // optional textures
    GLuint texture0 = 0, texture1 = 0;
    char *chan0 = change_file_extension(scene->shader_file, "channel0");
    if (access(chan0, R_OK) == 0) {
        texture0 = load_texture(chan0);
        if (!texture0) die("unable to load texture '%s'\n", chan0);
    }
    char *chan1 = change_file_extension(scene->shader_file, "channel1");
    if (access(chan1, R_OK) == 0) {
        texture1 = load_texture(chan1);
        if (!texture1) die("unable to load texture '%s'\n", chan1);
    }
    if (texture0) { glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texture0); }
    if (texture1) { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texture1); }

    // uniforms
    GLint time_loc  = glGetUniformLocation(program, "iTime");
    GLint dtym_loc  = glGetUniformLocation(program, "iTimeDelta");
    GLint frame_loc = glGetUniformLocation(program, "iFrame");
    GLint res_loc   = glGetUniformLocation(program, "iResolution");
    GLint c0_loc    = glGetUniformLocation(program, "iChannel0");
    GLint c1_loc    = glGetUniformLocation(program, "iChannel1");
    glUniform1i(c0_loc, 0);
    glUniform1i(c1_loc, 1);
    glUniform3f(res_loc, scene->width, scene->height, 0);

    // GL state for readbacks
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, scene->width, scene->height);

    // timing
    struct timespec start_time, end_time, orig_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    clock_gettime(CLOCK_MONOTONIC, &orig_time);
    unsigned long frame = 0;

    const size_t image_sz = (size_t)scene->width * scene->height * 4u;

    // queues and mapper
    enum { RING_SIZE = 8 };         // filled queue
    void *ring_filled_storage[RING_SIZE];
    spsc_ring_t ring_filled;
    spsc_init(&ring_filled, ring_filled_storage, RING_SIZE);

#if RENDER_USE_PBO
    // PBOs
    enum { PBO_COUNT = 3 };
    typedef struct { GLuint pbo; GLsync fence; } pbo_item_t;
    pbo_item_t pboq[PBO_COUNT];
    GLuint pbos[PBO_COUNT];
    glGenBuffers(PBO_COUNT, pbos);
    for (int i = 0; i < PBO_COUNT; ++i) {
        pboq[i].pbo   = pbos[i];
        pboq[i].fence = 0;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboq[i].pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, image_sz, NULL, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // CPU buffer pool and free ring
    enum { CPU_POOL = 8 };          // number of reusable CPU buffers
    uint8_t *cpu_pool[CPU_POOL];
    for (int i = 0; i < CPU_POOL; ++i) {
        cpu_pool[i] = (uint8_t*)aligned_alloc(64, image_sz);
        if (!cpu_pool[i]) die("failed to alloc CPU staging buffer\n");
    }
    void *ring_free_storage[CPU_POOL];
    spsc_ring_t ring_free;
    spsc_init(&ring_free, ring_free_storage, CPU_POOL);
    for (int i = 0; i < CPU_POOL; ++i) (void)spsc_push(&ring_free, cpu_pool[i]);

    mapper_ctx_t mctx = { .q_filled = &ring_filled, .q_free = &ring_free, .scene = scene, .run = 1 };
#else
    // CPU path uses same filled ring, no free ring needed if you do not reuse
    mapper_ctx_t mctx = { .q_filled = &ring_filled, .q_free = NULL, .scene = scene, .run = 1 };
#endif

    pthread_t mapper_th;
    if (pthread_create(&mapper_th, NULL, mapper_thread_main, &mctx) != 0) {
        die("failed to start mapper thread\n");
    }

    int slot = 0;

    // main loop
    while (scene->do_render) {
        frame++;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        float t  = (end_time.tv_sec - orig_time.tv_sec) + (end_time.tv_nsec - orig_time.tv_nsec) / 1e9f;
        float dt = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9f;

        glUseProgram(program);
        glUniform1f(time_loc,  t);
        glUniform1f(dtym_loc,  dt);
        glUniform1f(frame_loc, (float)frame);

        // draw the fullscreen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

#if RENDER_USE_PBO
        // queue async readback into current PBO, fence it
        pbo_item_t *cur = &pboq[slot];
        glBindBuffer(GL_PIXEL_PACK_BUFFER, cur->pbo);
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        cur->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // present after fence creation
        eglSwapBuffers(display, egl_surface);

        // harvest previous PBO if ready
        int prev_idx = (slot + PBO_COUNT - 1) % PBO_COUNT;
        pbo_item_t *prev = &pboq[prev_idx];
        if (prev->fence) {
            GLenum r = glClientWaitSync(prev->fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
            if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED) {
                glDeleteSync(prev->fence); prev->fence = 0;

                // pull a free CPU buffer
                uint8_t *dst = (uint8_t*)spsc_pop(&ring_free);
                if (dst) {
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, prev->pbo);
                    uint8_t *gpu_ptr = (uint8_t*)glMapBufferRange(
                        GL_PIXEL_PACK_BUFFER, 0, image_sz, GL_MAP_READ_BIT);
                    if (gpu_ptr) {
                        memcpy(dst, gpu_ptr, image_sz);
                        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

                        // enqueue to mapper
                        (void)spsc_try_push(&ring_filled, dst);
                    }
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                }
            } else if (r == GL_WAIT_FAILED) {
                glFinish();
                glDeleteSync(prev->fence); prev->fence = 0;
            }
        }

        slot = (slot + 1) % PBO_COUNT;
#else
        // simple CPU path: readback directly to a freshly malloc'd buffer
        uint8_t *dst = (uint8_t*)aligned_alloc(64, image_sz);
        if (LIKELY(dst)) {
            glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, dst);
            eglSwapBuffers(display, egl_surface);
            (void)spsc_try_push(&ring_filled, dst);
        } else {
            eglSwapBuffers(display, egl_surface);
        }
#endif

        long slept = calculate_fps(scene->fps, scene->show_fps);
        if (scene->auto_fps) {
            long single_time = 1000000 / scene->fps;
            float percent = 100.0f - (float)slept / (float)single_time * 100.0f;
            if (percent < 95.0f) scene->fps++;
            else if (percent > 97.0f) scene->fps--;
        }
    }

    // stop mapper and join
    mctx.run = 0;
    pthread_join(mapper_th, NULL);

    // cleanup
    glDeleteBuffers(1, &vbo);
#if RENDER_USE_PBO
    glDeleteBuffers(PBO_COUNT, pbos);
    // free CPU buffer pool
    for (int i = 0; i < CPU_POOL; ++i) free(cpu_pool[i]);
#endif
    eglDestroySurface(display, egl_surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    gbm_surface_destroy(surface);
    gbm_device_destroy(gbm);
    close(fd);
    return NULL;
}



// ------------- replacement function -------------
/*/
void *render_shader_old(void *arg) {
    scene_info *scene = (scene_info*)arg;
    debug("render shader %s\n", scene->shader_file);

    // DRM / GBM setup
    int fd = open("/dev/dri/card0", O_RDWR);
    if (fd < 0) die("Failed to open DRM device /dev/dri/card0\n");

    struct gbm_device *gbm = gbm_create_device(fd);
    struct gbm_surface *surface = gbm_surface_create(
        gbm, scene->width, scene->height,
        GBM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING);

    // EGL and GLES
    EGLDisplay display = eglGetDisplay(gbm);
    eglInitialize(display, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    // prefer ES3 for PBOs, fall back to ES2 if needed
    EGLConfig config; EGLint num_configs;
    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, // ES3 preferred
        EGL_SURFACE_TYPE,   EGL_WINDOW_BIT,
        EGL_RED_SIZE,       8,
        EGL_GREEN_SIZE,     8,
        EGL_BLUE_SIZE,      8,
        EGL_ALPHA_SIZE,     8,
        EGL_NONE
    };
    eglChooseConfig(display, attribs, &config, 1, &num_configs);

    // request ES 3 context to enable PBO + sync, will still work with ES2 with RENDER_USE_PBO 0
    static const EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    EGLSurface egl_surface = eglCreateWindowSurface(display, config, (EGLNativeWindowType)surface, NULL);
    eglMakeCurrent(display, egl_surface, egl_surface, context);

    // disable vsync so GPU is not capped
    eglSwapInterval(display, 0);

    // GL program
    printf("compiling GLSL shader...\n");
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);
    printf("compiled\n");

    // full screen quad
    static const GLfloat vertices[] = {
        -1.f,  1.f, 0.f,   -1.f, -1.f, 0.f,
         1.f,  1.f, 0.f,    1.f, -1.f, 0.f
    };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    GLint pos_attrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(pos_attrib);
    glVertexAttribPointer(pos_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);


    // textures, bind once
    GLuint texture0 = 0, texture1 = 0;
    char *chan0 = change_file_extension(scene->shader_file, "channel0");
    if (access(chan0, R_OK) == 0) {
        texture0 = load_texture(chan0);
        if (!texture0) die("unable to load texture '%s'\n", chan0);
    }
    char *chan1 = change_file_extension(scene->shader_file, "channel1");
    if (access(chan1, R_OK) == 0) {
        texture1 = load_texture(chan1);
        if (!texture1) die("unable to load texture '%s'\n", chan1);
    }
    if (texture0) { glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texture0); }
    if (texture1) { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texture1); }

    // uniforms locations
    GLint time_loc      = glGetUniformLocation(program, "iTime");
    GLint timed_loc     = glGetUniformLocation(program, "iTimeDelta");
    GLint frame_loc     = glGetUniformLocation(program, "iFrame");
    GLint res_loc       = glGetUniformLocation(program, "iResolution");
    GLint chan0_loc     = glGetUniformLocation(program, "iChannel0");
    GLint chan1_loc     = glGetUniformLocation(program, "iChannel1");

    glUniform1i(chan0_loc, 0);
    glUniform1i(chan1_loc, 1);
    glUniform3f(res_loc, scene->width, scene->height, 0);


    // pack alignment for readbacks
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, scene->width, scene->height);

    // frame timing
    struct timespec start_time, end_time, orig_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    clock_gettime(CLOCK_MONOTONIC, &orig_time);
    unsigned long frame = 0;

    // mapping pipeline
    const size_t image_sz = (size_t)scene->width * scene->height * 4u;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

#if RENDER_USE_PBO
    printf("PBO in use\n");
    
    // 3 PBOs
    enum { PBO_COUNT = 3 };
    typedef struct { GLuint pbo; GLsync fence; uint8_t *cpu; } pbo_item_t;
    pbo_item_t pboq[PBO_COUNT];
    map_job_t jobs[PBO_COUNT];

    GLuint pbos[PBO_COUNT];
    glGenBuffers(PBO_COUNT, pbos);
    for (int i=0;i<PBO_COUNT;i++) {
        pboq[i].fence = 0;
        pboq[i].cpu   = (uint8_t*)aligned_alloc(64, image_sz);
        pboq[i].pbo   = pbos[i];
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboq[i].pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, image_sz, NULL, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    // CPU buffer pool and rings
    enum { CPU_POOL = 8, RING_SIZE = 8 };
    uint8_t *cpu_pool[CPU_POOL];
    for (int i = 0; i < CPU_POOL; ++i) {
        cpu_pool[i] = (uint8_t*)aligned_alloc(64, image_sz);
        if (!cpu_pool[i]) die("failed to alloc CPU staging buffer\n");
    }

    void *ring_filled_storage[RING_SIZE];
    void *ring_free_storage[CPU_POOL];

    spsc_ring_t ring_filled, ring_free;
    spsc_init(&ring_filled, ring_filled_storage, RING_SIZE);
    spsc_init(&ring_free,   ring_free_storage,   CPU_POOL);

    // populate free ring with all buffers
    for (int i = 0; i < CPU_POOL; ++i) (void)spsc_push(&ring_free, cpu_pool[i]);

    // mapper thread
    mapper_ctx_t mctx = { .q_in = &ring_filled, .q_free = &ring_free, .run = 1 };
    pthread_t mapper_th;
    if (pthread_create(&mapper_th, NULL, mapper_thread_main, &mctx) != 0) {
        die("failed to start mapper thread\n");
    }

#else
    // CPU path
    enum { CPU_BUF_COUNT = 3, RING_SIZE = 8 };
    uint8_t *cpu_bufs[CPU_BUF_COUNT];
    for (int i = 0; i < CPU_BUF_COUNT; ++i) {
        cpu_bufs[i] = (uint8_t*)aligned_alloc(64, image_sz);
        if (!cpu_bufs[i]) die("failed to alloc CPU staging buffer\n");
    }
    map_job_t jobs[CPU_BUF_COUNT];

    void *ring_filled_storage[RING_SIZE];
    spsc_ring_t ring_filled;
    spsc_init(&ring_filled, ring_filled_storage, RING_SIZE);

    mapper_ctx_t mctx = { .q_in = &ring_filled, .q_free = NULL, .run = 1 };
    pthread_t mapper_th;
    if (pthread_create(&mapper_th, NULL, mapper_thread_main, &mctx) != 0) {
        die("failed to start mapper thread\n");
    }
#endif





#else
    // CPU staging buffers and jobs
    enum { CPU_BUF_COUNT = 3 };
    uint8_t *cpu_bufs[CPU_BUF_COUNT];
    for (int i = 0; i < CPU_BUF_COUNT; ++i) {
        cpu_bufs[i] = (uint8_t*)aligned_alloc(64, image_sz);
        if (!cpu_bufs[i]) die("failed to alloc CPU staging buffer\n");
    }
    map_job_t jobs[CPU_BUF_COUNT];
#endif

    // mapper thread and ring
    enum { RING_SIZE = 8 };
    void *ring_storage[RING_SIZE];
    spsc_ring_t ring;
    spsc_init(&ring, ring_storage, RING_SIZE);

    mapper_ctx_t mctx = { .q_in = &ring, .run = 1 };
    pthread_t mapper_th;
    if (pthread_create(&mapper_th, NULL, mapper_thread_main, &mctx) != 0) {
        die("failed to start mapper thread\n");
    }

    int slot = 0;

    while (scene->do_render) {
        frame++;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        float t  = (end_time.tv_sec - orig_time.tv_sec) + (end_time.tv_nsec - orig_time.tv_nsec) / 1e9f;
        float dt = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9f;

        glUseProgram(program);
        glUniform1f(time_loc,  t);
        glUniform1f(timed_loc, dt);
        glUniform1f(frame_loc, (float)frame);

        // render full-screen quad
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

#if RENDER_USE_PBO
        // 1) Kick async readback for current slot
        pbo_item_t *cur = &pboq[slot];
        glBindBuffer(GL_PIXEL_PACK_BUFFER, cur->pbo);
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        // make sure the GPU sees the commands soon
        glFlush();
        cur->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        eglSwapBuffers(display, egl_surface);

        // 2) Try to harvest the previous slot
        int prev_idx = (slot + PBO_COUNT - 1) % PBO_COUNT;
        pbo_item_t *prev = &pboq[prev_idx];

        if (prev->fence) {
            // ask once, also ask driver to flush if needed
            GLenum r = glClientWaitSync(prev->fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
            if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED) {
                glDeleteSync(prev->fence); prev->fence = 0;

                glBindBuffer(GL_PIXEL_PACK_BUFFER, prev->pbo);
                uint8_t *gpu_ptr = (uint8_t*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, image_sz, GL_MAP_READ_BIT);
                if (gpu_ptr) {
                    memcpy(prev->cpu, gpu_ptr, image_sz);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

                    // queue to mapper as CPU job
                    map_job_t *job = &jobs[prev_idx];
                    job->kind = JOB_CPU_PIXELS;
                    job->scene = scene;
                    job->size_bytes = image_sz;
                    job->u.cpu.pixels = prev->cpu;
                    // try to push, drop if full to avoid deadlock
                    if (!spsc_try_push(&ring, job)) {
                        printf("mapper ring full, dropping frame\n");
                    }
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            } else if (r == GL_WAIT_FAILED) {
                // driver issue or context problem, fallback: block once
                glFinish();
                glDeleteSync(prev->fence); prev->fence = 0;
            }
        }

        slot = (slot + 1) % PBO_COUNT;
#else
        // CPU path: read directly to a staging buffer, then enqueue pointer
        uint8_t *dst = cpu_bufs[slot];
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, dst);
        eglSwapBuffers(display, egl_surface);

        map_job_t *job = &jobs[slot];
        job->kind = JOB_CPU_PIXELS;
        job->scene = scene;
        job->size_bytes = image_sz;
        job->u.cpu.pixels = dst;

        if (!spsc_try_push(&ring, job)) {
            printf("mapper ring full, dropping frame\n");
        }
        slot = (slot + 1) % CPU_BUF_COUNT;
#endif

        // FPS pacing and adaptive fps, unchanged
        long slept = calculate_fps(scene->fps, scene->show_fps);
        if (scene->auto_fps) {
            long single_time = 1000000 / scene->fps;
            float percent = 100.0f - (float)slept / (float)single_time * 100.0f;
            if (percent < 95.0f) scene->fps++;
            else if (percent > 97.0f) scene->fps--;
        }
    }

    // stop mapper
    map_job_t quit = { .kind = JOB_QUIT, .scene = scene, .size_bytes = 0 };
    (void)spsc_try_push(&ring, &quit);
    pthread_join(mapper_th, NULL);

    // Cleanup GL
    glDeleteBuffers(1, &vbo);
#if RENDER_USE_PBO
    // glDeleteBuffers(PBO_COUNT, pboq);
#else
    for (int i = 0; i < CPU_BUF_COUNT; ++i) free(cpu_bufs[i]);
#endif
    eglDestroySurface(display, egl_surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    gbm_surface_destroy(surface);
    gbm_device_destroy(gbm);
    close(fd);
    return NULL;
}
    */