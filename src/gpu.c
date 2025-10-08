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


// ---------- full renderer ----------
void *render_shader(void *arg) {
    scene_info *scene = (scene_info*)arg;
    debug("render shader starting %s, %dx%d depth: %d\n", scene->shader_file, scene->width, scene->height, scene->bit_depth);

    cpu_set_affinity(1); // run this thread on CPU 2

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

    debug(" + shader program %d compiles\n", program);

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
    debug(" + textures loaded\n", program);

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
    debug(" + uniforms loaded\n", program);

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
    debug(" + ring buffer created\n", program);

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

    mapper_ctx_t *src = &scene->src_ctx;
    src->q_filled = &ring_filled;
    src->q_free = &ring_free;
    src->scene = scene;
    src->run = 1;
    debug(" + PBO created\n", program);
#else
    src->q_filled = &ring_filled;
    src->q_free = NULL;
    src->scene = scene;
    src->run = 1;
    debug(" + System Buffers created\n", program);
#endif

    pthread_t mapper_th;
    if (pthread_create(&mapper_th, NULL, mapper_thread_main, scene) != 0) {
        die("failed to start mapper thread\n");
    }

    debug(" + Mpping thread created\n", program);
    int slot = 0;

    printf("running GPU shader\n");
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

        (void)calculate_fps(scene);
        
    }
    printf("GPU shader complete\n");

    // stop mapper and join
    scene->src_ctx.run = 0;
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


