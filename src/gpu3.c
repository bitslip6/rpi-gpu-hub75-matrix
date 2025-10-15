/*
 * File: gpu.c
 * Description: Implements OpenGL ES and EGL initialization, shader compilation, texture management,
 * and a simple single-producer single-consumer (SPSC) ring buffer for GPU operations.
 */

#include <stddef.h>
#include <stdio.h>
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <sys/param.h>
#include <gbm.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <linux/time.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <semaphore.h>
#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MEMGUARD_OVERRIDE_STDLIB
#include "memguard2.h"

#include "rpihub75.h"
#include "util.h"
#include "pixels.h"
#include "spsc.h"

#ifdef USE_STB_IMAGE
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

/*
 * Test Shader Source:
 * This constant contains a simple GLSL fragment shader used for testing purposes.
 * It outputs a red color to the framebuffer.
 */
const char *test_shader_source =
    "#version 310 es\n"
    "precision mediump float;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    color = vec4(1.0, 0.0, 1.0, 1.0);\n" // Red color
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

/**
 * Load Texture:
 * Loads a PNG texture from the specified file path using stb_image.
 * Generates an OpenGL texture, sets texture parameters, and returns the texture ID.
 */
#ifdef USE_STB_IMAGE
GLuint load_texture(const char *filePath)
{
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Load the texture data from a PNG file using stb_image
    int width, height, nrChannels;
    unsigned char *data = stbi_load(filePath, &width, &height, &nrChannels, 0);
    if (data)
    {
        // Determine the format based on the number of channels in the PNG file
        GLenum format;
        if (nrChannels == 1)
            format = GL_RED;
        else if (nrChannels == 3)
            format = GL_RGB;
        else if (nrChannels == 4)
            format = GL_RGBA;
        else
        {
            printf("Unsupported number of channels in PNG: %d\n", nrChannels);
            stbi_image_free(data);
            return 0;
        }

        // Upload texture to GPU with mipmaps
        // internalformat parameter (3rd) is GLint; our chosen 'format' is GLenum, cast to GLint to silence -Wsign-conversion
        glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D); // Generate mipmaps for texture

        // Set texture parameters for wrapping and filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);      // Wrap horizontally
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);      // Wrap vertically
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // fast filter
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // fast filter
    }
    else
    {
        die("Failed to load texture: %s\n", filePath);
    }

    // Free image memory after loading into OpenGL
    stbi_image_free(data);

    return textureID;
}
#endif

/**
 * @brief Compiles a GLSL shader from the provided source code using the specified shader type.
 *
 * @param source The GLSL source code for the shader.
 * @param shader_type The type of shader (e.g., GL_VERTEX_SHADER or GL_FRAGMENT_SHADER).
 * @return The OpenGL shader ID if compilation is successful; otherwise, aborts execution on error.
 */
static GLuint compile_shader(const char *source, const GLenum shader_type)
{
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, NULL, info_log);
        die("Shader compilation error: %s\n", info_log);
    }

    return shader;
}

/**
 * @brief: Creates an OpenGL program by compiling and linking a vertex and fragment shader.
 *              The fragment shader is based on a shadertoy shader file with a header prepended.
 *
 * @param file The file path of the shadertoy shader source to load and compile.
 * @returns The OpenGL program ID of the linked shader program.
 */
static GLuint create_shadertoy_program(char *file)
{
    size_t filesize;
    char *src = file_get_contents(file, &filesize);
    if (filesize == 0)
    {
        die("Failed to read shader source\n");
    }

    char *src_with_header = (char *)malloc(filesize + 8192);
    if (src_with_header == NULL)
    {
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
    if (!success)
    {
        char info_log[512];
        glGetProgramInfoLog(program, 512, NULL, info_log);
        die("Program linking error: %s\n", info_log);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    SAFE_FREE(src_with_header);
    return program;
}

/**
 * @brief return a new string with the extension changed to new_extension
 *
 * @param filename
 * @param new_extension
 * @return char* - caller must free() the returned string
 */
char *change_file_extension(const char *filename, const char *new_extension)
{
    // Find the last dot in the filename
    const char *dot = strrchr(filename, '.');
    size_t new_filename_length = 0;

    // If there is no dot, simply append the new extension
    if (dot == NULL)
    {
        new_filename_length = strlen(filename) + strlen(new_extension) + 2; // +2 for dot and null terminator
    }
    else if (dot > filename)
    {
        new_filename_length = (unsigned)(dot - filename) + strlen(new_extension) + 2; // +2 for dot and null terminator
    }
    else
    {
        die("Invalid filename: %s\n", filename);
    }

    // Allocate memory for the new filename
    char *new_filename = (char *)malloc(new_filename_length);
    if (new_filename == NULL)
    {
        perror("Unable to allocate memory");
        return NULL;
    }

    // Copy the original filename up to the dot, if it exists
    if (dot == NULL)
    {
        strcpy(new_filename, filename);
    }
    else
    {
        strncpy(new_filename, filename, (size_t)(dot - filename));
        new_filename[dot - filename] = '\0'; // Null-terminate the string
    }

    // Append the new extension
    strcat(new_filename, ".");
    strcat(new_filename, new_extension);

    return new_filename;
}

/*
 * Macro: RENDER_USE_PBO
 * Description: Enables the use of Pixel Buffer Objects (PBO) for efficient GPU data transfers when available.
 * If not defined or GLES 3 is unavailable, the code will fallback to CPU pointer queueing.
 */
#ifndef RENDER_USE_PBO
#define RENDER_USE_PBO 1
#endif

// --------- simple SPSC ring for pointers ---------

/**
 * @brief Opens the DRM device for rendering, preferring /dev/dri/renderD128.
 * @return The file descriptor of the opened DRM device. - caller needs to call close()
 */
int open_dri_device()
{
    char path[256];
    snprintf(path, sizeof(path), "/dev/dri/renderD128");
    if (!file_exists("/dev/dri/renderD128"))
    {
        debug(" * v3d-pi5 not enabled, ensure config.txt contains: dtroverlay=vc4-kms-v3d-pi5 - for headless nodes disable hdmi via dtoverlay=vc4-kms-v3d-pi5,nohdmi\n");
        // fallback to /dev/dri/card0
        if (file_exists("/dev/dri/card0"))
        {
            snprintf(path, sizeof(path), "/dev/dri/card0");
        }
        else
        {
            die(" * DRM render node /dev/dri/renderD128 not found, /dev/dri/card0 also not found. ensure vc4-kms-v3d is loaded\n");
        }
    }
    // DRM / GBM
    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        die("Failed to open DRM device /dev/dri/card0\n");
    }

    return fd;
}

#ifdef USE_STB_IMAGE
void bind_tex(char *shader_file, char *texture_extension, GLuint unit)
{
    char *chan0 = change_file_extension(shader_file, texture_extension);
    if (access(chan0, R_OK) == 0)
    {
        GLuint texture = load_texture(chan0);
        if (!texture)
        {
            die("unable to load texture '%s'\n", chan0);
        }
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, texture);

        // force no-mipmap sampling and NPOT-safe wrap
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        glActiveTexture(GL_texture);
        glBindTexture(GL_TEXTURE_2D, texture);
    }

    SAFE_FREE(chan0);
}
#endif

/**
 * @brief update the time uniforms in the shader. call once per frame
 *
 * @param program
 */
void update_uniforms(GLuint program)
{

    static struct timespec end_time, orig_time, start_time;
    static GLint frame_loc = -1, time_loc = -1, dtym_loc = -1;
    static uint32_t frame = 0;

    // set the uniform locations once
    if (time_loc == -1 || dtym_loc == -1)
    {
        time_loc = glGetUniformLocation(program, "iTime");
        dtym_loc = glGetUniformLocation(program, "iTimeDelta");
        frame_loc = glGetUniformLocation(program, "iFrame");
        orig_time = start_time; // establish time origin
        // orig_time = start_time; // REMOVE ME: is this correct?
    }

    // update time uniforms
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float t = (float)(end_time.tv_sec - orig_time.tv_sec) + (float)(end_time.tv_nsec - orig_time.tv_nsec) / 1e9f;
    float dt = (float)(end_time.tv_sec - start_time.tv_sec) + (float)(end_time.tv_nsec - start_time.tv_nsec) / 1e9f;

    frame++;

    glUseProgram(program);
    glUniform1f(time_loc, t);
    glUniform1f(dtym_loc, dt);
    glUniform1i(frame_loc, (GLint)frame);
}

/**
 * @brief Structure to hold GPU initialization resources
 */
typedef struct
{
    int device_fd;
    struct gbm_device *gbm;
    struct gbm_surface *surface;
    EGLDisplay display;
    EGLContext context;
    EGLSurface egl_surface;
    // Offscreen render target
    GLuint fbo;
    GLuint color_tex;
} gpu_context_t;

/**
 * @brief Initializes GPU context including DRM, GBM, EGL and OpenGL ES
 *
 * @param width Surface width
 * @param height Surface height
 * @return gpu_context_t* Pointer to initialized GPU context, or NULL on failure
 */
gpu_context_t *init_gpu_context(unsigned int width, unsigned int height)
{
    gpu_context_t *ctx = malloc(sizeof(gpu_context_t));
    if (!ctx)
    {
        die("Failed to allocate GPU context\n");
    }

    // Open DRI device
    ctx->device_fd = open_dri_device();

    // Create GBM device and surface
    ctx->gbm = gbm_create_device(ctx->device_fd);
    ctx->surface = gbm_surface_create(
        ctx->gbm, width, height,
        GBM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING);

    // EGL / GLES setup
    ctx->display = eglGetDisplay(ctx->gbm);
    eglInitialize(ctx->display, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLConfig config;
    EGLint num_configs;
    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        // EGL_RED_SIZE, 8,
        // EGL_GREEN_SIZE, 8,
        // EGL_BLUE_SIZE, 8,
        // EGL_ALPHA_SIZE, 8,
        EGL_NONE};
    eglChooseConfig(ctx->display, attribs, &config, 1, &num_configs);

    EGLContext ectx = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT, (EGLint[]){EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE});

    // For surfaceless (preferred if supported)
    eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, ectx);
    ctx->context = ectx;
    ctx->egl_surface = EGL_NO_SURFACE;

    GLuint color_tex, fbo;
    glGenTextures(1, &color_tex);
    glBindTexture(GL_TEXTURE_2D, color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, color_tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        die("Failed to create framebuffer\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);

    // persist FBO/texture so callers can rebind explicitly before readback
    ctx->fbo = fbo;
    ctx->color_tex = color_tex;

    /*
    GLuint color_tex, fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                       GL_TEXTURE_2D, color_tex, 0);
    */

    /*
    static const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ctx->context = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT, ctx_attribs);
    ctx->egl_surface = eglCreateWindowSurface(ctx->display, config, (EGLNativeWindowType)ctx->surface, NULL);

    eglMakeCurrent(ctx->display, ctx->egl_surface, ctx->egl_surface, ctx->context);
    eglSwapInterval(ctx->display, 0); // uncapped
    */

    return ctx;
}

// Safer offscreen path: use a small PBuffer surface and EGL_DEFAULT_DISPLAY, no GBM surface.
gpu_context_t *init_gpu_context_pbuffer(unsigned int width, unsigned int height)
{
    // Use GBM-backed surfaceless EGL, which works headless on Pi
    gpu_context_t *ctx = calloc(1, sizeof(gpu_context_t));
    if (!ctx) die("Failed to allocate GPU context (surfaceless)\n");

    // Open DRM and create GBM device
    ctx->device_fd = open_dri_device();
    ctx->gbm = gbm_create_device(ctx->device_fd);
    ctx->surface = NULL;

    // Prefer eglGetPlatformDisplayEXT if available, else fallback
    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (getPlatformDisplay) {
        // Prefer surfaceless first for headless stability
        const EGLint attrs[] = { EGL_NONE };
        ctx->display = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, attrs);
        if (ctx->display == EGL_NO_DISPLAY) {
            // Try GBM platform as a fallback
            ctx->display = getPlatformDisplay(EGL_PLATFORM_GBM_KHR, (void*)ctx->gbm, attrs);
        }
    }
    if (ctx->display == EGL_NO_DISPLAY) {
        // Fallback to classic GBM display
        ctx->display = eglGetDisplay((EGLNativeDisplayType)ctx->gbm);
    }
    if (ctx->display == EGL_NO_DISPLAY) die("eglGetDisplay failed (GBM/surfaceless)\n");
        if (!eglInitialize(ctx->display, NULL, NULL)) die("eglInitialize failed\n");
    eglBindAPI(EGL_OPENGL_ES_API);

        // Query extensions to see if surfaceless is supported
        const char *exts = eglQueryString(ctx->display, EGL_EXTENSIONS);
        bool have_surfaceless = false;
        if (exts) {
                have_surfaceless = strstr(exts, "EGL_KHR_surfaceless_context") || strstr(exts, "EGL_MESA_platform_surfaceless");
        }

        // Try several config attribute sets in order
        EGLConfig config = 0; EGLint num = 0; bool got_cfg = false;
        const EGLint cfgs[][16] = {
                // ES3, no surface requirement
                { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                    EGL_NONE },
                // ES3, allow pbuffer
                { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                    EGL_NONE },
                // ES2 fallback, no surface requirement
                { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                    EGL_NONE },
                // ES2, allow pbuffer
                { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                    EGL_NONE }
        };
        for (size_t i = 0; i < sizeof(cfgs)/sizeof(cfgs[0]); ++i) {
                if (eglChooseConfig(ctx->display, cfgs[i], &config, 1, &num) && num >= 1) { got_cfg = true; break; }
        }
        if (!got_cfg) die("eglChooseConfig (surfaceless/pbuffer) failed\n");

        // Try ES3 context, fallback to ES2
        ctx->context = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT, (EGLint[]){EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE});
        if (ctx->context == EGL_NO_CONTEXT) {
                ctx->context = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT, (EGLint[]){EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE});
                if (ctx->context == EGL_NO_CONTEXT) die("eglCreateContext failed (ES3/ES2)\n");
        }

        if (have_surfaceless) {
                if (!eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx->context)) die("eglMakeCurrent (surfaceless) failed\n");
                ctx->egl_surface = EGL_NO_SURFACE;
        } else {
                // Create a tiny pbuffer if surfaceless not supported
                EGLint pb_attrs[] = { EGL_WIDTH, (EGLint)width, EGL_HEIGHT, (EGLint)height, EGL_NONE };
                ctx->egl_surface = eglCreatePbufferSurface(ctx->display, config, pb_attrs);
                if (ctx->egl_surface == EGL_NO_SURFACE) die("eglCreatePbufferSurface failed\n");
                if (!eglMakeCurrent(ctx->display, ctx->egl_surface, ctx->egl_surface, ctx->context)) die("eglMakeCurrent (pbuffer) failed\n");
        }

    // Create offscreen FBO/texture
    glGenTextures(1, &ctx->color_tex);
    glBindTexture(GL_TEXTURE_2D, ctx->color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &ctx->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ctx->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctx->color_tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) die("FBO incomplete (surfaceless)\n");

    glBindFramebuffer(GL_FRAMEBUFFER, ctx->fbo);
    glViewport(0, 0, width, height);
    return ctx;
}

/**
 * @brief Cleanup GPU context and free resources
 *
 * @param ctx GPU context to cleanup
 */
void cleanup_gpu_context(gpu_context_t *ctx)
{
    if (!ctx)
        return;
    if (ctx->fbo)
    {
        glDeleteFramebuffers(1, &ctx->fbo);
        ctx->fbo = 0;
    }
    if (ctx->color_tex)
    {
        glDeleteTextures(1, &ctx->color_tex);
        ctx->color_tex = 0;
    }
    if (ctx->egl_surface && ctx->egl_surface != EGL_NO_SURFACE)
    {
        eglDestroySurface(ctx->display, ctx->egl_surface);
    }
    if (ctx->context)
    {
        eglDestroyContext(ctx->display, ctx->context);
    }
    eglTerminate(ctx->display);
    if (ctx->surface) gbm_surface_destroy(ctx->surface);
    if (ctx->gbm) gbm_device_destroy(ctx->gbm);
    if (ctx->device_fd >= 0) close(ctx->device_fd);
    free(ctx);
}

// ---------- full renderer ----------
/**
 * @brief Primary rendering function that sets up DRM/GBM and EGL/GL contexts, compiles
 *              the shader program from a shadertoy file, sets up vertex buffers and textures,
 *              and enters the main rendering loop. It handles asynchronous readback using PBOs (if enabled)
 *              or CPU readback, and adjusts frame rate dynamically. This function is executed in a separate
 *              thread and uses the provided scene_info for configuration.
 *
 * @param arg A pointer to a scene_info structure containing rendering parameters such as shader file,
 *          dimensions, and FPS settings.
 */
void *render_shader(void *arg)
{
    scene_info *scene = (scene_info *)arg;
    debug(" ~~ render shader %s\n", scene->shader_file);
    pin_thread_to_cpu(2); // make sure we don't run

    scene->stride = 4;

    // Initialize GPU context
    gpu_context_t *gpu_ctx = init_gpu_context(scene->width, scene->height);
    static uint8_t *pending_dst = NULL; // persists across iterations

    // program and quad
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);

    static const GLfloat verts[] = {
        -1.f, 1.f, 0.f, -1.f, -1.f, 0.f,
        1.f, 1.f, 0.f, 1.f, -1.f, 0.f};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    GLuint pos_attrib = (GLuint)glGetAttribLocation(program, "position");
    if ((GLint)pos_attrib >= 0)
    {
        glEnableVertexAttribArray((GLuint)pos_attrib);
        glVertexAttribPointer((GLuint)pos_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }

    // IMPORTANT: ensure tight unpack before any texture uploads
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Optional textures
#ifdef USE_STB_IMAGE

    bind_tex(scene->shader_file, "channel0", 0);
    bind_tex(scene->shader_file, "channel1", 1);

    GLint c0_loc = glGetUniformLocation(program, "iChannel0");
    GLint c1_loc = glGetUniformLocation(program, "iChannel1");
    glUniform1i(c0_loc, 0);
    glUniform1i(c1_loc, 1);

#endif

    // uniforms
    GLint res_loc = glGetUniformLocation(program, "iResolution");
    glUniform3f(res_loc, scene->width, scene->height, 0);

    // GL state for readbacks
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, scene->width, scene->height);

    // timing
    unsigned long frame = 0;
    const GLsizeiptr image_sz = (GLsizeiptr)((size_t)scene->width * scene->height * 4u);
    int slot = 0;

    GLsync complete_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    uint8_t *staging = malloc(image_sz);

    // main loop
    while (scene->do_render)
    {
        // update the time uniforms
        update_uniforms(program);

        // ensure we draw/read from our FBO
        glBindFramebuffer(GL_FRAMEBUFFER, gpu_ctx->fbo);
        glViewport(0, 0, scene->width, scene->height);

        // draw
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // glFinish();

        // yield until the frame is complete...
        GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

        while (1)
        {
            GLenum w = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000);
            if (w == GL_ALREADY_SIGNALED || w == GL_CONDITION_SATISFIED)
                break;
            if (w == GL_WAIT_FAILED)
                break;
            sched_yield();
        }
        glDeleteSync(fence);

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
// be explicit about read buffer in GLES3
#ifdef GL_COLOR_ATTACHMENT0
        glReadBuffer(GL_COLOR_ATTACHMENT0);
#endif
        // pull the pixels back to the CPU using the CPU
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, staging);

        // present after queuing readback
        // eglSwapBuffers(gpu_ctx->display, gpu_ctx->egl_surface);

        uint32_t *dst = (uint32_t *)spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
        if (dst)
        {

            // #ifdef DEBUG
            uintptr_t base = (uintptr_t)scene->ring_buf_mapper->items;
            uintptr_t end = base + scene->ring_buf_mapper->buff_size;
            uintptr_t d = (uintptr_t)dst;
            if (!(d >= base && d + image_sz <= end))
            {
                die("SPSC slot pointer out of bounds\n");
            }
            // #endif

            // printf("image_sz=%zu, q->item_size=%zu, q->capacity=%zu - ", image_sz, scene->ring_buf_mapper->item_size, scene->ring_buf_mapper->capacity);
            // printf("dst=%p, staging=%p\n", dst, staging);
            //  printf("copy from %p to %p\n", staging, dst);
            memcpy(dst, staging, image_sz);
            // push the read frame onto the mapper thread
            spsc_push_ptr_commit(scene->ring_buf_mapper);
        }
        else
        {
            debug("dropping frame from OpenGL\n");
        }

        // always do a small sleep to allow other threads to run
        //struct timespec ts = {0, 200000};
        //nanosleep(&ts, NULL);

        long slept = calculate_fps(scene->fps, scene->show_fps);
        sched_yield();
    }

    debug(" ## GPU render thread exiting...\n");

    // cleanup
    glDeleteBuffers(1, &vbo);
    cleanup_gpu_context(gpu_ctx);

    return NULL;
}

// Minimal renderer to isolate GPU issues: no PBOs, no SPSC, no fences. Draws a solid clear
// and a simple shader quad to the offscreen FBO, reads back a single pixel, prints checksum, exits.
void *render_shader_minimal(void *arg)
{
    scene_info *scene = (scene_info *)arg;
    debug(" ~~ render shader (minimal) %s\n", scene->shader_file);

    // Environment-based toggles for isolation
    int fb_side = 0; const char *fb_env = getenv("HUB75_FB_SIZE");
    if (fb_env) { int v = atoi(fb_env); if (v > 0 && v <= 4096) fb_side = v; }
    unsigned fb_w = fb_side ? (unsigned)fb_side : scene->width;
    unsigned fb_h = fb_side ? (unsigned)fb_side : scene->height;
    int read_every = 1; const char *re_env = getenv("HUB75_READ_EVERY");
    if (re_env) { int v = atoi(re_env); if (v >= 0 && v <= 1000) read_every = v; }
    bool clear_only = getenv("HUB75_CLEAR_ONLY") != NULL;

    // Create EGL/GLES + offscreen FBO
    gpu_context_t *gpu_ctx = init_gpu_context_pbuffer(fb_w, fb_h);

    // Bind our FBO explicitly
    glBindFramebuffer(GL_FRAMEBUFFER, gpu_ctx->fbo);
    glViewport(0, 0, fb_w, fb_h);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // Print GL info for diagnostics
    const GLubyte *ven = glGetString(GL_VENDOR);
    const GLubyte *ren = glGetString(GL_RENDERER);
    const GLubyte *ver = glGetString(GL_VERSION);
    printf("[minimal] GL Vendor: %s\n", ven ? (const char*)ven : "(null)");
    printf("[minimal] GL Renderer: %s\n", ren ? (const char*)ren : "(null)");
    printf("[minimal] GL Version: %s\n", ver ? (const char*)ver : "(null)");

    // Phase 1: clear test
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    uint8_t px[4] = {0};
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("[minimal] clear test pixel RGBA=(%u,%u,%u,%u)\n", px[0], px[1], px[2], px[3]);

    // Phase 2: trivial shader draw
    GLuint prog = 0; GLuint vbo = 0; GLint pos_loc = -1; GLint res_loc = -1;
    if (!clear_only) {
        prog = create_shadertoy_program(scene->shader_file ? scene->shader_file : "");
        glUseProgram(prog);
    static const GLfloat verts[] = {
        -1.f, 1.f, 0.f,
        -1.f, -1.f, 0.f,
        1.f, 1.f, 0.f,
        1.f, -1.f, 0.f};
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        pos_loc = glGetAttribLocation(prog, "position");
        if (pos_loc >= 0) {
            glEnableVertexAttribArray((GLuint)pos_loc);
            glVertexAttribPointer((GLuint)pos_loc, 3, GL_FLOAT, GL_FALSE, 0, 0);
        }
        res_loc = glGetUniformLocation(prog, "iResolution");
        glUniform3f(res_loc, (GLfloat)fb_w, (GLfloat)fb_h, 0.0f);
    }

    uint32_t frame = 0;
    uint8_t p2[4] = {0};
    // uint32_t *staging = malloc(scene->width * scene->height * 4);
    while(scene->do_render) {
        // draw once and read back center pixel
        glBindFramebuffer(GL_FRAMEBUFFER, gpu_ctx->fbo);
        glViewport(0, 0, fb_w, fb_h);
        if (clear_only) {
            float pulse = (frame & 63) / 63.0f;
            glClearColor(pulse, 1.0f - pulse, 0.2f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        } else {
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        glFlush();
        if (read_every > 0 && (frame % (unsigned)read_every) == 0u) {
            int cx = (int)(fb_w / 2), cy = (int)(fb_h / 2);
            glReadPixels(cx, cy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p2);
        }

        GLenum err = glGetError();
        if (UNLIKELY(err != GL_NO_ERROR)) {
            printf("[minimal] GL error 0x%x at frame %u\n", err, frame);
            break;
        }

        if (frame++ % 100 == 1) {
            printf("render frame: %d\n", frame);
        }

    (void)calculate_fps(scene->fps, scene->show_fps);
    struct timespec ts={0,2*1000*1000}; nanosleep(&ts, NULL);
    }
    printf("[minimal] shader test pixel RGBA=(%u,%u,%u,%u)\n", p2[0], p2[1], p2[2], p2[3]);

    // cleanup minimal GL objects and context
    if (pos_loc >= 0) glDisableVertexAttribArray((GLuint)pos_loc);
    if (vbo) { glBindBuffer(GL_ARRAY_BUFFER, 0); glDeleteBuffers(1, &vbo); }
    if (prog) { glUseProgram(0); glDeleteProgram(prog); }
    cleanup_gpu_context(gpu_ctx);

    printf(" ## minimal GPU render exiting\n");
    return NULL;
}

void *render_shader_pbo(void *arg)
{
    scene_info *scene = (scene_info *)arg;
    debug(" ~~ render shader %s\n", scene->shader_file);

    // Initialize GPU context
    gpu_context_t *gpu_ctx = init_gpu_context(scene->width, scene->height);
    static uint8_t *pending_dst = NULL; // persists across iterations

    // program and quad
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);

    static const GLfloat verts[] = {
        -1.f, 1.f, 0.f, -1.f, -1.f, 0.f,
        1.f, 1.f, 0.f, 1.f, -1.f, 0.f};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    GLuint pos_attrib = (GLuint)glGetAttribLocation(program, "position");
    if ((GLint)pos_attrib >= 0)
    {
        glEnableVertexAttribArray((GLuint)pos_attrib);
        glVertexAttribPointer((GLuint)pos_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }

    // IMPORTANT: ensure tight unpack before any texture uploads
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Optional textures
#ifdef USE_STB_IMAGE

    bind_tex(scene->shader_file, "channel0", 0);
    bind_tex(scene->shader_file, "channel1", 1);

    GLint c0_loc = glGetUniformLocation(program, "iChannel0");
    GLint c1_loc = glGetUniformLocation(program, "iChannel1");
    glUniform1i(c0_loc, 0);
    glUniform1i(c1_loc, 1);

#endif

    // uniforms
    GLint res_loc = glGetUniformLocation(program, "iResolution");
    glUniform3f(res_loc, scene->width, scene->height, 0);

    // GL state for readbacks
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, scene->width, scene->height);

    // timing
    unsigned long frame = 0;

    const GLsizeiptr image_sz = scene->width * scene->height * 4u;

    // spsc_init(&ring_filled, ring_filled_storage, RING_SIZE);

#if RENDER_USE_PBO
    printf(" * Using PBO async readback\n");
    // triple PBOs
    enum
    {
        PBO_COUNT = 3
    };
    typedef struct
    {
        GLuint pbo;
        GLsync fence;
    } pbo_item_t;
    pbo_item_t pboq[PBO_COUNT];
    GLuint pbos[PBO_COUNT];
    glGenBuffers(PBO_COUNT, pbos);
    for (int i = 0; i < PBO_COUNT; ++i)
    {
        pboq[i].pbo = pbos[i];
        pboq[i].fence = 0;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboq[i].pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, image_sz, NULL, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

#endif

    int slot = 0;

    // main loop
    while (scene->do_render)
    {
        // update the time uniforms
        update_uniforms(program);

        // draw
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        uint8_t *dst = spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
        if (!dst)
        {
            debug("dropping frame from OpenGL\n");
            // eglSwapBuffers(gpu_ctx->display, gpu_ctx->egl_surface); // do we need this?
            continue;
        }

        // pull the pixels back to CPU using the GPU
#if RENDER_USE_PBO
        // queue async readback into current PBO, then fence
        pbo_item_t *cur = &pboq[slot];
        glBindBuffer(GL_PIXEL_PACK_BUFFER, cur->pbo);
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        cur->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // present after queuing readback
        // eglSwapBuffers(gpu_ctx->display, gpu_ctx->egl_surface);

        // harvest previous PBO if ready
        int prev_idx = (slot + PBO_COUNT - 1) % PBO_COUNT;
        pbo_item_t *prev = &pboq[prev_idx];

        if (prev->fence)
        {
            GLenum r = glClientWaitSync(prev->fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
            if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED)
            {

                glDeleteSync(prev->fence);
                prev->fence = 0;

                /* only now reserve a ring slot, reuse pending if we already have one */
                if (!pending_dst)
                {
                    pending_dst = spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
                    if (!pending_dst)
                    {
                        /* cannot deliver this frame, just drop it safely */
                        glBindBuffer(GL_PIXEL_PACK_BUFFER, prev->pbo);
                        (void)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, image_sz, GL_MAP_READ_BIT);
                        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                        slot = (slot + 1) % PBO_COUNT;
                        goto next_frame;
                    }
                }

                glBindBuffer(GL_PIXEL_PACK_BUFFER, prev->pbo);
                uint8_t *gpu_ptr = (uint8_t *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, image_sz, GL_MAP_READ_BIT);
                if (gpu_ptr)
                {
                    memcpy(pending_dst, gpu_ptr, (size_t)image_sz);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                    spsc_push_ptr_commit(scene->ring_buf_mapper);
                    pending_dst = NULL; /* we consumed the reservation */
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }
            else if (r == GL_WAIT_FAILED)
            {
                glFinish();
                glDeleteSync(prev->fence);
                prev->fence = 0;
            }
        }
        /*
        if (prev->fence)
        {
            GLenum r = glClientWaitSync(prev->fence, GL_SYNC_FLUSH_COMMANDS_BIT, 2);
            if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED)
            {
                glDeleteSync(prev->fence);
                prev->fence = 0;



                glBindBuffer(GL_PIXEL_PACK_BUFFER, prev->pbo);
                uint8_t *gpu_ptr = (uint8_t *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, image_sz, GL_MAP_READ_BIT);
                if (gpu_ptr)
                {
                    memcpy(dst, gpu_ptr, (size_t)image_sz);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                    spsc_push_ptr_commit(scene->ring_buf_mapper);
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }
            else if (r == GL_WAIT_FAILED)
            {
                glFinish();
                glDeleteSync(prev->fence);
                prev->fence = 0;
            }
        }
            */
        slot = (slot + 1) % PBO_COUNT;
    next_frame:;
#else
        // pull the pixels back to the CPU using the CPU
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, dst);

        eglSwapBuffers(gpu_ctx->display, gpu_ctx->egl_surface);
        // push the read frame onto the mapper thread
        spsc_push_ptr_commit(scene->ring_buf_mapper);
#endif

        long slept = calculate_fps(scene->fps, scene->show_fps);
    }

    debug(" ## GPU render thread exiting...\n");

    // cleanup
    glDeleteBuffers(1, &vbo);
#if RENDER_USE_PBO
    glDeleteBuffers(3, (GLuint[]){pboq[0].pbo, pboq[1].pbo, pboq[2].pbo});
#endif
    cleanup_gpu_context(gpu_ctx);

    return NULL;
}
