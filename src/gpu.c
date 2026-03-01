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
#include <string.h>
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

//#define MEMGUARD_OVERRIDE_STDLIB
// #include "memguard2.h"

#include "lowlevel.h"
#include "rpihub75.h"
#include "util.h"
#include "pixels.h"
#include "spsc.h"
#include "text_sdf.h"
#include "compositor.h"

/* Image loading is handled by util.c (image_read_rgba8) which supports PNG and JPEG */

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
 * Load Texture (supports PNG and JPEG via image_read_rgba8)
 */
static GLuint load_texture(const char *filePath)
{
    uint8_t *image_data = NULL;
    int width = 0, height = 0, stride = 0;
    if (image_read_rgba8(filePath, &image_data, &width, &height, &stride) != 0) {
        die("Failed to load texture: %s\n", filePath);
    }

    // Upload to GL
    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)width, (GLsizei)height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    debug("loaded texture %s [%dx%d]\n", filePath, width, height);
    free(image_data);
    return textureID;
}

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
    snprintf(path, sizeof(path), "/dev/dri/card0");
    if (!file_exists("/dev/dri/card0"))
    {
        debug(" * v3d-pi5 not enabled, or user doesn't have permission: /dev/dri/renderD128 ensure config.txt contains: dtroverlay=vc4-kms-v3d-pi5 - for headless nodes disable hdmi via dtoverlay=vc4-kms-v3d-pi5,nohdmi\n");
        // fallback to /dev/dri/card0
        if (file_exists("/dev/dri/card0"))
        {
            debug(" * falling back to /dev/dri/card0\n");
            if (!access("/dev/dri/card0", R_OK | W_OK)) {
                die(" * /dev/dri/card0 exists but cannot be opened for read/write. ensure vc4-kms-v3d is loaded and user is in the 'video' group\n");
            }
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

static void bind_tex(char *shader_file, char *texture_extension, GLuint unit)
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

        // trilinear mipmap filtering for clean downsampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }

    SAFE_FREE(chan0);
}

/**
 * @brief update the time uniforms in the shader. call once per frame
 *
 * @param program
 */
void update_uniforms(GLuint program, float time_scale)
{

    static struct timespec end_time, orig_time, last_time;
    static GLint frame_loc = -1;
    static GLint time_loc = -1;
    static GLint dtym_loc = -1;
    static uint32_t frame = 0;

    // set the uniform locations once
    if (time_loc == -1) //|| dtym_loc == -1)
    {
        // update time uniforms
        clock_gettime(CLOCK_MONOTONIC, &orig_time);
        clock_gettime(CLOCK_MONOTONIC, &last_time);

        time_loc  = glGetUniformLocation(program, "iTime");
        dtym_loc  = glGetUniformLocation(program, "iTimeDelta");
        frame_loc = glGetUniformLocation(program, "iFrame");

    }

    // update time uniforms
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float t = (float)(end_time.tv_sec - orig_time.tv_sec) + (float)(end_time.tv_nsec - orig_time.tv_nsec) / 1e9f;
    float dt = (float)(end_time.tv_sec - last_time.tv_sec) + (float)(end_time.tv_nsec - last_time.tv_nsec) / 1e9f;
    clock_gettime(CLOCK_MONOTONIC, &last_time);

    frame++;

    //glUseProgram(program);
    glUniform1f(time_loc, t * time_scale);
    glUniform1f(dtym_loc, dt * time_scale);
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
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)width, (GLsizei)height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
#ifdef GL_COLOR_ATTACHMENT0
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, color_tex, 0);
#endif
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        die("Failed to create framebuffer\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    // persist FBO/texture so callers can rebind explicitly before readback
    ctx->fbo = fbo;
    ctx->color_tex = color_tex;

    return ctx;
}

// Safer offscreen path: use a small PBuffer surface and EGL_DEFAULT_DISPLAY, no GBM surface.
gpu_context_t *init_gpu_context_pbuffer(unsigned int width, unsigned int height)
{
    // Use GBM-backed surfaceless EGL, which works headless on Pi
    gpu_context_t *ctx = calloc(1, sizeof(gpu_context_t));
    if (!ctx) {
        die("Failed to allocate GPU context (surfaceless)\n");
    }

    // Open DRM and create GBM device
    ctx->device_fd = open_dri_device();
    ctx->gbm = gbm_create_device(ctx->device_fd);
    ctx->surface = NULL;

    // Prefer eglGetPlatformDisplayEXT if available, else fallback
    PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (getPlatformDisplay) {
        debug(" [*] Using eglGetPlatformDisplayEXT for surfaceless/GBM EGL display\n");
        // Prefer surfaceless first for headless stability
        const EGLint attrs[] = { EGL_NONE };
        ctx->display = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, attrs);
        if (ctx->display == EGL_NO_DISPLAY) {
            // Try GBM platform as a fallback
            ctx->display = getPlatformDisplay(EGL_PLATFORM_GBM_KHR, (void*)ctx->gbm, attrs);
        }
    }
    if (ctx->display == EGL_NO_DISPLAY) {
        printf(" [*] eglGetPlatformDisplayEXT not available or failed, falling back to eglGetDisplay\n");
        // Fallback to classic GBM display
        ctx->display = eglGetDisplay((EGLNativeDisplayType)ctx->gbm);
    }
    if (ctx->display == EGL_NO_DISPLAY) {
        die(" [!] eglGetDisplay failed (GBM/surfaceless)\n");
    }
    if (!eglInitialize(ctx->display, NULL, NULL)) {
        die(" [!] eglInitialize failed\n");
    }
    // eglBindAPI(EGL_OPENGL_ES_API);

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
            if (eglChooseConfig(ctx->display, cfgs[i], &config, 1, &num) && num >= 1) {
                debug(" [*] eglChooseConfig number: [%d] success\n", i);
                got_cfg = true;
                break;
            }
    }
    if (!got_cfg) {
        die(" [!] eglChooseConfig (surfaceless/pbuffer) failed\n");
    }

    // Try ES3 context, fallback to ES2
    ctx->context = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT, (EGLint[]){EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE});
    if (ctx->context == EGL_NO_CONTEXT) {
        debug(" [.] eglCreateContext ES3 failed, trying ES2...\n");
        ctx->context = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT, (EGLint[]){EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE});
        if (ctx->context == EGL_NO_CONTEXT) {
            die(" [!] eglCreateContext failed (ES3/ES2)\n");
        }
    }

    if (have_surfaceless) {
            if (!eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx->context)) {
                die(" [!] eglMakeCurrent (surfaceless) failed\n");
            }
            ctx->egl_surface = EGL_NO_SURFACE;
    } else {
            // Create a tiny pbuffer if surfaceless not supported
            EGLint pb_attrs[] = { EGL_WIDTH, (EGLint)width, EGL_HEIGHT, (EGLint)height, EGL_NONE };
            ctx->egl_surface = eglCreatePbufferSurface(ctx->display, config, pb_attrs);
            if (ctx->egl_surface == EGL_NO_SURFACE) {
                die(" [!] eglCreatePbufferSurface failed\n");
            }
            if (!eglMakeCurrent(ctx->display, ctx->egl_surface, ctx->egl_surface, ctx->context)) {
                die(" [!] eglMakeCurrent (pbuffer) failed\n");
            }
    }

    eglSwapInterval(ctx->display, 0); // uncapped

    // Create offscreen FBO/texture
    glGenTextures(1, &ctx->color_tex);
    glBindTexture(GL_TEXTURE_2D, ctx->color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)width, (GLsizei)height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenFramebuffers(1, &ctx->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ctx->fbo);
#ifdef GL_COLOR_ATTACHMENT0
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctx->color_tex, 0);
#endif
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        die(" [!] FBO incomplete (surfaceless)\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, ctx->fbo);
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
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

static image_buffer_t *text_render_sdf(char *sdf_path, char *msg, float size_px, RGBA color, float weight, float outline_width, RGBA outline_color) {

    // Configure font: load and scale to 64px line height 
    sdf_font_t *font = sdf_font_load_scaled(sdf_path, 64.0f);
    if (!font) {
        fprintf(stderr, "[TEXT] Failed to load SDF font from '%s' (expect metrics.csv + PNGs).\n", sdf_path);
        return NULL;
    }

    // Create text object
    //const char msg[] = "Hello, HUB75 SDF Scroller!  ";
    sdf_text_t *txt = sdf_text_create(font, msg);
    txt->size_px = size_px;
    txt->color = color;
    txt->alpha = 254;
    txt->y = 2.0f;
    txt->x = 0.0f;
    txt->dir_x = -1.0f;
    txt->dir_y = 0.0f;
    txt->speed = 255.0f;
    txt->softness = 0.09f;
    txt->effects.weight         = weight;
    // txt->effects.glow_color     = (RGBA){192, 232, 245, 255};
    // txt->effects.glow_radius    = Normal_clamp(1.0f);
    txt->effects.outline_width  = Normal_clamp(outline_width); // set >0 to enable outline
    txt->effects.outline_color  = outline_color;
    txt->effects.outline_smooth = Normal_clamp(0.1f);
    txt->valign = SDF_VALIGN_BOTTOM;
         

    sdf_text_update(txt, 64);
    image_buffer_t *image = image_buffer_new(txt->dimensions.x, txt->dimensions.y);

    sdf_text_render(txt, (uint8_t*)image->data, txt->dimensions.x, txt->dimensions.y, 4, 0.0f);

    /* Cleanup */
    sdf_text_destroy(txt);
    sdf_font_free(font);
    return image;
}




// ---------- full renderer ----------
/**
 * @brief Primary rendering function that sets up DRM/GBM and EGL/GL contexts, compiles
 *              the shader program from a shadertoy file, sets up vertex buffers and textures,
 *              and enters the main rendering loop. It handles asynchronous readback using PBOs (if enabled)
 *              or CPU readback, and adjusts frame rate dynamically. This function is executed in a separate
 *              thread and uses the provided hub75_display_t for configuration.
 *
 * @param arg A pointer to a hub75_display_t structure containing rendering parameters such as shader file,
 *          dimensions, and FPS settings.
 */
void *main_render_shader(void *arg)
{
    hub75_display_t *scene = (hub75_display_t *)arg;
    debug(" ~~ render shader 2: %s\n", scene->shader_file);
    cpu_pin_thread(2); // make sure we don't run

    scene->stride = 4;

    // Initialize GPU context
    gpu_context_t *gpu_ctx = init_gpu_context_pbuffer(scene->width, scene->height);

    // program and quad
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);

    static const GLfloat verts[] = {
        -1.f,  1.f, 0.f,  // top-left
        -1.f, -1.f, 0.f,  // bottom-left
         1.f,  1.f, 0.f,  // top-right
         1.f, -1.f, 0.f   // bottom-right
    };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    GLuint pos_attrib = (GLuint)glGetAttribLocation(program, "position");
    printf("vertex pos_attrib=%d\n", (int)pos_attrib);
    if ((GLint)pos_attrib >= 0)
    {
        glEnableVertexAttribArray((GLuint)pos_attrib);
        //glVertexAttribPointer((GLuint)pos_attrib, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
        glVertexAttribPointer((GLuint)pos_attrib, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }

    // IMPORTANT: ensure tight unpack before any texture uploads
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Optional textures (PNG or JPEG)
    bind_tex(scene->shader_file, "channel0", 0);
    bind_tex(scene->shader_file, "channel1", 1);
    GLint c0_loc = glGetUniformLocation(program, "iChannel0");
    GLint c1_loc = glGetUniformLocation(program, "iChannel1");
    glUniform1i(c0_loc, 0);
    glUniform1i(c1_loc, 1);

    // uniforms
    GLint res_loc = glGetUniformLocation(program, "iResolution");
    printf("res loc: %d\n", (int)res_loc);
    glUniform3f(res_loc, scene->width, scene->height, 0);

    // GL state for readbacks
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, scene->width, scene->height);

    const uint16_t width = scene->width;
    const uint16_t height = scene->height;


    RGBA white = {255, 255, 255, 0};
    RGBA black = {0, 0, 255, 0};
    image_buffer_t *image = text_render_sdf("assets/roboto", "You are pretty good at this   ", 128.0f, white, 1.0f, 0.1f, black);

    // main loop
    while (scene->do_render)
    {
        // update the time uniforms
        update_uniforms(program, scene->time_scale);

        // ensure we draw/read from our FBO
        glBindFramebuffer(GL_FRAMEBUFFER, gpu_ctx->fbo);
        glViewport(0, 0, scene->width, scene->height);

        // draw
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glFinish();

        // yield until the frame is complete...
        /*
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
        */

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
// be explicit about read buffer in GLES3
#ifdef GL_COLOR_ATTACHMENT0
        glReadBuffer(GL_COLOR_ATTACHMENT0);
#endif
        // pull the pixels back to the CPU using the CPU

        // present after queuing readback
        // eglSwapBuffers(gpu_ctx->display, gpu_ctx->egl_surface);

        int32_t xpos = 0;
        uint32_t *dst = (uint32_t *)spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
        if (dst)
        {
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, dst);

            // Create image_buffer_t wrapper for destination
            image_buffer_t dst_buffer = {
                .dimensions = {scene->width, scene->height},
                .row_stride = scene->width * 4,
                .data = (RGBA*)dst
            };

            // Calculate rectangles for compositing
            vec4 dst_rect = {(float)MAX(0, xpos), 16.0f, (float)scene->width, 168.0f};
            int32_t spos = (xpos < 0) ? abs(xpos) : 0;
            vec4 src_rect = {(float)MIN(image->dimensions.x, spos), 0.0f, 
                             (float)image->dimensions.x, (float)image->dimensions.y};
            
            composite_rgba_over_rgba(&dst_buffer, image, dst_rect, src_rect);
            spsc_push_ptr_commit(scene->ring_buf_mapper);
        }
        else
        {
            debug("dropping frame from OpenGL\n");
        }

        calculate_fps(scene->fps, scene->show_fps);
        //long slept = calculate_fps(scene->fps, scene->show_fps);

    }

    debug(" ## GPU render thread exiting...\n");

    // cleanup
    glDeleteBuffers(1, &vbo);
    cleanup_gpu_context(gpu_ctx);

    return NULL;
}




