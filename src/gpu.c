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
#include <math.h>
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
#include "transformers.h"

/* Image loading is handled by util.c (image_read_rgba8) which supports PNG and JPEG */

/**
 * Read next record from stdin, delimited by blank line or EOF.
 * Returns malloc'd string (caller frees), or NULL on EOF.
 * Newlines within the record are replaced with spaces.
 */
static char *read_stdin_record(void) {
    char *buf = NULL;
    size_t total = 0;
    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '\n' || (line[0] == '\r' && line[1] == '\n')) {
            if (total > 0) break; // end of record
            continue;             // skip leading blank lines
        }
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = ' ';
        buf = realloc(buf, total + len + 1);
        memcpy(buf + total, line, len);
        total += len;
    }
    if (buf) buf[total] = '\0';
    return buf;
}

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
 * @brief GLSL type names and their zero-initializer values for GLES compatibility.
 * Desktop GLSL drivers zero-initialize variables; GLES does not guarantee this.
 */
typedef struct {
    const char *name;
    const char *zero;
    size_t name_len;
} glsl_type_zero_t;

static const glsl_type_zero_t glsl_types[] = {
    { "float",  "0.",        5 },
    { "int",    "0",         3 },
    { "uint",   "0u",        4 },
    { "bool",   "false",     4 },
    { "vec2",   "vec2(0)",   4 },
    { "vec3",   "vec3(0)",   4 },
    { "vec4",   "vec4(0)",   4 },
    { "ivec2",  "ivec2(0)",  5 },
    { "ivec3",  "ivec3(0)",  5 },
    { "ivec4",  "ivec4(0)",  5 },
    { "uvec2",  "uvec2(0)",  5 },
    { "uvec3",  "uvec3(0)",  5 },
    { "uvec4",  "uvec4(0)",  5 },
    { "mat2",   "mat2(0.)",  4 },
    { "mat3",   "mat3(0.)",  4 },
    { "mat4",   "mat4(0.)",  4 },
};
#define GLSL_TYPE_COUNT (sizeof(glsl_types) / sizeof(glsl_types[0]))

/**
 * @brief Check if character is a GLSL identifier character (alphanumeric or underscore)
 */
static inline int is_ident(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

/**
 * @brief Match a GLSL type keyword at position p, returning the type entry or NULL.
 * Requires a word boundary after the keyword (not followed by alphanumeric or underscore).
 */
static const glsl_type_zero_t *match_type(const char *p) {
    for (int i = 0; i < (int)GLSL_TYPE_COUNT; i++) {
        if (strncmp(p, glsl_types[i].name, glsl_types[i].name_len) == 0 &&
            !is_ident(p[glsl_types[i].name_len])) {
            return &glsl_types[i];
        }
    }
    return NULL;
}

/**
 * @brief Preprocess GLSL source for GLES compatibility.
 *
 * Desktop GLSL drivers zero-initialize uninitialized variables, but GLES does not.
 * This function scans for variable declarations and adds explicit zero initializers
 * where missing, allowing Shadertoy shaders to be copy-pasted and run on GLES.
 *
 * Handles patterns like:
 *   float i,d,s,m,k,t = iTime;  ->  float i=0.,d=0.,s=0.,m=0.,k=0.,t = iTime;
 *   vec3 p;                      ->  vec3 p=vec3(0);
 *   int i;                       ->  int i=0;
 *
 * Correctly skips function declarations, function parameters, struct members,
 * comments, preprocessor directives, and already-initialized variables.
 *
 * @param src  The original shader source (null-terminated).
 * @return     A new malloc'd string with preprocessed source. Caller must free.
 */
static char *shader_preprocess(const char *src)
{
    size_t src_len = strlen(src);
    // generous allocation: each var might get "=vec4(0)" (~8 chars) added
    size_t out_cap = src_len * 3 + 1024;
    char *out = (char *)malloc(out_cap);
    if (!out) return NULL;
    size_t o = 0;

    int in_block_comment = 0;
    int struct_depth = 0;
    char last_boundary = ';'; // treat start of source as a statement boundary

    const char *p = src;
    while (*p) {
        // --- block comment state ---
        if (in_block_comment) {
            if (p[0] == '*' && p[1] == '/') {
                out[o++] = *p++; out[o++] = *p++;
                in_block_comment = 0;
            } else {
                out[o++] = *p++;
            }
            continue;
        }
        if (p[0] == '/' && p[1] == '*') {
            out[o++] = *p++; out[o++] = *p++;
            in_block_comment = 1;
            continue;
        }

        // --- line comment: copy to end of line ---
        if (p[0] == '/' && p[1] == '/') {
            while (*p && *p != '\n') out[o++] = *p++;
            continue;
        }

        // --- preprocessor directive: copy to end of line ---
        if (*p == '#' && (last_boundary == ';' || last_boundary == '{' || last_boundary == '}' || last_boundary == '\n')) {
            while (*p && *p != '\n') out[o++] = *p++;
            continue;
        }

        // --- track statement boundary characters ---
        if (*p == ';' || *p == '{' || *p == '}') {
            // track struct depth: '}' after struct opens closes struct body
            if (*p == '{' && struct_depth > 0) struct_depth++;
            if (*p == '}' && struct_depth > 0) struct_depth--;
            last_boundary = *p;
            out[o++] = *p++;
            continue;
        }

        // --- detect 'struct' keyword to track struct bodies ---
        if (strncmp(p, "struct", 6) == 0 && !is_ident(p[6])) {
            // mark that the next '{' opens a struct body
            struct_depth = 1;
            // but we haven't hit '{' yet, so set to a marker value
            // struct_depth will be incremented to 1 when we see '{'
            struct_depth = 0;
            // write "struct" and find the '{' naturally
            // Actually: set a flag, and when we see '{' with this flag, enter struct mode
            // Simpler: just scan ahead to check if '{' comes before ';'
            const char *ahead = p + 6;
            while (*ahead == ' ' || *ahead == '\t' || *ahead == '\n' || is_ident(*ahead)) ahead++;
            if (*ahead == '{') struct_depth = 1; // will be incremented when we hit the '{'
            // temporarily set to 1 so that when '{' is hit, it increments to the right depth
            // Actually let's just set it so '{' pushes it up. Set struct_depth = 1 now,
            // the '{' handler above will increment it to 2, but we want depth 1 while inside.
            // Simpler approach: set struct_depth=1 to mean "we're about to enter a struct"
            // and '{' increments it. '}' decrements it. When it goes to 0, we're out.
            if (*ahead == '{') struct_depth = 1;
            else struct_depth = 0;
            // copy the struct keyword
            for (int j = 0; j < 6; j++) out[o++] = *p++;
            continue;
        }

        // --- whitespace/newlines: track boundary and copy ---
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
            if (*p == '\n') last_boundary = '\n';
            out[o++] = *p++;
            continue;
        }

        // --- try to match a type keyword for variable declaration ---
        if (struct_depth == 0 &&
            (last_boundary == ';' || last_boundary == '{' || last_boundary == '}' || last_boundary == '\n'))
        {
            const glsl_type_zero_t *type = match_type(p);
            if (type) {
                // look past type keyword + whitespace to find the identifier
                const char *after_type = p + type->name_len;
                while (*after_type == ' ' || *after_type == '\t') after_type++;

                // must start with a letter or underscore (identifier)
                if (is_ident(*after_type) && !(*after_type >= '0' && *after_type <= '9')) {
                    // skip the identifier
                    const char *after_id = after_type;
                    while (is_ident(*after_id)) after_id++;
                    // skip whitespace after identifier
                    const char *check = after_id;
                    while (*check == ' ' || *check == '\t') check++;

                    // if '(' follows, it's a function declaration — skip
                    // if ';', ',', '=', or '[' follows, it's a variable declaration
                    if (*check != '(') {
                        // find the terminating ';' (tracking paren depth for nested expressions)
                        const char *semi = after_type;
                        int pdepth = 0;
                        while (*semi && !(*semi == ';' && pdepth == 0)) {
                            if (*semi == '(') pdepth++;
                            else if (*semi == ')') pdepth--;
                            semi++;
                        }
                        if (*semi == ';') {
                            // write the type keyword
                            for (size_t j = 0; j < type->name_len; j++) out[o++] = *p++;
                            out[o++] = ' ';
                            // skip whitespace between type and declarator list
                            while (*p == ' ' || *p == '\t') p++;

                            // process each comma-separated declarator
                            const char *end = semi;  // points at ';'
                            while (p < end) {
                                // find the end of this declarator: next ',' at paren_depth 0, or end
                                const char *dstart = p;
                                const char *dend = p;
                                pdepth = 0;
                                while (dend < end) {
                                    if (*dend == '(') pdepth++;
                                    else if (*dend == ')') pdepth--;
                                    else if (*dend == ',' && pdepth == 0) break;
                                    dend++;
                                }

                                // check if this declarator has '=' at paren_depth 0
                                int has_init = 0;
                                pdepth = 0;
                                for (const char *c = dstart; c < dend; c++) {
                                    if (*c == '(') pdepth++;
                                    else if (*c == ')') pdepth--;
                                    else if (*c == '=' && pdepth == 0) { has_init = 1; break; }
                                }

                                // copy the declarator text
                                // trim leading whitespace for clean output
                                const char *dtrim = dstart;
                                while (dtrim < dend && (*dtrim == ' ' || *dtrim == '\t' || *dtrim == '\n' || *dtrim == '\r')) dtrim++;
                                // trim trailing whitespace
                                const char *dtrim_end = dend;
                                while (dtrim_end > dtrim && (dtrim_end[-1] == ' ' || dtrim_end[-1] == '\t' || dtrim_end[-1] == '\n' || dtrim_end[-1] == '\r')) dtrim_end--;

                                // check for array brackets — don't zero-init arrays
                                int is_array = 0;
                                for (const char *c = dtrim; c < dtrim_end; c++) {
                                    if (*c == '[') { is_array = 1; break; }
                                }

                                // write the declarator
                                for (const char *c = dtrim; c < dtrim_end; c++) out[o++] = *c;

                                // inject zero initializer if missing
                                if (!has_init && !is_array && dtrim < dtrim_end) {
                                    out[o++] = '=';
                                    for (const char *z = type->zero; *z; z++) out[o++] = *z;
                                }

                                // write comma separator if not last
                                if (dend < end && *dend == ',') {
                                    out[o++] = ',';
                                    dend++; // skip the comma
                                }
                                p = dend;
                            }

                            out[o++] = ';';
                            p = semi + 1; // skip past ';'
                            last_boundary = ';';
                            continue;
                        }
                    }
                }
            }
        }

        // --- default: copy character ---
        last_boundary = 0; // not a boundary character
        out[o++] = *p++;
    }

    out[o] = '\0';
    return out;
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

    // preprocess: zero-initialize uninitialized variables for GLES compatibility
    char *preprocessed = shader_preprocess(src);
    if (preprocessed) {
        SAFE_FREE(src);
        src = preprocessed;
        filesize = strlen(src);
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
    txt->valign = SDF_VALIGN_BASELINE;

    sdf_text_update(txt, 64);
    image_buffer_t *image = image_buffer_new(txt->dimensions.x, txt->dimensions.y);

    sdf_text_render(txt, (uint8_t*)image->data, txt->dimensions.x, txt->dimensions.y, image->row_stride, 0.0f);

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


    // If the image mapper is flip or mirror_flip, we handle the vertical flip
    // here (before text compositing) so text renders right-side-up. Clear the
    // mapper so the mapper thread doesn't double-flip.
    bool gpu_flip_rows = false;
    uint8_t *flip_row_buf = NULL;
    if (scene->image_mapper == flip_mapper || scene->image_mapper == mirror_flip_mapper) {
        gpu_flip_rows = true;
        flip_row_buf = malloc((size_t)width * 4);
        if (scene->image_mapper == flip_mapper) {
            scene->image_mapper = NULL;
        } else {
            scene->image_mapper = mirror_mapper;
        }
    }

    // Text overlay: only set up if CLI provided -e
    image_buffer_t *text_image = NULL;
    sdf_font_t *text_font = NULL;
    sdf_text_t *txt = NULL;
    float scroll_speed = (scene->text_overlay) ? scene->text_overlay->scroll_speed : 128.0f;
    float scroll_wrap_mod = 0.0f;
    // If stdin_text mode, read the first record from stdin
    if (scene->text_overlay && scene->text_overlay->stdin_text && !scene->text_overlay->text) {
        char *first = read_stdin_record();
        if (first) {
            scene->text_overlay->text = first;
        }
    }
    if (scene->text_overlay && scene->text_overlay->text) {
        char font_path[128];
        snprintf(font_path, sizeof(font_path), "assets/%s", scene->text_overlay->font_name);
        text_font = sdf_font_load_scaled(font_path, scene->text_overlay->font_size);
        if (text_font) {
            txt = sdf_text_create(text_font, scene->text_overlay->text);
            txt->size_px = scene->text_overlay->font_size;
            txt->color = scene->text_overlay->color;
            txt->alpha = 254;
            txt->x = 0.0f;
            txt->y = 0.0f;
            txt->dir_x = 0.0f;
            txt->dir_y = 0.0f;
            txt->speed = 0.0f;
            txt->softness = 0.09f;
            txt->effects.weight = scene->text_overlay->weight;
            txt->effects.outline_color = scene->text_overlay->outline_color;
            txt->effects.outline_width = Normal_clamp(scene->text_overlay->outline_width);
            txt->valign = SDF_VALIGN_BASELINE;

            sdf_text_update(txt, scene->width);

            //fprintf(stderr, "[TEXT-DBG] txt after update: x=%.1f y=%.1f x0=%.1f y0=%.1f speed=%.1f dir=(%.1f,%.1f) dims=%dx%d wrap_mod=%.3f\n", txt->x, txt->y, txt->x0, txt->y0, txt->speed, txt->dir_x, txt->dir_y, txt->dimensions.x, txt->dimensions.y, txt->wrap_mod);

            text_image = image_buffer_new(txt->dimensions.x, txt->dimensions.y);
            sdf_text_render(txt, (uint8_t*)text_image->data,
                txt->dimensions.x, txt->dimensions.y, text_image->row_stride, 0.0f);

            // Check if any non-zero pixels were rendered
            uint32_t nonzero = 0;
            for (int i = 0; i < txt->dimensions.x * txt->dimensions.y; i++) {
                uint32_t px = ((uint32_t*)text_image->data)[i];
                if (px != 0) nonzero++;
            }
            // fprintf(stderr, "[TEXT-DBG] text_image %dx%d, non-zero pixels: %u / %d\n", text_image->dimensions.x, text_image->dimensions.y, nonzero, txt->dimensions.x * txt->dimensions.y);

            // Scrolling: text enters from right, exits left, then wraps
            // Use text_image dimensions (allocation size) not txt->dimensions
            // (which sdf_text_render may have shrunk to actual rendered bounds)
            scroll_wrap_mod = (float)(scene->width + text_image->dimensions.x) / scroll_speed;
            // fprintf(stderr, "[TEXT-DBG] scroll_speed=%.1f scroll_wrap_mod=%.3f\n", scroll_speed, scroll_wrap_mod);
        }
    }

    float this_time = 0.0f;
    float scroll_time_offset = 0.0f;  // subtracted from this_time to reset scroll on text cycle
    float prev_wrap_time = 0.0f;
    int dbg_frame = 0;

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

        glPixelStorei(GL_PACK_ALIGNMENT, 1);
#ifdef GL_COLOR_ATTACHMENT0
        glReadBuffer(GL_COLOR_ATTACHMENT0);
#endif

        uint32_t *dst = (uint32_t *)spsc_push_ptr_begin(scene->ring_buf_mapper, 200);
        if (dst)
        {
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, dst);

            // OpenGL renders Y-flipped (origin at bottom-left). Flip rows here
            // so the shader is right-side-up before text compositing. This means
            // the mapper thread's flip_mapper becomes a no-op for GPU output and
            // text composites in normal screen coordinates.
            if (gpu_flip_rows) {
                const size_t row_bytes = (size_t)width * 4;
                uint8_t *top = (uint8_t *)dst;
                uint8_t *bot = top + (height - 1) * row_bytes;
                while (top < bot) {
                    // swap using the scratch row buffer
                    memcpy(flip_row_buf, top, row_bytes);
                    memcpy(top, bot, row_bytes);
                    memcpy(bot, flip_row_buf, row_bytes);
                    top += row_bytes;
                    bot -= row_bytes;
                }
            }

            // Composite text overlay if configured
            if (text_image) {
                image_buffer_t dst_buffer = {
                    .dimensions = {scene->width, scene->height},
                    .row_stride = scene->width * 4,
                    .data = (RGBA*)dst
                };

                // Scroll from right to left: xpos starts at scene->width, decreases
                float scroll_time = this_time - scroll_time_offset;
                float wrap_time = fmodf(scroll_time, scroll_wrap_mod);

                // Detect scroll wrap and cycle to next stdin record
                if (wrap_time < prev_wrap_time && scene->text_overlay->stdin_text && txt) {
                    char *new_text = read_stdin_record();
                    if (new_text) {
                        sdf_text_set_text(txt, new_text);
                        sdf_text_update(txt, scene->width);

                        int new_w = txt->dimensions.x;
                        int new_h = txt->dimensions.y;
                        if (new_w != text_image->dimensions.x || new_h != text_image->dimensions.y) {
                            free(text_image);
                            text_image = image_buffer_new(new_w, new_h);
                        } else {
                            memset(text_image->data, 0, (size_t)new_h * text_image->row_stride);
                        }

                        txt->y = 0;
                        sdf_text_render(txt, (uint8_t*)text_image->data, new_w, new_h,
                                        text_image->row_stride, 0.0f);
                        scroll_wrap_mod = (float)(scene->width + text_image->dimensions.x) / scroll_speed;
                        scroll_time_offset = this_time;  // reset scroll origin to now
                        wrap_time = 0.0f;
                        free(new_text);
                    }
                }
                prev_wrap_time = wrap_time;
                float xpos = (float)scene->width - scroll_speed * wrap_time;

                // Compute matching src/dst widths so the compositor
                // does a 1:1 blit with no scaling
                float dx0 = fmaxf(0, xpos);
                float sx0 = (xpos < 0) ? -xpos : 0;
                float visible_w = fminf((float)scene->width - dx0,
                                        (float)text_image->dimensions.x - sx0);

                float dy0 = (float)scene->text_overlay->y_pos;
                float dy1 = fminf((float)scene->height, dy0 + (float)text_image->dimensions.y);
                vec4 dst_rect = {dx0, dy0, dx0 + visible_w, dy1};
                vec4 src_rect = {sx0, 0.0f, sx0 + visible_w,
                                 (float)text_image->dimensions.y};

                if (dbg_frame < 5 || (dbg_frame % 90 == 0)) {
                    // fprintf(stderr, "[TEXT-DBG] frame=%d xpos=%.1f visible_w=%.0f dst=(%.0f,%.0f,%.0f,%.0f) src=(%.0f,%.0f,%.0f,%.0f)\n", dbg_frame, xpos, visible_w, dst_rect.x, dst_rect.y, dst_rect.z, dst_rect.w, src_rect.x, src_rect.y, src_rect.z, src_rect.w);
                }
                dbg_frame++;

                if (visible_w > 0) {
                    composite_rgba_over_rgba(&dst_buffer, text_image, dst_rect, src_rect);
                }
            }

            spsc_push_ptr_commit(scene->ring_buf_mapper);
        }
        else
        {
            debug("dropping frame from OpenGL\n");
        }

        this_time = calculate_fps(scene->fps, scene->show_fps);
    }

    debug(" ## GPU render thread exiting...\n");

    // cleanup text overlay
    if (text_image) free(text_image);
    if (txt) sdf_text_destroy(txt);
    if (text_font) sdf_font_free(text_font);
    free(flip_row_buf);

    // cleanup
    glDeleteBuffers(1, &vbo);
    cleanup_gpu_context(gpu_ctx);

    return NULL;
}




