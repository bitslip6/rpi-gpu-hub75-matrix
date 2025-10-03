#define _GNU_SOURCE
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

#include <stdlib.h>

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



// helpers: fixed-point RGB scaler for an axis-aligned rectangle, RGBA8
static inline void scale_rect_rgb_q8(uint8_t *__restrict__ pixels,
                                     int width, int height,
                                     int x0, int y0, int w, int h,
                                     uint16_t red_q8, uint16_t green_q8, uint16_t blue_q8) // 256 == 1.0
{
    if (x0 < 0) { w += x0; x0 = 0; }
    if (y0 < 0) { h += y0; y0 = 0; }
    if (x0 + w > width)  w = width  - x0;
    if (y0 + h > height) h = height - y0;

    const int row_stride = width * 4;
    uint8_t *row = pixels + y0 * row_stride + x0 * 4;

    for (int y = 0; y < h; y++) {
        uint8_t *p = row;
        for (int x = 0; x < w; x++, p += 4) {
            // R,G,B scaled, A unchanged
            p[0] = (uint8_t)((p[0] * red_q8) >> 8);
            p[1] = (uint8_t)((p[1] * green_q8) >> 8);
            p[2] = (uint8_t)((p[2] * blue_q8) >> 8);
        }
        row += row_stride;
    }
}

// apply a 3x3 per-panel brightness table to a 192x192 RGBA image
// panel_q8[row][col] in Q8, 256 == 1.0, 179 ~ 0.70, etc.
static inline void apply_panel_brightness_q8(uint8_t *__restrict__ pixels,
                                             scene_info *scene)
{
    for (int py = 0; py < scene->num_ports; py++) {
        for (int px = 0; px < scene->num_chains; px++) {
            int idx = (py * scene->num_chains) + px;
            int panel_type = scene->panel_types[idx] - 1;
            if (panel_type < 0 || panel_type >= scene->num_panel_types) {
                continue;
            }

            uint8_t r = scene->panel_scale[panel_type].red_q8;
            uint8_t g = scene->panel_scale[panel_type].green_q8;
            uint8_t b = scene->panel_scale[panel_type].blue_q8;

            scale_rect_rgb_q8(pixels, scene->width, scene->height,
                              px * scene->panel_width, py * scene->panel_height,
                              scene->panel_width, scene->panel_height,
                              r, g, b);
        }
    }
}




/**
 * @brief render the shadertoy compatible shader source code in the 
 * file pointed to at scene->shader_file
 * 
 * exits if shader is unable to be rendered
 * 
 * loop exits and memory is freed if/when scene->do_render becomes false
 * 
 * frame delay is adaptive and updates to current scene->fps on each frame update
 * 
 * 
 * @param arg pointer to the current scene_info object
 */
void *render_shader(void *arg) {
    scene_info *scene = (scene_info*)arg;
    debug("render shader %s\n", scene->shader_file);

    // Open a file descriptor to the DRM device
    int fd = open("/dev/dri/card0", O_RDWR);
    if (fd < 0) {
        die("Failed to open DRM device /dev/dri/card0\n");
    }


	static const EGLint context_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};

    // Create GBM device and surface
    struct gbm_device *gbm = gbm_create_device(fd);
    struct gbm_surface *surface = gbm_surface_create(gbm, scene->width, scene->height, GBM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING);

    // Initialize EGL
    EGLDisplay display = eglGetDisplay(gbm);
    eglInitialize(display, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);

    // Create EGL context and surface.
    // TODO experiment with 565 color
    EGLConfig config;
    EGLint num_configs;
    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    eglChooseConfig(display, attribs, &config, 1, &num_configs);

    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    EGLSurface egl_surface = eglCreateWindowSurface(display, config, (EGLNativeWindowType)surface, NULL);
    eglMakeCurrent(display, egl_surface, egl_surface, context);

    // Set up OpenGL ES
    printf("compiling GLSL shader...\n");
    GLuint program = create_shadertoy_program(scene->shader_file);
    glUseProgram(program);

    // Define a square with two triangles. This is a rendering surface for our fragment shader
    GLfloat vertices[] = {
        -1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f
    };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLint posAttrib = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 0, 0);


    // setup the timers for frame delays
    struct timespec start_time, end_time, orig_time;
    // uint32_t frame_time_us = 1000000 / scene->fps;
    size_t image_buf_sz = scene->width * (scene->height) * sizeof(uint32_t);

    // RGBA format (4 bytes per pixel)
    GLubyte *restrict pixelsA __attribute__((aligned(16))) = (GLubyte*)malloc(image_buf_sz*(MAX(scene->motion_blur_frames+1,10)));
    if (pixelsA == NULL) {
        die("unable to allocate %d bytes memory for shader frames...\n", image_buf_sz * (MAX(scene->motion_blur_frames+1,10)));
    }
    GLubyte *pixels = pixelsA;

    // pointer to the current motion blur buffer
    GLubyte *pixelsO = pixelsA+(image_buf_sz * scene->motion_blur_frames+1);

    // uniforms point to information we will pass to the GLSL shader
    GLint timeLocation = glGetUniformLocation(program, "iTime");
    GLint timeDeltaLocation = glGetUniformLocation(program, "iTimeDelta");
    GLint frameLocation = glGetUniformLocation(program, "iFrame");
    GLint resLocation = glGetUniformLocation(program, "iResolution");
    GLint chan0Location = glGetUniformLocation(program, "iChannel0");
    GLint chan1Location = glGetUniformLocation(program, "iChannel1");
    


    // some variables for each frame iteration
    float motion_blur[scene->motion_blur_frames+1];
    float time1, time2 = 0.0f;
    unsigned long frame= 0;
    int frame_num = 0;


    // calculate motion blur frame weights  (decreasing frame weights)
    float sum = 0;
    for (int i = 0; i < scene->motion_blur_frames; i++) {
        motion_blur[i] = powf(0.5f, i);  // Exponents symmetric around zero
        sum += motion_blur[i];
    }

    // now average each frame so that the sum of all frames adds to 1.0
    for (int i = 0; i < scene->motion_blur_frames; i++) {
        motion_blur[i] /= sum;
    }


    GLuint texture0 = 0, texture1 = 0;
    char *chan0 = change_file_extension(scene->shader_file, "channel0");
    if (access(chan0, R_OK) == 0) {
        printf("loading texture %s\n", chan0);
        texture0 = load_texture(chan0);
        if (texture0 == 0) {
            die("unable to load texture '%s'\n", chan0);
        }
    }

    char *chan1 = change_file_extension(scene->shader_file, "channel1");
    if (access(chan1, R_OK) == 0) {
        printf("loading texture %s\n", chan1);
        texture1 = load_texture(chan1);
        if (texture1 == 0) {
            die("unable to load texture '%s'\n", chan1);
        }
    }



    //printf("GLSL shader compiled. rendering...\n");
    // loop until do_render is false. most likely never exit...
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    clock_gettime(CLOCK_MONOTONIC, &orig_time);
    //float level = scene->brightness * 
    while(scene->do_render) {
        frame++;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        time1 = (end_time.tv_sec - orig_time.tv_sec) + (end_time.tv_nsec - orig_time.tv_nsec) / 1000000000.0f;
        time2 = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0f;
        glUseProgram(program);

        if (texture0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture0);

            if (texture1) {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, texture1);
            }
        }

        glUniform1f(timeLocation, time1);
        glUniform1f(timeDeltaLocation, time2);
        glUniform1f(frameLocation, frame);
        glUniform1i(chan0Location, 0);
        glUniform1i(chan1Location, 1);
        glUniform3f(resLocation, scene->width, (scene->height), 0);

        // Render
        glViewport(0, 0, scene->width, scene->height);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        eglSwapBuffers(display, egl_surface);

        // switch between pixels buffers A-F based on frame number
        pixels = pixelsA + (frame_num * image_buf_sz);
        glReadPixels(0, 0, scene->width, scene->height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

        if (scene->panel_types[0] != 0) {
            apply_panel_brightness_q8(pixels, scene);
        }
            

        // apply motion blur in the CPU
        if (scene->motion_blur_frames > 0) {
            for (int i = 0; i < scene->width * scene->height * 4; i++) {
                float accum = 0;
                for (int f = 0; f < scene->motion_blur_frames; f++) {
                    GLubyte *frame_idx = pixelsA + (((f + frame_num) % scene->motion_blur_frames)* image_buf_sz);
                    accum += ((float)(frame_idx[i])) * motion_blur[(f + frame_num) % scene->motion_blur_frames];
                }

                pixelsO[i] = (uint8_t)(accum);
            }
            scene->bcm_mapper(scene, pixelsO);
            frame_num = frame % scene->motion_blur_frames;
        }
        // skip motion blur ....
        else {
            scene->bcm_mapper(scene, pixels);
        }

        // calculate the current FPS and delay to achieve fram rate
        calculate_fps(scene->fps, scene->show_fps);
    }


    // Cleanup
    glDeleteBuffers(1, &vbo);
    eglDestroySurface(display, egl_surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    gbm_surface_destroy(surface);
    gbm_device_destroy(gbm);

    free(pixelsA);
    close(fd);
    return NULL;
}

