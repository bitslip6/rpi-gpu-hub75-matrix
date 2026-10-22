#define _GNU_SOURCE
/* Recoverable media adapter shared by native runtime and Python. It does not
 * call the legacy CLI shader/video loops (which retain fatal CLI policies). */
#include "runtime_internal.h"
#include "util.h"
#include "video_fit.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <gbm.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern const char *shadertoy_header, *vertex_shader_source;
extern char *h75_shader_preprocess(const char *src);
struct h75_media {
    bool video, flushing, shader_claim, preserve_aspect;
    uint32_t width, height, frame_index;
    const atomic_bool *stop;
    int fd, stream;
    struct gbm_device *gbm;
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    GLuint program, texture, fbo, vbo, channels[4];
    uint8_t *pixels;
    AVFormatContext *format;
    AVCodecContext *codec;
    AVFrame *frame;
    AVPacket *packet;
    struct SwsContext *scaler;
    int64_t first_pts;
    float fallback_fps, last_pts;
};
static atomic_bool shader_active = false;
static int interrupted(void *arg) {
    h75_media *m = arg;
    return atomic_load(m->stop);
}
static char *read_shader(const char *path, h75_error *e) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        h75_fail(e, H75_IO, "opening shader %s: %s", path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fileno(f), &st) || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
        st.st_size > 4 * 1024 * 1024) {
        fclose(f);
        h75_fail(e, H75_INVALID, "shader must be a regular nonempty file <=4MiB");
        return NULL;
    }
    size_t n = (size_t)st.st_size;
    char *src = malloc(n + 1);
    if (!src) {
        fclose(f);
        h75_fail(e, H75_MEMORY, "allocating shader source");
        return NULL;
    }
    if (fread(src, 1, n, f) != n) {
        free(src);
        fclose(f);
        h75_fail(e, H75_IO, "reading shader");
        return NULL;
    }
    fclose(f);
    src[n] = 0;
    if (strlen(src) != n) {
        free(src);
        h75_fail(e, H75_INVALID, "embedded NUL in shader");
        return NULL;
    }
    char *processed = h75_shader_preprocess(src);
    if (processed) {
        free(src);
        src = processed;
    }
    return src;
}
static GLuint compile(const char *src, GLenum type, h75_error *e) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[440];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        h75_fail(e, H75_INVALID, "shader compilation: %s", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}
static h75_status shader_open(h75_media *m, const char *path, h75_error *e) {
    bool expected = false;
    if (!atomic_compare_exchange_strong(&shader_active, &expected, true))
        return h75_fail(e, H75_BUSY, "one active EGL shader session is supported per process");
    m->shader_claim = true;
    char *src = read_shader(path, e);
    if (!src)
        return e ? e->code : H75_IO;
    PFNEGLGETPLATFORMDISPLAYEXTPROC platform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (platform)
        m->display = platform(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    if (m->display == EGL_NO_DISPLAY || !eglInitialize(m->display, NULL, NULL)) {
        m->display = EGL_NO_DISPLAY;
        m->fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
        if (m->fd < 0)
            m->fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
        if (m->fd >= 0)
            m->gbm = gbm_create_device(m->fd);
        if (m->gbm)
            m->display = platform ? platform(EGL_PLATFORM_GBM_KHR, m->gbm, NULL)
                                  : eglGetDisplay((EGLNativeDisplayType)m->gbm);
        if (m->display == EGL_NO_DISPLAY || !eglInitialize(m->display, NULL, NULL)) {
            free(src);
            return h75_fail(
                e, H75_IO, "EGL initialization failed (0x%x); check render-node permissions/driver",
                eglGetError());
        }
    }
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        free(src);
        return h75_fail(e, H75_IO, "binding OpenGL ES");
    }
    EGLConfig config;
    EGLint count = 0;
    EGLint attrs[] = {EGL_RENDERABLE_TYPE,
                      EGL_OPENGL_ES3_BIT,
                      EGL_SURFACE_TYPE,
                      EGL_PBUFFER_BIT,
                      EGL_RED_SIZE,
                      8,
                      EGL_GREEN_SIZE,
                      8,
                      EGL_BLUE_SIZE,
                      8,
                      EGL_ALPHA_SIZE,
                      8,
                      EGL_NONE};
    if (!eglChooseConfig(m->display, attrs, &config, 1, &count) || !count) {
        free(src);
        return h75_fail(e, H75_IO, "no compatible EGL ES3 configuration");
    }
    m->context = eglCreateContext(m->display, config, EGL_NO_CONTEXT,
                                  (EGLint[]){EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE});
    if (m->context == EGL_NO_CONTEXT) {
        free(src);
        return h75_fail(e, H75_IO, "creating ES3 context");
    }
    m->surface = eglCreatePbufferSurface(
        m->display, config,
        (EGLint[]){EGL_WIDTH, (EGLint)m->width, EGL_HEIGHT, (EGLint)m->height, EGL_NONE});
    if (m->surface == EGL_NO_SURFACE ||
        !eglMakeCurrent(m->display, m->surface, m->surface, m->context)) {
        free(src);
        return h75_fail(e, H75_IO, "creating/making EGL pbuffer current");
    }
    size_t len = strlen(src) + 8192;
    char *full = malloc(len);
    if (!full) {
        free(src);
        return h75_fail(e, H75_MEMORY, "shader header allocation");
    }
    snprintf(full, len, shadertoy_header, src);
    free(src);
    GLuint vs = compile(vertex_shader_source, GL_VERTEX_SHADER, e), fs = 0;
    if (vs)
        fs = compile(full, GL_FRAGMENT_SHADER, e);
    free(full);
    if (!vs || !fs) {
        if (vs)
            glDeleteShader(vs);
        if (fs)
            glDeleteShader(fs);
        return e ? e->code : H75_INVALID;
    }
    m->program = glCreateProgram();
    glAttachShader(m->program, vs);
    glAttachShader(m->program, fs);
    glLinkProgram(m->program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(m->program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[440];
        glGetProgramInfoLog(m->program, sizeof(log), NULL, log);
        return h75_fail(e, H75_INVALID, "shader linking: %s", log);
    }
    glUseProgram(m->program);
    const GLfloat vertices[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glGenBuffers(1, &m->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    GLint pos = glGetAttribLocation(m->program, "position");
    if (pos >= 0) {
        glEnableVertexAttribArray((GLuint)pos);
        glVertexAttribPointer((GLuint)pos, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }
    glGenTextures(1, &m->texture);
    glBindTexture(GL_TEXTURE_2D, m->texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)m->width, (GLsizei)m->height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &m->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m->texture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        return h75_fail(e, H75_IO, "incomplete shader framebuffer");
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glViewport(0, 0, (GLsizei)m->width, (GLsizei)m->height);
    glUniform3f(glGetUniformLocation(m->program, "iResolution"), (float)m->width, (float)m->height,
                1);
    for (unsigned i = 0; i < 4; i++) {
        /* Extend the CLI's sibling texture convention to all four channels. */
        const char *dot = strrchr(path, '.'), *slash = strrchr(path, '/');
        size_t base = dot && (!slash || dot > slash) ? (size_t)(dot - path) : strlen(path);
        char *texture_path = malloc(base + 16);
        if (!texture_path)
            return h75_fail(e, H75_MEMORY, "texture path");
        snprintf(texture_path, base + 16, "%.*s.channel%u", (int)base, path, i);
        uint8_t *rgba = NULL;
        int w = 1, h = 1, stride = 0;
        if (access(texture_path, R_OK) == 0) {
            if (image_read_rgba8(texture_path, &rgba, &w, &h, &stride)) {
                free(texture_path);
                return h75_fail(e, H75_IO, "cannot load shader channel texture");
            }
        }
        const uint8_t black[4] = {0, 0, 0, 255};
        glActiveTexture(GL_TEXTURE0 + i);
        glGenTextures(1, &m->channels[i]);
        glBindTexture(GL_TEXTURE_2D, m->channels[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     rgba ? rgba : black);
        free(rgba);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        free(texture_path);
        char uniform[32];
        snprintf(uniform, sizeof(uniform), "iChannelResolution[%u]", i);
        glUniform3f(glGetUniformLocation(m->program, uniform), (float)w, (float)h, 1);
        snprintf(uniform, sizeof(uniform), "iChannel%u", i);
        glUniform1i(glGetUniformLocation(m->program, uniform), (GLint)i);
    }
    m->pixels = malloc((size_t)m->width * m->height * 4);
    if (!m->pixels)
        return h75_fail(e, H75_MEMORY, "shader readback storage");
    return H75_OK;
}
static h75_status video_open(h75_media *m, const char *path, h75_error *e) {
    struct stat st;
    if (stat(path, &st) || !S_ISREG(st.st_mode))
        return h75_fail(e, H75_IO, "video must be an existing local regular file");
    m->format = avformat_alloc_context();
    if (!m->format)
        return h75_fail(e, H75_MEMORY, "format context");
    m->format->interrupt_callback = (AVIOInterruptCB){interrupted, m};
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "protocol_whitelist", "file", 0);
    int result = avformat_open_input(&m->format, path, NULL, &opts);
    av_dict_free(&opts);
    if (result < 0)
        return h75_fail(e, H75_IO, "cannot open video (%d)", result);
    if (avformat_find_stream_info(m->format, NULL) < 0)
        return h75_fail(e, H75_IO, "cannot read video stream info");
    const AVCodec *codec = NULL;
    m->stream = av_find_best_stream(m->format, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (m->stream < 0 || !codec)
        return h75_fail(e, H75_INVALID, "no supported video stream");
    m->codec = avcodec_alloc_context3(codec);
    if (!m->codec)
        return h75_fail(e, H75_MEMORY, "decoder context");
    if (avcodec_parameters_to_context(m->codec, m->format->streams[m->stream]->codecpar) < 0)
        return h75_fail(e, H75_IO, "decoder parameters");
    m->codec->thread_count = 2;
    m->codec->thread_type = FF_THREAD_FRAME;
    m->codec->max_pixels = 8192 * 8192;
    if (avcodec_open2(m->codec, codec, NULL) < 0)
        return h75_fail(e, H75_IO, "opening decoder");
    m->frame = av_frame_alloc();
    m->packet = av_packet_alloc();
    if (!m->frame || !m->packet)
        return h75_fail(e, H75_MEMORY, "decoder frame/packet");
    AVRational rate = av_guess_frame_rate(m->format, m->format->streams[m->stream], NULL);
    m->fallback_fps = (rate.num > 0 && rate.den > 0) ? (float)av_q2d(rate) : 30;
    m->first_pts = AV_NOPTS_VALUE;
    return H75_OK;
}
h75_status h75_media_open(h75_media **out, const char *path, bool video, uint32_t w, uint32_t h,
                          bool preserve_aspect, const atomic_bool *stop, h75_error *e) {
    *out = NULL;
    h75_media *m = calloc(1, sizeof(*m));
    if (!m)
        return h75_fail(e, H75_MEMORY, "allocating media context");
    m->video = video;
    m->preserve_aspect = preserve_aspect;
    m->width = w;
    m->height = h;
    m->stop = stop;
    m->fd = -1;
    h75_status s = video ? video_open(m, path, e) : shader_open(m, path, e);
    if (s) {
        h75_media_close(m);
        return s;
    }
    *out = m;
    return H75_OK;
}
int h75_media_next(h75_media *m, uint8_t *rgba, float time, float dt, float *pts, h75_error *e) {
    if (atomic_load(m->stop)) {
        h75_fail(e, H75_CANCELLED, "media stopped");
        return -1;
    }
    if (!m->video) {
        glUseProgram(m->program);
        glBindFramebuffer(GL_FRAMEBUFFER, m->fbo);
        glUniform1f(glGetUniformLocation(m->program, "iTime"), time);
        glUniform1f(glGetUniformLocation(m->program, "iGlobalTime"), time);
        glUniform1f(glGetUniformLocation(m->program, "iTimeDelta"), dt);
        glUniform1i(glGetUniformLocation(m->program, "iFrame"), (GLint)m->frame_index++);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glReadPixels(0, 0, (GLsizei)m->width, (GLsizei)m->height, GL_RGBA, GL_UNSIGNED_BYTE,
                     m->pixels);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            h75_fail(e, H75_IO, "GL render/readback error 0x%x", error);
            return -1;
        }
        /* Normalize GL's bottom-up origin before compositing text or CPU layers. */
        size_t row = (size_t)m->width * 4;
        for (uint32_t y = 0; y < m->height; y++)
            memcpy(rgba + (size_t)y * row, m->pixels + (size_t)(m->height - 1 - y) * row, row);
        *pts = -1;
        return 0;
    }
    for (;;) {
        if (atomic_load(m->stop)) {
            h75_fail(e, H75_CANCELLED, "decoder stopped");
            return -1;
        }
        int result = avcodec_receive_frame(m->codec, m->frame);
        if (result == 0) {
            int w, h, x, y;
            AVRational sar = av_guess_sample_aspect_ratio(m->format,
                                m->format->streams[m->stream], m->frame);
            video_fit(m->frame->width, m->frame->height, sar.num, sar.den,
                      (int)m->width, (int)m->height, m->preserve_aspect,
                      &w, &h, &x, &y);
            m->scaler = sws_getCachedContext(m->scaler, m->frame->width, m->frame->height,
                                             m->frame->format, w, h,
                                             AV_PIX_FMT_RGBA, SWS_BILINEAR, NULL, NULL, NULL);
            if (!m->scaler) {
                h75_fail(e, H75_MEMORY, "video scaling context");
                return -1;
            }
            /* The producer reuses buffers: clear padding on every frame and
             * keep it opaque so overlays composite over black, not stale pixels. */
            if (m->preserve_aspect) {
                memset(rgba, 0, (size_t)m->width * m->height * 4);
                for (size_t i = 3; i < (size_t)m->width * m->height * 4; i += 4)
                    rgba[i] = 255;
            }
            uint8_t *dst[] = {rgba + ((size_t)y * m->width + (size_t)x) * 4, NULL, NULL, NULL};
            int pitch[] = {(int)m->width * 4, 0, 0, 0};
            if (sws_scale(m->scaler, (const uint8_t *const *)m->frame->data, m->frame->linesize, 0,
                          m->frame->height, dst, pitch) <= 0) {
                h75_fail(e, H75_IO, "video scaling failed");
                return -1;
            }
            int64_t timestamp = m->frame->best_effort_timestamp;
            if (timestamp != AV_NOPTS_VALUE) {
                if (m->first_pts == AV_NOPTS_VALUE)
                    m->first_pts = timestamp;
                AVRational tb = m->format->streams[m->stream]->time_base;
                uint64_t ticks =
                    timestamp >= m->first_pts ? (uint64_t)timestamp - (uint64_t)m->first_pts : 0;
                *pts = (float)ticks * (float)tb.num / (float)tb.den;
            } else
                *pts = (float)m->frame_index / m->fallback_fps;
            m->last_pts = *pts;
            m->frame_index++;
            av_frame_unref(m->frame);
            return 0;
        }
        if (result == AVERROR_EOF) {
            *pts = m->last_pts + 1.0f / m->fallback_fps;
            return 1;
        }
        if (result != AVERROR(EAGAIN)) {
            h75_fail(e, H75_IO, "decoding video (%d)", result);
            return -1;
        }
        if (m->flushing) {
            h75_fail(e, H75_IO, "decoder stalled after EOF");
            return -1;
        }
        do {
            result = av_read_frame(m->format, m->packet);
            if (result < 0)
                break;
            if (m->packet->stream_index == m->stream)
                break;
            av_packet_unref(m->packet);
        } while (!atomic_load(m->stop));
        if (result == AVERROR_EOF) {
            m->flushing = true;
            result = avcodec_send_packet(m->codec, NULL);
        } else if (result < 0) {
            h75_fail(e, H75_IO, "reading video (%d)", result);
            return -1;
        } else {
            result = avcodec_send_packet(m->codec, m->packet);
            av_packet_unref(m->packet);
        }
        if (result < 0 && result != AVERROR_EOF) {
            h75_fail(e, H75_IO, "sending decoder packet (%d)", result);
            return -1;
        }
    }
}
void h75_media_close(h75_media *m) {
    if (!m)
        return;
    if (m->packet)
        av_packet_free(&m->packet);
    if (m->frame)
        av_frame_free(&m->frame);
    if (m->scaler)
        sws_freeContext(m->scaler);
    if (m->codec)
        avcodec_free_context(&m->codec);
    if (m->format)
        avformat_close_input(&m->format);
    if (m->display != EGL_NO_DISPLAY) {
        if (m->context != EGL_NO_CONTEXT && eglGetCurrentContext() == m->context) {
            if (m->vbo)
                glDeleteBuffers(1, &m->vbo);
            if (m->program)
                glDeleteProgram(m->program);
            if (m->fbo)
                glDeleteFramebuffers(1, &m->fbo);
            if (m->texture)
                glDeleteTextures(1, &m->texture);
            glDeleteTextures(4, m->channels);
            eglMakeCurrent(m->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        }
        if (m->surface != EGL_NO_SURFACE)
            eglDestroySurface(m->display, m->surface);
        if (m->context != EGL_NO_CONTEXT)
            eglDestroyContext(m->display, m->context);
        eglTerminate(m->display);
    }
    if (m->gbm)
        gbm_device_destroy(m->gbm);
    if (m->fd >= 0)
        close(m->fd);
    if (m->shader_claim)
        atomic_store(&shader_active, false);
    free(m->pixels);
    free(m);
}
