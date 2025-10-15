#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <assert.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

#include "rpihub75.h"
 
/* a dummy compute shader that does nothing */
#define COMPUTE_SHADER_SRC "          \
#version 310 es\n                                                       \
                                                                        \
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;       \
                                                                        \
void main(void) {                                                       \
   /* awesome compute code here */                                      \
}                                                                       \
"

static EGLDisplay egl_display;
static EGLContext egl_context;
static EGLSurface egl_surface; /* may be EGL_NO_SURFACE when surfaceless */

typedef struct {
    GLuint fbo;
    GLuint color_tex;
    int width;
    int height;
} offscreen_fbo_t;


static int make_fbo(offscreen_fbo_t *fb, int width, int height)
{
    fb->width = width;
    fb->height = height;

    glGenTextures(1, &fb->color_tex);
    glBindTexture(GL_TEXTURE_2D, fb->color_tex);

    /* very important: use non-mipmap filters so the texture is "complete" */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    /* RGBA8 is color-renderable in GLES 3.0 */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glGenFramebuffers(1, &fb->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fb->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, fb->color_tex, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    /* correct constant is GL_FRAMEBUFFER_COMPLETE, not GL_FRAME_BUFFER_COMPLETE */
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        return 0;
    }

    glViewport(0, 0, width, height);
    glBindTexture(GL_TEXTURE_2D, 0);
    return 1;
}


static int make_offscreen_context(int width, int height)
{

    printf("render 128\n"); fflush(stdout);
    int32_t fd = open ("/dev/dri/renderD128", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open(/dev/dri/renderD128) failed: %s\n", strerror(errno));
        return 0;
    }

    printf("gbm 1: %d\n", fd); fflush(stdout);
    printf("calling gbm_create_device...\n"); fflush(stdout);
    struct gbm_device *gbm = gbm_create_device (fd);
    printf("gbm 2: %p\n", gbm); fflush(stdout);
    if (gbm == NULL) {
        fprintf(stderr, "gbm_create_device returned NULL\n");
        close(fd);
        return 0;
    }
 
   printf("get platform display\n");
    // setup EGL from the GBM device 
    EGLDisplay egl_dpy = eglGetPlatformDisplay (EGL_PLATFORM_GBM_MESA, gbm, NULL);
    assert (egl_dpy != NULL);
    /* record the display we just initialized for later calls */
    egl_display = egl_dpy;
   assert (egl_dpy != NULL);
 
   printf("egl init\n");
   bool res = eglInitialize (egl_dpy, NULL, NULL);
   if (!res) {
         fprintf(stderr, "eglInitialize failed: %d\n", eglGetError());
         eglGetError(); // clear error
   }
   assert (res);

   /*
    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) { printf("EGL NO DISPLAY\n"); return 0; }
    if (!eglInitialize(egl_display, NULL, NULL)) { printf("EGL initialization failed\n"); return 0; }
    */
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLConfig cfg;
    EGLint n = 0;
    const EGLint cfg_attrs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,   /* fallback if surfaceless unsupported */
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_ALPHA_SIZE,      8,
        EGL_NONE
    };
    printf("egl attrs 1\n");
    if (!eglChooseConfig(egl_display, cfg_attrs, &cfg, 1, &n) || n == 0) { printf("cant set attributes\n"); return 0; }
 
    printf("egl attrs 2\n");
    const EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    egl_context = eglCreateContext(egl_display, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (egl_context == EGL_NO_CONTEXT) { printf("EGL context creation failed\n"); return 0; }

    printf("egl surface\n");
    /* try surfaceless first if supported */
    egl_surface = EGL_NO_SURFACE;
#ifdef EGL_KHR_surfaceless_context
    if (eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context)) {
        printf("no surface success\n");
        return 1;
    }

#endif

    printf("egl pb attrs\n");
    /* fallback to a tiny pbuffer (not used for rendering) */
    const EGLint pb_attrs[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    egl_surface = eglCreatePbufferSurface(egl_display, cfg, pb_attrs);
    if (egl_surface == EGL_NO_SURFACE) { printf("EGL pbuffer surface creation failed\n"); return 0; }
    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) { printf("EGL make current failed\n"); return 0; }

    printf("complete\n");
    return 1;
}
 
void* mini_gpu (void *arg) {
   scene_info *scene = (scene_info *)arg;
   bool res;

   /* once */
   if (!make_offscreen_context(scene->width, scene->height)) { printf("failed to make offscreen context\n"); return NULL; }
   offscreen_fbo_t fb = {0};
   if (!make_fbo(&fb, scene->width, scene->height)) { printf("failed to make FBO\n"); return NULL; }

   //pin_thread_to_cpu(3); // make sure we don't run on CPU 3
 
   /*
   int32_t fd = open ("/dev/dri/renderD128", O_RDWR);
   assert (fd > 0);
 
   struct gbm_device *gbm = gbm_create_device (fd);
   assert (gbm != NULL);
 
   // setup EGL from the GBM device 
   EGLDisplay egl_dpy = eglGetPlatformDisplay (EGL_PLATFORM_GBM_MESA, gbm, NULL);
   assert (egl_dpy != NULL);
 
   res = eglInitialize (egl_dpy, NULL, NULL);
   if (!res) {
         fprintf(stderr, "eglInitialize failed: %d\n", eglGetError());
         eglGetError(); // clear error
   }
   assert (res);
 
   const char *egl_extension_st = eglQueryString (egl_dpy, EGL_EXTENSIONS);
   assert (strstr (egl_extension_st, "EGL_KHR_create_context") != NULL);
   assert (strstr (egl_extension_st, "EGL_KHR_surfaceless_context") != NULL);
 
      //EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
   static const EGLint config_attribs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_NONE
   };
   EGLConfig cfg;
   EGLint count;
 
   res = eglChooseConfig (egl_dpy, config_attribs, &cfg, 1, &count);
   assert (res);
 
   res = eglBindAPI (EGL_OPENGL_ES_API);
   assert (res);
 
   static const EGLint attribs[] = {
      EGL_CONTEXT_MAJOR_VERSION, 3,
      EGL_CONTEXT_MINOR_VERSION, 1,
      EGL_NONE
   };
   EGLContext core_ctx = eglCreateContext (egl_dpy,
                                           cfg,
                                           EGL_NO_CONTEXT,
                                           attribs);
   assert (core_ctx != EGL_NO_CONTEXT);
 
   res = eglMakeCurrent (egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, core_ctx);
   assert (res);
 
   // setup a compute shader 
   GLuint compute_shader = glCreateShader (GL_COMPUTE_SHADER);
   assert (glGetError () == GL_NO_ERROR);
 
   const char *shader_source = COMPUTE_SHADER_SRC;
   glShaderSource (compute_shader, 1, &shader_source, NULL);
   assert (glGetError () == GL_NO_ERROR);
 
   glCompileShader (compute_shader);
   assert (glGetError () == GL_NO_ERROR);
 
   GLuint shader_program = glCreateProgram ();
 
   glAttachShader (shader_program, compute_shader);
   assert (glGetError () == GL_NO_ERROR);
 
   glLinkProgram (shader_program);
   assert (glGetError () == GL_NO_ERROR);

   int frame = 0;
    GLuint color_tex = 0, fbo = 0;
    glGenTextures(1, &color_tex);
    glBindTexture(GL_TEXTURE_2D, color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scene->width, scene->height, 0,
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
    glViewport(0, 0, scene->width, scene->height);

   */



   // clear-only loop: avoid invalid draw without a graphics program/VAO
   unsigned int frame = 0;
   uint8_t *dst = malloc(scene->width * scene->height * 4);
   while(scene->do_render) {
      glBindFramebuffer(GL_FRAMEBUFFER, fb.fbo);
      glViewport(0, 0, scene->width, scene->height);
      float pulse = (float)((frame & 63)) / 63.0f;
      glClearColor(pulse, 1.0f - pulse, 0.2f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      GLenum err = glGetError();
      if (err != GL_NO_ERROR) {
         fprintf(stderr, "[mini_gpu] GL error 0x%x at frame %d\n", err, frame);
         break;
      }

      /* fence with yield, avoids hard stalls */
      GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      for (;;) {
         GLenum r = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000);
         if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED) break;
         if (r == GL_WAIT_FAILED) break;
         sched_yield();
      }
      glDeleteSync(fence);

      /* read back */
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadPixels(0, 0, fb.width, fb.height, GL_RGBA, GL_UNSIGNED_BYTE, dst);

      if ((++frame % 100) == 0) {
         printf("mini GPU frame %d\n", frame);
      }
      usleep(2500);
   }

   printf("[mini_gpu] exiting after %d frames\n", frame);
   free(dst);

 
   /*
   glDeleteShader (compute_shader);
 
   glUseProgram (shader_program);
   assert (glGetError () == GL_NO_ERROR);
 
   // dispatch computation 
   glDispatchCompute (1, 1, 1);
   assert (glGetError () == GL_NO_ERROR);
 
   printf ("Compute shader dispatched and finished successfully\n");
 
   // free stuff 
   if (fbo) glDeleteFramebuffers(1, &fbo);
   if (color_tex) glDeleteTextures(1, &color_tex);
   glDeleteProgram (shader_program);
   // unbind context before destroying
   EGLBoolean success = eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   if (!success) {
       fprintf(stderr, "eglMakeCurrent failed during cleanup\n");
   }
   eglDestroyContext (egl_dpy, core_ctx);
   eglTerminate (egl_dpy);
   if (gbm) gbm_device_destroy (gbm);
   if (fd >= 0) close (fd);
 
   */
   return NULL;
}