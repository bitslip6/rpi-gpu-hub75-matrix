/* hub_api.c: Implements a scene-bound wrapper table so FFI users can call
 * primitives without threading a scene pointer through every call. */

#include <stdint.h>
#include "hub75gpu.h"
#include "pixels.h"

extern scene_info *g_scene;

/* Forward declarations */
extern scene_info *scene_new();
extern scene_info *scene_parse(int argc, char **argv);
extern void scene_start(scene_info *scene);
extern void hub75_request_shutdown(scene_info *scene);
extern void hub75_wait_shutdown(scene_info *scene);
//extern void map_byte_image_to_bcm(scene_info *scene, uint8_t *image);
extern float calculate_fps(uint16_t target_fps, bool show_fps);

/* Drawing helpers */
extern void hub_pixel(scene_info *scene, int x, int y, RGB pixel);
extern void hub_pixel_factor(scene_info *scene, int x, int y, RGB pixel, float factor);
extern void hub_pixel_alpha(scene_info *scene, int x, int y, RGBA pixel);
extern void hub_fill(scene_info *scene, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color);
extern void hub_line(scene_info *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, RGB color);
extern void hub_line_aa(scene_info *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, const RGB color);
extern void hub_circle(scene_info *scene, uint16_t cx, uint16_t cy, uint16_t radius, RGB color);

/* Singleton instance (mutable scene + function pointers) */
static hub75_api g_api;

/* ---- Wrapper implementations that capture g_api.scene ---- */
static void api_start(void)                 { if (g_api.scene) scene_start(g_api.scene); }
static void api_request_shutdown(void)      { if (g_api.scene) hub75_request_shutdown(g_api.scene); }
static void api_wait_shutdown(void)         { if (g_api.scene) hub75_wait_shutdown(g_api.scene); }
static void api_map_image(uint8_t *image)   { if (g_api.scene) map_byte_image_to_bcm(g_api.scene, image); }

static void api_pixel(int x, int y, RGB p)                   { if (g_api.scene) hub_pixel(g_api.scene, x, y, p); }
static void api_pixel_factor(int x, int y, RGB p, float f)    { if (g_api.scene) hub_pixel_factor(g_api.scene, x, y, p, f); }
static void api_pixel_alpha(int x, int y, RGBA p)             { if (g_api.scene) hub_pixel_alpha(g_api.scene, x, y, p); }
static void api_fill(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB c) { if (g_api.scene) hub_fill(g_api.scene, x1, y1, x2, y2, c); }
static void api_line(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c)   { if (g_api.scene) hub_line(g_api.scene, x0, y0, x1, y1, c); }
static void api_line_aa(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c){ if (g_api.scene) hub_line_aa(g_api.scene, x0, y0, x1, y1, c); }
static void api_circle(uint16_t cx,uint16_t cy,uint16_t r, RGB c){ if (g_api.scene) hub_circle(g_api.scene,cx,cy,r,c); }
static int api_fps_get() { return (int)calculate_fps(g_scene->fps, g_scene->show_fps); }

const hub75_api *hub75_get_api(scene_info *scene) {
    if (g_api.version == 0) {
        g_api.version = 2; /* new layout with wrappers */
        g_api.new_scene        = scene_new;
        g_api.parse_scene      = scene_parse;
        g_api.start            = api_start;
        g_api.request_shutdown = api_request_shutdown;
        g_api.wait_shutdown    = api_wait_shutdown;
        g_api.map_image        = api_map_image;
        g_api.fps_calculate    = calculate_fps;
        g_api.pixel            = api_pixel;
        g_api.pixel_factor     = api_pixel_factor;
        g_api.pixel_alpha      = api_pixel_alpha;
        g_api.fill             = api_fill;
        g_api.line             = api_line;
        g_api.line_aa          = api_line_aa;
        g_api.circle           = api_circle;
        g_api.fps_get          = api_fps_get;
    }
    if (scene) g_api.scene = scene;
    return &g_api;
}
