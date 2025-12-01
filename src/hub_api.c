/* hub_api.c: Implements a scene-bound wrapper table so FFI users can call
 * primitives without threading a scene pointer through every call. */

#include <stdint.h>
#include "hub75gpu.h"
#include "pixels.h"

extern hub75_display_t *g_scene;

/* Forward declarations */
extern hub75_display_t *hub75_display_new();
extern hub75_display_t *scene_parse(int argc, char **argv);
extern void scene_start(hub75_display_t *scene);
extern void hub75_request_shutdown(hub75_display_t *scene);
extern void hub75_wait_shutdown(hub75_display_t *scene);
//extern void map_byte_image_to_bcm(hub75_display_t *scene, uint8_t *image);
extern float calculate_fps(uint16_t target_fps, bool show_fps);

/* Drawing helpers */
extern void draw_pixel(hub75_display_t *scene, int x, int y, RGBA pixel);
extern void draw_pixel_alpha(hub75_display_t *scene, int x, int y, RGBA pixel);
extern void hub_fill(hub75_display_t *scene, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB color);
extern void draw_line(hub75_display_t *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, RGBA color);
extern void draw_line_aa(hub75_display_t *scene, const uint16_t x0, const uint16_t y0, const uint16_t x1, const uint16_t y1, const RGBA color);
extern void hub_circle(hub75_display_t *scene, uint16_t cx, uint16_t cy, uint16_t radius, RGB color);

/* Singleton instance (mutable scene + function pointers) */
static hub75_api g_api;

/* ---- Wrapper implementations that capture g_api.scene ---- */
static void api_start(void)                 { if (g_api.scene) scene_start(g_api.scene); }
static void api_request_shutdown(void)      { if (g_api.scene) hub75_request_shutdown(g_api.scene); }
static void api_wait_shutdown(void)         { if (g_api.scene) hub75_wait_shutdown(g_api.scene); }
static void api_map_image(uint8_t *image)   { if (g_api.scene) map_byte_image_to_bcm(g_api.scene, image); }

static void api_pixel(int x, int y, RGBA p)                   { if (g_api.scene) { draw_pixel(g_api.scene, x, y, p); } }
static void api_fill(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGB c) { if (g_api.scene) hub_fill(g_api.scene, x1, y1, x2, y2, c); }
static void api_line(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c)   { if (g_api.scene) { RGBA px = {c.r, c.g, c.b, 255}; draw_line(g_api.scene, x0, y0, x1, y1, px); } }
static void api_line_aa(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1, RGB c){ if (g_api.scene) { RGBA px = {c.r, c.g, c.b, 255}; draw_line_aa(g_api.scene, x0, y0, x1, y1, px); } }
static void api_circle(uint16_t cx,uint16_t cy,uint16_t r, RGB c){ if (g_api.scene) hub_circle(g_api.scene,cx,cy,r,c); }
static int api_fps_get() { return (int)calculate_fps(g_scene->fps, g_scene->show_fps); }

const hub75_api *hub75_get_api(hub75_display_t *scene) {
    if (g_api.version == 0) {
        g_api.version = 2; /* new layout with wrappers */
        g_api.new_scene        = hub75_display_new;
        g_api.parse_scene      = scene_parse;
        g_api.start            = api_start;
        g_api.request_shutdown = api_request_shutdown;
        g_api.wait_shutdown    = api_wait_shutdown;
        g_api.map_image        = api_map_image;
        g_api.fps_calculate    = calculate_fps;
        g_api.pixel            = api_pixel;
        g_api.fill             = api_fill;
        g_api.line             = api_line;
        g_api.line_aa          = api_line_aa;
        g_api.circle           = api_circle;
        g_api.fps_get          = api_fps_get;
    }
    if (scene) g_api.scene = scene;
    return &g_api;
}
