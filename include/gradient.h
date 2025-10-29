#include "hub75gpu.h"

void gradient_fill(hub75_display_t *scene, int y, int x0, int x1, 
                                             const SimpleGradient *gradient, 
                                             int minx, int miny, int maxx, int maxy);

void gradient_polygon(hub75_display_t *scene, Polygonf_t *poly, SimpleGradient gradient);
 