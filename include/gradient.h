#include "hub75gpu.h"

void gradient_fill(scene_info *scene, int y, int x0, int x1, 
                                             const SimpleGradient *gradient, 
                                             int minx, int miny, int maxx, int maxy);

void gradient_polygon(scene_info *scene, Polygonf_t *poly, SimpleGradient gradient);
 