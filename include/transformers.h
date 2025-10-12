#include <stdint.h>
#include "rpihub75.h"

#ifndef __TRANSFORMERS_H__
#define __TRANSFORMERS_H__

/**
 * @brief invert the image vertically
 */
uint8_t *flip_mapper(const uint8_t *image,
                             uint8_t *image_out,
                             const scene_info *scene);

/**
 * @brief mirror and flip an image
 */
uint8_t *mirror_flip_mapper(const uint8_t *image,
                            uint8_t * image_out,
                            const struct scene_info *scene);
                            
uint8_t* u_mapper_impl(const uint8_t *image, uint8_t *output_image, const scene_info *scene);
uint8_t *mirror_mapper(const uint8_t *image, uint8_t *image_out, const struct scene_info *scene);


#endif
