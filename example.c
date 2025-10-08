/**
 * This is an example program using the rpi-gpu-hub75 library.
 * To compile:
 * gcc -O3 -Wall -lrpihub75_gpu example.c -o example
 * ./example
 * # this should display command line options
 * # example for 4 64x64 panels (128x128 pixels) 2 chains connected to 2 ports, 
 * # 48 pwm color bits, 192 brightness, 2.2 gamma, render the cartoon.glsl shader
 * # at 120 fps with 
 * ./example -w 128 -h 128 -p 2 -c 2 -d 48 -b 192 -g 2.2 -f 120 -s shaders/cartoon.glsl
 *
 */
// TODO: add make file flag for enabling asserts
//#define ENABLE_ASSEERTS 0
// replace #define with scene debug printing options...
// NOTE: this is defined in rpihub75.h and must be recompiled and reinstalled with the lib....
//#define CONSOLE_DEBUG 1

#include <pthread.h>
#include <signal.h>
#include <rpihub75/rpihub75.h>
#include <rpihub75/util.h>
#include <rpihub75/gpu.h>
#include <rpihub75/video.h>
#include <rpihub75/pixels.h>


unsigned int ri(unsigned int max) {
	return rand() % max;
}

// our CPU rendering implementation, see gpu.c for shader rendering details
void* render_cpu(void *arg) {
    // get the current scene info
    scene_info *scene = (scene_info*)arg;
    // allocate  memory for image data, we can also use the preallocated scene->image if that's easier
    uint8_t *img = malloc(scene->width * scene->height * scene->stride);
    memset(img, 0, scene->width * scene->height * scene->stride);

    debug("rendering on CPU\n");
    // need to pause a second for gpio to be setup
    usleep(50000);
    while(scene->do_render) {
        // darken every pixel in the image for each byte of R,G,B data
        if (1) {
            for (int i=0; i<scene->height*scene->width*scene->stride; i++) {
                scene->image[i] = (uint8_t)scene->image[i] * 0.96f;
            }
        }

        // generate some random points on the screen
        uint16_t x1 = ri(scene->width);
        uint16_t x2 = ri(scene->width);
        uint16_t x3 = ri(scene->width);
        uint16_t y1 = ri(scene->height);
        uint16_t y2 = ri(scene->height);
        uint16_t y3 = ri(scene->height);

        // generate a random color
        RGB color = {ri(250), ri(250), ri(250)};

        hub_triangle_aa(scene, x1, y1, x2, y2, x3, y3, color);

        // draw a line
        //hub_line(scene, x1, y1, x2, y2, color);

        // draw a rectangle
        //hub_fill(scene, x1, y1, x2, y2, color);

        // draw a circle
        //hub_circle(scene, x1, y1, y3 % 20, color);

        // render the RGB data to the active BCM buffers.
        map_byte_image_to_bcm(scene, scene->image);

        // calcualte_fps will delay execution to achieve the desired frames per second
        (void)calculate_fps(scene);
    }
    free(img);
    return NULL;
}


int main(int argc, char **argv)
{
    printf("rpi-gpu-hub75 v0.2 example program %s pin out configuration\n", ADDRESS_TYPE);
    srand(time(NULL));
    // parse command line options to define the scene
    // use -h for help, see this function in util.c for more information on command line parsing
    scene_info *scene = default_scene(argc, argv);

    // you can also just hard code the scene description this into your project EG:
    // scene_info *scene = new_scene(); // get a new scene with default values
    //scene->num_chains = 2; // change some settings ....
    //scene->num_ports = 2;
    //scene->width = 128;

    // ensure that the scene is valid - allocate memory for internal buffers
    // install signal handlers for graceful shutdown
    start_scene(scene);

    
    // create another thread to run the frame drawing function (GPU or CPU)
    pthread_t update_thread;
    // use the CPU renderer if no shader or video file was passed
    if (scene->shader_file == NULL) {
        pthread_create(&update_thread, NULL, render_cpu, scene);
    }
    // use the gpu shader or video renderer if we have one, else use the cpu renderer above
    else if (access(scene->shader_file, R_OK) == 0) {
        // render a glsl shader
        if (has_extension(scene->shader_file, "glsl")) {
            scene->stride = 4;
            pthread_create(&update_thread, NULL, render_shader, scene);
        }
        // treat everything else as a video file
        else {
            pthread_create(&update_thread, NULL, render_video_fn, scene);
        }
    } else {
        die("unable to open file %s\n", scene->shader_file);
    }


    // the render functions will all spawn the necessary threads to run the display
    // Wait for the rendering thread to complete or for shutdown signal
    printf("Press Ctrl-C to exit gracefully...\n");
    pthread_join(update_thread, NULL);
    printf("joined thread\n");

    // wait...
    while(scene->do_render) {
        usleep(100000);
    }
    
    printf("Program terminated successfully.\n");
    return 0;
}
