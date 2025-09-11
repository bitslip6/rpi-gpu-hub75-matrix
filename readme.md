This library implements HUB75 protocol on the rpi5
==================================================
Library for rendering advanced GPU graphics and video on HUB75 panels on Raspberry 5 with a 9600Hz refresh for a single panel. Supports up to 6 chained panels on 3 ports for 18 panels, 384x192 pixels. 64bit BCM for 64 color levels, = 262144 colors. 255 levels of brightness control. GLSL shader support. Many tone mappings and gamma correction combinations are possible for complete HDR image control.

Support for Raspberry Pi 4 is provided with significantly degraded performance. The GPIO rise time on Pi4 is 1.5x slower than Pi5 (24ns vs 16ns) and does not have the ability to force bit settings, only to clear OR enable bit settings. This means it takes 2 operations on Pi4 (1 to set and 1 to clear) vs a single operation on Pi5 (1 to set and clear) and both operations are 1.5x slower. This results in a maximum clock speed of <10Mhz. Unfortunately because of the long "settle" time on the GPIO lines, we can only achieve stable operation of about 1-1.5Mhz. Honestly, just upgrade to a pi5 if frame rate or color performance is important for your project at all. Also 64bit color depth results in significant flickering and brightness control is only available via BCM control.

Support for anti-aliased TTF fonts is coming soon with GPU rendering. This should allow for some nice effects like 120fps floating point sinus text scrollers with fully animated texture maps and nice anti-aliases edges. Support will add an additional external binary dependency to generate the sine distance fields for the fonts with [msdfgen](https://github.com/Chlumsky/msdfgen).

This is loosely based on the work done by hzeller adding HUB75 support to RPI (www.github.com/hzeller/rpi-rgb-led-matrix)
as well as the work of Harry Fairhead documenting the peripheral address space for rpi5(https://www.i-programmer.info/programming/148-hardware/16887-raspberry-pi-iot-in-c-pi-5-memory-mapped-gpio.html). Also thanks to nothings stb image loader library which is used
for loading textures for shaders. https://github.com/nothings/stb/blob/master/stb_image.h

If you are a Python programmer and can help create a cython wrapper, there is some work done creating a Python wrapper but Python error messages are cryptic and there are lots of idiosyncrasies that just need the touch of an experienced Python programmer (I have very little python experience). Drop an issue and I'll help you along with the structs and functions that need wraping to implement.


shaders/cartoon.glsl
------------------
![Shader Demo 1](https://raw.githubusercontent.com/bitslip6/rpi-gpu-hub75-matrix/refs/heads/main/assets/shader_1.jpg)

shaders/cyber.glsl
--------------------
![Shader Demo 2](https://raw.githubusercontent.com/bitslip6/rpi-gpu-hub75-matrix/refs/heads/main/assets/shader_2.jpg)

shaders/voronoi.glsl (video)
--------------------
[![Shader Demo Video](https://img.youtube.com/vi/1Jd2xxN-WbU/0.jpg)](https://www.youtube.com/watch?v=1Jd2xxN-WbU)


Overview
--------
BitBang HUB75 data at steady 20Mhz. It supports a 9600Hz refresh rate on Pi5, and 1500Hz on Pi4, for a single 64x64 panel. Supports up to 3 ports with 2 pixels 
per port per clock cycle. The library handles the double buffering of frame data. Support for 24bpp RGB and 32bpp RGBA
source image data. Frame rates of >120Hz with 64 bits of BCM data are easily possible with chain lengths of 3 or more. 
Support for up to 64 bits of binary code modulation data (1/64 pwm cycle for 64 different color levels for each RGB value).

GPU support using Linux's Generic Buffer Manager (gbm), GLESv2, and EGL is also included. This means you can use 
OpenGL fragment shaders to render PWM data to the hub75 panel. Several shadertoy.org shaders are included in the shaders
directory.

Multiple tone mapping implementations are provided including ACES, reinhard, and exposure as well as saturation and 
contrast controls. Tone mapping compresses the upper and lower end of the linear sRGB data to provide a more natural
and balanced image on the LED panel. You can implement your own tone mapping by implementing the func_tone_mapper_t
function and setting it in the active "scene_info". Tone mapping changes take effect on the next frame update and 
do not add any delay after initial BCM mapping. 

Gamma correction is also provided. Global gamma can be controlled on the command line. Each red, green, and blue color
channel also has its own gamma correction to help improve color balance. In practice gamma of about 2.2 produces
generally good results. red, green, and blue gamma are multiplied against base gamma for each color channel. For 
factory color-balanced panels (very common), these should all be set to 1 as #define in include/rpihub75.h (default).

Linear color correction is also provided. You can linearly add + or - red, green and blue to the color channels 
by adjusting these values in the scene controls. This will affect the generated BCM data that is mapped on every frame.

No hardware clocks are required for operation so you can run the code with only group gpio
privileges. Operation is mostly flicker-free, however, you should see the improved response by running with nice -n -20
and running the real-time PREEMPT_RT patch on the kernel (6.6) as of this writing. PREEMPT_RT is mainline in 6.12
so hopefully no patches are required on the next raspbian release!

This implementation only supports rpi5 at the moment. It should be simple to add support for other PIs as only
the memory-mapped peripheral address for the GPIO pins is required. Preliminary GPIO peripheral offsets are in
rpihub75.h. There is a #ifdef PI3, PI4 and it defaults to PI5. If you are inclined, please test on an earlier PI
and send a PR with the correct offsets for ZERO, PI3, and PI4.

Please read HZeller's excellent write-up on wiring the PI to the HUB75 display.  I highly recommend using one of his
active 3 port boards to ensure proper level translation and to map the address lines, OE, and clock pins to all 3 boards



Glossary
--------
### PWM 
modulates the pulse width of a signal (i.e., the "on" time vs "off" time) to control the average power delivered to a device, typically using fixed frequency and variable duty cycles. PWM is sometimes used in place of BCM in this document.

### BCM 
modulates the signal based on binary values. It uses a binary sequence, where each bit's on/off duration is proportional to its weight in the sequence. BCM is more efficient at low brightness levels than PWM, as it distributes "on" periods more evenly.

### Gamma
mapping linear color space to a compressed color space where colors in the lower band (say 0-64 for 8bit RGB)
are spread out so that the color values ramp exponentially. This more closely matches human eye perception but does
remove dynamic range at the low end of the spectrum producing image quantization in darker regions.

### Dithering 
When you apply gamma correction, you compress the dynamic range of your RGB values, especially for dark colors. The colors that should have been represented by a smooth gradient are reduced to just a few distinct levels. Dithering helps by distributing the error introduced by this quantization process over neighboring pixels.

###Tone Mapping
Tone mapping takes normalized RGB data (0-1) and maps each RGB channel to a different value accoring to the
mapping function. These functions are designed to compress the upper and lower regions of an image to preserve High Dynamic
Range in a lower dynamic range format. Several are provided and they are easy to implement. Experiment with what works for
your scene. Reference: https://www.cl.cam.ac.uk/teaching/2122/AGIP/07_HDR_and_tone_mapping_1pp.pdf

###GLSL 
(OpenGL Shading Languate) GLSL is a high-level shading language based on the C programming language, designed for programming shaders in OpenGL API. It allows custom vertex, fragment and compute shaders to control rendering pipeline of 3D graphics applications, enabling effects such as lighting, texture mapping, and other visual effects.

###SDF 
(Sine Distance Field): a function that defines a shape in 3d space. The function produces outputs such that points outside the object are positive values, the surface of the shape is at point 0 and points inside the shape are negative.

Build & Install - Raspbian Lite
-------------------------------
``` bash
sudo apt update
sudo apt upgrade
# build dependencies for rpi-gpu-hub75-matrix
sudo apt install build-essential gcc make libgles2-mesa-dev libgbm-dev libegl1-mesa-dev libavformat-dev libswscale-dev

# for compiling the video player support (optional) (currently in devlopment)
sudo apt install libavformat-dev libswscale-dev ffmpeg
# you must reboot after installing ffmepg libraries to use opengl.
# not sure why this is, but you should reboot after intalling the ffmpeg dependencies


git clone https://github.com/bitslip6/rpi-gpu-hub75-matrix
cd rpi-gpu-hub75-matrix

# NOTE: you cannot compile for multiple boards. pin configuration is defined at compile time
# compile with supports for hzeller's 3 port board (default):
make DEF="-DHZELLER=1"
# OR compile with support for ada fruit hub75 hat:
make DEF="-DADA_HAT=1"
# OR edit include/rpihub75.h and edit teh #define for pin mapping if using a different board or pin configuration

# install the library in /usr/local
sudo make install

# compile the example program
gcc -Wall -O3 -mtune=native -lrpihub75_gpu example.c -o example
# or use the makefile
make example

# get command line help
./example
# render a shader to 1 64x64 panel, bit depth 32, 120 fps, gamma 1.6, 50% brightness
./example -x 64 -y 64 -d 32 -f 120 -g 1.6 -b 128 -s shaders/cartoon.glsl
# render a IQ's "happy jumping" shader to 128x128 panel, bit depth 64, 2 ports, 2 panels, 60 fps, 
# mirrored and flipped, 255 brightness (100%), gamma 2.4, saturation tone mapping level 1.8
./example -x 128 -y 128 -d 64 -p 2 -c 2 -f 60 -i mirror_flip -b 255 -g 2.4 -t saturation:1.8 -s shaders/happy_jump.glsl
# render the triangle demo on the CPU, see example.c for basic library usage.
./example -x 128 -y 128 -d 64 -f 60
# render the lines gpu shader, 24 bit, with strong Floyd Steinberg dithering (-l 1-254), gamma 2.2
./example -x 128 -y 128 -d 24 -f 60 -l 250 -g 2.2 -s shaders/lines.glsl
# coming soon, mp4 drawing, network drawing, improved dithering
``` 

Compile RealTime Kernel
-----------------------
``` bash
# for compiling the PREEMPT_RT kernel patch (optional)
sudo apt install git vim bc bison flex libssl-dev libncurses5-dev

# real-time kernel patch to remove any flicker:
# compile real-time patch for rpi5 6.6 kernel
# make sure these instructions are updated, visit https://raspberrypi.com/documentation/computers/linux_kernel.html#building

mkdir kernel
git clone --depth=1 --branch rpi-6.6.y https:/github.com/raspberrypi/linux
wget https://mirrors.edge.kernel.org/pub/linux/kernel/projects/rt/6.6/patch-6.6.58-rt45.patch.gz
cd linux
zcat ../patch-6.6.58-rt45.patch.gz | patch -p1 --dry-run # make sure patch applies correctly
zcat ../patch-6.6.58-rt45.patch.gz | patch -p1
KERNEL=kernel_2712                                       # use kernel_8 for rpi1-4
make bcm2712_defconfig                                   # use bcm2711_defconfig for rpi1-4 

make menuconfig                                          # General -> Preemption Model -> select Real Time option
vi .config                                               # custome CONFIG_LOCALVERSION (helps you identify your kernel when runing uname -a)
make -j4 Image.gz modules dtbs
# wait about 30-45 minutes
echo $KERNEL
sudo make -j4 modules_install
sudo cp /boot/firmware/$KERNEL.img /boot/firmware/$KERNEL-backup.img
sudo cp arch/arm64/boot/Image.gz /boot/firmware/$KERNEL.img
sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/
sudo reboot
```


Hub75 Operation
---------------
All documentation reefers to 64x64 panels. Your Mileage May Vary.
Pins: r1,r2,g1,g2,b1,b2, this is the color data. each LED is actually 3 leds (red, green and blue). The LED can be
on or off. We will be pulsing them on/off very quickly to achieve the illusion of different color values. Color data
is sent 2 pixels at a time beginning on column 0. r1,g1,b1 is the pixel in the upper 32 rows, r2,g2,b2 is the pixel
in the lower 32 rows.

A,B,C,D,E are the address lines. These 5 pins represent the row address. 2^5 = 32 so depending on which bitmask is
set on the address lines, the 2 corresponding led rows will be addressed for shifting in data. Data is shifted in on
the falling edge of CLK. so after setting the address line, we set pixel value for column 0 along with clock, and then
we pull the clock low. That is pixel 0. We now shift in the next pixel and so on 64 times. If we have multiple panels we
simply continue shifting in data (in 64-column chunks) for as many panels as we have.

To actually update the panel, we must bring OE (output enable) line high (to disable to display) and toggle the latch
pin. data for one row is now latched. we advance the address row lines drop the enable pin low (turn the display on)
and begin the process again.


Library Operation:
-----------------
See the header files for function definitions.

The library bit bangs the data out to the HUB75 panel at a steady 20Mhz. This is significantly faster on my scope than
hzeller's implementation by up to 10x. The software forks a thread that pulls from the PWM data and continuously pulses 
the rgb pins and the clock line. After each row, the Output Enable pin is driven high and the data is latched and the next
row is advanced. After 32 rows (or 1/2 panel height) are written to all 3 ports, the BCM buffer is advanced and the 
update begins again for the next "bit plane".

Since we are clocking in 2 rows of RGB data per clock cycle (R1, R2, G1, G2, B1, B2) at 20Mhz it takes 2048 clock cycles
to shift in data for 1 64x64 panel. This translates to a single 64x64 panel refresh rate of 9700Hz and allows us to chain
4 panels together per port at >2400Hz. 

Rather than call "SetPixel", you draw directly to a 24bpp or 32bpp buffer and then call this library's function
map_byte_image_to_bcm() to translate the 24bpp RGB buffer to the BCM signal. The buffer that this function writes the BCM
data is read from on another thread via the render_forever() method.

The render_forever() method will run until scene->do_render is set to false.

Minimum Program
---------------
```c
#include <pthread.h>
#include <rpihub75/rpihub75.h>
#include <rpihub75/util.h>
#include <rpihub75/video.h>
#include <rpihub75/pixels.h>

void* render_loop(void *arg) {
    // get the current scene info
    scene_info *scene = (scene_info*)arg;
    for (;;) {
        int red = 255, green = 128, blue = 64;
	RGB color = {red, green, blue};
	hub_pixel(scene, 32, 16, color);         // render to scene internal image buffer
	hub_pixel(scene, 32, 16, color);
	hub_line(scene, 5, 5, 32, 32, color);
	hub_line_aa(scene, 5, 5, 32, 32, color); // anti-aliased
	//hub_triangle(scene, x1, y1, x2, y2, x3, y3, color);
	//hub_triangle_aa(scene, x1, y1, x2, y2, x3, y3, color); // anti-aliased
	scene->bcm_mapper(scene, NULL); // render image bugger in scene
	calculate_fps(scene->fps, scene->show_fps);
    }
}

int main(int argc, char **argv) {
	scene_info *scene = default_scene(argc, argv);
	check_scene(scene);
	pthread_t update_thread;
	pthread_create(&update_thread, NULL, render_loop, scene);
	render_forever(scene); // does not return
}

```


Example using your own drawing buffer:
-------------------------------------
```c
scene_info *scene = default_scene(argc, argv);
// example scene->stride is 3 for 24bpp (3 bytes per pixel)
uint8_t *imageRGB = (uint8_t*)malloc(scene->width * scene->height * scene->stride); 

int x = 32;
int y = 16;
uint8_t red = 255; 
uint8_t green = 128; 
uint8_t blue = 64; 

// scene->stride is 3
imageRGB[((y*scene->width) +x *scene->stride)] = red;
imageRGB[((y*scene->width) +x *scene->stride)+1] = green;
imageRGB[((y*scene->width) +x *scene->stride)+2] = blue;

scene->bcm_mapper(scene, imageRGB);   // pass the imange buffer here. supports RGB with scene->stride = 3 and RGBA with scene->stride = 4

calculate_fps(scene->fps);
```


Users can update either 24bpp RGB or 32bpp RGBA frame buffers directly and then call scene->bcm_mapper() after rendering
a new frame. Calling this method will translate the RGB data to BCM bit data. BCM data is organized as a multi dimensional
array of uint32_t data. Each uint32_t stores a bitmask for the r1,r2,g1,g2,b1,b2 pins for the current pixel's bit-plane. There
is no need to call any other functions as the "render_forever()" code pulls directly from this buffer.


RGB to BCM Mapping
------------------

Data is indexed as follows where bcm is the current bit plane index (0 - bit_depth):

int offset = ((y * scene->width + (x)) * scene->bit_depth) + bcm;

using a linear mapping for RGB (255, 128, 0), the bcm data for a single pixel would map to:
r: 1,1,1,1,1,1,1,1,1,1,1...
g: 1,0,1,0,1,0,1,0,1,0,1...
b: 0,0,0,0,0,0,0,0,0,0,0...

these values would be precomputed after every frame and toggled for each display update (9600-2400Hz)


each bit plane (that is a uint32_t with all of the pin toggles for all 3 output ports for a particular pixel on a single 
bit plane, there are bit_depth number of bit planes per image) is updated atomically in a single write. This means there
is no need for double buffering to achieve a flicker-free display. Simply call map_byte_image_to_bcm with your new image
buffer as often as you like. The data will be overwritten and the new PWM data will be updated immediately. This allows you
to draw to the display at up to 9600Hz (depending on the number of chained displays) however frame rates of about 120fps seem 
to produce excellent results and higher frame rates have diminishing returns after that.

Because we have a 9600-2400Hz refresh rate we can use up to 64bit PWM cycles. That means that each RGB value
can have 64 levels of brightness or 64*64*64 = 262144 colors. In practice though, values at the lower end (more bits off
than on) have a much more perceptible effect on brightness than values at the higher end (more on than off). This is because
the human eye is much more sensitive to brightness changes at darker levels than at brighter levels. To correct this I have
added gamma correction. See color calibration further in this document for details.


brightness is controlled via a 9K "jitter mask". 9K of random bytes are generated and if each random value is > brightness
level (uint8_t) the OE pin is toggled for the mask. when OE is toggled high, the display is toggled off. By applying this 
mask for every pixel, we are able to output our normal BCM color data and toggle the brightness value on or off randomly
averaging out to the current brightness level. This provides fine-tuned brightness control (255 levels) while
maintaining excellent color balance.

Alternatively, you can encode brightness data directly into the PWM data, however, this yields  poor results for low
brightness levels even when using 64 bits of BCM data. This feature is primarly for Pi3 & 4 models.


The mapping from 24bpp (or 32bpp) RGB data to BCM data is very optimized. It uses 128-bit SIMD vectors for the innermost
loop. All 3 output ports are mapped in a single line of code, allowing the compiler to compute the value of all pins
and then update memory in 1 atomic operation.


```c
for (int j=0; j<bit_depth; j++) {
        // mask off just this bit plane's data
        uint64_t mask = 1L << j;

        bcm_signal[bcm_offset++] =
            // PORT 0, top pixel
            (!!(bits[image[0]] & mask)) << ADDRESS_P0_R1 |
            (!!(bits[image[1]] & mask)) << ADDRESS_P0_G1 |
            (!!(bits[image[2]] & mask)) << ADDRESS_P0_B1 |

            // PORT 0, bottom pixel
            (!!(bits[image[p0b+0]] & mask)) << ADDRESS_P0_R2 |
            (!!(bits[image[p0b+1]] & mask)) << ADDRESS_P0_G2 |
            (!!(bits[image[p0b+2]] & mask)) << ADDRESS_P0_B2 |

            // PORT 1, bottom pixel
            (!!(bits[image[p1t+0]] & mask)) << ADDRESS_P1_R1 |
            (!!(bits[image[p1t+1]] & mask)) << ADDRESS_P1_G1 |
            (!!(bits[image[p1t+2]] & mask)) << ADDRESS_P1_B1 |

            // PORT 1, bottom pixel
            (!!(bits[image[p1b+0]] & mask)) << ADDRESS_P1_R2 |
            (!!(bits[image[p1b+1]] & mask)) << ADDRESS_P1_G2 |
            (!!(bits[image[p1b+2]] & mask)) << ADDRESS_P1_B2 |

            // PORT 2, bottom pixel
            (!!(bits[image[p2t+0]] & mask)) << ADDRESS_P1_R1 |
            (!!(bits[image[p2t+1]] & mask)) << ADDRESS_P1_G1 |
            (!!(bits[image[p2t+2]] & mask)) << ADDRESS_P1_B1 |

            // PORT 2, bottom pixel
            (!!(bits[image[p2b+0]] & mask)) << ADDRESS_P2_R2 |
            (!!(bits[image[p2b+1]] & mask)) << ADDRESS_P2_G2 |
            (!!(bits[image[p2b+2]] & mask)) << ADDRESS_P2_B2;
    }
```


GPU Support
-----------
To add GPU shader support you will need to install glesv2, gbm and mesagl.
sudo apt-get install libgles2-mesa-dev libgbm-dev libegl1-mesa-dev

support for single buffer shadertoy shaders is already added so just pass your shader via the -s command line
parameter. This will set the path to the shader in the scene_info->shader string. render_shader() in gpu.c
will look for a shader on the filesystem at path scene_info->shader and attempt to compile it. It will update
glUniforms iTime and iResolution like shadertoy, however no support for additional buffers or textures has
been added as of yet. Send a PR if you are inclined.

After rendering the shader, the frame buffer is read using glReadPixels() and pwm_mapped the same as the
CPU renderer. the render_shader loop does not return. It will attempt to usleep until scene->fps is matched.
If the GPU can not keep up with the current fps, no sleep is performed.



Compiling and Installing
------------------------
to build and install library and header files to /usr/local run:


```bash
# compile version without GPU support
make lib
# compile version with GPU support
make libgpu
# compile both versions
make 
# install headers and libraries in /usr/local
sudo make install
# you may need to manullay run "sudo ldconfig" depending on your OS environment

# to compile the example app without GPU support:
gcc -O3 -Wall -lrpihub75 example.c -o example 

# to compile the example app with GPU support:
gcc -O3 -Wall -lrpihub75_gpu example.c -o example 

# print command line configuration help
./example 

# run glsl shader app for 1 64x64 panel on port0, 120fps, 48 bits bcm depth, gamma 2.2, 50% brightness
./example -p 1 -c 1 -x 64 -y 64 -d 48 -g 2.2 -f 120 -b 128 -s shaders/cartoon.glsl

```

Example Program
---------------

```c
// see main.c for this example
#include <pthread.h>
#include <rpihub75/rpihub75.h>
#include <rpihub75/util.h>
#include <rpihub75/gpu.h>

unsigned int ri(unsigned int max) {
	return rand() % max;
}

// our CPU rendering implementation, see gpu.c for shader rendering details
void* render_cpu(void *arg) {
    // get the current scene info
    const scene_info *scene = (scene_info*)arg;
    const int buffer_sz     = scene->width * scene->height * scene->stride;
    uint8_t *image          = (uint8_t*)malloc(buffer_sz);

    // loop forever on this thread
    for(;;) {
        // darken every pixel in the image
        for (int x=0; x<buffer_sz; x++) {
            image[x] = (uint8_t)(image[x] * 0.96f);
        }

        uint16_t x1 = ri(scene->width);
        uint16_t x2 = ri(scene->width);
        uint16_t x3 = ri(scene->width);
        uint16_t y1 = ri(scene->height);
        uint16_t y2 = ri(scene->height);
        uint16_t y3 = ri(scene->height);

        RGB color = {ri(250), ri(250), ri(250)};

        // Draw a random antialiased triangle, see pixels.h for drawing primitives
        // drawing primitives begin with "hub_" and draw to scene->image buffer
        hub_triangle_aa(scene, x1, y1, x2, y2, x3, y3, color);

        // render the RGB data to the active PWM buffers. sleep delay the frame to sync with scene->fps
        // You can optionally draw directly into *image, and then pass the image
        // to scene->bcm_mapper(scene, image, TRUE);
        scene->bcm_mapper(scene, NULL, TRUE);
    }


int main(int argc, char **argv)
{
    // parse command line options to define the scene
    // use -h for help, see this function in util.c for more information on command line parsing
    scene_info *scene = default_scene(argc, argv);

    // Ensure that the scene is valid
    check_scene(scene);
    
    // Create another thread to run the frame drawing function (GPU or CPU)
    pthread_t update_thread;
    // Use the GPU shader renderer if we have one, else use the CPU renderer above
    if (scene->shader_file == NULL) {
        pthread_create(&update_thread, NULL, render_cpu, scene);
    } else {
        pthread_create(&update_thread, NULL, render_shader, scene);
    }

    // This function will never return
    render_forever(scene);
}
```


Command Line Arguments
----------------------
You can configure your setup for your application from the command line if you so choose by adding the call: 
```c
    scene_info *scene = default_scene(argc, argv);
```

This will parse the following command line parameters and setup your scene_info configuration for you.
If you prefer you can also hard code this configuration or load it from a configuration file. This structure
is required to call render_forever() and the scene->bcm_mapper() function pointer points to the current
pwm_mapping function for the scene. (see func_bcm_mapper_t)


```txt
 Usage: ./example
     -s <file>         GPU fragment shader or video file (mp2, mp4, etc) to render
     -x <width>        image width              (16-384)
     -y <height>       image height             (16-384)
     -w <width>        panel width              (16/32/64)
     -h <height>       panel height             (16/32/64)
     -f <fps>          frames per second        (1-255)
     -p <num ports>    number of ports          (1-3)
     -c <num chains>   number of chains         (1-16)
     -g <gamma>        gamma correction         (1.0-2.8)
     -d <bit depth>    bit depth                (4-64) multiple of 4
     -b <brightness>   overall brightness level (0-254)
     -m <frames>       motion blur frames       (0-32)
     -l <dither>       dither strength, 0 = off (0.0-10.0)
     -i <mapper>       image mapper (mirror, flip, mirror_flip) (need to add support for U and V mapping)
      // both sigmoid and saturation tone mappers accept a level ie: saturation:2.0
     -t <tone_mapper>  (aces, reinhard, none, saturation:0.5-5.0, sigmoid:0.5-2.0, hable)
     -j                adjust brightness in BCM data, only for pi3-4
     -z                run LED calibration script
     -o                display FPS counters and panel refresh rate in Hz
```



Odds and Ends
-------------

I have considered adding image dithering. Since we are going from 8-bit data (24bpp) down to 5 or 6-bit data (32 - 64 bit
pwm values) we are losing 2-3 bits of data per pixel. What we lose in temporal data (value) we can reintroduce to the image
spatially. Those 2-3 bits of information can be added to the neighboring pixels using ordered dithering or floyd Steinberg 
dithering by slightly increasing or decreasing the RGB values of the neighboring pixels based on this loss of information.
There is some code to achieve this but I have not had the results I would like to see so this is still a work in progress.
If this is something you are interested in, drop me a line, send me a link to relevant information implementations or send
a PR.

I am currently investigating chaining multiple rp2040 chips to multiple chains of HUB75 panels to improve the image 
stability and the number of supported chained panels. Current thinking is rpi5 has 2 spi lines which are capable of 50Mhz
operation. Split between 4 rp2040s that allows for 25Mhz data update to each rp2040. With 8 panels attached to each rp2040
we could address 32 panels with 100Mhz total bandwidth. If we downsample frame data to a 256 color palette, we can send
data to 1 pixel as a single byte. This would allow us to send a single frame of 8 panels in 32KB. at 60fps we have 1.9MB/s
per rp2040. This data can be sent at 16Mhz. This could allow us to send SPI data to up to 6 rp2040 chips at 16Mhz each
and 60fps with 8 panels on each rp2040.

The frame data would have a 256-entry 32bcm palette for a total of 1024 bytes at the beginning on the frame describing
the 256 colors in the frame's palette. each pixel could then be sent as a single byte as a lookup to the palette color.
This would allow us to send data 3x faster to the rp2040 chips.

Once frame data was loaded on the rp2040 chip. pio would use a lookup table from raw frame bytes to bcm data and shift out
to the 6 color registers on the current BCM cycle. This data format would allow us to store a huge amount of image data in
the rp2040 256KB sRAM which still maintains good color response and excellent refresh rates. This method would also allow
us to shift in data faster than the pi5 20Mhz upper limit on GPIO and clock data in at the limit of the HUB75 panels (32-40Mhz)

So far I have only been working on the math to see if this makes any sense. Since it seems possible I might make a rp2040 
implementation and see if 50Mhz bandwidth is achievable over grounded cat5e. After that, programming the PIO to perform the
BCM palette lookup and shit out the data at 20Mhz+ will be the final test to see if this is a viable method of controlling
additional panels at a high refresh rate.

Compiler Flags
--------------
* be sure to add -O3 to gcc when compiling your source.
* -ffast-math reduces some precision but improves performance for some cases.
* -mtune=native will compile for your CPU optimizations.
* -Wdouble-promotion can help identify a lot of places where you are actually using double precision (64 bit) not single precision (float)
   obviously, 64bit floats are at least 2x slower than single precision.
* -fopt-info-vec will show you loops the compiler was able to vectorize for you (SIMD, AVX, SSE, ETC)



investigation:
blue noise dithering


From a new raspibian lite system. install rpi-gpu-hub75-matrix and Linux realtime kernel for
silky smooth hub75 panels

