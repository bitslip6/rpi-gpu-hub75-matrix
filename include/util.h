#include <unistd.h>
#include <stdint.h>


#ifndef __UTIL_H__
#define __UTIL_H__
#include "rpihub75.h"


/**
 * @brief used to set the pin mode of a GPIO pin using mmaped /dev/gpiomem0 
 * 
 */
typedef struct{
    uint32_t status;
    uint32_t ctrl; 
}GPIOregs;
#define GPIO ((GPIOregs*)GPIOBase)

/**
 * @brief helper struct for accessing the RIO registers of a GPIO pin
 * 
 */
typedef struct
{
    volatile uint32_t Out;
    volatile uint32_t OE;
    volatile uint32_t In;
    volatile uint32_t InSync;
} rioregs;


/**
 * @brief printf() a message to stderr and exit with a non-zero status
 * 
 * @param message 
 * @param ... 
 */
void die(const char *message, ...);

/**
 * @brief display a message to stderr if CONSOLE_DEBUG is defined
 * 
 * @param format 
 * @param ... 
 */
#define HAVE_DEBUG_FUNC 1
void debug(const char *format, ...);

/**
 * @brief return the difference between two timespecs in microseconds
 */
int64_t ts_diff_us(const struct timespec *a, const struct timespec *b);


/**
 * @brief Adds a specified number of milliseconds to a timespec structure, handling the overflow
 *              of nanoseconds to seconds.
 **/
void timespec_add_ms(struct timespec *ts, long ms);

/**
 * @brief safe free a pointer and set it to NULL
 * 
 * @param pp pointer to the pointer to free
 */
void safe_free(void **pp);

#ifndef SAFE_FREE
#define SAFE_FREE(p)                                      \
    do {                                                  \
        void **__pp = (void**)&(p);                       \
        if (__pp && *__pp) {                              \
            free(*__pp);  /* or mg_free(*__pp); */        \
            *__pp = NULL;                                 \
        }                                                 \
    } while (0)
#endif



/**
 * @brief  calculate a jitter mask for the OE pin that should randomly toggle the OE pin on/off acording to brightness
 * TODO: look for rows of > 3 bits that are all the same and spread these bits out. this will reduce flicker on the display
 * 
 * @param jitter_size  prime number > 1024 < 4096
 * @param brightness   larger values produce brighter output, max 255
 * @return uint32_t*   a pointer to the jitter mask. caller must release memory
 */
uint32_t *jitter_create(const uint16_t jitter_size, const uint8_t brightness, bool jitter_brightness);

/**
 * @brief write data to a file, exit on any failure
 * 
 * @param filename filename to write to (wb)
 * @param data data to write
 * @param size number of bytes to write
 * @return int number of bytes written, -1 on error
 */

size_t file_put_contents(const char *filename, const void *data, const size_t size);

/**
 * @brief read in a file, allocate memory and return the data. caller must free.
 * this function will set filesize, you do not need to pass in filesize.
 * exit on any failure
 * 
 * @param filename - file to read
 * @param filesize - pointer to the size of the file. will be set after the call. ugly i know
 * @return char* - pointer to read data. NOTE: caller must free
 */
char *file_get_contents(const char *filename, size_t *filesize);

/**
 * @brief print a 32 bit number in binary format to stdout
 * 
 * @param fd 
 * @param number 
 */
void binary32(FILE *fd, const uint32_t number);

/**
 * @brief print a 64 bit number in binary format to stdout
 * 
 * @param fd 
 * @param number 
 */
void binary64(FILE *fd, const uint64_t number);

/**
 * @brief read size random data from /dev/urandom into the buffer
 * 
 * @param buffer 
 * @param size 
 * @return int - always 0
 */
int rnd(unsigned char *buffer, const size_t size);

/**
 * @brief compute exponential moving average
 */
float math_ema(float new_value, float old_value, float alpha);

/**
 * @brief map the gpio pins to memory
 * 
 * @param offset 
 * @return uint32_t* 
 */
uint32_t* map_gpio(int version);

/**
 * @brief remove whitespace from string
 * @param s 
 * @return char* 
 */
char *str_trim_spaces(char *s);

int parse_float(const char *s, float *out);
uint8_t math_norm_q8(float x);
Normal normalize(float x, float period);


/**
 * @brief set the GPIO pins for hub75 operation.  this is based on hzeller's active board pinouts
 * @see https://github.com/hzeller/rpi-rgb-led-matrix
 * 
 * @param PERIBase 
 */
void configure_gpio(uint32_t *PERIBase, int version);

/**
 * @brief display command line scene configuration options and exit
 * 
 * @param argc 
 * @param argv 
 */
void usage(int argc, char **argv);


/**
 * @brief draw various test patterns to the display
 * 
 * @param arg 
 * @return void* 
 */
void *calibrate_panels(void *arg);

/**
 * @brief function to crete a udp server and pull raw frame data. see the udp_packet struct
 * for info on the data format
 * 
 * exits on any error
 * @param arg 
 * @return void* 
 */
void* receive_udp_data(void *arg);


/**
 * @brief test if a file exists
 */
bool file_exists(const char *filename);


#endif
