#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include <math.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <netinet/in.h>

#include "util.h"
#include "rpihub75.h"
#include "pixels.h"
#include "mymath.h"
#include "spsc.h"


extern char *optarg;

/**
 * @brief die with a message
 * 
 * @param message 
 */
void die(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    exit(1);
}

/**
 * @brief a wrapper around fprintf(stderr, ...) that also prints the current time
 * accepts var args
 * 
 * @param ... 
 * @return void 
 */
void debug(const char *format, ...) {
    if (CONSOLE_DEBUG) {
        va_list args;
        va_start(args, format);
        vfprintf(stderr, format, args);
        va_end(args);
    }
}



/**
 * @brief  test if a file has a specific extension
 * 
 * @param filename 
 * @param extension 
 * @return true|false
 */
bool has_extension(const char *filename, const char *extension) {
    // Find the last occurrence of '.' in the filename
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) {
        // No extension found or filename starts with '.'
        return false;
    }

    // Compare the found extension with the known extension (case-sensitive)
    return strcmp(dot + 1, extension) == 0;
}

/**
 * @brief write data to a file, exit on any failure
 * 
 * @param filename filename to write to (wb)
 * @param data data to write
 * @param size number of bytes to write
 * @return int number of bytes written, -1 on error
 */
int file_put_contents(const char *filename, const void *data, const size_t size) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        die("unable to write to file: %s\n", filename);
    }
    size_t written = fwrite(data, 1, size, file);
    if (written != size) {
        die("failed to write bytes to file: %s, [%d of %d]\n", filename, written, size);
    }
    if (fclose(file) != 0) {
        die("failed to close file: %s\n", filename);
    }
    return written;
}

/**
 * @brief read in a file, allocate memory and return the data. caller must free.
 * this function will set filesize, you do not need to pass in filesize.
 * exit on any failure
 * 
 * @param filename - file to read
 * @param filesize - pointer to the size of the file. will be set after the call. ugly i know
 * @return char* - pointer to read data. NOTE: caller must free
 */
char *file_get_contents(const char *filename, long *filesize) {
    // Open the file in binary mode
    FILE *file = fopen(filename, "rb"); 
    if (file == NULL) {
        die("Could not open file: %s\n", filename);
    }

    // Seek to the end of the file to determine its size
    fseek(file, 0, SEEK_END);
    *filesize = ftell(file);
    // Go back to the beginning of the file
    rewind(file);

    // Allocate memory for the file contents + null terminator
    char *buffer = (char *)malloc(*filesize + 1);
    if (buffer == NULL) {
        die("Memory allocation failed reading file %s of %d bytes\n", filename, *filesize);
    }

    // Read the file into the buffer
    size_t read_size = fread(buffer, 1, *filesize, file);
    if (read_size != *filesize) {
        die("Failed to read the complete file %s, read %d of %d bytes\n", filename, read_size, *filesize);
    }

    // Null-terminate the string
    buffer[*filesize] = '\0';

    if (fclose(file) != 0) {
        die("failed to close file: %s\n", filename);
    }
    return buffer;
}

/**
 * @brief print a 32 bit number in binary format to stdout
 * 
 * @param fd 
 * @param number 
 */
void binary32(FILE *fd, const uint32_t number) {
    // Print the number in binary format, ensure it only shows 11 bits
    for (int i = 31; i >= 0; i--) {
        fprintf(fd, "%d", (number >> i) & 1);
    }
}


/**
 * @brief print a 64 bit number in binary format to stdout
 * 
 * @param fd 
 * @param number 
 */
void binary64(FILE *fd, const uint64_t number) {
    // Print the number in binary format, ensure it only shows 11 bits
    for (int i = 63; i >= 0; i--) {
        fprintf(fd, "%ld", (number >> i) & 1);
    }
}

/**
 * @brief read size random data from /dev/urandom into the buffer
 * 
 * @param buffer 
 * @param size 
 * @return int - always 0
 */
int rnd(unsigned char *buffer, const size_t size) {
    // Open /dev/urandom for strong random data
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        die("unable to open /dev/urandom\n");
    }

    // Read 'size' bytes from /dev/urandom into the buffer
    ssize_t bytes_read = read(fd, buffer, size);
    if (bytes_read != size) {
        die("unable to read %d bytes from /dev/urandom\n", size);
    }

    // Close the file descriptor
    if (close(fd) != 0) {
        die("failed to close /dev/urandom\n");
    }
    return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* rnd(void *buf, size_t n) must fill buf with cryptographic random bytes from /dev/urandom */

#ifndef PIN_OE
#define PIN_OE  (1u << 0)   /* adjust to your actual OE bit mask */
#endif

/* rotate array in place by rot elements, 0 <= rot < n */
static void rotate_u32(uint32_t *a, size_t n, size_t rot) {
    if (!n || !rot) return;
    rot %= n;
    size_t m = 0;
    for (size_t start = 0; m < n; ++start) {
        uint32_t prev = a[start];
        size_t i = start;
        do {
            size_t next = i + rot;
            if (next >= n) next -= n;
            uint32_t tmp = a[next];
            a[next] = prev;
            prev = tmp;
            i = next;
            ++m;
        } while (i != start);
    }
}

/**
 * Create a jitter mask of length jitter_size, where each entry is either 0 or PIN_OE.
 * We choose exactly K entries to be the "rare" state and distribute them nearly evenly.
 * This reduces long runs and visible flicker at low brightness.
 *
 * brightness: 0..255, higher means brighter, so fewer PIN_OE blanks.
 *
 * Returns malloc()'d array of uint32_t, caller must free().
 */
uint32_t *create_jitter_mask(const uint16_t jitter_size, const uint8_t brightness) {
    const size_t n = (size_t)jitter_size;
    uint32_t *jitter = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!jitter) return NULL;

    /* Probability that a given frame is blanked by OE:
       p_blank = (255 - brightness) / 256.0
       Choose K = floor(p_blank * n) exactly, to reduce variance and clumping. */
    const uint32_t blanks_k = ((uint32_t)(255 - brightness) * n) >> 8; /* matches threshold behavior */
    const uint32_t ons_k    = n - blanks_k;

    /* Decide which state is rare and should be stratified */
    const int rare_is_blank = (blanks_k <= ons_k);

    const uint32_t rare_value     = rare_is_blank ? PIN_OE : 0u;
    const uint32_t default_value  = rare_is_blank ? 0u     : PIN_OE; /* flip these two if OE polarity is opposite */

    const uint32_t rare_k = rare_is_blank ? blanks_k : ons_k;

    /* Fast fill with the default state */
    for (size_t i = 0; i < n; ++i) jitter[i] = default_value;

    if (rare_k == 0) {
        /* all default, no flicker possible */
        return jitter;
    }
    if (rare_k >= n) {
        /* all rare, degenerate but valid */
        for (size_t i = 0; i < n; ++i) jitter[i] = rare_value;
        return jitter;
    }

    /* Stratified placement:
       Partition the sequence into rare_k contiguous spans whose sizes differ by at most 1.
       In each span, place exactly one rare element at a random offset within the span.
       This guarantees spacing and avoids long runs. */
    const size_t base_span = n / rare_k;          /* floor */
    const size_t extra     = n % rare_k;          /* first 'extra' spans have size base_span+1 */

    size_t cursor = 0;
    for (uint32_t r = 0; r < rare_k; ++r) {
        size_t span = base_span + (r < extra ? 1 : 0);

        /* draw one byte for offset, keep inside span */
        uint8_t rnd_byte = 0;
        if (span > 1) rnd(&rnd_byte, 1);
        size_t off = (span > 1) ? (rnd_byte % span) : 0;

        size_t pos = cursor + off;     /* guaranteed pos < n and unique */
        jitter[pos] = rare_value;
        cursor += span;
    }

    /* Optional: rotate by a random offset so multiple rows are out of phase */
    {
        uint16_t rot16 = 0;
        rnd((uint8_t*)&rot16, sizeof(rot16));
        size_t rot = (size_t)(rot16 % n);
        rotate_u32(jitter, n, rot);
    }

    return jitter;
}

/**
 * @brief  calculate a jitter mask for the OE pin that should randomly toggle the OE pin on/off acording to brightness
 * TODO: look for rows of > 3 bits that are all the same and spread these bits out. this will reduce flicker on the display
 * 
 * @param jitter_size  prime number > 1024 < 4096
 * @param brightness   larger values produce brighter output, max 255
 * @return uint32_t*   a pointer to the jitter mask. caller must release memory
 */
uint32_t *create_jitter_mask2(const uint16_t jitter_size, const uint8_t brightness) {
    srand(time(NULL));
    uint32_t *jitter  = (uint32_t*)malloc(jitter_size*sizeof(uint32_t));
    uint8_t *raw_data = (uint8_t*) malloc(jitter_size);

    // read random data from urandom into the raw_data buffer
    rnd(raw_data, jitter_size);

    // map raw data to the global OE jitter mask (toggle the OE pin on/off for JITTERS_SIZE frames)
    for (int i=0; i<jitter_size; i++) {
        if (raw_data[i] > brightness) {
            jitter[i] = PIN_OE;
        }
    }

    // look for runs of 4 or more 1 or 0. where we find a run, regenerate the data
    // make 3 passes at long run reduction, #defines in rpihub75.h  ....
    for (int j = 0; j < JITTER_PASSES; j++) {
        // Detect and redistribute runs of >4 identical bits
        for (int i = 0; i < jitter_size - JITTER_MAX_RUN_LEN; i++) {
            int run_length = 1;

            // Check if we have a run of similar bits (either all set or all clear)
            while (i + run_length < jitter_size && jitter[i] == jitter[i + run_length]) {
                run_length++;
            }

            // If the run length is more than 3, recalculate the bits
            if (run_length >= JITTER_MAX_RUN_LEN) {
                // recreate these bits, enchancement: pull bits from /dev/urandom 
                for (int j = i; j < i + run_length; j++) {
                    jitter[j] = ((rand() % 255) > brightness) ? PIN_OE : 0;  // Randomly set or clear the OE pin
                }
            }

            // Skip over the run we just processed
            i += run_length - 1;
        }
    }

    free(raw_data);
    return jitter;
}


/**
 * @brief count number of times this function is called, 1 every second output
 * the number of times called and reset the counter. This function can not
 * be called from multiple locations. It is not thread safe.
 * 
 * @param target_fps - target a sleep time to achieve this fps
 * @return long - returns sleep time in microseconds
 */
long calculate_fps(scene_info *scene) {
    // Variables to track FPS
    static unsigned int frame_count = 0;
    static time_t       last_time_s = 0;
    long                sleep_time  = 0;
    uint32_t target_frame_time_us   = (1000000 / scene->fps);
    time_t current_time_s           = time(NULL);
    struct timespec this_time;
    static struct timespec last_time;


    clock_gettime(CLOCK_MONOTONIC, &this_time);
 
    long frame_time = (this_time.tv_sec - last_time.tv_sec) * 1000000L +
                        (this_time.tv_nsec - last_time.tv_nsec) / 1000L;

    // Sleep for the remainder of the frame time to achieve target_fps
    sleep_time = target_frame_time_us - frame_time;

    if (sleep_time > 10 && sleep_time < 1000000L) {
        usleep(sleep_time);
    }

    frame_count++;

    // If one second has passed
    if (current_time_s != last_time_s && last_time_s != 0) {
        // Output FPS
        if (scene->show_fps) {
            double percent = 100-(double)((double)sleep_time / (double)target_frame_time_us)*100;
            printf("FPS: %d, micro second sleep per frame: %ld, CPU: %.1f%%\n", frame_count, sleep_time, percent);
        }

        // Reset frame count and update last_time
        frame_count = 0;
    } 
    last_time_s = current_time_s;

    clock_gettime(CLOCK_MONOTONIC, &last_time);
    if (scene->auto_fps) {
        long single_time = 1000000 / scene->fps;
        float percent = 100.0f - (float)sleep_time / (float)single_time * 100.0f;
        if (percent < 94.0f && scene->fps < 240) {
            scene->fps++;
        }
        else if (percent > 97.0f) {
            scene->fps--;
        }
    }

    return sleep_time;
}


/**
 * @brief map the gpio pins to memory
 * 
 * @param offset 
 * @return uint32_t* 
 */
uint32_t* map_gpio(uint32_t offset, int version) {

    loff_t peri = 0;
    int mem_fd = 0; 
    if (version == 4) {
	    peri = PERI4_BASE;
     	    mem_fd = open("/dev/gpiomem", O_RDWR | O_SYNC);
    } else if (version == 3) {
	    peri = PERI3_BASE;
     	    mem_fd = open("/dev/gpiomem", O_RDWR | O_SYNC);
    } else if (version == 5) {
	    peri = PERI5_BASE;
     	    mem_fd = open("/dev/gpiomem0", O_RDWR | O_SYNC);
    } else {
	    die("unknown pi version\n");
    }

    if (CONSOLE_DEBUG) {
        printf("peripheral address: %lx\n", peri);
    }
    asm volatile ("" : : : "memory");  // Prevents optimization
    uint32_t *map = (uint32_t *)mmap(
        NULL,
        64 * 1024 * 1024,
        (PROT_READ | PROT_WRITE),
        MAP_SHARED,
        mem_fd,
        peri
    );
        //0x1f00000000 (on pi5 for root access to /dev/mem use this address)
    if (map == MAP_FAILED) {
        die("mmap failed: %s [%ld] / [%lx]\n", strerror(errno), peri, peri);
    }
    if (mem_fd != 0) {
    	close(mem_fd);
    }
    if (CONSOLE_DEBUG) {
        printf("gpio mapped\n");
    }
    return map;
}

// convert normalized float 0.0-1.0 to q8 1-255
uint8_t math_norm_q8(float x) {
    if (x <= 0) return 0;
    if (x >= 1) return 255;
    return (uint8_t)((x * 253.0f) + 1.0f);
}



// Function to get a character without needing Enter
char getch() {
    struct termios oldt, newt;
    char ch;
    tcgetattr(STDIN_FILENO, &oldt);           // Get current terminal attributes
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);         // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);  // Set new attributes
    ch = getchar();                           // Get the character
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // Restore old attributes
    return ch;
}


/**
 * @brief remove whitespace from string
 * @param s 
 * @return char* 
 */
char *str_trim_spaces(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;
    char *e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}


/**
 * @brief parse a string to a float, return -1 on error
 * @param s the string to parse into a float
 * @param out pointer to the float to parse into
 * @return int 
 */
int parse_float(const char *s, float *out) {
    errno = 0;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || errno == ERANGE) return -1;
    // ignore trailing spaces
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') return -1;
    *out = (float)v;
    return 0;
}


/* Parse optarg like "1:1:1,2:1:3" into dst[0..*out_count-1].
Missing panels default to 1. Returns 0 on success, -1 on error.
@param max_out maximum panel type number (can not exceed number of defined panel types)
*/
int parse_panel_types(const char *arg,
                       uint8_t *dst,
                       size_t max_out,
                       uint8_t *out_count)
{
    if (!arg || !dst || max_out == 0) {
        return -1;
    }

    char *buf = strdup(arg);
    if (!buf) {
        return -1;
    }


    size_t count = 0;
    size_t max_type = 0;
    char *save_outer = NULL;
    for (char *tok = strtok_r(buf, ",", &save_outer);
         tok;
         tok = strtok_r(NULL, ",", &save_outer))
    {
        char *panel = str_trim_spaces(tok);

        // split r:g:b
        int idx = 0;
        char *save_inner = NULL;
        for (char *ctok = strtok_r(panel, ":", &save_inner);
             ctok;
             ctok = strtok_r(NULL, ":", &save_inner), idx++)
        {
            if (count >= MAX_PANELS) {
                break;
            }
            int v;
            v = atoi(str_trim_spaces(ctok));
            if (v > max_type) {
                max_type = v;
            }
            dst[count++] = v;
        }

    }

    // output the maximum panel type number
    if (out_count) {
        *out_count = max_type;
    }
    return 0;
}

/* Parse optarg like "1:1:1,1:.79:.89" into dst[0..*out_count-1].
   Missing channels default to 1.0. Returns 0 on success, -1 on error. */
int parse_panel_scales(const char *arg,
                       panel_rgb_scale *dst,
                       size_t max_out,
                       int *out_count)
{
    if (!arg || !dst || max_out == 0) {
        return -1;
    }

    char *buf = strdup(arg);
    if (!buf) {
        return -1;
    }


    size_t count = 0;
    char *save_outer = NULL;
    for (char *tok = strtok_r(buf, ",", &save_outer);
         tok && count < max_out;
         tok = strtok_r(NULL, ",", &save_outer))
    {
        char *panel = str_trim_spaces(tok);

        // defaults
        panel_rgb_scale s = {255, 255, 255};

        // split r:g:b
        int idx = 0;
        char *save_inner = NULL;
        for (char *ctok = strtok_r(panel, ":", &save_inner);
             ctok && idx < 3;
             ctok = strtok_r(NULL, ":", &save_inner), idx++)
        {
            float v;
            if (parse_float(str_trim_spaces(ctok), &v) != 0) {
                free(buf);
                return -1;
            }
            if      (idx == 0) s.red_q8   = math_norm_q8(v);
            else if (idx == 1) s.green_q8 = math_norm_q8(v);
            else               s.blue_q8  = math_norm_q8(v);
        }

        dst[count++] = s;
    }

    free(buf);
    if (out_count) {
        *out_count = count;
    }
    return 0;
}


/* Parse optarg like "-12:24:22,-30:-32:45" into dst[0..*out_count-1].
   Missing channels default to 0. Returns 0 on success, -1 on error. */
int parse_panel_offsets(const char *arg,
                       panel_rgb_offset *dst,
                       size_t max_out,
                       int *out_count)
{
    if (!arg || !dst || max_out == 0) {
        return -1;
    }

    char *buf = strdup(arg);
    if (!buf) {
        return -1;
    }


    size_t count = 0;
    char *save_outer = NULL;
    for (char *tok = strtok_r(buf, ",", &save_outer);
         tok && count < max_out;
         tok = strtok_r(NULL, ",", &save_outer))
    {
        char *panel = str_trim_spaces(tok);

        // defaults
        panel_rgb_offset s = {0, 0, 0};

        // split r:g:b
        int idx = 0;
        char *save_inner = NULL;
        for (char *ctok = strtok_r(panel, ":", &save_inner);
             ctok && idx < 3;
             ctok = strtok_r(NULL, ":", &save_inner), idx++)
        {
            int v = atoi(str_trim_spaces(ctok));
            if      (idx == 0) s.red_q8   = MAX(-127, MIN(v, 127));
            else if (idx == 1) s.green_q8 = MAX(-127, MIN(v, 127));
            else               s.blue_q8  = MAX(-127, MIN(v, 127));
        }

        dst[count++] = s;
    }

    free(buf);
    if (out_count) {
        *out_count = count;
    }
    return 0;
}



 
void configure_gpio_4(uint32_t *PERIBase, int version) {

    for (uint32_t pin_num=2; pin_num<28; pin_num++) {
        asm volatile ("" : : : "memory");  // Prevents optimization
					   
	uint32_t sel_reg = (pin_num / 10);
	uint32_t sel_shift = ((pin_num % 10) * 3);

	PERIBase[sel_reg] = (PERIBase[sel_reg] & ~(0b111 << sel_shift)) | (0b001 << sel_shift);
	// set pin to output mode
	//PERIBase[sel_reg] &= ~(0b111 << sel_shift);
	//PERIBase[sel_reg] |= (0b111 << sel_shift);
	printf("%d output mode enabled\n", pin_num);

	/*
	PERIBase[138] &= ~(1 << (pin_num % 32));

	printf("%d fast slew enabled\n", pin_num);
	*/

	uint pdn_reg   = (57 + (pin_num / 16));
	uint pdn_shift = ((pin_num % 16) * 2);

	PERIBase[pdn_reg] &= ~(0b11 << pdn_shift);
	PERIBase[pdn_reg] |= 0b10 << pdn_shift;
	printf("%d pull down enabled\n", pin_num);

	uint drive_reg   = (129 + (pin_num / 16));
	uint drive_shift = ((pin_num % 16) * 2);

	PERIBase[drive_reg] &= ~(0b11 << drive_shift);
	PERIBase[drive_reg] |= 0b11 << drive_shift;
	printf("%d pull down enabled\n", pin_num);




    }
}

/**
 * @brief set the GPIO pins for hub75 operation.  this is based on hzeller's active board pinouts
 * @see https://github.com/hzeller/rpi-rgb-led-matrix
 * 
 * @param PERIBase 
 */
void configure_gpio(uint32_t *PERIBase, int version) {
    // set all GPIO pins to output mode, fast slew rate, pull down enabled
    // https://www.i-programmer.info/programming/148-hardware/16887-raspberry-pi-iot-in-c-pi-5-memory-mapped-gpio.html
    //

    uint32_t pad_off = PAD5_OFFSET; 
    uint32_t gpio_off = GPIO5_OFFSET; 
    uint32_t rio_off = RIO5_OFFSET; 
    if (version <= 4) {
	    configure_gpio_4(PERIBase, version);
	    return;
    }

    for (uint32_t pin_num=2; pin_num<28; pin_num++) {
        asm volatile ("" : : : "memory");  // Prevents optimization
        uint32_t *PADBase  = PERIBase + pad_off;
        uint32_t *pad = PADBase + 1;   
        uint32_t *GPIOBase = PERIBase + gpio_off;
        uint32_t *RIOBase = PERIBase + rio_off;

        GPIO[pin_num].ctrl = 5;
        uint8_t slew_rate = 0x15;
        if (pin_num == PIN_CLK || pin_num == PIN_LATCH) {
            slew_rate = 0x01; // slower slew rate for clock and latch
        }
        pad[pin_num] = slew_rate;

        rioSET->OE = 0x01<<pin_num;     // these 2 lines actually set the pin to output mode
        rioSET->Out = 0x01<<pin_num;
    }
}

/**
 * @brief display command line scene configuration options and exit
 * 
 * @param argc 
 * @param argv 
 */
void usage(int argc, char **argv) {
    die(
        "Usage: %s\n"
        "     -s <file>         GPU fragment shader, or mp4 file to render\n"
        "     -x <width>        total image width         (16-512)\n"
        "     -y <height>       total image height        (16-512)\n"
        "     -w <width>        panel width               (16/32/64/128) default 64\n"
        "     -h <height>       panel height              (16/32/64) default 64\n"
        "     -o <RGB>          panel pixel order         (RGB, RBG, BGR, BRG, GRB, GBR)\n"
        "     -f <fps>          target frames per second  (1-255) - omit to auto scale fps to 95%% CPU\n"
        "     -p <num ports>    number of ports           (1-3)\n"
        "     -c <num chains>   number of panels chained  (1-16)\n"
        "     -g <gamma>        gamma correction          (1.0-2.8)\n"
        "     -d <bit depth>    bit depth                 (8-64)\n"
        "     -b <brightness>   overall brightness level  (1-254)\n"
        "     -q                enable quantization dithering\n"
        "     -l <dither>       spatial dithering intensity level (0-10)\n"
        "     -m <frames>       motion blur frames        (0-32)\n"
        "     -i <mapper>       image mapper (mirror, flip, mirror_flip)\n"
        "     -t <tone_mapper>  (aces, reinhard, none, saturation, sigmoid, hable)\n"
        "     -j                adjust brightness in pixel BCM, only for Pi3-4\n"
        "     -z                run LED calibration script\n"
        "     -n                display data from UDP server on port %d (untested)\n"
        "     -v                display current FPS and Panel refresh Hz\n"
        "     -O <r:g:b,r:g:b>  panel color correction offset ammount, +-128 for each color, comma delimited\n"
        "                       add or subtract this ammount to each panels rgb channels\n"
        "     -T <r:g:b,r:g:b>  panel color correction multiple factor, 0.0-1.0 for each color, comma delimited\n"
        "                       multiply this ammount to each panels rgb channels\n"
        "     -P <T:T:T,T:T:T>  panel type mapping, each T is a panel type number (0-7).\n"
        "                       columns are : sepearted, rows are , separated (0:0:0,0:1:1,1:1:1)\n"
        "     -?                this help\n"
        "     NOTE: if runing as root, it will automatically use the real-time scheduler which reduces flicker\n"
        , argv[0], SERVER_PORT);
}


/**
 * @brief Get the nth token of str split by delimiter
 * not thread safe, returns a pointer to a static buffer
 * 
 * @param str 
 * @param delimiter 
 * @param position 
 * @return char* 
 */
char *get_nth_token(const char *str, char delimiter, int position) {
    const int MAX_TOKEN_LENGTH = 256;
    static char result[256]; // Stack-allocated buffer for the token
    char temp[strlen(str) + 1]; // Temporary copy of the input string
    strncpy(temp, str, sizeof(temp));
    temp[sizeof(temp) - 1] = '\0'; // Ensure null termination

    char *token = strtok(temp, &delimiter);
    int index = 0;

    // Traverse tokens until the n-th token is found
    while (token != NULL) {
        if (index == position) {
            strncpy(result, token, MAX_TOKEN_LENGTH - 1);
            result[MAX_TOKEN_LENGTH - 1] = '\0'; // Ensure null termination
            return result; // Return stack-allocated token
        }
        token = strtok(NULL, &delimiter);
        index++;
    }

    // Return an empty string if n-th token is not found
    result[0] = '\0';
    return result;
}

/**
 * @brief creae a new scene with default values
 */
scene_info *new_scene() {
    // setup all scene configuration info
    scene_info *scene = (scene_info*)malloc(sizeof(scene_info));
    memset(scene, 0, sizeof(scene_info));
    scene->width = IMG_WIDTH;
    scene->height = IMG_HEIGHT;
    scene->panel_height = PANEL_HEIGHT;
    scene->panel_width = PANEL_WIDTH;
    scene->num_chains = 0;
    scene->num_ports = 0;
    // scene->buffer_ptr = 0;
    scene->stride = 3;
    scene->gamma = GAMMA;
    scene->red_gamma = RED_GAMMA_SCALE;
    scene->green_gamma = GREEN_GAMMA_SCALE;
    scene->blue_gamma =BLUE_GAMMA_SCALE; 
    scene->red_linear = RED_SCALE;
    scene->green_linear = GREEN_SCALE;
    scene->blue_linear = BLUE_SCALE;
    scene->jitter_brightness = true;

    scene->bit_depth = 64;
    scene->pixel_order = PIXEL_ORDER_RGB;
    scene->panel_order = PANEL_RGB;
    scene->tone_mapper = copy_tone_mapperF;
    scene->brightness = 200;
    scene->motion_blur_frames = 0;
    scene->do_render = TRUE;
    scene->dither = 0.0f;
    scene->quant_dither = false;
    scene->panel_types[0] = 0;
    scene->num_panel_types = 0;
    scene->frame_index = 0;
    scene->auto_fps = true;

    // default to 60 fps
    scene->fps = 60;
    scene->show_fps = FALSE;

    return scene;
}

/**
 * @brief create a default scene setup using the #DEFINE values
 * parse command line options to override. This is a great way
 * to test your setup easily from command line
 * 
 * @param argc 
 * @param argv 
 * @return scene_info* 
 */
scene_info *default_scene(int argc, char **argv) {

    // initialize the new scene
    scene_info *scene = new_scene();

    // print usage if no arguments
    if (argc < 2) { 
        usage(argc, argv);
    }

    // Parse command-line options
    int opt = 0;
    int num_scales = 0;
    while ((opt = getopt(argc, argv, "O:T:P:o:x:y:w:h:s:f:p:c:g:d:m:b:t:l:i:jzvq?")) != -1) {
        switch (opt) {
        case 's':
            scene->shader_file = optarg;
            break;
        case 'x':
            scene->width = atoi(optarg);
            break;
        case 'y':
            scene->height = atoi(optarg);
            break;
        case 'w':
            scene->panel_width = atoi(optarg);
            break;
        case 'h':
        case '?':
            scene->panel_height = atoi(optarg);
            if (scene->panel_height <= 1) {
                usage(argc, argv);
            }
            break;
        case 'f':
            scene->fps = atoi(optarg);
            scene->auto_fps = false;
            break;
        case 'p':
            scene->num_ports = atoi(optarg);
            break;
        case 'c':
            uint8_t t = (uint8_t)atoi(optarg);
            printf("set chains to %d\n", t);
            scene->num_chains = t;
            break;
        case 'g':
            scene->gamma = atof(optarg);
            break;
        case 'd':
            scene->bit_depth = atoi(optarg);
            break;
        case 'b':
            scene->brightness = atoi(optarg);
            break;
        case 'm':
            scene->motion_blur_frames = atoi(optarg);
            break;
        case 'l':
            scene->dither = atof(optarg);
            scene->dither = clampf(scene->dither, 0.0f, 10.0f);
            break;
        case 'q':
            scene->quant_dither = true;
            break;
        case 'j':
            scene->jitter_brightness = false;
            break;
        case 'T':
            if (parse_panel_scales(optarg, scene->panel_scale, MAX_PANEL_TYPES, &num_scales) != 0) {
                fprintf(stderr, "invalid -T format: %s\n", optarg);
                exit(1);
            }
            break;
        case 'O':
            if (parse_panel_offsets(optarg, scene->panel_offset, MAX_PANEL_TYPES, &num_scales) != 0) {
                fprintf(stderr, "invalid -O format: %s\n", optarg);
                exit(1);
            }
            break;
        case 'P':
            if (parse_panel_types(optarg, scene->panel_types, MAX_PANEL_TYPES, &scene->num_panel_types) != 0) {
                fprintf(stderr, "invalid -T format: %s\n", optarg);
                exit(1);
            }
            break;
        case 'z':
            scene->gamma = -99.0f;
            break;
        case 'v':
            scene->show_fps = TRUE;
            break;
        case 't':
            char *lvl = get_nth_token(optarg, ':', 1);
            // printf("lvl: %s\n", lvl);
            float level = atof(lvl);
            char *tone = get_nth_token(optarg, ':', 0);
            scene->tone_level = 1.0f;
            if (strcmp(tone, "aces") == 0) {
                scene->tone_mapper = aces_tone_mapperF;
            } else if (strcmp(tone, "reinhard") == 0) {
                if (level < 0.1f || level  > 5.0f) {
                    level = 1.0f;
                }
                debug("reinhard level %f\n", (double)level);
                scene->tone_level = level;
                scene->tone_mapper = reinhard_tone_mapperF;
            } else if (strcmp(tone, "hable") == 0) {
                scene->tone_mapper = hable_tone_mapperF;
            } else if (strcmp(tone, "none") == 0) {
                scene->tone_mapper = copy_tone_mapperF;
            } else if (strcmp(tone, "saturation") == 0) {
                if (level < 0.1f || level  > 5.0f) {
                    level = 1.0f;
                }
                debug("saturation level %f\n", (double)level);
                scene->tone_level = level;
                scene->tone_mapper = saturation_tone_mapperF;
            } else if (strcmp(tone, "sigmoid") == 0) {
                if (level < 0.1f || level  > 5.0f) {
                    level = 1.0f;
                }
                scene->tone_level = level;
                scene->tone_mapper = sigmoid_tone_mapperF;
            } else {
                die("Unknown tone mapper: %s, must be one of (aces, reinhard, hable, saturation, sigmoid, none)\n", optarg);
            }
            break;
        case 'i':
            if (strcasecmp(optarg, "u") == 0) {
                scene->image_mapper = u_mapper_impl;
            }
            else if (strcasecmp(optarg, "flip") == 0) {
                scene->image_mapper = flip_mapper;
            }
            else if (strcasecmp(optarg, "mirror") == 0) {
                scene->image_mapper = mirror_mapper;
            }
            else if (strcasecmp(optarg, "mirror_flip") == 0) {
                scene->image_mapper = mirror_flip_mapper;
            } else {
                die("Unknown image mapper: %s, must be one of (u, mirror, flip, mirror_flip)\n", optarg);
            }
            break;
        case 'o':
            if (strcasecmp(optarg, "RGB") == 0) {
                scene->panel_order = PANEL_RGB;  
            }
            else if (strcasecmp(optarg, "RBG") == 0) {
                scene->panel_order = PANEL_RBG;  
            }
            else if (strcasecmp(optarg, "BGR") == 0) {
                scene->panel_order = PANEL_BGR;  
            }
            else if (strcasecmp(optarg, "BRG") == 0) {
                scene->panel_order = PANEL_BRG;  
            }
            else if (strcasecmp(optarg, "GRB") == 0) {
                scene->panel_order = PANEL_GRB;  
            }
            else if (strcasecmp(optarg, "GBR") == 0) {
                scene->panel_order = PANEL_GBR;  
            }
            else {
                die("Unknown panel pixel order: %s, must be one of (RGB, RBG, BGR, BRG, GRB, GBR)\n", optarg);
            }
            break;

        default:
            usage(argc, argv);
        }
    }
    if (opt == 0) {
        usage(argc, argv);
    }

    if (scene->num_chains == 0) {
        int calcualted_chains = ceil(scene->width / scene->panel_width);
        printf("calculated: %d chains\n", calcualted_chains);
        scene->num_chains = MAX(scene->num_chains, calcualted_chains);
    }

    if (scene->num_ports == 0) {
        int calcualted_ports = MIN(3, ceil(scene->width / scene->panel_width));
        printf("calculated: %d ports\n", calcualted_ports);
        scene->num_ports = MAX(scene->num_ports, calcualted_ports);
    }



    if (num_scales != 0 && num_scales != scene->num_panel_types) {
        fprintf(stderr, "-T does not match -P !!\n\n");
        fprintf(stderr, "each R:G:B comma 'pair' passed to -T represents a single panel type scaling for RGB planes\n");
        fprintf(stderr, "each output port has it's panel type defined in -P\n");
        fprintf(stderr, "seperate panel types for each port with a colon, separate ports with a comma\n\n");
        fprintf(stderr, "to define 2 output ports of 3 panels each with the second row scaled to 80%% brightness, type:\n");
        fprintf(stderr, "  define 2 panel 'types': -T 1:1:1,.8:.8:.8\n");
        fprintf(stderr, "  map the 6 panels to their 'types': -P 0:0:0,1:1:1\n\n");
        exit(1);
    }

    return scene;
}

/**
 * @brief draw various test patterns to the display
 * 
 * @param arg 
 * @return void* 
 */
void *calibrate_panels(void *arg) {
    scene_info *scene = (scene_info*)arg;
    printf("Point your browser to: calibration.html the rpi_gpu_hub75_matrix directory\n");
    printf("After calibrating each set of vertical bar.  press any key on your browser window to continue\n\n");

    printf("Keyboard commands:\n");
    printf("a - lower gamma,          A - raise gamma\n");
    printf("r - lower red linearly,   R - raise red linearly\n");
    printf("g - lower green linearly, G - raise green linearly\n");
    printf("b - lower blue linearly,  B - raise blue linearly\n");
    printf("t - lower red gamma,      T - raise red gamma\n");
    printf("h - lower green gamma,    H - raise green gamma\n");
    printf("n - lower blue gamma,     N - raise blue gamma\n");
    printf("<enter> next color\n");
	float rgb_bars[12][3] = {
        {1.0, 1.0, 1.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {0.0, 0.4, 0.8},
        {0.8, 0.4, 1.0},
        {0.8, 0.0, 0.4},
        {0.0, 0.8, 0.4},
        {0.4, 0.8, 0.0},
        {0.4, 0.0, 0.8},
        {0.4, 0.6, 0.8}
    };

    printf("created rgb bars\n");

    RGB c1;/* = {0x04, 0x20, 0x7c};
    RGB c2 = {0x7c, 0x96, 0x09};
    RGB c3 = {0x9e, 0x01, 0x38};
    RGB c4 = {0xed, 0x12, 0xb2};
    RGB c5 = {0xce, 0x9e, 0x00};
    */

    // const scene_info *scene = (scene_info*)arg;
    uint8_t *image          = (uint8_t*)malloc(scene->width * scene->height * scene->stride);
    memset(image, 0, scene->width * scene->height * scene->stride);
    uint8_t j = 0, num_col = 5;
    printf("malloc memory %dx%d\n", scene->width, scene->height);

    while (j < 12) {

        for (int y = 0; y < scene->height; y++) {
            for (int x=0; x<scene->panel_width; x++) {
                RGB *pixel = (RGB *)(image + ((y * scene->panel_width) + x) * scene->stride);
                //printf("pixel ...%dx%d\n", x, y);
                int i = x / floor(scene->panel_width / num_col);
                //if (y == 0) { printf("x = %d, i = %d = v:%d\n", x, i, (254 / (i+1))); }
                c1.r = rgb_bars[j][0] * (254 / (i+1));
                c1.g = rgb_bars[j][1] * (254 / (i+1));
                c1.b = rgb_bars[j][2] * (254 / (i+1));
                *pixel = c1;
            }
        }

        map_byte_image_to_bcm(scene, image);
        char ch = getch();
        if (ch == 'a') {
            scene->gamma -= 0.01f;
            printf("gamma        down = %f\n", (double)scene->gamma);
        } else if (ch == 'A') {
            scene->gamma += 0.01f;
            printf("gamma        up   = %f\n", (double)scene->gamma);
        } else if (ch == 'g') {
            scene->green_linear -= 0.01f;
            printf("green_linear down = %f\n", (double)scene->green_linear);
        } else if (ch == 'G') {
            scene->green_linear += 0.01f;
            printf("green_linear up   = %f\n", (double)scene->green_linear);
        } else if (ch == 'h') {
            scene->green_gamma -= 0.01f;
            printf("green_gamma down  = %f\n", (double)scene->green_gamma);
        } else if (ch == 'H') {
            scene->green_gamma += 0.01f;
            printf("green_gamma up    = %f\n", (double)scene->green_gamma);
        } else if (ch == 't') {
            scene->red_gamma -= 0.01f;
            printf("red_gamma down    = %f\n", (double)scene->red_gamma);
        } else if (ch == 'T') {
            scene->red_gamma += 0.01f;
            printf("red_gamma up      = %f\n", (double)scene->red_gamma);
        } 
         else if (ch == 'n') {
            scene->blue_gamma -= 0.01f;
            printf("blue_gamma down   = %f\n", (double)scene->blue_gamma);
        } else if (ch == 'N') {
            scene->blue_gamma += 0.01f;
            printf("blue_gamma up     = %f\n", (double)scene->blue_gamma);
        } 
         else if (ch == 'b') {
            scene->blue_linear -= 0.01f;
            printf("blue_linear down  = %f\n", (double)scene->blue_linear);
        } else if (ch == 'B') {
            scene->blue_linear += 0.01f;
            printf("blue_linear up    = %f\n", (double)scene->blue_linear);
        } else if (ch == 'R') {
            scene->red_linear += 0.01f;
            printf("red_linear up     = %f\n", (double)scene->red_linear);
        } else if (ch == 'r') {
            scene->red_linear -= 0.01f;
            printf("red_linear down   = %f\n", (double)scene->red_linear);
        }
        else if (ch == '\n') {
            // show all values
            printf("gamma: %f, red_linear: %f, green_linear: %f, blue_linear: %f\n", (double)scene->gamma, (double)scene->red_linear, (double)scene->green_linear, (double)scene->blue_linear);
            // show lal gamma values
            printf("red_gamma: %f, green_gamma: %f, blue_gamma: %f\n", (double)scene->red_gamma, (double)scene->green_gamma, (double)scene->blue_gamma);
            j++;
            if (j >= 12) {
                die("calibration complete\n");
            }
        }
        scene->tone_mapper = (scene->tone_mapper == NULL) ? copy_tone_mapperF : NULL;
    }

    return NULL;
}




/**
 * @brief function to crete a udp server and pull raw frame data. see the udp_packet struct
 * for info on the data format
 * 
 * exits on any error
 * @param arg 
 * @return void* 
 */
void* receive_udp_data(void *arg) {
    scene_info *scene = (scene_info *)arg; // dereference the scene info
    int sock;
    struct sockaddr_in server_addr;
    struct udp_packet packet;
    uint16_t max_frame_sz  = scene->width * scene->height * scene->stride;
    uint16_t max_packet_id = ceilf((float)max_frame_sz / (float)(PACKET_SIZE - 10));

    uint8_t *image_data = malloc(max_frame_sz * 16);

    // Create UDP socket
    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        die("Server socket creation failed\n");
    }

    // Set up server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket
    if (bind(sock, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        die("Bind failed");
    }

    for(;;) {
        socklen_t len = sizeof(server_addr);
        int n = recvfrom(sock, &packet, sizeof(packet), 0, (struct sockaddr *)&server_addr, &len);
        if (n < 0) {
            close(sock);
            die("Receive failed");
        }

        // Check preamble for data alignment
        if (ntohl(packet.preamble) != PREAMBLE) {
            printf("Invalid preamble received\n");
            continue;
        }

        const uint16_t packet_id = ntohs(packet.packet_id);
        const uint16_t total_packets = ntohs(packet.total_packets);


        const uint16_t frame_num = ntohs(packet.frame_num) % 8;
        const uint16_t frame_off = MIN(MIN(packet_id, max_packet_id) * PACKET_SIZE, max_frame_sz-1);

        memcpy(image_data + ((frame_num * max_frame_sz) + frame_off), packet.data, PACKET_SIZE - 10);
        if (packet_id == total_packets) {
            // map to pwm data
            map_byte_image_to_bcm(scene, image_data + (frame_num * max_frame_sz));
        }

    }

    close(sock);
    return NULL;
}
