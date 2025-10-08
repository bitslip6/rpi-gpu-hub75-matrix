CFLAGS_BASE := -O3 -g -pipe -fno-math-errno -ffast-math
#CFLAGS_BASE := -O2 -g -pipe 
CFLAGS_CPU  ?= $(shell sh detect_flags.sh)
CFLAGS      := $(CFLAGS_BASE) $(CFLAGS_CPU)


# Compiler and flags
CC = gcc
#CFLAGS = -DNDEBUG=1 -std=gnu2x -fPIC -ffast-math -fopt-info-vec -funroll-loops -ftree-vectorize -mtune=native -O3 -Wall -Wpedantic -Wdouble-promotion -Iinclude
CFLAGS = -DNDEBUG=1 -std=gnu2x -fPIC -funroll-loops -ftree-vectorize -Wall -Wpedantic -Iinclude 
#CFLAGS = -DNDEBUG=1 -std=gnu2x -fPIC -funroll-loops -ftree-vectorize -Wall -Wpedantic -Iinclude -fno-omit-frame-pointer -fsanitize=address,undefined,leak -fno-common -U_FORTIFY_SOURCE
LDFLAGS = -lpthread -lrt -lm -lc -lavformat -lavcodec -lswscale -lavutil 
CFLAGS += $(DEF)
CFLAGS += $(CFLAGS_BASE)
CFLAGS += `pkg-config --cflags libavcodec` 

# Directories
PREFIX = /usr/local
INCLUDEDIR = $(PREFIX)/include/rpihub75
LIBDIR = $(PREFIX)/lib
BUILDDIR = build

# Source files
SRC_COMMON = src/util.c src/pixels.c src/rpihub75.c
SRC_GPU = src/gpu.c src/video.c

# Library output names
LIB_NO_GPU = librpihub75.so
LIB_GPU = librpihub75_gpu.so

# Object files
# OBJ_COMMON = $(SRC_COMMON:%.c=$(BUILDDIR)/%.o)
# OBJ_GPU = $(SRC_GPU:%.c=$(BUILDDIR)/%.o)
OBJ_COMMON = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(SRC_COMMON))
OBJ_GPU = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(SRC_GPU))

# Check for OpenGL ES, GBM, and EGL libraries
GLESV2_FOUND := $(shell pkg-config --exists glesv2 && echo yes || echo no)
GBM_FOUND := $(shell pkg-config --exists gbm && echo yes || echo no)
EGL_FOUND := $(shell pkg-config --exists egl && echo yes || echo no)
AVCODEC_FOUND := $(shell pkg-config --exists libavcodec && echo yes || echo no)
SWSCALE_FOUND := $(shell pkg-config --exists libswscale && echo yes || echo no)
AVUTIL_FOUND := $(shell pkg-config --exists libavutil && echo yes || echo no)

# Targets
.PHONY: all clean install check-libs example

# Default target to build both libraries
all: check-libs $(LIB_NO_GPU) $(LIB_GPU)

lib: $(LIB_NO_GPU)

libgpu: $(LIB_GPU)

# Create build directory if not exists
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Check for required GPU libraries
check-libs:
ifeq ($(GLESV2_FOUND),no)
    $(error "GLESv2 library not found. Please install it. sudo apt-get install libgles2-mesa-dev")
endif
ifeq ($(GBM_FOUND),no)
    $(error "GBM library not found. Please install it. sudo apt-get install libgbm-dev")
endif
ifeq ($(EGL_FOUND),no)
    $(error "EGL library not found. Please install it. sudo apt-get install libegl1-mesa-dev")
endif
ifeq ($(AVCODEC_FOUND),no)
    $(error "AVCodec library not found. Please install it. sudo apt-get install libavcodec-dev")
endif
ifeq ($(SWSCALE_FOUND),no)
    $(error "SWscale library not found. Please install it. sudo apt-get install libswscale-dev")
endif
ifeq ($(AVUTIL_FOUND),no)
    $(error "SWscale library not found. Please install it. sudo apt-get install libavutil-dev")
endif

# No-GPU library (without gpu.c, no OpenGL)
$(LIB_NO_GPU): $(OBJ_COMMON) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(CFLAGS_CPU) -shared -o $@ $(OBJ_COMMON)  $(LDFLAGS)

# GPU-enabled library (with gpu.c, linked to OpenGL)
$(LIB_GPU): $(OBJ_COMMON) $(OBJ_GPU) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(CFLAGS_CPU) -shared -o $@ $(OBJ_COMMON) $(OBJ_GPU) $(LDFLAGS) `pkg-config --libs glesv2 gbm egl`

# New example target to compile example.c
example: example.c $(LIB_GPU)
	$(CC) example.c -Wall -O3 -lrpihub75_gpu -o example

# New example target to compile scratch.c
scratch: tests/scratch.c $(LIB_GPU)
	$(CC) tests/scratch.c -Wall -O3 -lrpihub75_gpu -o scratch


# Install target
install: all
	# Create directories
	mkdir -p $(INCLUDEDIR)
	mkdir -p $(LIBDIR)
	# Copy header files
	chmod +x *.so
	cp include/rpihub75.h $(INCLUDEDIR)
	cp include/util.h $(INCLUDEDIR)
	cp include/gpu.h $(INCLUDEDIR)
	cp include/pixels.h $(INCLUDEDIR)
	cp include/video.h $(INCLUDEDIR)
	cp include/spsc.h $(INCLUDEDIR)
	# Copy libraries
	cp $(LIB_NO_GPU) $(LIB_GPU) $(LIBDIR)
	ldconfig

# Clean target
clean:
	rm -rf $(BUILDDIR)
	rm -f $(OBJ_COMMON) $(OBJ_GPU) $(LIB_NO_GPU) $(LIB_GPU) example



# Object file compilation rules
$(BUILDDIR)/%.o: src/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(CFLAGS_CPU) -c $< -o $@

# Dependencies (optional)
$(BUILDDIR)/util.o: src/util.c include/util.h
$(BUILDDIR)/pixels.o: src/pixels.c include/rpihub75.h include/pixels.h
$(BUILDDIR)/video.o: src/video.c include/rpihub75.h
$(BUILDDIR)/gpio.o: src/gpio.c include/rpihub75.h
$(BUILDDIR)/gpu.o: src/gpu.c include/rpihub75.h include/stb_image.h
