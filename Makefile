#############################################
# rpi-gpu-hub75 build system
#  Features / Improvements:
#   * Unified override-friendly flags
#   * Automatic header dependency generation (-MMD -MP)
#   * Separate CPU vs GPU+FFmpeg shared libs
#   * Optional static lib (CPU subset)
#   * pkg-config discovery for ffmpeg + GLES/GBM/EGL
#   * Quiet vs verbose (V=1)
#   * DESTDIR aware install/uninstall
#   * Example app with rpath to run from build tree
#   * Print useful variables (make print-vars)
#############################################

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

VERSION        ?= 0.2.0
CC             ?= gcc
AR             ?= ar
PKG_CONFIG     ?= pkg-config

# Build type: release (performance) or debug
# Usage: make BUILD=debug   or   make BUILD=release
BUILD          ?= release

# Variant suffix & directory separation so both builds can coexist
ifeq ($(BUILD),debug)
	VARIANT_SUFFIX := _dbg
	BUILDDIR       ?= build/debug
	OPT_FLAGS      := -O0 -g -fno-omit-frame-pointer
	FEATURE_FLAGS  := -DDEBUG -fPIC -pthread
	# Extra debug aids (comment out if undesired)
	DEBUG_EXTRA    ?= 
else ifeq ($(BUILD),release)
	VARIANT_SUFFIX :=
	BUILDDIR       ?= build/release
	OPT_FLAGS      := -O3 -g -pipe -fno-math-errno -ffast-math -funroll-loops -ftree-vectorize
	FEATURE_FLAGS  := -DNDEBUG=1 -fPIC -pthread
	DEBUG_EXTRA    ?=
else
	$(error Unknown BUILD='$(BUILD)' (expected 'release' or 'debug'))
endif

# CPU tuning flags auto-detected (ignore failure)
CFLAGS_CPU     ?= $(shell sh detect_flags.sh 2>/dev/null)

# Core flag groups (user can append via EXTRA_CFLAGS)
STD_FLAGS      ?= -std=gnu2x
WARN_FLAGS     ?= -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion
INCLUDE_FLAGS  ?= -Iinclude

# External packages
PKG_FFMPEG = libavcodec libavformat libswscale libavutil
PKG_GPU    = glesv2 gbm egl
# Prefer libpng; fall back to libpng16 if needed
PNG_PKG    := $(shell $(PKG_CONFIG) --exists libpng   && echo libpng   || \
						($(PKG_CONFIG) --exists libpng16 && echo libpng16 || echo none))

PKG_CFLAGS       := $(shell $(PKG_CONFIG) --cflags $(PKG_FFMPEG) $(PKG_GPU) 2>/dev/null)
PKG_LIBS_FFMPEG  := $(shell $(PKG_CONFIG) --libs $(PKG_FFMPEG) 2>/dev/null)
PKG_LIBS_GPU     := $(shell $(PKG_CONFIG) --libs $(PKG_GPU) 2>/dev/null)
PNG_CFLAGS       := $(shell if [ "$(PNG_PKG)" != "none" ]; then $(PKG_CONFIG) --cflags $(PNG_PKG); fi 2>/dev/null)
PKG_LIBS_PNG     := $(shell if [ "$(PNG_PKG)" != "none" ]; then $(PKG_CONFIG) --libs $(PNG_PKG); fi 2>/dev/null)

CFLAGS ?= $(STD_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) $(FEATURE_FLAGS) $(INCLUDE_FLAGS) $(CFLAGS_CPU) $(PKG_CFLAGS) $(PNG_CFLAGS) $(DEBUG_EXTRA) $(EXTRA_CFLAGS)

LDLIBS_COMMON = -pthread -lm -lrt
LDLIBS_FFMPEG = $(PKG_LIBS_FFMPEG)
LDLIBS_GPU    = $(PKG_LIBS_GPU)
LDLIBS_PNG    = $(PKG_LIBS_PNG)

# Verbosity
V ?= 0
ifeq ($(V),0)
 Q=@
else
 Q=
endif

# Directories (DESTDIR for packaging)
PREFIX      ?= /usr/local
INCLUDEDIR  ?= $(PREFIX)/include/rpihub75
LIBDIR      ?= $(PREFIX)/lib
# (BUILDDIR set per BUILD above)

# Sources
SRC_COMMON = src/util.c src/spsc.c src/lowlevel.c src/pixels.c src/rpihub75.c src/api.c src/scene.c src/transformers.c src/3d.c src/gradient.c src/text_sdf.c src/functions.c src/compositor.c
SRC_GPU    = src/gpu.c src/video.c

# Library names
LIB_BASENAME = rpihub75
LIB_NO_GPU   = lib$(LIB_BASENAME)$(VARIANT_SUFFIX).so
LIB_GPU      = lib$(LIB_BASENAME)_gpu$(VARIANT_SUFFIX).so
STATIC_LIB   = lib$(LIB_BASENAME)$(VARIANT_SUFFIX).a

# Objects
OBJ_COMMON = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(SRC_COMMON))
OBJ_GPU    = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(SRC_GPU))
OBJ_ALL    = $(OBJ_COMMON) $(OBJ_GPU)
DEPS       = $(OBJ_ALL:.o=.d)

# Library presence checks
GLESV2_FOUND    := $(shell $(PKG_CONFIG) --exists glesv2      && echo yes || echo no)
GBM_FOUND       := $(shell $(PKG_CONFIG) --exists gbm         && echo yes || echo no)
EGL_FOUND       := $(shell $(PKG_CONFIG) --exists egl         && echo yes || echo no)
AVCODEC_FOUND   := $(shell $(PKG_CONFIG) --exists libavcodec  && echo yes || echo no)
AVFORMAT_FOUND  := $(shell $(PKG_CONFIG) --exists libavformat && echo yes || echo no)
SWSCALE_FOUND   := $(shell $(PKG_CONFIG) --exists libswscale  && echo yes || echo no)
AVUTIL_FOUND    := $(shell $(PKG_CONFIG) --exists libavutil   && echo yes || echo no)
PNG_FOUND       := $(shell test "$(PNG_PKG)" != "none" && echo yes || echo no)
EFENCE_FOUND    := $(shell echo "int main(){}" | $(CC) -x c - -o /dev/null -lefence >/dev/null 2>&1 && echo yes || echo no)
EFENCE_LIB      := $(if $(filter yes,$(EFENCE_FOUND)),-lefence,)

.PHONY: all lib libgpu static clean distclean install uninstall check-libs example scratch print-vars debug perf example-debug example-perf example-fast test-sampler

all: check-libs $(LIB_NO_GPU) $(LIB_GPU) 
ifeq ($(BUILD),debug)
	@echo "[INFO] Built DEBUG variant (suffix $(VARIANT_SUFFIX))"
else
	@echo "[INFO] Built RELEASE variant"
endif

lib: $(LIB_NO_GPU)
libgpu: $(LIB_GPU)
static: $(STATIC_LIB)

print-vars:
	@echo VERSION=$(VERSION)
	@echo CC=$(CC)
	@echo CFLAGS=$(CFLAGS)
	@echo OBJ_COMMON=$(OBJ_COMMON)
	@echo OBJ_GPU=$(OBJ_GPU)
	@echo PKG_CFLAGS=$(PKG_CFLAGS)
	@echo PKG_LIBS_FFMPEG=$(PKG_LIBS_FFMPEG)
	@echo PKG_LIBS_GPU=$(PKG_LIBS_GPU)
	@echo PNG_PKG=$(PNG_PKG)
	@echo PNG_FOUND=$(PNG_FOUND)
	@echo PNG_CFLAGS=$(PNG_CFLAGS)
	@echo PKG_LIBS_PNG=$(PKG_LIBS_PNG)

$(BUILDDIR):
	$(Q)mkdir -p $(BUILDDIR)

check-libs:
ifeq ($(GLESV2_FOUND),no)
  $(error Missing glesv2 (sudo apt-get install libgles2-mesa-dev))
endif
ifeq ($(GBM_FOUND),no)
  $(error Missing gbm (sudo apt-get install libgbm-dev))
endif
ifeq ($(EGL_FOUND),no)
  $(error Missing egl (sudo apt-get install libegl1-mesa-dev))
endif
ifeq ($(AVCODEC_FOUND),no)
  $(error Missing libavcodec (sudo apt-get install libavcodec-dev))
endif
ifeq ($(AVFORMAT_FOUND),no)
  $(error Missing libavformat (sudo apt-get install libavformat-dev))
endif
ifeq ($(SWSCALE_FOUND),no)
  $(error Missing libswscale (sudo apt-get install libswscale-dev))
endif
ifeq ($(AVUTIL_FOUND),no)
  $(error Missing libavutil (sudo apt-get install libavutil-dev))
endif

# Shared libs
$(LIB_NO_GPU): $(OBJ_COMMON) | $(BUILDDIR)
	@echo "[LINK] $@ (CPU)"
	$(Q)$(CC) -shared -o $@ $(OBJ_COMMON) $(LDLIBS_COMMON) $(LDLIBS_PNG)

$(LIB_GPU): $(OBJ_COMMON) $(OBJ_GPU) | $(BUILDDIR)
	@echo "[LINK] $@ (GPU+FFmpeg)"
	$(Q)$(CC) -shared -o $@ $(OBJ_COMMON) $(OBJ_GPU) $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG)

# Static lib (CPU part only)
$(STATIC_LIB): $(OBJ_COMMON)
	@echo "[AR ] $@"
	$(Q)$(AR) rcs $@ $(OBJ_COMMON)

# Example program
# Fast path: link against already-installed libs (e.g. /usr/local/lib) if USE_SYSTEM_LIBS=1
#   make USE_SYSTEM_LIBS=1 example
# Convenience alias:
example-fast:
	$(MAKE) USE_SYSTEM_LIBS=1 example

ifeq ($(USE_SYSTEM_LIBS),1)
SYSTEM_LIBDIR ?= $(LIBDIR)
example: example.c
	@echo "[CC ] $@ (system libs)"
	$(Q)$(CC) $(CFLAGS) -Wl,-rpath,$(SYSTEM_LIBDIR) -o $@ $< \
		-l$(LIB_BASENAME)_gpu$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG)
else
example: example.c $(LIB_GPU)
	@echo "[CC ] $@ (local libs)"
	$(Q)$(CC) $(CFLAGS) -L. -Wl,-rpath,'$$ORIGIN' -o $@ $< \
		-l$(LIB_BASENAME)_gpu$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG)
endif

# Debug example (links with Electric Fence if available)
# Convenience wrappers that invoke recursive make with BUILD override
debug:
	$(MAKE) BUILD=debug clean all example
perf:
	$(MAKE) BUILD=release clean all example

# Legacy names retained for compatibility
example-debug:
	$(MAKE) BUILD=debug example
example-perf:
	$(MAKE) BUILD=release example

# Scratch test (optional if exists)
scratch: tests/scratch.c $(LIB_GPU)
	@echo "[CC ] $@"
	$(Q)$(CC) $(CFLAGS) -L. -Wl,-rpath,'$$ORIGIN' -o $@ $< -l$(LIB_BASENAME)_gpu $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) || echo "scratch build skipped (missing file)"

# Install / Uninstall
install: all
	@echo "[INSTALL] headers -> $(DESTDIR)$(INCLUDEDIR)"
	$(Q)mkdir -p $(DESTDIR)$(INCLUDEDIR)
	@echo "[INSTALL] libs -> $(DESTDIR)$(LIBDIR)"
	$(Q)mkdir -p $(DESTDIR)$(LIBDIR)
	#$(Q)cp include/rpihub75.h include/util.h include/gpu.h include/spsc.h include/pixels.h include/video.h include/scene.h $(DESTDIR)$(INCLUDEDIR) 
	$(Q)cp include/hub75gpu.h include/spsc.h include/pixels.h include/util.h $(DESTDIR)$(INCLUDEDIR) 
	$(Q)cp $(LIB_NO_GPU) $(LIB_GPU) $(DESTDIR)$(LIBDIR)
	@echo "Consider running ldconfig (root) if not found at runtime."

uninstall:
	@echo "[UNINSTALL] removing libs + headers"
	$(Q)rm -f $(DESTDIR)$(LIBDIR)/$(LIB_NO_GPU) $(DESTDIR)$(LIBDIR)/$(LIB_GPU)
	$(Q)rm -f $(DESTDIR)$(INCLUDEDIR)/rpihub75.h \
	              $(DESTDIR)$(INCLUDEDIR)/util.h \
	              $(DESTDIR)$(INCLUDEDIR)/gpu.h \
	              $(DESTDIR)$(INCLUDEDIR)/pixels.h \
	              $(DESTDIR)$(INCLUDEDIR)/scene.h \
	              $(DESTDIR)$(INCLUDEDIR)/spsc.h \
	              $(DESTDIR)$(INCLUDEDIR)/hub75gpu.h \
	              $(DESTDIR)$(INCLUDEDIR)/video.h || true

# Cleaning
clean:
	@echo "[CLEAN] objects"
	$(Q)rm -rf build/debug build/release
	$(Q)rm -f $(OBJ_COMMON) $(OBJ_GPU) $(DEPS)
	$(Q)rm -f lib$(LIB_BASENAME).so lib$(LIB_BASENAME)_gpu.so lib$(LIB_BASENAME)_gpu_dbg.so lib$(LIB_BASENAME)_dbg.so example example_dbg example-perf scratch

distclean: clean
	@echo "[CLEAN] distribution extras (none)"

# Compile objects with dependency generation
$(BUILDDIR)/%.o: src/%.c | $(BUILDDIR)
	@echo "[CC ] $<"
	$(Q)$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Auto-generated dependency files
-include $(DEPS)

# Convenience meta targets
.PHONY: test run

run: example
	./example -h || true

test: example
	@echo "No test suite defined yet." 

# Unit test for grayscale bilinear sampler
tests/test_sampler: tests/unit/test_sampler.c $(LIB_NO_GPU)
	@echo "[CC ] $@"
	$(Q)$(CC) $(CFLAGS) -Iinclude -L. -Wl,-rpath,'$$ORIGIN/..' -o $@ $< \
		-l$(LIB_BASENAME)$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_PNG)

test-sampler: tests/test_sampler
	@echo "[RUN] $<"
	$(Q)./tests/test_sampler
