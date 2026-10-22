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

VERSION        ?= 0.3.0
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
# Prefer libjpeg; fall back to libjpeg-turbo if needed
JPEG_PKG   := $(shell $(PKG_CONFIG) --exists libjpeg  && echo libjpeg  || \
						($(PKG_CONFIG) --exists libjpeg-turbo && echo libjpeg-turbo || echo none))

PKG_CFLAGS       := $(shell $(PKG_CONFIG) --cflags $(PKG_FFMPEG) $(PKG_GPU) 2>/dev/null)
PKG_LIBS_FFMPEG  := $(shell $(PKG_CONFIG) --libs $(PKG_FFMPEG) 2>/dev/null)
PKG_LIBS_GPU     := $(shell $(PKG_CONFIG) --libs $(PKG_GPU) 2>/dev/null)
PNG_CFLAGS       := $(shell if [ "$(PNG_PKG)" != "none" ]; then $(PKG_CONFIG) --cflags $(PNG_PKG); fi 2>/dev/null)
PKG_LIBS_PNG     := $(shell if [ "$(PNG_PKG)" != "none" ]; then $(PKG_CONFIG) --libs $(PNG_PKG); fi 2>/dev/null)
JPEG_CFLAGS      := $(shell if [ "$(JPEG_PKG)" != "none" ]; then $(PKG_CONFIG) --cflags $(JPEG_PKG); fi 2>/dev/null)
PKG_LIBS_JPEG    := $(shell if [ "$(JPEG_PKG)" != "none" ]; then $(PKG_CONFIG) --libs $(JPEG_PKG); fi 2>/dev/null)

CFLAGS ?= $(STD_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) $(FEATURE_FLAGS) $(INCLUDE_FLAGS) $(CFLAGS_CPU) $(PKG_CFLAGS) $(PNG_CFLAGS) $(JPEG_CFLAGS) $(DEBUG_EXTRA) $(EXTRA_CFLAGS)

# Auto-define HAVE_LIBPNG / HAVE_LIBJPEG when detected
ifneq ($(PNG_PKG),none)
CFLAGS += -DHAVE_LIBPNG=1
endif
ifneq ($(JPEG_PKG),none)
CFLAGS += -DHAVE_LIBJPEG=1
endif

LDLIBS_COMMON = -pthread -lm -lrt
LDLIBS_FFMPEG = $(PKG_LIBS_FFMPEG)
LDLIBS_GPU    = $(PKG_LIBS_GPU)
LDLIBS_PNG    = $(PKG_LIBS_PNG)
LDLIBS_JPEG   = $(PKG_LIBS_JPEG)

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
INSTALL_HEADERS = include/hub75gpu.h include/hub75_runtime.h include/spsc.h include/pixels.h include/util.h include/rpihub75.h include/lists.h include/debug.h
# (BUILDDIR set per BUILD above)

# Sources
SRC_COMMON = src/util.c src/spsc.c src/lowlevel.c src/pixels.c src/rpihub75.c src/api.c src/scene.c src/transformers.c src/3d.c src/gradient.c src/text_sdf.c src/functions.c src/compositor.c src/runtime.c src/render_commands.c
SRC_GPU    = src/gpu.c src/video.c src/runtime_media.c

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
JPEG_FOUND      := $(shell test "$(JPEG_PKG)" != "none" && echo yes || echo no)
EFENCE_FOUND    := $(shell echo "int main(){}" | $(CC) -x c - -o /dev/null -lefence >/dev/null 2>&1 && echo yes || echo no)
EFENCE_LIB      := $(if $(filter yes,$(EFENCE_FOUND)),-lefence,)

.PHONY: all lib libgpu static clean distclean install uninstall check-libs example scratch print-vars debug perf example-debug example-perf example-fast test-sampler pio-dma-bench pio-clock-scope pio-hub75-scope

all: check-libs $(LIB_NO_GPU) $(LIB_GPU) 
ifeq ($(BUILD),debug)
	@echo "[INFO] Built DEBUG variant (suffix $(VARIANT_SUFFIX))"
else
	@echo "[INFO] Built RELEASE variant"
endif

lib: $(LIB_NO_GPU)
libgpu: check-libs $(LIB_GPU)
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
	@echo JPEG_PKG=$(JPEG_PKG)
	@echo JPEG_FOUND=$(JPEG_FOUND)
	@echo JPEG_CFLAGS=$(JPEG_CFLAGS)
	@echo PKG_LIBS_JPEG=$(PKG_LIBS_JPEG)

$(BUILDDIR):
	$(Q)mkdir -p $(BUILDDIR)

check-libs:
	$(Q)for pkg in $(PKG_GPU) $(PKG_FFMPEG); do \
		$(PKG_CONFIG) --exists $$pkg || { echo "Missing GPU dependency: $$pkg" >&2; exit 1; }; \
	done

# Shared libs
$(LIB_NO_GPU): $(OBJ_COMMON) | $(BUILDDIR)
	@echo "[LINK] $@ (CPU)"
	$(Q)$(CC) $(LDFLAGS) -shared -o $@ $(OBJ_COMMON) $(LDLIBS_COMMON) $(LDLIBS_PNG) $(LDLIBS_JPEG)

$(LIB_GPU): $(OBJ_COMMON) $(OBJ_GPU) | $(BUILDDIR)
	@echo "[LINK] $@ (GPU+FFmpeg)"
	$(Q)$(CC) $(LDFLAGS) -shared -o $@ $(OBJ_COMMON) $(OBJ_GPU) $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG) $(LDLIBS_JPEG)

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
example: example.c examples/runtime_demo.h
	@echo "[CC ] $@ (system libs)"
	$(Q)$(CC) $(CFLAGS) -Wl,-rpath,$(SYSTEM_LIBDIR) -o $@ $< \
		-l$(LIB_BASENAME)_gpu$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG) $(LDLIBS_JPEG)
else
example: example.c examples/runtime_demo.h $(LIB_GPU)
	@echo "[CC ] $@ (local libs)"
	$(Q)$(CC) $(CFLAGS) -L. -Wl,-rpath,'$$ORIGIN' -o $@ $< \
		-l$(LIB_BASENAME)_gpu$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG) $(LDLIBS_JPEG)
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

# Install / Uninstall (prefix/link settings can change without source timestamps)
.PHONY: force-pkgconfig
force-pkgconfig:
$(BUILDDIR)/rpihub75.pc: rpihub75.pc.in Makefile force-pkgconfig | $(BUILDDIR)
	$(Q)sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' \
		-e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(VERSION)|g' \
		-e 's|@NAME@|rpihub75|g' -e 's|@LIBRARY@|$(LIB_BASENAME)$(VARIANT_SUFFIX)|g' \
		-e 's|@PRIVATE_LIBS@|$(LDLIBS_COMMON) $(LDLIBS_PNG) $(LDLIBS_JPEG)|g' $< > $@

$(BUILDDIR)/rpihub75_gpu.pc: rpihub75.pc.in Makefile force-pkgconfig | $(BUILDDIR)
	$(Q)sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' \
		-e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(VERSION)|g' \
		-e 's|@NAME@|rpihub75_gpu|g' -e 's|@LIBRARY@|$(LIB_BASENAME)_gpu$(VARIANT_SUFFIX)|g' \
		-e 's|@PRIVATE_LIBS@|$(LDLIBS_COMMON) $(LDLIBS_FFMPEG) $(LDLIBS_GPU) $(LDLIBS_PNG) $(LDLIBS_JPEG)|g' $< > $@

.PHONY: install-cpu
install-cpu: lib $(BUILDDIR)/rpihub75.pc
	$(Q)mkdir -p $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(LIBDIR)/pkgconfig
	$(Q)cp $(INSTALL_HEADERS) $(DESTDIR)$(INCLUDEDIR)
	$(Q)cp $(LIB_NO_GPU) $(DESTDIR)$(LIBDIR)
	$(Q)cp $(BUILDDIR)/rpihub75.pc $(DESTDIR)$(LIBDIR)/pkgconfig

install: all $(BUILDDIR)/rpihub75.pc $(BUILDDIR)/rpihub75_gpu.pc
	@echo "[INSTALL] headers -> $(DESTDIR)$(INCLUDEDIR)"
	$(Q)mkdir -p $(DESTDIR)$(INCLUDEDIR)
	@echo "[INSTALL] libs -> $(DESTDIR)$(LIBDIR)"
	$(Q)mkdir -p $(DESTDIR)$(LIBDIR)
	$(Q)cp $(INSTALL_HEADERS) $(DESTDIR)$(INCLUDEDIR)
	$(Q)cp $(LIB_NO_GPU) $(LIB_GPU) $(DESTDIR)$(LIBDIR)
	$(Q)mkdir -p $(DESTDIR)$(LIBDIR)/pkgconfig
	$(Q)cp $(BUILDDIR)/rpihub75.pc $(BUILDDIR)/rpihub75_gpu.pc $(DESTDIR)$(LIBDIR)/pkgconfig
	@echo "Consider running ldconfig (root) if not found at runtime."

uninstall:
	@echo "[UNINSTALL] removing libs + headers"
	$(Q)rm -f $(DESTDIR)$(LIBDIR)/$(LIB_NO_GPU) $(DESTDIR)$(LIBDIR)/$(LIB_GPU)
	$(Q)rm -f $(DESTDIR)$(LIBDIR)/pkgconfig/rpihub75.pc $(DESTDIR)$(LIBDIR)/pkgconfig/rpihub75_gpu.pc
	$(Q)rm -f $(DESTDIR)$(INCLUDEDIR)/hub75_runtime.h $(DESTDIR)$(INCLUDEDIR)/lists.h $(DESTDIR)$(INCLUDEDIR)/debug.h
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

# Validation must not inherit finite-math assumptions from the SIMD renderer.
$(BUILDDIR)/runtime.o $(BUILDDIR)/render_commands.o $(BUILDDIR)/runtime_media.o $(BUILDDIR)/text_sdf.o: CFLAGS += -fno-fast-math

# Auto-generated dependency files
-include $(DEPS)

# Convenience meta targets
.PHONY: test run

run: example
	./example -h || true

test: test-runtime test-rio-scan test-compositor-sine

.PHONY: test-runtime test-python example-runtime
$(BUILDDIR)/example_runtime: examples/runtime.c examples/runtime_demo.h $(LIB_NO_GPU)
	$(Q)$(CC) $(CFLAGS) -Iexamples -L$(dir $(LIB_NO_GPU)) -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../..' -o $@ $< \
		-l$(LIB_BASENAME)$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_PNG) $(LDLIBS_JPEG)

example-runtime: $(BUILDDIR)/example_runtime
$(BUILDDIR)/test_runtime: tests/unit/test_runtime.c src/runtime.c $(LIB_NO_GPU) include/hub75_runtime.h
	@echo "[CC ] $@ (pin-free runtime)"
	$(Q)$(CC) $(CFLAGS) -fno-fast-math -UNDEBUG -DH75_TESTING -L$(dir $(LIB_NO_GPU)) -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../..' -o $@ $< src/runtime.c \
		-l$(LIB_BASENAME)$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_PNG) $(LDLIBS_JPEG)

test-runtime: $(BUILDDIR)/test_runtime
	$(Q)./$<

# Requires an installed/in-place Python package. Never activates GPIO.
test-python:
	$(Q)PYTHONPATH=python $(PYTHON) -X faulthandler -m unittest discover -s python/tests -v

PYTHON ?= python3

# Unit test for grayscale bilinear sampler
$(BUILDDIR)/test_sampler: tests/unit/test_sampler.c $(LIB_NO_GPU)
	@echo "[CC ] $@"
	$(Q)$(CC) $(CFLAGS) -UNDEBUG -L$(dir $(LIB_NO_GPU)) -Wl,-rpath,'$$ORIGIN:$$ORIGIN/../..' -o $@ $< \
		-l$(LIB_BASENAME)$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_PNG) $(LDLIBS_JPEG)

test-sampler: $(BUILDDIR)/test_sampler
	@echo "[RUN] $<"
	$(Q)./$<

# Video fitting geometry test (no FFmpeg or GPIO required).
.PHONY: test-video-fit
test-video-fit: $(BUILDDIR)/test_video_fit
	$(Q)./$<

$(BUILDDIR)/test_video_fit: tests/test_video_fit.c include/video_fit.h
	@mkdir -p $(dir $@)
	$(Q)$(CC) -std=gnu2x -Wall -Wextra -UNDEBUG -Iinclude -o $@ $<

# Sine compositor regression test (no GPIO access).
.PHONY: test-compositor-sine
$(BUILDDIR)/test_compositor_sine: tests/unit/test_compositor_sine.c $(LIB_NO_GPU) include/compositor.h
	@echo "[CC ] $@"
	$(Q)$(CC) $(CFLAGS) -L. -Wl,-rpath,'$$ORIGIN/../..' -o $@ $< \
		-l$(LIB_BASENAME)$(VARIANT_SUFFIX) $(LDLIBS_COMMON) $(LDLIBS_PNG) $(LDLIBS_JPEG)

test-compositor-sine: $(BUILDDIR)/test_compositor_sine
	@echo "[RUN] $< (no GPIO access)"
	$(Q)./$<

# Pin-free RIO ownership tests: no display library, GPU/FFmpeg, or ARM-only flags.
.PHONY: test-rio-scan
RIO_SCAN_TEST_CFLAGS ?= $(STD_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) -pthread -Iinclude $(EXTRA_CFLAGS)

$(BUILDDIR)/test_rio_scan: tests/unit/test_rio_scan.c src/rio_scan.c include/rio_scan.h | $(BUILDDIR)
	@echo "[CC ] $@ (pin-free)"
	$(Q)$(CC) $(RIO_SCAN_TEST_CFLAGS) -o $@ tests/unit/test_rio_scan.c src/rio_scan.c

test-rio-scan: $(BUILDDIR)/test_rio_scan
	@echo "[RUN] $< (no GPIO access)"
	$(Q)./$<

# Standalone RIO scope benchmark. Build only; GPIO requires explicit opt-in.
.PHONY: rio-scope test-rio-scope test-rio-isolate test-rio-environment test-configure-system

$(BUILDDIR)/rio_scope: tests/rio_scope.c tests/rio_scope_io.h tests/rio_scope_lock.h src/rio_scan.c include/rio_scan.h | $(BUILDDIR)
	@echo "[CC ] $@ (safe RAM backend by default)"
	$(Q)$(CC) $(RIO_SCAN_TEST_CFLAGS) -Itests -o $@ tests/rio_scope.c src/rio_scan.c

rio-scope: $(BUILDDIR)/rio_scope
	@echo "Safe RAM smoke test: ./$< --cycles 3"
	@echo "Read tests/README-rio-scope.md before any --output run."

$(BUILDDIR)/test_rio_scope_io: tests/unit/test_rio_scope_io.c tests/rio_scope_io.h | $(BUILDDIR)
	@echo "[CC ] $@ (pin-free MMIO trace model)"
	$(Q)$(CC) $(RIO_SCAN_TEST_CFLAGS) -Itests -o $@ tests/unit/test_rio_scope_io.c

$(BUILDDIR)/test_rio_scope_setup: tests/unit/test_rio_scope_setup.c tests/rio_scope.c tests/rio_scope_io.h tests/rio_scope_lock.h src/rio_scan.c include/rio_scan.h | $(BUILDDIR)
	@echo "[CC ] $@ (pin-free setup/cleanup model)"
	$(Q)$(CC) $(RIO_SCAN_TEST_CFLAGS) -Itests -o $@ tests/unit/test_rio_scope_setup.c src/rio_scan.c

$(BUILDDIR)/test_rio_scope_lock: tests/unit/test_rio_scope_lock.c tests/rio_scope_lock.h | $(BUILDDIR)
	@echo "[CC ] $@ (temporary-file advisory-lock test; no GPIO)"
	$(Q)$(CC) $(RIO_SCAN_TEST_CFLAGS) -Itests -o $@ tests/unit/test_rio_scope_lock.c

test-rio-scope: test-configure-system $(BUILDDIR)/test_rio_scope_io $(BUILDDIR)/test_rio_scope_setup $(BUILDDIR)/test_rio_scope_lock $(BUILDDIR)/rio_scope
	$(Q)./$(BUILDDIR)/test_rio_scope_io
	$(Q)./$(BUILDDIR)/test_rio_scope_setup
	$(Q)./$(BUILDDIR)/test_rio_scope_lock
	$(Q)RIO_SCOPE_EXE=$(abspath $(BUILDDIR)/rio_scope) python3 -m unittest discover -s tests/unit -p 'test_rio*.py' -v

test-rio-isolate:
	@echo "[RUN] isolation tests (temporary filesystem/fake affinities; no host tuning)"
	$(Q)python3 -m unittest discover -s tests/unit -p 'test_rio_isolate.py' -v

test-rio-environment:
	@echo "[RUN] boot/runtime controller tests (temporary files/fake affinities; no host tuning)"
	$(Q)python3 -m unittest discover -s tests/unit -p 'test_rio_environment.py' -v

test-configure-system:
	@echo "[RUN] persistent system setup tests (fake systemd/affinities and temporary files only)"
	$(Q)python3 -m unittest discover -s tests/unit -p 'test_configure_system.py' -v

# Standalone RP1 PIO hardware-DMA throughput benchmark (does not drive GPIO).
$(BUILDDIR)/pio_dma_bench: tests/pio_dma_bench.c | $(BUILDDIR)
	@echo "[CC ] $@"
	$(Q)$(CC) -O2 -std=gnu2x -Wall -Wextra -o $@ $<

pio-dma-bench: $(BUILDDIR)/pio_dma_bench
	@echo "Run: ./$< [seconds] [clock_divider] [chunk_KiB] [--gpio BASE|--project-pins] [--autopull|--pack4]"

# Scope-only GPIO17 clock test: no DMA, no other GPIO outputs.
$(BUILDDIR)/pio_clock_scope: tests/pio_clock_scope.c | $(BUILDDIR)
	@echo "[CC ] $@"
	$(Q)$(CC) -O2 -std=gnu2x -Wall -Wextra -o $@ $<

pio-clock-scope: $(BUILDDIR)/pio_clock_scope
	@echo "Run (panels disconnected): ./$< [seconds] [clock_divider]"

# One DMA word per pixel clock: 18 RGB pins + side-set GPIO17 CLK.
$(BUILDDIR)/pio_hub75_scope: tests/pio_hub75_scope.c | $(BUILDDIR)
	@echo "[CC ] $@"
	$(Q)$(CC) -O2 -std=gnu2x -Wall -Wextra -o $@ $<

pio-hub75-scope: $(BUILDDIR)/pio_hub75_scope
	@echo "Run (panels disconnected): ./$< [seconds] [clock_divider] [chunk_KiB] [--light-dma] --output"
