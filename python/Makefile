# Minimal Makefile to build the CPython extension without setuptools

PYTHON ?= python3
CC     ?= gcc

EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PY_CFLAGS  := $(shell $(PYTHON)-config --cflags 2>/dev/null || $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("CFLAGS"))')
PY_LDFLAGS := $(shell $(PYTHON)-config --ldflags 2>/dev/null || echo "")

CFLAGS := -fPIC -pthread -I../include $(PY_CFLAGS)
LDFLAGS := -shared -pthread -L.. -lrpihub75_gpu -lrpihub75 -Wl,-rpath,'$$ORIGIN/..' $(PY_LDFLAGS)

TARGET := hub75gpu$(EXT_SUFFIX)
OBJS := hub75gpu_py.o

all: $(TARGET)
	@echo "Built $(TARGET)"

%.o: %.c
	@echo "[CC ] $<"
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
	@echo "[LD ] $@"
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	@echo "[CLEAN]"
	@rm -f $(OBJS) $(TARGET)
