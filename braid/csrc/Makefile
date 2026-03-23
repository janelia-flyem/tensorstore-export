CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic

# Use CONDA_PREFIX for headers/libs if available (pixi/conda envs)
ifneq ($(CONDA_PREFIX),)
  CFLAGS += -I$(CONDA_PREFIX)/include
  LDFLAGS += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

.DEFAULT_GOAL := all
all: libdvid_decompress.so libbraid_codec.so

# Legacy: standalone DVID decompressor (no external deps)
libdvid_decompress.so: dvid_decompress.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $<

# Combined codec: DVID decompress + cseg encode, with zstd/zlib support
libbraid_codec.so: dvid_decompress.c cseg_encode.c
	$(CC) $(CFLAGS) $(LDFLAGS) -shared -fPIC -o $@ $^ -lzstd -lz

clean:
	rm -f libdvid_decompress.so libbraid_codec.so

.PHONY: clean all
