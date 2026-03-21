CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
TARGET = libdvid_decompress.so

$(TARGET): dvid_decompress.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: clean
