CC ?= cc
CFLAGS ?= -O2 -Wall

# A shared library with an entry point: executable as a launcher, loadable via LD_PRELOAD.
lowvram: lowvram.c
	$(CC) $(CFLAGS) -shared -fPIC -Wl,-e,lowvram_main -o $@ $< -ldl -lpthread

clean:
	rm -f lowvram

.PHONY: clean
