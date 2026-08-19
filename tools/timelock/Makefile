CC      = gcc
# _POSIX_C_SOURCE exposes clock_gettime/CLOCK_MONOTONIC under -std=c11
# (strict ISO C hides them otherwise; needed on newer gcc/glibc).
CFLAGS  = -O2 -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -Wpedantic
LIBS    = -lgmp -lcrypto

SRCS    = main.c generate.c solve.c attack.c crypto.c puzzle_io.c primes.c
TARGET  = minroot-timelock

# On AION: load modules before building
#   module load compiler/GCC/13.2.0
#   module load math/GMP/6.3.0-GCCcore-13.2.0
# OpenSSL is available system-wide on AION nodes.

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
