include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall

NVCFLAGS = -O2 -use_fast_math

BIN2CFLAGS = -c -st

LDFLAGS = -L$(CUDA_HOME)/lib64
LOADLIBES = ../libcumultigpu.a
LDLIBS = -lcuda -lcudart -lrt

VPATH = ../include

.PHONY: all clean distclean

TARGETS = bandwidth-test block-test context-test cumemalloc-test event flop-test flop-word gpu-bandwidth kernel-test kernel-test-rt module-test multigpu pitch stream-test timer

all: $(TARGETS)

clean:
	$(RM) flop-test.fatbin.c kernel-test.fatbin.c

distclean: clean
	$(RM) $(TARGETS) flop-test.cubin flop-test.fatbin flop-test.ptx kernel-test.ptx

bandwidth-test: bandwidth-test.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

block-test: block-test.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) $(<)

context-test: context-test.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cumemalloc-test: cumemalloc-test.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

event: event.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

flop-test: flop-test.c flop-test.fatbin.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

flop-word: flop-word.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

gpu-bandwidth: gpu-bandwidth.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

kernel-test: kernel-test.c kernel-test.fatbin.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

kernel-test-rt: kernel-test.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) $(<)

module-test: module-test.c flop-test.fatbin.c flop-test.fatbin flop-test.cubin flop-test.ptx error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

multigpu: multigpu.c kernel-test.ptx error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

pitch: pitch.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

stream-test: stream-test.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

timer: CPPFLAGS += -D_POSIX_C_SOURCE=199309L
timer: timer.c error.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
