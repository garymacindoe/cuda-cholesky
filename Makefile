include make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -Iinclude -I$(CUDA_HOME)/include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall

.PHONY: all blas lapack clean distclean

VPATH = include

all: libcumultigpu.a libcumultigpu_seq.a blas lapack

blas:
	cd blas && $(MAKE) all

lapack:
	cd lapack && $(MAKE) all

clean:
	cd blas && $(MAKE) clean
# 	cd lapack && $(MAKE) clean
	$(RM) src/*.o

distclean: clean
	cd blas && $(MAKE) distclean
# 	cd lapack && $(MAKE) distclean
	$(RM) libcumultigpu.a libcumultigpu_seq.a

libcumultigpu.a: src/error.o src/multigpu.o
libcumultigpu_seq.a: src/error.o src/multigpu_seq.o

src/error.c: error.h
src/multigpu.c: error.h cumultigpu.h
src/multigpu_seq.c: error.h cumultigpu.h
