include make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -Iinclude -I$(CUDA_HOME)/include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall

.PHONY: all test clean distclean

VPATH = include

all: libcumultigpu.a libcumultigpu_seq.a libblas.a liblapack.a

test: libcumultigpu.a libcumultigpu_seq.a libblas.a liblapack.a
	cd test && $(MAKE)

clean:
	cd blas && $(MAKE) clean
	cd lapack && $(MAKE) clean
	cd multigpu && $(MAKE) clean
	cd test && $(MAKE) clean

distclean: clean
	$(RM) libcumultigpu.a libcumultigpu_seq.a libblas.a liblapack.a

libblas.a:
	cd blas && $(MAKE) all

liblapack.a: libblas.a
	cd lapack && $(MAKE) all

libcumultigpu.a libcumultigpu_seq.a:
	cd multigpu && $(MAKE) ../$(@)
