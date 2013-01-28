include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall

VPATH = . ../include

TARGETS ?= libcumultigpu.a libcumultigpu_seq.a

.PHONY: all clean distclean

all: $(TARGETS)

clean:
	$(RM) *.o

distclean: clean
	$(RM) $(TARGETS)

libcumultigpu.a: error.o multigpu.o
libcumultigpu_seq.a: error.o multigpu_seq.o

error.o: error.h
multigpu.o: cumultigpu.h error.h
multigpu_seq.o: cumultigpu.h error.h
