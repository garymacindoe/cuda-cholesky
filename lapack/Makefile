include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -ffast-math -fopenmp
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp

NVCFLAGS = -O2 -use_fast_math

ifdef verbose
  NVCFLAGS += -Xptxas=-v
  CFLAGS += -ftree-vectorizer-verbose=$(verbose)
#   CFLAGS += -vec-report=$(verbose)
endif

BIN2CFLAGS = -c -st

TARGET = ../liblapack.a

OBJECTS = slauum.o spotrf.o spotri.o strtri.o slogdet.o

VPATH = ../include

.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) $(OBJECTS) slogdet.fatbin slogdet.fatbin.c

$(TARGET): $(OBJECTS)

slauum.o: lapack.h blas.h cumultigpu.h error.h config.h
spotrf.o: lapack.h blas.h cumultigpu.h error.h config.h
spotri.o: lapack.h blas.h cumultigpu.h error.h
strtri.o: lapack.h blas.h cumultigpu.h error.h config.h

slogdet.o: lapack.h blas.h cumultigpu.h error.h slogdet.fatbin.c

slogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_11,sm_13 -arch=compute_11
