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

OBJECTS = handle.o \
          slauum.o spotrf.o spotri.o strtri.o \
          slogdet.o

FATBINS = spotrf.fatbin slauum.fatbin strtri.fatbin slogdet.fatbin

VPATH = ../include

.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) $(OBJECTS) $(FATBINS) $(addsuffix .c,$(FATBINS))

$(TARGET): $(OBJECTS)

handle.o: lapack.h blas.h cumultigpu.h handle.h error.h

slauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h slauum.fatbin.c
spotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h spotrf.fatbin.c
spotri.o: lapack.h blas.h cumultigpu.h error.h
strtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h strtri.fatbin.c

slogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h slogdet.fatbin.c

slogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_11,sm_13 -arch=compute_11
slauum.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
spotrf.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
spotri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
strtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
spotrf.fatbin: strtri.cu
