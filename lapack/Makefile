include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -Werror -ftree-vectorize -ffast-math -fopenmp
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
          slauum.o spotrf.o spotri.o strtri.o slogdet.o \
          dlauum.o dpotrf.o dpotri.o dtrtri.o dlogdet.o
#           clauum.o cpotrf.o cpotri.o ctrtri.o \
#           zlauum.o zpotrf.o zpotri.o ztrtri.o

FATBINS = slogdet.fatbin dlogdet.fatbin

VPATH = ../include

.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) $(OBJECTS) $(FATBINS) $(addsuffix .c,$(FATBINS))

$(TARGET): $(OBJECTS)

handle.o: lapack.h blas.h cumultigpu.h handle.h error.h

slauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
spotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
spotri.o: lapack.h blas.h cumultigpu.h error.h
strtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
slogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h slogdet.fatbin.c
slogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_11,sm_13 -arch=compute_11

dlauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
dpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
dpotri.o: lapack.h blas.h cumultigpu.h error.h
dtrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
dlogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h dlogdet.fatbin.c
dlogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_13 -arch=compute_13

# clauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
# cpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
# cpotri.o: lapack.h blas.h cumultigpu.h error.h
# ctrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h

# zlauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
# zpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
# zpotri.o: lapack.h blas.h cumultigpu.h error.h
# ztrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h
