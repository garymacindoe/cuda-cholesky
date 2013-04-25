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

FATBINS = slauum.fatbin spotrf.fatbin strtri.fatbin slogdet.fatbin \
          dlauum.fatbin dpotrf.fatbin dtrtri.fatbin dlogdet.fatbin
#           clauum.fatbin cpotrf.fatbin ctrtri.fatbin \
#           zlauum.fatbin zpotrf.fatbin ztrtri.fatbin

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
slauum.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
spotrf.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
strtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
slogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_11,sm_13 -arch=compute_11

dlauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h dlauum.fatbin.c
dpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h dpotrf.fatbin.c
dpotri.o: lapack.h blas.h cumultigpu.h error.h
dtrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h dtrtri.fatbin.c
dlogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h dlogdet.fatbin.c
dlauum.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
dpotrf.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
dtrtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
dlogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_13 -arch=compute_13

# clauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h clauum.fatbin.c
# cpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h cpotrf.fatbin.c
# cpotri.o: lapack.h blas.h cumultigpu.h error.h
# ctrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h ctrtri.fatbin.c
# clauum.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
# cpotrf.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
# ctrtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11

# zlauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h zlauum.fatbin.c
# zpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h zpotrf.fatbin.c
# zpotri.o: lapack.h blas.h cumultigpu.h error.h
# ztrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h ztrtri.fatbin.c
# zlauum.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
# zpotrf.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
# ztrtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
