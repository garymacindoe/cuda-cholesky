include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include

CC = gcc
CFLAGS = -march=native -ggdb -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -ffast-math -fopenmp
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
          dlauum.o dpotrf.o dpotri.o dtrtri.o \
          clauum.o cpotrf.o cpotri.o ctrtri.o \
          zlauum.o zpotrf.o zpotri.o ztrtri.o \
          slogdet.o dlogdet.o clogdet.o zlogdet.o

FATBINS = spotrf.fatbin slauum.fatbin strtri.fatbin \
          dpotrf.fatbin dlauum.fatbin dtrtri.fatbin \
          cpotrf.fatbin clauum.fatbin ctrtri.fatbin \
          zpotrf.fatbin zlauum.fatbin ztrtri.fatbin

FATBINS_EXTRA = slogdet.fatbin dlogdet.fatbin clogdet.fatbin zlogdet.fatbin

VPATH = ../include

.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) $(OBJECTS) $(FATBINS) $(addsuffix .c,$(FATBINS)) $(FATBINS_EXTRA) $(addsuffix .c,$(FATBINS_EXTRA))

$(TARGET): $(OBJECTS)

handle.o: lapack.h blas.h cumultigpu.h handle.h error.h

slauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h slauum.fatbin.c
spotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h spotrf.fatbin.c
spotri.o: lapack.h blas.h cumultigpu.h error.h
strtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h strtri.fatbin.c

dlauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h dlauum.fatbin.c
dpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h dpotrf.fatbin.c
dpotri.o: lapack.h blas.h cumultigpu.h error.h
dtrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h dtrtri.fatbin.c

clauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h clauum.fatbin.c
cpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h cpotrf.fatbin.c
cpotri.o: lapack.h blas.h cumultigpu.h error.h
ctrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h ctrtri.fatbin.c

zlauum.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h zlauum.fatbin.c
zpotrf.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h zpotrf.fatbin.c
zpotri.o: lapack.h blas.h cumultigpu.h error.h
ztrtri.o: lapack.h blas.h cumultigpu.h handle.h error.h config.h ztrtri.fatbin.c

slogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h slogdet.fatbin.c
dlogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h dlogdet.fatbin.c
clogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h clogdet.fatbin.c
zlogdet.o: lapack.h blas.h cumultigpu.h handle.h error.h zlogdet.fatbin.c

spotrf.fatbin slauum.fatbin strtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
dpotrf.fatbin dlauum.fatbin dtrtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13
cpotrf.fatbin clauum.fatbin ctrtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_11,sm_13 -arch=compute_11
zpotrf.fatbin zlauum.fatbin ztrtri.fatbin: NVCFLAGS += -maxrregcount=32 -code=sm_13 -arch=compute_13

$(FATBINS): blas.h cumultigpu.h

slogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_11,sm_13 -arch=compute_11
clogdet.fatbin: NVCFLAGS += -maxrregcount=10 -code=sm_11,sm_13 -arch=compute_11
dlogdet.fatbin: NVCFLAGS += -maxrregcount=24 -code=sm_13 -arch=compute_13
zlogdet.fatbin: NVCFLAGS += -maxrregcount=12 -code=sm_13 -arch=compute_13
