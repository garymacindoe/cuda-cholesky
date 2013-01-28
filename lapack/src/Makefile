include ../../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I../../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include -I../../include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -ffast-math -fopenmp
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp

NVCFLAGS = -O2 -use_fast_math -maxrregcount=32

BIN2CFLAGS = -c -st

VPATH = . ../include ../../include

.PHONY: all clean distclean

all: ../liblapack.a

clean:
	$(RM) *.o *.fatbin *.fatbin.c

distclean:
	$(RM) ../liblapack.a

../liblapack.a: spotrf.o spotri.o strtri.o \
                dpotrf.o dpotri.o dtrtri.o \
                cpotrf.o cpotri.o ctrtri.o \
                zpotrf.o zpotri.o ztrtri.o

spotrf.o: lapack.h blas.h cumultigpu.h error.h config.h spotrf.fatbin.c
spotri.o: lapack.h blas.h cumultigpu.h error.h config.h spotri.fatbin.c
strtri.o: lapack.h blas.h cumultigpu.h

dpotrf.o: lapack.h blas.h cumultigpu.h error.h config.h dpotrf.fatbin.c
dpotri.o: lapack.h blas.h cumultigpu.h error.h config.h dpotri.fatbin.c
dtrtri.o: lapack.h blas.h cumultigpu.h

cpotrf.o: lapack.h blas.h cumultigpu.h error.h config.h cpotrf.fatbin.c
cpotri.o: lapack.h blas.h cumultigpu.h error.h config.h cpotri.fatbin.c
ctrtri.o: lapack.h blas.h cumultigpu.h

zpotrf.o: lapack.h blas.h cumultigpu.h error.h config.h zpotrf.fatbin.c
zpotri.o: lapack.h blas.h cumultigpu.h error.h config.h zpotri.fatbin.c
ztrtri.o: lapack.h blas.h cumultigpu.h

spotrf.fatbin spotri.fatbin: NVCFLAGS = -code=sm_11,sm_13 -arch=compute_11
dpotrf.fatbin dpotri.fatbin: NVCFLAGS = -code=sm_13 -arch=compute_13
cpotrf.fatbin cpotri.fatbin: NVCFLAGS = -code=sm_11,sm_13 -arch=compute_11
zpotrf.fatbin zpotri.fatbin: NVCFLAGS = -code=sm_13 -arch=compute_13

spotrf.fatbin: blas.h cumultigpu.h
spotri.fatbin: blas.h cumultigpu.h

dpotrf.fatbin: blas.h cumultigpu.h
dpotri.fatbin: blas.h cumultigpu.h

cpotrf.fatbin: blas.h cumultigpu.h
cpotri.fatbin: blas.h cumultigpu.h

zpotrf.fatbin: blas.h cumultigpu.h
zpotri.fatbin: blas.h cumultigpu.h
