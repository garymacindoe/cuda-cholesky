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

all: ../libblas.a

clean:
	$(RM) *.o *.fatbin *.fatbin.c

distclean:
	$(RM) ../libblas.a

../libblas.a: handle.o xerbla.o \
              sgemm.o ssyrk.o strmm.o strsm.o \
              dgemm.o dsyrk.o dtrmm.o dtrsm.o \
              cgemm.o cherk.o ctrmm.o ctrsm.o \
              zgemm.o zherk.o ztrmm.o ztrsm.o

handle.o: blas.h cumultigpu.h handle.h error.h
xerbla.o: blas.h cumultigpu.h

ssyrk.o: blas.h cumultigpu.h error.h handle.h config.h ssyrk.fatbin.c
sgemm.o: blas.h cumultigpu.h error.h handle.h config.h sgemm.fatbin.c
strmm.o: blas.h cumultigpu.h error.h handle.h config.h strmm.fatbin.c
strsm.o: blas.h cumultigpu.h error.h handle.h config.h strsm.fatbin.c

dsyrk.o: blas.h cumultigpu.h error.h handle.h config.h dsyrk.fatbin.c
dgemm.o: blas.h cumultigpu.h error.h handle.h config.h dgemm.fatbin.c
dtrmm.o: blas.h cumultigpu.h error.h handle.h config.h dtrmm.fatbin.c
dtrsm.o: blas.h cumultigpu.h error.h handle.h config.h dtrsm.fatbin.c

cherk.o: blas.h cumultigpu.h error.h handle.h config.h cherk.fatbin.c
cgemm.o: blas.h cumultigpu.h error.h handle.h config.h cgemm.fatbin.c
ctrmm.o: blas.h cumultigpu.h error.h handle.h config.h ctrmm.fatbin.c
ctrsm.o: blas.h cumultigpu.h error.h handle.h config.h ctrsm.fatbin.c

zherk.o: blas.h cumultigpu.h error.h handle.h config.h zherk.fatbin.c
zgemm.o: blas.h cumultigpu.h error.h handle.h config.h zgemm.fatbin.c
ztrmm.o: blas.h cumultigpu.h error.h handle.h config.h ztrmm.fatbin.c
ztrsm.o: blas.h cumultigpu.h error.h handle.h config.h ztrsm.fatbin.c

sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin: NVCFLAGS = -code=sm_11,sm_13 -arch=compute_11
dgemm.fatbin dsyrk.fatbin dtrmm.fatbin dtrsm.fatbin: NVCFLAGS = -code=sm_13 -arch=compute_13
cgemm.fatbin cherk.fatbin ctrmm.fatbin ctrsm.fatbin: NVCFLAGS = -code=sm_11,sm_13 -arch=compute_11
zgemm.fatbin zherk.fatbin ztrmm.fatbin ztrsm.fatbin: NVCFLAGS = -code=sm_13 -arch=compute_13

sgemm.fatbin: blas.h cumultigpu.h
ssyrk.fatbin: blas.h cumultigpu.h
strmm.fatbin: blas.h cumultigpu.h
strsm.fatbin: blas.h cumultigpu.h

dgemm.fatbin: blas.h cumultigpu.h
dsyrk.fatbin: blas.h cumultigpu.h
dtrmm.fatbin: blas.h cumultigpu.h
dtrsm.fatbin: blas.h cumultigpu.h

cgemm.fatbin: blas.h cumultigpu.h
cherk.fatbin: blas.h cumultigpu.h
ctrmm.fatbin: blas.h cumultigpu.h
ctrsm.fatbin: blas.h cumultigpu.h

zgemm.fatbin: blas.h cumultigpu.h
zherk.fatbin: blas.h cumultigpu.h
ztrmm.fatbin: blas.h cumultigpu.h
ztrsm.fatbin: blas.h cumultigpu.h
