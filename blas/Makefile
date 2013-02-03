include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -ffast-math -fopenmp
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp

LOADLIBES = ../libcumultigpu.a
LDLIBS = -lcuda

NVCFLAGS = -O2 -use_fast_math -maxrregcount=32

ifdef verbose
  NVCFLAGS += -Xptxas=-v
  CFLAGS += -ftree-vectorizer-verbose=$(verbose)
#   CFLAGS += -vec-report=$(verbose)
endif

BIN2CFLAGS = -c -st

TARGET = ../libblas.a

OBJECTS = handle.o xerbla.o \
          sgemm.o ssyrk.o strmm.o strsm.o \
          dgemm.o dsyrk.o dtrmm.o dtrsm.o \
          cgemm.o cherk.o ctrmm.o ctrsm.o \
          zgemm.o zherk.o ztrmm.o ztrsm.o

FATBINS = sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin \
          dgemm.fatbin dsyrk.fatbin dtrmm.fatbin dtrsm.fatbin \
          cgemm.fatbin cherk.fatbin ctrmm.fatbin ctrsm.fatbin \
          zgemm.fatbin zherk.fatbin ztrmm.fatbin ztrsm.fatbin

VPATH = ../include

.NOTPARALLEL: ../include/config.h
.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) config $(OBJECTS) $(FATBINS) $(addsuffix .c,$(FATBINS))

$(TARGET): $(OBJECTS)

config: config.c error.h blas.h cumultigpu.h | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LOADLIBES) $(LDLIBS)

../include/config.h: config
	@echo "#ifndef CONFIG_H" > $(@)
	@echo "#define CONFIG_H" >> $(@)
	./$(<) >> $(@)
	@echo "#endif" >> $(@)

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

sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin: NVCFLAGS += -code=sm_11,sm_13 -arch=compute_11
dgemm.fatbin dsyrk.fatbin dtrmm.fatbin dtrsm.fatbin: NVCFLAGS += -code=sm_13 -arch=compute_13
cgemm.fatbin cherk.fatbin ctrmm.fatbin ctrsm.fatbin: NVCFLAGS += -code=sm_11,sm_13 -arch=compute_11
zgemm.fatbin zherk.fatbin ztrmm.fatbin ztrsm.fatbin: NVCFLAGS += -code=sm_13 -arch=compute_13

$(FATBINS): blas.h cumultigpu.h
