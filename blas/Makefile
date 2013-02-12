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
          sgemm.o ssyrk.o strmm.o strsm.o

FATBINS = sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin

VPATH = ../include

.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) config $(OBJECTS) $(FATBINS) $(addsuffix .c,$(FATBINS))

$(TARGET): $(OBJECTS)

config: config.c error.h blas.h cumultigpu.h | sgemm.fatbin
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

sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin: NVCFLAGS += -code=sm_11,sm_13 -arch=compute_11

$(FATBINS): blas.h cumultigpu.h
