include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include

CC = gcc
CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -ffast-math -fopenmp
# CC = icc
# CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp

VPATH = ../include

.PHONY: all clean

all: ../libcumultigpu.a ../libcumultigpu_seq.a

clean:
	$(RM) error.o multigpu.o multigpu_seq.o

../libcumultigpu.a: error.o multigpu.o
../libcumultigpu_seq.a: error.o multigpu_seq.o

error.o: error.h
multigpu.o: cumultigpu.h error.h
multigpu_seq.o: cumultigpu.h error.h
