include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -I../include -I$(CUDA_HOME)/include

CC = gcc
CFLAGS = -ggdb -O0 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion
# CC = icc
# CFLAGS = -O0 -pipe -std=c99 -Wall -openmp

LDFLAGS = -L$(CUDA_HOME)/lib64
LDLIBS = -lcuda -lgomp -lrt

VPATH = ../include

MULTIGPU_SRC = $(wildcard multigpu/*.c)
MULTIGPU_TARGETS = $(basename $(notdir $(MULTIGPU_SRC)))
$(MULTIGPU_TARGETS): LOADLIBES = ../libcumultigpu.a

BLAS_SRC = $(wildcard blas/*.c)
BLAS_TARGETS = $(basename $(notdir $(BLAS_SRC)))
$(BLAS_TARGETS): LOADLIBES = ../libblas.a ../libcumultigpu.a

LAPACK_SRC = $(wildcard lapack/*.c)
LAPACK_TARGETS = $(basename $(notdir $(LAPACK_SRC)))
$(LAPACK_TARGETS): LOADLIBES = ../liblapack.a ../libblas.a ../libcumultigpu.a

TARGETS = $(MULTIGPU_TARGETS) $(BLAS_TARGETS) $(LAPACK_TARGETS)

.PHONY: all clean

all: $(TARGETS)

clean:
	$(RM) $(TARGETS) $(MULTIGPU_SRC:.c=.d) $(BLAS_SRC:.c=.d) $(LAPACK_SRC:.c=.d)

$(TARGETS):
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) $(<) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

-include $(MULTIGPU_SRC:.c=.d)
-include $(BLAS_SRC:.c=.d)
-include $(LAPACK_SRC:.c=.d)

%.d: %.c
	@set -e; $(RM) $@; \
	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($(notdir $*)\)\.o[ :]*,\1 $@: ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$
