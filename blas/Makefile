include ../make.inc

CUDA_HOME = /opt/cuda

CPPFLAGS = -Iinclude -I../include -I$(CUDA_HOME)/include

CC = gcc
CFLAGS = -O0 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion
# CC = icc
# CFLAGS = -O0 -pipe -std=c99 -Wall

LOADLIBES = libblas.a ../libcumultigpu.a
LDFLAGS = -L$(CUDA_HOME)/lib64
LDLIBS = -lcuda -lgomp -lrt

VPATH = test include ../include

TEST_PROGS = sgemm ssyrk strmm strmm2 strsm \
             dgemm dsyrk dtrmm dtrmm2 dtrsm \
             cgemm cherk ctrmm ctrmm2 ctrsm \
             zgemm zherk ztrmm ztrmm2 ztrsm \
             cusgemm cusgemm2 cussyrk custrmm2 custrsm \
             cudgemm cudgemm2 cudsyrk cudtrmm2 cudtrsm \
             cucgemm cucgemm2 cucherk cuctrmm2 cuctrsm \
             cuzgemm cuzgemm2 cuzherk cuztrmm2 cuztrsm \
             cumultigpusgemm cumultigpussyrk cumultigpustrsm cumultigpustrmm \
             cumultigpudgemm cumultigpudsyrk cumultigpudtrsm cumultigpudtrmm \
             cumultigpucgemm cumultigpucherk cumultigpuctrsm cumultigpuctrmm \
             cumultigpuzgemm cumultigpuzherk cumultigpuztrsm cumultigpuztrmm

TEST_SRC = $(addprefix,test/,$(addsuffix,$(TEST_PROGS),.c))

.PHONY: all test clean distclean

all: libblas.a

clean:
	cd src && $(MAKE) clean
	$(RM) *.o test/*.d

distclean: clean
	cd src && $(MAKE) distclean
	$(RM) $(TEST_PROGS)

test: $(TEST_PROGS)

libblas.a:
	cd src && $(MAKE)

$(TEST_PROGS): $(LOADLIBES)

-include $(TEST_SRC:.c=.d)
