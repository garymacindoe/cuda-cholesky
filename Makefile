CUDA_HOME = /opt/cuda
INTEL_HOME = /opt/intel/composerxe-2013_update1.1.117

CC = gcc
NVCC = nvcc
NVCFLAGS = -O2 -use_fast_math -maxrregcount=32

CPPFLAGS = -Iinclude -I$(CUDA_HOME)/include
NVCPPFLAGS = -Iinclude
LDFLAGS = -L$(CUDA_HOME)/lib64
LDLIBS = -lcuda -lrt

# TODO:  separate no-opt CFLAGS for testing code.
# TODO:  implement hacks in C codes to vectorise all possible loops.
ifeq ($(notdir $(CC)), icc)
  CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp
  LDFLAGS += -L$(INTEL_HOME)/compiler/lib/intel64
  LDLIBS += -liomp5
else
  CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -ffast-math -fopenmp
  LDLIBS += -lgomp
endif

ifdef verbose
  NVCFLAGS += -Xptxas=-v
  ifeq ($(notdir $(CC)), icc)
    CFLAGS += -vec-report=$(verbose)
  else
    CFLAGS += -ftree-vectorizer-verbose=$(verbose)
  endif
endif

ifdef bank_conflicts
  CPPFLAGS += -D__BANK_CONFLICTS__=$(bank_conflicts)
endif

ifdef mgpu_seq
  CPPFLAGS += -DMGPU_SEQ
else
  LDLIBS += -lpthread
endif

RM = rm -f
RMDIR = rm -rf
MKDIR = mkdir

VPATH = include src/blas src/lapack

OBJDIR = obj

.NOTPARALLEL: $(OBJDIR) $(OBJDIR)/src $(OBJDIR)/src/blas $(OBJDIR)/test

.PHONY: all clean distclean

all: cgemm cherk ctrmm2 ctrmm ctrsm \
     dgemm dsyrk dtrmm2 dtrmm dtrsm \
     sgemm ssyrk strmm2 strmm strsm \
     zgemm zherk ztrmm2 ztrmm ztrsm \
     cucgemm cucgemm2 cucherk cuctrmm2 cuctrsm \
     cudgemm cudgemm2 cudsyrk cudtrmm2 cudtrsm \
     cusgemm cusgemm2 cussyrk custrmm2 custrsm \
     cuzgemm cuzgemm2 cuzherk cuztrmm2 cuztrsm \
     cumultigpucgemm cumultigpucherk cumultigpuctrmm cumultigpuctrsm \
     cumultigpudgemm cumultigpudsyrk cumultigpudtrmm cumultigpudtrsm \
     cumultigpusgemm cumultigpussyrk cumultigpustrmm cumultigpustrsm \
     cumultigpuzgemm cumultigpuzherk cumultigpuztrmm cumultigpuztrsm

clean:
	$(RMDIR) $(OBJDIR)

distclean: clean
	$(RM) cgemm cherk ctrmm2 ctrmm ctrsm \
	      dgemm dsyrk dtrmm2 dtrmm dtrsm \
	      sgemm ssyrk strmm2 strmm strsm \
	      zgemm zherk ztrmm2 ztrmm ztrsm
	$(RM) cucgemm cucgemm2 cucherk cuctrmm2 cuctrsm \
	      cudgemm cudgemm2 cudsyrk cudtrmm2 cudtrsm \
	      cusgemm cusgemm2 cussyrk custrmm2 custrsm \
	      cuzgemm cuzgemm2 cuzherk cuztrmm2 cuztrsm
	$(RM) cumultigpucgemm cumultigpucherk cumultigpuctrmm cumultigpuctrsm \
	      cumultigpudgemm cumultigpudsyrk cumultigpudtrmm cumultigpudtrsm \
	      cumultigpusgemm cumultigpussyrk cumultigpustrmm cumultigpustrsm \
	      cumultigpuzgemm cumultigpuzherk cumultigpuztrmm cumultigpuztrsm
	$(RM) cpotrf cpotri ctrtri ctrtri2 \
	      dpotrf dpotri dtrtri dtrtri2 \
	      spotrf spotri strtri strtri2 \
	      zpotrf zpotri ztrtri ztrtri2
	$(RM) cgemm.fatbin cherk.fatbin ctrmm.fatbin ctrsm.fatbin \
	      dgemm.fatbin dsyrk.fatbin dtrmm.fatbin dtrsm.fatbin \
	      sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin \
	      zgemm.fatbin zherk.fatbin ztrmm.fatbin ztrsm.fatbin
	$(RM) cpotrf.fatbin \
	      dpotrf.fatbin \
	      spotrf.fatbin \
	      zpotrf.fatbin

cgemm:  $(OBJDIR)/test/cgemm.o  $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cherk:  $(OBJDIR)/test/cherk.o  $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ctrmm2: $(OBJDIR)/test/ctrmm2.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ctrmm:  $(OBJDIR)/test/ctrmm.o  $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ctrsm:  $(OBJDIR)/test/ctrsm.o  $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

dgemm:  $(OBJDIR)/test/dgemm.o  $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dsyrk:  $(OBJDIR)/test/dsyrk.o  $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dtrmm2: $(OBJDIR)/test/dtrmm2.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dtrmm:  $(OBJDIR)/test/dtrmm.o  $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dtrsm:  $(OBJDIR)/test/dtrsm.o  $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

sgemm:  $(OBJDIR)/test/sgemm.o  $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ssyrk:  $(OBJDIR)/test/ssyrk.o  $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
strmm2: $(OBJDIR)/test/strmm2.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
strmm:  $(OBJDIR)/test/strmm.o  $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
strsm:  $(OBJDIR)/test/strsm.o  $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

zgemm:  $(OBJDIR)/test/zgemm.o  $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
zherk:  $(OBJDIR)/test/zherk.o  $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ztrmm2: $(OBJDIR)/test/ztrmm2.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ztrmm:  $(OBJDIR)/test/ztrmm.o  $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ztrsm:  $(OBJDIR)/test/ztrsm.o  $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cucgemm:  $(OBJDIR)/test/cucgemm.o  $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | cgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cucgemm2: $(OBJDIR)/test/cucgemm2.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | cgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cucherk:  $(OBJDIR)/test/cucherk.o  $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | cherk.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cuctrmm2: $(OBJDIR)/test/cuctrmm2.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | ctrmm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cuctrsm:  $(OBJDIR)/test/cuctrsm.o  $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | ctrsm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cudgemm:  $(OBJDIR)/test/cudgemm.o  $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | dgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cudgemm2: $(OBJDIR)/test/cudgemm2.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | dgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cudsyrk:  $(OBJDIR)/test/cudsyrk.o  $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | dsyrk.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cudtrmm2: $(OBJDIR)/test/cudtrmm2.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | dtrmm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cudtrsm:  $(OBJDIR)/test/cudtrsm.o  $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | dtrsm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cusgemm:  $(OBJDIR)/test/cusgemm.o  $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cusgemm2: $(OBJDIR)/test/cusgemm2.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cussyrk:  $(OBJDIR)/test/cussyrk.o  $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | ssyrk.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
custrmm2: $(OBJDIR)/test/custrmm2.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | strmm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
custrsm:  $(OBJDIR)/test/custrsm.o  $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | strsm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cuzgemm:  $(OBJDIR)/test/cuzgemm.o  $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cuzgemm2: $(OBJDIR)/test/cuzgemm2.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cuzherk:  $(OBJDIR)/test/cuzherk.o  $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | zherk.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cuztrmm2: $(OBJDIR)/test/cuztrmm2.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | ztrmm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cuztrsm:  $(OBJDIR)/test/cuztrsm.o  $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | ztrsm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cumultigpucgemm: $(OBJDIR)/test/cumultigpucgemm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpucherk: $(OBJDIR)/test/cumultigpucherk.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpuctrmm: $(OBJDIR)/test/cumultigpuctrmm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpuctrsm: $(OBJDIR)/test/cumultigpuctrsm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cumultigpudgemm: $(OBJDIR)/test/cumultigpudgemm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpudsyrk: $(OBJDIR)/test/cumultigpudsyrk.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpudtrmm: $(OBJDIR)/test/cumultigpudtrmm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpudtrsm: $(OBJDIR)/test/cumultigpudtrsm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cumultigpusgemm: $(OBJDIR)/test/cumultigpusgemm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpussyrk: $(OBJDIR)/test/cumultigpussyrk.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpustrmm: $(OBJDIR)/test/cumultigpustrmm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpustrsm: $(OBJDIR)/test/cumultigpustrsm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cumultigpuzgemm: $(OBJDIR)/test/cumultigpuzgemm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpuzherk: $(OBJDIR)/test/cumultigpuzherk.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpuztrmm: $(OBJDIR)/test/cumultigpuztrmm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cumultigpuztrsm: $(OBJDIR)/test/cumultigpuztrsm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/error.o | sgemm.fatbin dgemm.fatbin cgemm.fatbin zgemm.fatbin
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

cpotrf:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/test/cpotrf.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
cpotri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/cpotri.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/cpotri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ctrtri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/ctrtri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ctrtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/ctrtri2.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

dpotrf:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/test/dpotrf.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dpotri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dpotri.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/dpotri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dtrtri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/dtrtri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
dtrtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/dtrtri2.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

spotrf:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/test/spotrf.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
spotri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/spotri.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/spotri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
strtri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/strtri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
strtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/strtri2.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

zpotrf:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/test/zpotrf.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
zpotri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/zpotri.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/zpotri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ztrtri:  $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/ztrtri.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)
ztrtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/handle.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/ztrtri2.o
	$(CC) $(^) -o $(@) $(LDFLAGS) $(LOADLIBES) $(LDLIBS)

$(OBJDIR):
	$(MKDIR) $(@)

$(OBJDIR)/src: | $(OBJDIR)
	$(MKDIR) $(@)
$(OBJDIR)/src/error.o: error.h | $(OBJDIR)/src
$(OBJDIR)/src/multigpu.o: cumultigpu.h error.h | $(OBJDIR)/src

$(OBJDIR)/src/blas: | $(OBJDIR)/src
	$(MKDIR) $(@)
$(OBJDIR)/src/blas/handle.o: blas.h cumultigpu.h handle.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/xerbla.o: blas.h cumultigpu.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cgemm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cherk.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrmm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrsm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dgemm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dsyrk.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrmm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrsm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/sgemm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ssyrk.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strmm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strsm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zgemm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zherk.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrmm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrsm.o: blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/blas

$(OBJDIR)/src/lapack: | $(OBJDIR)/src
	$(MKDIR) $(@)
$(OBJDIR)/src/lapack/cpotrf.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/cpotri.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/ctrtri.o: lapack.h blas.h cumultigpu.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dpotrf.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dpotri.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dtrtri.o: lapack.h blas.h cumultigpu.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/spotrf.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/spotri.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/strtri.o: lapack.h blas.h cumultigpu.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/zpotrf.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/zpotri.o: lapack.h blas.h cumultigpu.h error.h handle.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/ztrtri.o: lapack.h blas.h cumultigpu.h | $(OBJDIR)/src/lapack

cgemm.fatbin cherk.fatbin ctrmm.fatbin ctrsm.fatbin: FATBINFLAGS = -code=sm_11,sm_13 -arch=compute_11
dgemm.fatbin dsyrk.fatbin dtrmm.fatbin dtrsm.fatbin: FATBINFLAGS = -code=sm_13 -arch=compute_13
sgemm.fatbin ssyrk.fatbin strmm.fatbin strsm.fatbin: FATBINFLAGS = -code=sm_11,sm_13 -arch=compute_11
zgemm.fatbin zherk.fatbin ztrmm.fatbin ztrsm.fatbin: FATBINFLAGS = -code=sm_13 -arch=compute_13
cgemm.fatbin: blas.h cumultigpu.h
cherk.fatbin: blas.h cumultigpu.h
ctrmm.fatbin: blas.h cumultigpu.h
ctrsm.fatbin: blas.h cumultigpu.h
dgemm.fatbin: blas.h cumultigpu.h
dsyrk.fatbin: blas.h cumultigpu.h
dtrmm.fatbin: blas.h cumultigpu.h
dtrsm.fatbin: blas.h cumultigpu.h
sgemm.fatbin: blas.h cumultigpu.h
ssyrk.fatbin: blas.h cumultigpu.h
strmm.fatbin: blas.h cumultigpu.h
strsm.fatbin: blas.h cumultigpu.h
zgemm.fatbin: blas.h cumultigpu.h
zherk.fatbin: blas.h cumultigpu.h
ztrmm.fatbin: blas.h cumultigpu.h
ztrsm.fatbin: blas.h cumultigpu.h

cpotrf.fatbin spotrf.fatbin: FATBINFLAGS = -code=sm_11,sm_13 -arch=compute_11
dpotrf.fatbin zpotrf.fatbin: FATBINFLAGS = -code=sm_13 -arch=compute_13
cpotrf.fatbin: blas.h cumultigpu.h
dpotrf.fatbin: blas.h cumultigpu.h
spotrf.fatbin: blas.h cumultigpu.h
zpotrf.fatbin: blas.h cumultigpu.h

$(OBJDIR)/test: | $(OBJDIR)
	$(MKDIR) $(@)

$(OBJDIR)/test/cgemm.o:  blas.h cumultigpu.h test/cgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cherk.o:  blas.h cumultigpu.h test/cherk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ctrmm2.o: blas.h cumultigpu.h test/ctrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ctrmm.o:  blas.h cumultigpu.h test/ctrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ctrsm.o:  blas.h cumultigpu.h test/ctrsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/dgemm.o:  blas.h cumultigpu.h test/dgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/dsyrk.o:  blas.h cumultigpu.h test/dsyrk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/dtrmm2.o: blas.h cumultigpu.h test/dtrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/dtrmm.o:  blas.h cumultigpu.h test/dtrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/dtrsm.o:  blas.h cumultigpu.h test/dtrsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/sgemm.o:  blas.h cumultigpu.h test/sgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ssyrk.o:  blas.h cumultigpu.h test/ssyrk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/strmm2.o: blas.h cumultigpu.h test/strmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/strmm.o:  blas.h cumultigpu.h test/strmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/strsm.o:  blas.h cumultigpu.h test/strsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/zgemm.o:  blas.h cumultigpu.h test/zgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/zherk.o:  blas.h cumultigpu.h test/zherk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ztrmm2.o: blas.h cumultigpu.h test/ztrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ztrmm.o:  blas.h cumultigpu.h test/ztrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/ztrsm.o:  blas.h cumultigpu.h test/ztrsm_ref.c | $(OBJDIR)/test

$(OBJDIR)/test/cucgemm2.o: blas.h cumultigpu.h error.h test/cgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cucgemm.o:  blas.h cumultigpu.h error.h test/cgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cucherk.o:  blas.h cumultigpu.h error.h test/cherk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuctrmm2.o: blas.h cumultigpu.h error.h test/ctrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuctrsm.o:  blas.h cumultigpu.h error.h test/ctrsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cudgemm2.o: blas.h cumultigpu.h error.h test/dgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cudgemm.o:  blas.h cumultigpu.h error.h test/dgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cudsyrk.o:  blas.h cumultigpu.h error.h test/dsyrk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cudtrmm2.o: blas.h cumultigpu.h error.h test/dtrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cudtrsm.o:  blas.h cumultigpu.h error.h test/dtrsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cusgemm2.o: blas.h cumultigpu.h error.h test/sgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cusgemm.o:  blas.h cumultigpu.h error.h test/sgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cussyrk.o:  blas.h cumultigpu.h error.h test/ssyrk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/custrmm2.o: blas.h cumultigpu.h error.h test/strmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/custrsm.o:  blas.h cumultigpu.h error.h test/strsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuzgemm2.o: blas.h cumultigpu.h error.h test/zgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuzgemm.o:  blas.h cumultigpu.h error.h test/zgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuzherk.o:  blas.h cumultigpu.h error.h test/zherk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuztrmm2.o: blas.h cumultigpu.h error.h test/ztrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cuztrsm.o:  blas.h cumultigpu.h error.h test/ztrsm_ref.c | $(OBJDIR)/test

$(OBJDIR)/test/cumultigpucgemm.o: blas.h cumultigpu.h error.h test/cgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucherk.o: blas.h cumultigpu.h error.h test/cherk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuctrmm.o: blas.h cumultigpu.h error.h test/ctrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuctrsm.o: blas.h cumultigpu.h error.h test/ctrsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudgemm.o: blas.h cumultigpu.h error.h test/dgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudsyrk.o: blas.h cumultigpu.h error.h test/dsyrk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudtrmm.o: blas.h cumultigpu.h error.h test/dtrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudtrsm.o: blas.h cumultigpu.h error.h test/dtrsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpusgemm.o: blas.h cumultigpu.h error.h test/sgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpussyrk.o: blas.h cumultigpu.h error.h test/ssyrk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpustrmm.o: blas.h cumultigpu.h error.h test/strmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpustrsm.o: blas.h cumultigpu.h error.h test/strsm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzgemm.o: blas.h cumultigpu.h error.h test/zgemm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzherk.o: blas.h cumultigpu.h error.h test/zherk_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuztrmm.o: blas.h cumultigpu.h error.h test/ztrmm_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuztrsm.o: blas.h cumultigpu.h error.h test/ztrsm_ref.c | $(OBJDIR)/test

$(OBJDIR)/test/spotrf.o:  lapack.h blas.h cumultigpu.h test/spotrf_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/spotri.o:  lapack.h blas.h cumultigpu.h test/spotri_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/strtri.o:  lapack.h blas.h cumultigpu.h test/strtri_ref.c | $(OBJDIR)/test
$(OBJDIR)/test/strtri2.o: lapack.h blas.h cumultigpu.h test/strtri_ref.c | $(OBJDIR)/test

# $(PTXDIR):
# 	$(MKDIR) $(@)
#
# $(PTXDIR)/src: | $(OBJDIR)
# 	$(MKDIR) $(@)
#
# $(PTXDIR)/src/error.o: error.h | $(OBJDIR)/src
#
# $(PTXDIR)/src/blas: | $(OBJDIR)/src
# 	$(MKDIR) $(@)
#
# BLAS_PTXS = $(foreach prec,s d,$(prec)gemm.ptx $(prec)syrk.ptx $(prec)trsm.ptx $(prec)trmm.ptx) \
#             $(foreach prec,c z,$(prec)gemm.ptx $(prec)herk.ptx $(prec)trsm.ptx $(prec)trmm.ptx)
# $(foreach blas,$(BLAS_PTXS),$(eval $(PTXDIR)/src/blas/$(blas): blas.h | $(PTXDIR)/src/blas))


# Rules
# Shell files
# %.c: %.sh
# 	$(SHELL) $(.SHELLFLAGS) $(<) > $(@)

# Object files
$(OBJDIR)/%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

# Archive files
# %.a:
# 	$(AR) $(ARFLAGS) $(@) $(^)

# PTX files
# %.ptx: %.cu
# 	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -ptx $(<)
#
# define ptx_template
# %.compute_$(1).ptx: %.cu
# 	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -arch=compute_$(1) -o $$(@) -ptx $$(<)
# endef
# $(foreach code,10 11 12 13 20,$(eval $(call ptx_template,$(code))))

# Cubins
# %.cubin: %.cu
# 	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -o $(@) -cubin $(<)
#
# %.cubin: %.ptx
# 	$(PTXAS) $(PTXASFLAGS) -o $(@) $(<)
#
# define cubin_template
# %.sm_$(1).cubin: %.compute_$(1).ptx
# 	$(PTXAS) $(PTXASFLAGS) -arch=sm_$(1) -o $$(@) $$(<)
# %.sm_$(1).cubin: %.cu
# 	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -arch=sm_$(1) -o $$(@) -cubin $$(<)
# endef
# $(foreach arch,10 11 12 13 20,$(eval $(call cubin_template,$(arch))))
#
# define cubin20_template
# %.sm_$(1).cubin: %.compute_20.ptx
# 	$(PTXAS) $(PTXASFLAGS) -arch=sm_$(1) -o $$(@) $$(<)
# %.sm_$(1).cubin: %.cu
# 	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -arch=sm_$(1) -o $$(@) -cubin $$(<)
# endef
# $(foreach arch,21 22 23,$(eval $(call cubin20_template,$(arch))))

# FATBINs - optional ptx followed by cubins
# e.g. sgemm.fatbin: sgemm.ptx sgemm.sm_11.cubin sgemm.sm_13.cubin
# nvcc 4.x can create FATBINs directly from .cu
%.fatbin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) $(FATBINFLAGS) -o $(@) -fatbin $(<)

# %.fatbin:
# 	$(FATBINARY) $(FATBINARYFLAGS) -cuda --create $(@) $(if $(filter %.ptx,$(<)),--ptx $(filter %.ptx,$(<))) $(foreach cubin,$(filter %.cubin,$(^)),--image profile=$(lastword $(subst ., ,$(basename $(cubin)))),file=$(cubin))

# Dynamic loading of FATBINs
# Converts the fatbins to linkable object files with the symbols _path_to_fatbin
# and _path_to_fatbin_size
# $(OBJDIR)/%.o : $(FATBINDIR)/%.fatbin
# 	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_start=_$(subst -,_,$(subst .,_,$(notdir $(<)))) --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_size=_$(subst -,_,$(subst .,_,$(notdir $(<))))_size --strip-symbol _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_end $(<) $(@)
