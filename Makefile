CUDA_HOME = /opt/cuda
INTEL_HOME = /opt/intel/composerxe-2011.6.233

AR = ar
CC = gcc
NVCC = nvcc
PTXAS = ptxas
FATBINARY = fatbinary
NVCFLAGS = -O2 -use_fast_math -maxrregcount=32
PTXASFLAGS = -O2 -maxrregcount=32
CUBIN_ARCHES = sm_11,sm_13
PTX_ARCH = compute_11
CUBIN_ARCHES_DOUBLE = sm_13
PTX_ARCH_DOUBLE = compute_13

ARFLAGS = crs
CPPFLAGS = -Iinclude -I$(CUDA_HOME)/include
NVCPPFLAGS = -Iinclude
LDFLAGS = -rdynamic -L$(CUDA_HOME)/lib64
LDLIBS = -lcuda -lrt -ldl

# TODO:  separate no-opt CFLAGS for testing code.
# TODO:  implement hacks in C codes to vectorise all possible loops.
ifeq ($(notdir $(CC)), icc)
  CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp
  LDFLAGS += -L$(INTEL_HOME)
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

RM = rm -f
RMDIR = rm -rf
MKDIR = mkdir

VPATH = include src src/blas src/lapack

OBJDIR = obj
PTXDIR = ptx

.NOTPARALLEL: $(OBJDIR) $(OBJDIR)/src $(OBJDIR)/src/blas $(OBJDIR)/test

.PHONY: all clean distclean

TEST_PROGS = cutask cutaskqueue cuthread \
             sgemm dgemm cgemm zgemm \
             ssyrk dsyrk cherk zherk \
             strsm dtrsm ctrsm ztrsm \
             strmm dtrmm ctrmm ztrmm \
             strmm2 dtrmm2 ctrmm2 ztrmm2 \
             cusgemm cudgemm cucgemm cuzgemm \
             cusgemm2 cudgemm2 cucgemm2 cuzgemm2 \
             cussyrk cudsyrk cucherk cuzherk \
             custrsm cudtrsm cuctrsm cuztrsm \
             custrmm2 cudtrmm2 cuctrmm2 cuztrmm2 \
             cumultigpusgemm cumultigpudgemm cumultigpucgemm cumultigpuzgemm \
             cumultigpussyrk cumultigpudsyrk cumultigpucherk cumultigpuzherk \
             cumultigpustrmm cumultigpudtrmm cumultigpuctrmm cumultigpuztrmm \
             cumultigpustrsm cumultigpudtrsm cumultigpuctrsm cumultigpuztrsm \
             spotrf dpotrf cpotrf zpotrf \
             strtri dtrtri ctrtri ztrtri \
             strtri2 dtrtri2 ctrtri2 ztrtri2 \
             cuspotrf cudpotrf cucpotrf cuzpotrf \
             cumultigpuspotrf cumultigpudpotrf cumultigpucpotrf cumultigpuzpotrf
TEST_FATBINS = sgemm.fatbin cgemm.fatbin \
               ssyrk.fatbin cherk.fatbin \
               strmm.fatbin ctrmm.fatbin \
               strsm.fatbin ctrsm.fatbin \
               spotrf.fatbin cpotrf.fatbin \
               $(TEST_FATBINS_DOUBLE)
TEST_FATBINS_DOUBLE = dgemm.fatbin zgemm.fatbin \
                      dsyrk.fatbin zherk.fatbin \
                      dtrmm.fatbin ztrmm.fatbin \
                      dtrsm.fatbin ztrsm.fatbin \
                      dpotrf.fatbin zpotrf.fatbin

MULTIGPU_OBJS = $(OBJDIR)/task.o $(OBJDIR)/taskqueue.o $(OBJDIR)/thread.o

all: $(TEST_PROGS)

clean:
	$(RMDIR) $(OBJDIR)

distclean: clean
	$(RM) $(TEST_PROGS) $(TEST_FATBINS)

$(TEST_PROGS):
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

$(TEST_FATBINS_DOUBLE): PTX_ARCH = $(PTX_ARCH_DOUBLE)
$(TEST_FATBINS_DOUBLE): CUBIN_ARCHES = $(CUBIN_ARCHES_DOUBLE)

cutask: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/test/cutask.o

cutaskqueue: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/src/taskqueue.o $(OBJDIR)/test/cutaskqueue.o

cuthread: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/src/taskqueue.o $(OBJDIR)/src/thread.o $(OBJDIR)/test/cuthread.o

sgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/sgemm.o
dgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/dgemm.o
cgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cgemm.o
zgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/zgemm.o
ssyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/test/ssyrk.o
dsyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/test/dsyrk.o
cherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/test/cherk.o
zherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/test/zherk.o
strmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/test/strmm.o
dtrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/test/dtrmm.o
ctrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/test/ctrmm.o
ztrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/test/ztrmm.o
strmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/test/strmm2.o
dtrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/test/dtrmm2.o
ctrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/test/ctrmm2.o
ztrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/test/ztrmm2.o
strsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/test/strsm.o
dtrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/test/dtrsm.o
ctrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/test/ctrsm.o
ztrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/test/ztrsm.o

spotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/spotrf.o
dpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/dpotrf.o
cpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/cpotrf.o
zpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/zpotrf.o

strtri: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/strtri.o
dtrtri: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/dtrtri.o
ctrtri: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/ctrtri.o
ztrtri: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/ztrtri.o
strtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/strtri2.o
dtrtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/dtrtri2.o
ctrtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/ctrtri2.o
ztrtri2: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/ztrtri2.o

cusgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/cusgemm.o | sgemm.fatbin
cudgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/cudgemm.o | dgemm.fatbin
cucgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cucgemm.o | cgemm.fatbin
cuzgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/cuzgemm.o | zgemm.fatbin
cusgemm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/cusgemm2.o | sgemm.fatbin
cudgemm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/cudgemm2.o | dgemm.fatbin
cucgemm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cucgemm2.o | cgemm.fatbin
cuzgemm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/cuzgemm2.o | zgemm.fatbin
cussyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/test/cussyrk.o | ssyrk.fatbin
cudsyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/test/cudsyrk.o | dsyrk.fatbin
cucherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/test/cucherk.o | cherk.fatbin
cuzherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/test/cuzherk.o | zherk.fatbin
custrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/test/custrmm2.o | strmm.fatbin
cudtrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/test/cudtrmm2.o | dtrmm.fatbin
cuctrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/test/cuctrmm2.o | ctrmm.fatbin
cuztrmm2: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/test/cuztrmm2.o | ztrmm.fatbin
custrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/test/custrsm.o | strsm.fatbin
cudtrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/test/cudtrsm.o | dtrsm.fatbin
cuctrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/test/cuctrsm.o | ctrsm.fatbin
cuztrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/test/cuztrsm.o | ztrsm.fatbin

cuspotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/src/lapack/strtri.o $(OBJDIR)/test/cuspotrf.o | sgemm.fatbin ssyrk.fatbin strsm.fatbin strmm.fatbin spotrf.fatbin
cudpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/src/lapack/dtrtri.o $(OBJDIR)/test/cudpotrf.o | dgemm.fatbin dsyrk.fatbin dtrsm.fatbin dtrmm.fatbin dpotrf.fatbin
cucpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/src/lapack/ctrtri.o $(OBJDIR)/test/cucpotrf.o | cgemm.fatbin cherk.fatbin ctrsm.fatbin ctrmm.fatbin cpotrf.fatbin
cuzpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/src/lapack/ztrtri.o $(OBJDIR)/test/cuzpotrf.o | zgemm.fatbin zherk.fatbin ztrsm.fatbin ztrmm.fatbin zpotrf.fatbin

cumultigpusgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/src/taskqueue.o $(OBJDIR)/src/thread.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/cumultigpusgemm.o | sgemm.fatbin
cumultigpudgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/src/taskqueue.o $(OBJDIR)/src/thread.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/cumultigpudgemm.o | dgemm.fatbin
cumultigpucgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/src/taskqueue.o $(OBJDIR)/src/thread.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cumultigpucgemm.o | cgemm.fatbin
cumultigpuzgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/task.o $(OBJDIR)/src/taskqueue.o $(OBJDIR)/src/thread.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/cumultigpuzgemm.o | zgemm.fatbin
cumultigpussyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/test/cumultigpussyrk.o | sgemm.fatbin ssyrk.fatbin
cumultigpudsyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/test/cumultigpudsyrk.o | dgemm.fatbin dsyrk.fatbin
cumultigpucherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/test/cumultigpucherk.o | cgemm.fatbin cherk.fatbin
cumultigpuzherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/test/cumultigpuzherk.o | zgemm.fatbin zherk.fatbin
cumultigpustrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strmm.o $(OBJDIR)/test/cumultigpustrmm.o | sgemm.fatbin strmm.fatbin
cumultigpudtrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrmm.o $(OBJDIR)/test/cumultigpudtrmm.o | dgemm.fatbin dtrmm.fatbin
cumultigpuctrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrmm.o $(OBJDIR)/test/cumultigpuctrmm.o | cgemm.fatbin ctrmm.fatbin
cumultigpuztrmm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrmm.o $(OBJDIR)/test/cumultigpuztrmm.o | zgemm.fatbin ztrmm.fatbin
cumultigpustrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/test/cumultigpustrsm.o | sgemm.fatbin strsm.fatbin
cumultigpudtrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/test/cumultigpudtrsm.o | dgemm.fatbin dtrsm.fatbin
cumultigpuctrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/test/cumultigpuctrsm.o | cgemm.fatbin ctrsm.fatbin
cumultigpuztrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/test/cumultigpuztrsm.o | zgemm.fatbin ztrsm.fatbin

cumultigpuspotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/test/cumultigpuspotrf.o | sgemm.fatbin ssyrk.fatbin strsm.fatbin
cumultigpudpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/test/cumultigpudpotrf.o | dgemm.fatbin dsyrk.fatbin dtrsm.fatbin
cumultigpucpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/test/cumultigpucpotrf.o | cgemm.fatbin cherk.fatbin ctrsm.fatbin
cumultigpuzpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/test/cumultigpuzpotrf.o | zgemm.fatbin zherk.fatbin ztrsm.fatbin

$(OBJDIR):
	$(MKDIR) $(@)

$(OBJDIR)/src: | $(OBJDIR)
	$(MKDIR) $(@)

$(OBJDIR)/src/error.o: error.h | $(OBJDIR)/src

$(OBJDIR)/src/util: | $(OBJDIR)/src
	$(MKDIR) $(@)

$(OBJDIR)/src/util/arrayqueue.o: util/arrayqueue.h | $(OBJDIR)/src/util
$(OBJDIR)/src/util/task.o: util/task.h | $(OBJDIR)/src/util
$(OBJDIR)/src/util/thread.o: util/thread.h util/arrayqueue.h util/task.h | $(OBJDIR)/src/util

$(OBJDIR)/src/blas: | $(OBJDIR)/src
	$(MKDIR) $(@)

$(OBJDIR)/src/blas/xerbla.o: blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/sgemm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dgemm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cgemm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zgemm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ssyrk.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dsyrk.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cherk.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zherk.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strsm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrsm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrsm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrsm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strmm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrmm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrmm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrmm.o: blas.h cuthread.h cutask.h cutaskqueue.h error.h | $(OBJDIR)/src/blas

$(OBJDIR)/src/lapack: | $(OBJDIR)/src
	$(MKDIR) $(@)

$(OBJDIR)/src/lapack/spotrf.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dpotrf.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/cpotrf.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/zpotrf.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/spotri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dpotri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/cpotri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/zpotri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/strtri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dtrtri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/ctrtri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/ztrtri.o: lapack.h blas.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/src/lapack

$(OBJDIR)/test: | $(OBJDIR)
	$(MKDIR) $(@)

$(OBJDIR)/test/cutask.o: cutask.h | $(OBJDIR)/test
$(OBJDIR)/test/cutaskqueue.o: cutaskqueue.h cutask.h | $(OBJDIR)/test
$(OBJDIR)/test/cuthread.o: cuthread.h cutaskqueue.h cutask.h | $(OBJDIR)/test

$(OBJDIR)/test/sgemm.o: test/sgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dgemm.o: test/dgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cgemm.o: test/cgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/zgemm.o: test/zgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ssyrk.o: test/ssyrk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dsyrk.o: test/dsyrk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cherk.o: test/cherk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/zherk.o: test/zherk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/strsm.o: test/strsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dtrsm.o: test/dtrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ctrsm.o: test/ctrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ztrsm.o: test/ztrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/strmm.o: test/strmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dtrmm.o: test/dtrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ctrmm.o: test/ctrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ztrmm.o: test/ztrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/strmm2.o: test/strmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dtrmm2.o: test/dtrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ctrmm2.o: test/ctrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ztrmm2.o: test/ztrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cusgemm.o: test/sgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudgemm.o: test/dgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucgemm.o: test/cgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzgemm.o: test/zgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cusgemm2.o: test/sgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudgemm2.o: test/dgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucgemm2.o: test/cgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzgemm2.o: test/zgemm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cussyrk.o: test/ssyrk_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudsyrk.o: test/dsyrk_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucherk.o: test/cherk_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzherk.o: test/zherk_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/custrsm.o: test/strsm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudtrsm.o: test/dtrsm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuctrsm.o: test/ctrsm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuztrsm.o: test/ztrsm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/custrmm.o: test/strmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudtrmm.o: test/dtrmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuctrmm.o: test/ctrmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuztrmm.o: test/ztrmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/custrmm2.o: test/strmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudtrmm2.o: test/dtrmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuctrmm2.o: test/ctrmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuztrmm2.o: test/ztrmm_ref.c error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpusgemm.o: test/sgemm_ref.c blas.h cutask.h cutaskqueue.h cuthread.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudgemm.o: test/dgemm_ref.c blas.h cutask.h cutaskqueue.h cuthread.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucgemm.o: test/cgemm_ref.c blas.h cutask.h cutaskqueue.h cuthread.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzgemm.o: test/zgemm_ref.c blas.h cutask.h cutaskqueue.h cuthread.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpussyrk.o: test/ssyrk_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudsyrk.o: test/dsyrk_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucherk.o: test/cherk_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzherk.o: test/zherk_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpustrsm.o: test/strsm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudtrsm.o: test/dtrsm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuctrsm.o: test/ctrsm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuztrsm.o: test/ztrsm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpustrmm.o: test/strmm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudtrmm.o: test/dtrmm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuctrmm.o: test/ctrmm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuztrmm.o: test/ztrmm_ref.c multigpu.h cumultigpu.h error.h | $(OBJDIR)/test

$(OBJDIR)/test/spotrf.o: test/spotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dpotrf.o: test/dpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cpotrf.o: test/cpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/zpotrf.o: test/zpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuspotrf.o: test/spotrf_ref.c lapack.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudpotrf.o: test/dpotrf_ref.c lapack.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucpotrf.o: test/cpotrf_ref.c lapack.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzpotrf.o: test/zpotrf_ref.c lapack.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuspotrf.o: test/spotrf_ref.c lapack.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudpotrf.o: test/dpotrf_ref.c lapack.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucpotrf.o: test/cpotrf_ref.c lapack.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzpotrf.o: test/zpotrf_ref.c lapack.h multigpu.h cumultigpu.h error.h | $(OBJDIR)/test

$(OBJDIR)/test/strtri.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dtrtri.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ctrtri.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ztrtri.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/strtri2.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dtrtri2.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ctrtri2.o: lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/ztrtri2.o: lapack.h blas.h error.h | $(OBJDIR)/test

$(TEST_FATBINS): blas.h

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
%.c: %.sh
	$(SHELL) $(.SHELLFLAGS) $(<) > $(@)

# Object files
$(OBJDIR)/%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

# Archive files
%.a:
	$(AR) $(ARFLAGS) $(@) $(^)

# PTX files
%.ptx: %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -ptx $(<)

define ptx_template
%.compute_$(1).ptx: %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -arch=compute_$(1) -o $$(@) -ptx $$(<)
endef
$(foreach code,10 11 12 13 20,$(eval $(call ptx_template,$(code))))

# Cubins
%.cubin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -o $(@) -cubin $(<)

%.cubin: %.ptx
	$(PTXAS) $(PTXASFLAGS) -o $(@) $(<)

define cubin_template
%.sm_$(1).cubin: %.compute_$(1).ptx
	$(PTXAS) $(PTXASFLAGS) -arch=sm_$(1) -o $$(@) $$(<)
%.sm_$(1).cubin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -arch=sm_$(1) -o $$(@) -cubin $$(<)
endef
$(foreach arch,10 11 12 13 20,$(eval $(call cubin_template,$(arch))))

define cubin20_template
%.sm_$(1).cubin: %.compute_20.ptx
	$(PTXAS) $(PTXASFLAGS) -arch=sm_$(1) -o $$(@) $$(<)
%.sm_$(1).cubin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -arch=sm_$(1) -o $$(@) -cubin $$(<)
endef
$(foreach arch,21 22 23,$(eval $(call cubin20_template,$(arch))))

# FATBINs - optional ptx followed by cubins
# e.g. sgemm.fatbin: sgemm.ptx sgemm.sm_11.cubin sgemm.sm_13.cubin
# nvcc 4.x can create FATBINs directly from .cu
%.fatbin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) $(if $(CUBIN_ARCHES),-code=$(foreach arch,$(CUBIN_ARCHES),$(arch))) $(if $(PTX_ARCH),-arch=$(PTX_ARCH)) -o $(@) -fatbin $(<)

%.fatbin:
	$(FATBINARY) $(FATBINARYFLAGS) -cuda --create $(@) $(if $(filter %.ptx,$(<)),--ptx $(filter %.ptx,$(<))) $(foreach cubin,$(filter %.cubin,$(^)),--image profile=$(lastword $(subst ., ,$(basename $(cubin)))),file=$(cubin))

# Dynamic loading of FATBINs
# Converts the fatbins to linkable object files with the symbols _path_to_fatbin
# and _path_to_fatbin_size
$(OBJDIR)/%.o : $(FATBINDIR)/%.fatbin
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_start=_$(subst -,_,$(subst .,_,$(notdir $(<)))) --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_size=_$(subst -,_,$(subst .,_,$(notdir $(<))))_size --strip-symbol _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_end $(<) $(@)
