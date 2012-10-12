CUDA_HOME = /opt/cuda
INTEL_HOME = /opt/intel/composerxe-2011.6.233

CC = gcc
NVCC = nvcc
PTXAS = ptxas
FATBINARY = fatbinary
NVCFLAGS = -O2 -use_fast_math -maxrregcount=32
PTXASFLAGS = -O2 -maxrregcount=32

CPPFLAGS = -Iinclude -I$(CUDA_HOME)/include
NVCPPFLAGS = -Iinclude
LDFLAGS = -rdynamic -L$(CUDA_HOME)/lib64
LDLIBS = -lcuda -lrt -ldl

# TODO:  separate no-opt CFLAGS for testing code.
# TODO:  implement hacks in C codes to vectorise all possible loops.
ifeq ($(notdir $(CC)), icc)
  CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp -vec-report=2
  LDFLAGS += -L$(INTEL_HOME)
  LDLIBS += -liomp5
else
  CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -fopenmp -ftree-vectorizer-verbose=2 -ffast-math
  LDLIBS += -lgomp
endif

ifeq ($(bench), 1)
  CPPFLAGS += -DBENCH
endif

RM = rm -f
RMDIR = rm -rf
MKDIR = mkdir

VPATH = include src src/blas

OBJDIR = obj
PTXDIR = ptx

.NOTPARALLEL: $(OBJDIR) $(OBJDIR)/src $(OBJDIR)/src/blas $(OBJDIR)/test

.PHONY: all clean distclean

TEST_PROGS = sgemm dgemm cgemm zgemm \
             ssyrk dsyrk cherk zherk \
             strsm dtrsm ctrsm ztrsm \
             cusgemm cudgemm cucgemm cuzgemm \
             cussyrk cudsyrk cucherk cuzherk \
             custrsm cudtrsm cuctrsm cuztrsm \
             cumultigpusgemm cumultigpudgemm cumultigpucgemm cumultigpuzgemm \
             cumultigpussyrk cumultigpudsyrk cumultigpucherk cumultigpuzherk \
             cumultigpustrsm cumultigpudtrsm cumultigpuctrsm cumultigpuztrsm \
             spotrf dpotrf cpotrf zpotrf \
             cuspotrf cudpotrf cucpotrf cuzpotrf \
             cumultigpuspotrf cumultigpudpotrf cumultigpucpotrf cumultigpuzpotrf
TEST_CUBINS = sgemm.cubin dgemm.cubin cgemm.cubin zgemm.cubin \
              ssyrk.cubin dsyrk.cubin cherk.cubin zherk.cubin \
              strsm.cubin dtrsm.cubin ctrsm.cubin ztrsm.cubin

all: $(TEST_PROGS)

clean:
	$(RMDIR) obj

distclean: clean
	$(RM) $(TEST_PROGS) $(TEST_CUBINS)

$(TEST_PROGS):
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

sgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/sgemm.o
dgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/dgemm.o
cgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cgemm.o
zgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/zgemm.o
ssyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/test/ssyrk.o
dsyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/test/dsyrk.o
cherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/test/cherk.o
zherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/test/zherk.o
strsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/test/strsm.o
dtrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/test/dtrsm.o
ctrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/test/ctrsm.o
ztrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/test/ztrsm.o

spotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/test/spotrf.o
dpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/test/dpotrf.o
cpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/test/cpotrf.o
zpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/test/zpotrf.o

cusgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/cusgemm.o | sgemm.cubin
cudgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/cudgemm.o | dgemm.cubin
cucgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cucgemm.o | cgemm.cubin
cuzgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/cuzgemm.o | zgemm.cubin
cussyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/test/cussyrk.o | ssyrk.cubin
cudsyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/test/cudsyrk.o | dsyrk.cubin
cucherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/test/cucherk.o | cherk.cubin
cuzherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/test/cuzherk.o | zherk.cubin
custrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/test/custrsm.o | strsm.cubin
cudtrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/test/cudtrsm.o | dtrsm.cubin
cuctrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/test/cuctrsm.o | ctrsm.cubin
cuztrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/test/cuztrsm.o | ztrsm.cubin

cuspotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/test/cuspotrf.o | sgemm.cubin ssyrk.cubin strsm.cubin
cudpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/test/cudpotrf.o | dgemm.cubin dsyrk.cubin dtrsm.cubin
cucpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/test/cucpotrf.o | cgemm.cubin cherk.cubin ctrsm.cubin
cuzpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/test/cuzpotrf.o | zgemm.cubin zherk.cubin ztrsm.cubin

cumultigpusgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/test/cumultigpusgemm.o | sgemm.cubin
cumultigpudgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/test/cumultigpudgemm.o | dgemm.cubin
cumultigpucgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/test/cumultigpucgemm.o | cgemm.cubin
cumultigpuzgemm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/test/cumultigpuzgemm.o | zgemm.cubin
cumultigpussyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/test/cumultigpussyrk.o | sgemm.cubin ssyrk.cubin
cumultigpudsyrk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/test/cumultigpudsyrk.o | dgemm.cubin dsyrk.cubin
cumultigpucherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/test/cumultigpucherk.o | cgemm.cubin cherk.cubin
cumultigpuzherk: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/test/cumultigpuzherk.o | zgemm.cubin zherk.cubin
cumultigpustrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/test/cumultigpustrsm.o | sgemm.cubin strsm.cubin
cumultigpudtrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/test/cumultigpudtrsm.o | dgemm.cubin dtrsm.cubin
cumultigpuctrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/test/cumultigpuctrsm.o | cgemm.cubin ctrsm.cubin
cumultigpuztrsm: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/test/cumultigpuztrsm.o | zgemm.cubin ztrsm.cubin

cumultigpuspotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sgemm.o $(OBJDIR)/src/blas/ssyrk.o $(OBJDIR)/src/blas/strsm.o $(OBJDIR)/src/lapack/spotrf.o $(OBJDIR)/test/cumultigpuspotrf.o | sgemm.cubin ssyrk.cubin strsm.cubin
cumultigpudpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/dgemm.o $(OBJDIR)/src/blas/dsyrk.o $(OBJDIR)/src/blas/dtrsm.o $(OBJDIR)/src/lapack/dpotrf.o $(OBJDIR)/test/cumultigpudpotrf.o | dgemm.cubin dsyrk.cubin dtrsm.cubin
cumultigpucpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/cgemm.o $(OBJDIR)/src/blas/cherk.o $(OBJDIR)/src/blas/ctrsm.o $(OBJDIR)/src/lapack/cpotrf.o $(OBJDIR)/test/cumultigpucpotrf.o | cgemm.cubin cherk.cubin ctrsm.cubin
cumultigpuzpotrf: $(OBJDIR)/src/error.o $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/zgemm.o $(OBJDIR)/src/blas/zherk.o $(OBJDIR)/src/blas/ztrsm.o $(OBJDIR)/src/lapack/zpotrf.o $(OBJDIR)/test/cumultigpuzpotrf.o | zgemm.cubin zherk.cubin ztrsm.cubin

$(OBJDIR):
	$(MKDIR) $(@)

$(OBJDIR)/src: | $(OBJDIR)
	$(MKDIR) $(@)

$(OBJDIR)/src/error.o: error.h | $(OBJDIR)/src

$(OBJDIR)/src/blas: | $(OBJDIR)/src
	$(MKDIR) $(@)

$(OBJDIR)/src/blas/xerbla.o: blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/sgemm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dgemm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cgemm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zgemm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ssyrk.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dsyrk.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cherk.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zherk.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strsm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrsm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrsm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrsm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strmm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrmm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrmm.o: blas.h error.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrmm.o: blas.h error.h | $(OBJDIR)/src/blas

$(OBJDIR)/src/lapack: | $(OBJDIR)/src
	$(MKDIR) $(@)

$(OBJDIR)/src/lapack/spotrf.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dpotrf.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/cpotrf.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/zpotrf.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/spotri.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/dpotri.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/cpotri.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack
$(OBJDIR)/src/lapack/zpotri.o: lapack.h blas.h error.h | $(OBJDIR)/src/lapack

$(OBJDIR)/test: | $(OBJDIR)
	$(MKDIR) $(@)

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
$(OBJDIR)/test/cusgemm.o: test/sgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudgemm.o: test/dgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucgemm.o: test/cgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzgemm.o: test/zgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cussyrk.o: test/ssyrk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudsyrk.o: test/dsyrk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucherk.o: test/cherk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzherk.o: test/zherk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/custrsm.o: test/strsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudtrsm.o: test/dtrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuctrsm.o: test/ctrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuztrsm.o: test/ztrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/custrmm.o: test/strmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudtrmm.o: test/dtrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuctrmm.o: test/ctrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuztrmm.o: test/ztrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpusgemm.o: test/sgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudgemm.o: test/dgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucgemm.o: test/cgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzgemm.o: test/zgemm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpussyrk.o: test/ssyrk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudsyrk.o: test/dsyrk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucherk.o: test/cherk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzherk.o: test/zherk_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpustrsm.o: test/strsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudtrsm.o: test/dtrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuctrsm.o: test/ctrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuztrsm.o: test/ztrsm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpustrmm.o: test/strmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudtrmm.o: test/dtrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuctrmm.o: test/ctrmm_ref.c blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuztrmm.o: test/ztrmm_ref.c blas.h error.h | $(OBJDIR)/test

$(OBJDIR)/test/spotrf.o: test/spotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/dpotrf.o: test/dpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cpotrf.o: test/cpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/zpotrf.o: test/zpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuspotrf.o: test/spotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cudpotrf.o: test/dpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cucpotrf.o: test/cpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cuzpotrf.o: test/zpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuspotrf.o: test/spotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpudpotrf.o: test/dpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpucpotrf.o: test/cpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/cumultigpuzpotrf.o: test/zpotrf_ref.c lapack.h blas.h error.h | $(OBJDIR)/test

$(TEST_CUBINS): blas.h

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
%.c : %.sh
	$(SHELL) $(.SHELLFLAGS) $(<) > $(@)

# Object files
$(OBJDIR)/%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

# PTX files
$(PTXDIR)/%.ptx : %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -ptx $(<)

define ptx_template
$(PTXDIR)/%.compute_$(1).ptx: %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -arch compute_$(1) -o $$(@) -ptx $$(<)
endef
$(foreach code,10 11 12 13 20,$(eval $(call ptx_template,$(code))))

# Cubins
%.cubin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -arch=sm_13 -o $(@) -cubin $(<)

%.cubin: $(PTXDIR)/%.ptx
	$(PTXAS) $(PTXASFLAGS) -o $(@) $(<)

define cubin_template
%.sm_$(1).cubin: $(PTXDIR)/%.compute_$(1).ptx
	$(PTXAS) $(PTXASFLAGS) -arch sm_$(1) -o $$(@) $$(<)
endef
$(foreach arch,10 11 12 13 20,$(eval $(call cubin_template,$(arch))))

define cubin20_template
%.sm_$(1).cubin: $(PTXDIR)/%.compute_20.ptx
	$(PTXAS) $(PTXASFLAGS) -arch sm_$(1) -o $$(@) $$(<)
endef
$(foreach arch,21 22 23,$(eval $(call cubin20_template,$(arch))))

# FATBINs - optional ptx followed by cubins
# e.g. sgemm.fatbin: sgemm.ptx sgemm.sm_11.cubin sgemm.sm_13.cubin
%.fatbin:
	$(FATBINARY) $(FATBINARYFLAGS) -cuda --create $(@) $(if $(filter %.ptx,$(<)),--ptx $(filter %.ptx,$(<))) $(foreach cubin,$(filter %.cubin,$(^)),--image profile=$(lastword $(subst ., ,$(basename $(cubin)))),file=$(cubin))

# Dynamic loading of FATBINs
# Converts the fatbins to linkable object files with the symbols _path_to_fatbin
# and _path_to_fatbin_size
$(OBJDIR)/%.o : $(FATBINDIR)/%.fatbin
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_start=_$(subst -,_,$(subst .,_,$(notdir $(<)))) --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_size=_$(subst -,_,$(subst .,_,$(notdir $(<))))_size --strip-symbol _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_end $(<) $(@)
