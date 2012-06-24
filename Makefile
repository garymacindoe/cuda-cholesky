CUDA_HOME = /opt/cuda
CUDA_LIBS = -lcuda -lcublas
INTEL_HOME = /opt/intel/composerxe-2011.6.233
INTEL_LIBS = -liomp5 -lpthread
MKL_HOME = /opt/intel/composerxe-2011.5.220/mkl
MKL_LIBS = -lmkl_core -lmkl_intel_ilp64 -lmkl_intel_thread

ifeq ($(origin CC), default)
  CC = gcc
endif

NVCC = nvcc
PTXAS = ptxas
FATBINARY = fatbinary
NVCFLAGS = -O2 -use_fast_math -arch compute_13
FATBINARYFLAGS = -cuda
PTXASFLAGS = -O2 -arch sm_13

CPPFLAGS = -Iinclude -I$(CUDA_HOME)/include
NVCPPFLAGS = -Iinclude
LDFLAGS = -rdynamic -L$(CUDA_HOME)/lib64
LDLIBS = $(CUDA_LIBS) -lpthread -lm -lrt -ldl -lcunit -lcurses

ifdef ccbin
  NVCC += -ccbin=$(ccbin)
endif

ifeq ($(notdir $(CC)), gcc)
  CFLAGS = -march=native -O2 -pipe -std=c99 -Wall -Wextra -Wconversion -pedantic -ftree-vectorize -floop-block -floop-parallelize-all -ftree-parallelize-loops=8
  LDLIBS += -lgomp
else ifeq ($(notdir $(CC)), icc)
  CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -parallel
  LDLIBS += $(INTEL_LIBS)
else
  CFLAGS = -O2 -pipe -std=c99 -Wall
  LDLIBS += -lgomp
endif

ifdef CULA_HOME
  CPPFLAGS += -I$(CULA_HOME)/include -DHAS_CULA
  LDFLAGS += -L$(CULA_HOME)/lib64
  LDLIBS += -lcula
endif

ifeq ($(mkl), 1)
  CPPFLAGS += -I$(MKL_HOME)/include
  LDFLAGS += -L$(MKL_HOME)/lib/intel64 -L$(INTEL_HOME)/compiler/lib/intel64
  LDLIBS += $(MKL_LIBS) $(INTEL_LIBS)
else
  CPPFLAGS += $(shell pkg-config --cflags blas lapack)
  LDFLAGS += $(shell pkg-config --libs-only-L blas lapack)
  LDLIBS += $(shell pkg-config --libs-only-l blas lapack)
endif

RM = rm -f
RMDIR = rm -rf
MKDIR = mkdir

VPATH = . include

SRCDIR = src
OBJDIR = obj
PTXDIR = ptx
CUBINDIR = cubin
FATBINDIR = fatbin

TARGETS = thinner memset32
TEST_TARGETS = vector-test matrix-test blas1test blas2test blas3test lapack-test

.PHONY: all test debug debug-test $(foreach target,test $(TARGETS) $(TEST_TARGETS),debug-$(target)) clean distclean
.NOTPARALLEL: $(OBJDIR)/src $(OBJDIR)/test $(PTXDIR) $(PTXDIR)/src $(CUBINDIR) $(CUBINDIR)/src $(FATBINDIR) $(FATBINDIR)/src

define debug_template =
debug-$(1): CFLAGS += -O0 -g
# debug-$(1): NVCFLAGS += -G
# debug-$(1): FATBINARYFLAGS += -g
# debug-$(1): PTXASFLAGS = -O0 -g
debug-$(1): $(1)
endef

all: $(TARGETS)
test: $(TEST_TARGETS)

debug: CFLAGS += -O0 $(DEBUGFLAG)
debug: NVCFLAGS += -G
debug: FATBINARYFLAGS += -g
debug: PTXASFLAGS = -O0 -g
debug: $(TARGETS)

debug-test: CFLAGS += -O0 $(DEBUGFLAG)
debug-test: NVCFLAGS += -G
debug-test: FATBINARYFLAGS += -g
debug-test: PTXASFLAGS = -O0 -g
debug-test: $(TEST_TARGETS)

$(foreach target,test $(TARGETS) $(TEST_TARGETS),$(eval $(call debug_template,$(target))))

clean:
	$(RMDIR) $(OBJDIR) $(PTXDIR) $(CUBINDIR) $(FATBINDIR)

distclean: clean
	$(RM) $(TARGETS) $(TEST_TARGETS)

vector-test: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/vector/vector.o $(OBJDIR)/src/vector/vectorf.fatbin.o $(OBJDIR)/src/vector/vectord.fatbin.o $(OBJDIR)/src/vector/vectoru32.fatbin.o $(OBJDIR)/src/vector/vectoru64.fatbin.o $(OBJDIR)/src/vector/vectorCf.fatbin.o $(OBJDIR)/src/vector/vectorCd.fatbin.o $(OBJDIR)/test/vector.o $(OBJDIR)/test/main.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

matrix-test: $(OBJDIR)/src/error.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/vector/vector.o $(OBJDIR)/src/matrix/matrix.o $(OBJDIR)/src/matrix/matrixf.fatbin.o $(OBJDIR)/src/matrix/matrixd.fatbin.o $(OBJDIR)/src/matrix/matrixCf.fatbin.o $(OBJDIR)/src/matrix/matrixCd.fatbin.o $(OBJDIR)/test/matrix.o $(OBJDIR)/test/main.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

blas1test: $(OBJDIR)/src/error.o $(OBJDIR)/src/gaussian.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/vector/vector.o $(foreach prec,f d Cf Cd, $(OBJDIR)/src/vector/vector$(prec).fatbin.o) $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/blas/sscal.o $(OBJDIR)/src/blas/dscal.o $(OBJDIR)/src/blas/cscal.o $(OBJDIR)/src/blas/zscal.o $(OBJDIR)/src/blas/csscal.o $(OBJDIR)/src/blas/zdscal.o $(OBJDIR)/src/blas/sdot.o $(OBJDIR)/src/blas/ddot.o $(OBJDIR)/src/blas/cdotu.o $(OBJDIR)/src/blas/zdotu.o $(OBJDIR)/src/blas/cdotc.o $(OBJDIR)/src/blas/zdotc.o $(OBJDIR)/src/blas/sscal.fatbin.o $(OBJDIR)/src/blas/dscal.fatbin.o $(OBJDIR)/src/blas/cscal.fatbin.o $(OBJDIR)/src/blas/zscal.fatbin.o $(OBJDIR)/src/blas/csscal.fatbin.o $(OBJDIR)/src/blas/zdscal.fatbin.o $(OBJDIR)/src/blas/sdot.fatbin.o $(OBJDIR)/src/blas/ddot.fatbin.o $(OBJDIR)/src/blas/cdotu.fatbin.o $(OBJDIR)/src/blas/zdotu.fatbin.o $(OBJDIR)/src/blas/cdotc.fatbin.o $(OBJDIR)/src/blas/zdotc.fatbin.o $(OBJDIR)/test/blas1.o $(OBJDIR)/test/main.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

blas2test: $(OBJDIR)/src/error.o $(OBJDIR)/src/gaussian.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/vector/vector.o $(foreach prec,f d Cf Cd, $(OBJDIR)/src/vector/vector$(prec).fatbin.o) $(OBJDIR)/src/blas/xerbla.o $(OBJDIR)/src/matrix/matrix.o $(foreach prec,f d Cf Cd, $(OBJDIR)/src/matrix/matrix$(prec).fatbin.o) $(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)gemv.o) $(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)gemv.fatbin.o) $(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)trmv.o) $(OBJDIR)/test/blas2.o $(OBJDIR)/test/main.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

blas3test: $(OBJDIR)/src/error.o $(OBJDIR)/src/gaussian.o $(OBJDIR)/src/multigpu.o $(OBJDIR)/src/matrix/matrix.o $(foreach prec,f d Cf Cd, $(OBJDIR)/src/matrix/matrix$(prec).fatbin.o) $(OBJDIR)/src/blas/xerbla.o $(foreach op,gemm syrk trsm trmm,$(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)$(op).o) $(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)$(op).fatbin.o)) $(foreach op,herk,$(foreach prec,c z, $(OBJDIR)/src/blas/$(prec)$(op).o) $(foreach prec,c z, $(OBJDIR)/src/blas/$(prec)$(op).fatbin.o)) $(OBJDIR)/test/blas3.o $(OBJDIR)/test/main.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

lapack-test: $(OBJDIR)/src/error.o $(OBJDIR)/src/gaussian.o $(OBJDIR)/src/multigpu.o  $(OBJDIR)/src/matrix/matrix.o $(foreach prec,f d Cf Cd, $(OBJDIR)/src/matrix/matrix$(prec).fatbin.o) $(foreach op,gemm syrk trsm trmm,$(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)$(op).o) $(foreach prec,s d c z, $(OBJDIR)/src/blas/$(prec)$(op).fatbin.o))  $(foreach op,potrf potri,$(foreach prec,s d c z, $(OBJDIR)/src/lapack/$(prec)$(op).o) $(foreach prec,s d c z, $(OBJDIR)/src/lapack/$(prec)$(op).fatbin.o)) $(OBJDIR)/test/lapack.o $(OBJDIR)/test/main.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

memset32: $(OBJDIR)/src/error.o $(OBJDIR)/memset32.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

thinner: thinner.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $(@) $(<) $(LOADLIBES) $(LDLIBS)

$(OBJDIR)/src: | $(OBJDIR)
$(OBJDIR)/src/error.o: src/error.c error.h | $(OBJDIR)/src
$(OBJDIR)/src/gaussian.o: src/gaussian.c gaussian.h | $(OBJDIR)/src
$(OBJDIR)/src/multigpu.o: src/multigpu.c multigpu.h | $(OBJDIR)/src

$(OBJDIR)/src/vector: | $(OBJDIR)/src
$(OBJDIR)/src/vector/vector.o: src/vector/vector.c src/vector/vector_source.c vector.h templates_on.h templates_off.h multigpu.h error.h | $(OBJDIR)/src/vector
$(foreach prec,f d u32 u64 Cf Cd,$(eval $(OBJDIR)/src/vector/vector$(prec).fatbin.o: $(FATBINDIR)/src/vector/vector$(prec).fatbin | $(OBJDIR)/src/vector))

$(OBJDIR)/src/matrix: | $(OBJDIR)/src
$(OBJDIR)/src/matrix/matrix.o: src/matrix/matrix.c src/matrix/matrix_source.c matrix.h matrix_header.h vector.h vector_header.h templates_on.h templates_off.h module.h align.h error.h | $(OBJDIR)/src/matrix
$(foreach prec,f d Cf Cd,$(eval $(OBJDIR)/src/matrix/matrix$(prec).fatbin.o: $(FATBINDIR)/src/matrix/matrix$(prec).fatbin | $(OBJDIR)/src/matrix))

$(OBJDIR)/src/blas: | $(OBJDIR)/src
$(OBJDIR)/src/blas/xerbla.o: src/blas/xerbla.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/sgemm.o: src/blas/sgemm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ssyrk.o: src/blas/ssyrk.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strsm.o: src/blas/strsm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/strmm.o: src/blas/strmm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dgemm.o: src/blas/dgemm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dsyrk.o: src/blas/dsyrk.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrsm.o: src/blas/dtrsm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/dtrmm.o: src/blas/dtrmm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cgemm.o: src/blas/sgemm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/cherk.o: src/blas/cherk.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrsm.o: src/blas/ctrsm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ctrmm.o: src/blas/ctrmm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zgemm.o: src/blas/zgemm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/zherk.o: src/blas/zherk.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrsm.o: src/blas/ztrsm.c blas.h | $(OBJDIR)/src/blas
$(OBJDIR)/src/blas/ztrmm.o: src/blas/ztrmm.c blas.h | $(OBJDIR)/src/blas

$(OBJDIR)/test: | $(OBJDIR)
$(OBJDIR)/test/vector.o: test/vector.c test/vector_test.c vector.h vector_header.h templates_on.h templates_off.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/matrix.o: test/matrix.c test/matrix_test.c matrix.h matrix_header.h vector.h vector_header.h templates_on.h templates_off.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/blas1.o: test/blas1.c test/testscal.c test/testscalc.c test/testdot.c test/testdotu.c test/testdotc.c vector.h vector_header.h matrix.h matrix_header.h blas.h blas_types.h templates_on.h templates_off.h multigpu.h gaussian.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/blas2.o: test/blas2.c test/testgemv.c test/testtrmv.c vector.h vector_header.h matrix.h matrix_header.h blas.h blas_types.h templates_on.h templates_off.h multigpu.h gaussian.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/blas3.o: test/blas3.c test/testgemm.c test/testsyrk.c test/testtrmm.c test/testtrsm.c vector.h vector_header.h matrix.h matrix_header.h blas.h blas_types.h templates_on.h templates_off.h multigpu.h gaussian.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/lapack.o: test/lapack.c test/testlacgv.c test/testlauu2.c test/testlauum.c test/testpotf2.c test/testpotrf.c test/testpotri.c test/testtrti2.c test/testtrtri.c vector.h vector_header.h matrix.h matrix_header.h blas.h blas_types.h lapack.h templates_on.h templates_off.h multigpu.h gaussian.h error.h | $(OBJDIR)/test
$(OBJDIR)/test/main.o: test/main.c error.h | $(OBJDIR)/test

$(PTXDIR)/src: | $(PTXDIR)
$(PTXDIR)/src/vector: | $(PTXDIR)/src
$(foreach prec,f d u32 u64,$(eval $(PTXDIR)/src/vector/vector$(prec).ptx: src/vector/vector$(prec).cu src/vector/vector.cu | $(PTXDIR)/src/vector))
$(foreach prec,Cf Cd,$(eval $(PTXDIR)/src/vector/vector$(prec).ptx: src/vector/vector$(prec).cu src/vector/vector.cu src/cuComplex.cuh | $(PTXDIR)/src/vector))
$(foreach cc,10 11 12 13 20,$(foreach prec,f d u32 u64,$(eval $(PTXDIR)/src/vector/vector$(prec).compute_$(cc).ptx: src/vector/vector$(prec).cu src/vector/vector.cu | $(PTXDIR)/src/vector)))
$(foreach cc,10 11 12 13 20,$(foreach prec,Cf Cd,$(eval $(PTXDIR)/src/vector/vector$(prec).compute_$(cc).ptx: src/vector/vector$(prec).cu src/vector/vector.cu src/cuComplex.cuh | $(PTXDIR)/src/vector)))

$(PTXDIR)/src/matrix: | $(PTXDIR)/src
$(foreach prec,f d,$(eval $(PTXDIR)/src/matrix/matrix$(prec).ptx: src/matrix/matrix$(prec).cu src/matrix/matrix.cu | $(PTXDIR)/src/matrix))
$(foreach prec,Cf Cd,$(eval $(PTXDIR)/src/matrix/matrix$(prec).ptx: src/matrix/matrix$(prec).cu src/matrix/matrix.cu src/cuComplex.cuh | $(PTXDIR)/src/matrix))
$(foreach cc,10 11 12 13 20,$(foreach prec,f d,$(eval $(PTXDIR)/src/matrix/matrix$(prec).compute_$(cc).ptx: src/matrix/matrix$(prec).cu src/matrix/matrix.cu | $(PTXDIR)/src/matrix)))
$(foreach cc,10 11 12 13 20,$(foreach prec,Cf Cd,$(eval $(PTXDIR)/src/matrix/matrix$(prec).compute_$(cc).ptx: src/matrix/matrix$(prec).cu src/matrix/matrix.cu src/cuComplex.cuh | $(PTXDIR)/src/matrix)))

$(PTXDIR)/src/blas: | $(PTXDIR)/src
$(PTXDIR)/src/blas/sgemm.ptx: src/blas/sgemm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/ssyrk.ptx: src/blas/ssyrk.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/strsm.ptx: src/blas/strsm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/strmm.ptx: src/blas/strmm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/dgemm.ptx: src/blas/dgemm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/dsyrk.ptx: src/blas/dsyrk.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/dtrsm.ptx: src/blas/dtrsm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/dtrmm.ptx: src/blas/dtrmm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/cgemm.ptx: src/blas/sgemm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/cherk.ptx: src/blas/cherk.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/ctrsm.ptx: src/blas/ctrsm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/ctrmm.ptx: src/blas/ctrmm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/zgemm.ptx: src/blas/zgemm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/zherk.ptx: src/blas/zherk.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/ztrsm.ptx: src/blas/ztrsm.cu blas.h | $(PTXDIR)/src/blas
$(PTXDIR)/src/blas/ztrmm.ptx: src/blas/ztrmm.cu blas.h | $(PTXDIR)/src/blas

$(CUBINDIR)/src/blas: | $(CUBINDIR)/src
$(CUBINDIR)/src/blas/sgemm.cubin: $(PTXDIR)/src/blas/sgemm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/ssyrk.cubin: $(PTXDIR)/src/blas/ssyrk.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/strsm.cubin: $(PTXDIR)/src/blas/strsm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/strmm.cubin: $(PTXDIR)/src/blas/strmm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/dgemm.cubin: $(PTXDIR)/src/blas/dgemm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/dsyrk.cubin: $(PTXDIR)/src/blas/dsyrk.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/dtrsm.cubin: $(PTXDIR)/src/blas/dtrsm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/dtrmm.cubin: $(PTXDIR)/src/blas/dtrmm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/cgemm.cubin: $(PTXDIR)/src/blas/sgemm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/cherk.cubin: $(PTXDIR)/src/blas/cherk.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/ctrsm.cubin: $(PTXDIR)/src/blas/ctrsm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/ctrmm.cubin: $(PTXDIR)/src/blas/ctrmm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/zgemm.cubin: $(PTXDIR)/src/blas/zgemm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/zherk.cubin: $(PTXDIR)/src/blas/zherk.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/ztrsm.cubin: $(PTXDIR)/src/blas/ztrsm.ptx | $(CUBINDIR)/src/blas
$(CUBINDIR)/src/blas/ztrmm.cubin: $(PTXDIR)/src/blas/ztrmm.ptx | $(CUBINDIR)/src/blas

$(CUBINDIR)/src: | $(CUBINDIR)
$(CUBINDIR)/src/vector: | $(CUBINDIR)/src
$(foreach prec,f d Cf Cd u32 u64,$(eval $(CUBINDIR)/src/vector/vector$(prec).cubin: $(PTXDIR)/src/vector/vector$(prec).ptx | $(CUBINDIR)/src/vector))
$(foreach sm,10 11 12 13 20,$(foreach prec,f d Cf Cd u32 u64,$(eval $(CUBINDIR)/src/vector/vector$(prec).sm_$(sm).cubin: $(PTXDIR)/src/vector/vector$(prec).compute_$(sm).ptx | $(CUBINDIR)/src/vector)))
$(foreach sm,21 22 23,$(foreach prec,f d Cf Cd u32 u64,$(eval $(CUBINDIR)/src/vector/vector$(prec).sm_$(sm).cubin: $(PTXDIR)/src/vector/vector$(prec).compute_20.ptx | $(CUBINDIR)/src/vector)))

$(CUBINDIR)/src/matrix: | $(CUBINDIR)/src
$(foreach prec,f d Cf Cd u32 u64,$(eval $(CUBINDIR)/src/matrix/matrix$(prec).cubin: $(PTXDIR)/src/matrix/matrix$(prec).ptx | $(CUBINDIR)/src/matrix))
$(foreach sm,10 11 12 13 20,$(foreach prec,f d Cf Cd u32 u64,$(eval $(CUBINDIR)/src/matrix/matrix$(prec).sm_$(sm).cubin: $(PTXDIR)/src/matrix/matrix$(prec).compute_$(sm).ptx | $(CUBINDIR)/src/matrix)))
$(foreach sm,21 22 23,$(foreach prec,f d Cf Cd u32 u64,$(eval $(CUBINDIR)/src/matrix/matrix$(prec).sm_$(sm).cubin: $(PTXDIR)/src/matrix/matrix$(prec).compute_20.ptx | $(CUBINDIR)/src/matrix)))

$(FATBINDIR)/src: | $(FATBINDIR)
$(FATBINDIR)/src/vector: | $(FATBINDIR)/src
$(FATBINDIR)/src/vector/vectorf.fatbin: $(PTXDIR)/src/vector/vectorf.ptx $(CUBINDIR)/src/vector/vectorf.sm_11.cubin $(CUBINDIR)/src/vector/vectorf.sm_13.cubin | $(FATBINDIR)/src/vector
$(FATBINDIR)/src/vector/vectord.fatbin: $(PTXDIR)/src/vector/vectord.compute_13.ptx $(CUBINDIR)/src/vector/vectord.sm_13.cubin | $(FATBINDIR)/src/vector
$(FATBINDIR)/src/vector/vectorCf.fatbin: $(PTXDIR)/src/vector/vectorCf.ptx $(CUBINDIR)/src/vector/vectorCf.sm_11.cubin $(CUBINDIR)/src/vector/vectorCf.sm_13.cubin | $(FATBINDIR)/src/vector
$(FATBINDIR)/src/vector/vectorCd.fatbin: $(PTXDIR)/src/vector/vectorCd.compute_13.ptx $(CUBINDIR)/src/vector/vectorCd.sm_13.cubin | $(FATBINDIR)/src/vector
$(FATBINDIR)/src/vector/vectoru32.fatbin: $(PTXDIR)/src/vector/vectoru32.ptx $(CUBINDIR)/src/vector/vectoru32.sm_11.cubin $(CUBINDIR)/src/vector/vectoru32.sm_13.cubin | $(FATBINDIR)/src/vector
$(FATBINDIR)/src/vector/vectoru64.fatbin: $(PTXDIR)/src/vector/vectoru64.ptx $(CUBINDIR)/src/vector/vectoru64.sm_11.cubin $(CUBINDIR)/src/vector/vectoru64.sm_13.cubin | $(FATBINDIR)/src/vector

$(FATBINDIR)/src/matrix: | $(FATBINDIR)/src
$(FATBINDIR)/src/matrix/matrixf.fatbin: $(PTXDIR)/src/matrix/matrixf.ptx $(CUBINDIR)/src/matrix/matrixf.sm_11.cubin $(CUBINDIR)/src/matrix/matrixf.sm_13.cubin | $(FATBINDIR)/src/matrix
$(FATBINDIR)/src/matrix/matrixd.fatbin: $(PTXDIR)/src/matrix/matrixd.compute_13.ptx $(CUBINDIR)/src/matrix/matrixd.sm_13.cubin | $(FATBINDIR)/src/matrix
$(FATBINDIR)/src/matrix/matrixCf.fatbin: $(PTXDIR)/src/matrix/matrixCf.ptx $(CUBINDIR)/src/matrix/matrixCf.sm_11.cubin $(CUBINDIR)/src/matrix/matrixCf.sm_13.cubin | $(FATBINDIR)/src/matrix
$(FATBINDIR)/src/matrix/matrixCd.fatbin: $(PTXDIR)/src/matrix/matrixCd.compute_13.ptx $(CUBINDIR)/src/matrix/matrixCd.sm_13.cubin | $(FATBINDIR)/src/matrix

$(FATBINDIR)/src/blas: | $(FATBINDIR)/src
$(foreach prec,s c cs,$(eval $(FATBINDIR)/src/blas/$(prec)scal.fatbin: $(PTXDIR)/src/blas/$(prec)scal.ptx $(CUBINDIR)/src/blas/$(prec)scal.sm_11.cubin $(CUBINDIR)/src/blas/$(prec)scal.sm_13.cubin | $(FATBINDIR)/src/blas))
$(foreach prec,d z zd,$(eval $(FATBINDIR)/src/blas/$(prec)scal.fatbin: $(PTXDIR)/src/blas/$(prec)scal.compute_13.ptx $(CUBINDIR)/src/blas/$(prec)scal.sm_13.cubin | $(FATBINDIR)/src/blas))
$(FATBINDIR)/src/blas/sdot.fatbin: $(PTXDIR)/src/blas/sdot.ptx $(CUBINDIR)/src/blas/sdot.sm_11.cubin $(CUBINDIR)/src/blas/sdot.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/cdotu.fatbin: $(PTXDIR)/src/blas/cdotu.ptx $(CUBINDIR)/src/blas/cdotu.sm_11.cubin $(CUBINDIR)/src/blas/cdotu.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/cdotc.fatbin: $(PTXDIR)/src/blas/cdotc.ptx $(CUBINDIR)/src/blas/cdotc.sm_11.cubin $(CUBINDIR)/src/blas/cdotc.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/cherk.fatbin: $(PTXDIR)/src/blas/cherk.ptx $(CUBINDIR)/src/blas/cherk.sm_11.cubin $(CUBINDIR)/src/blas/cherk.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/ddot.fatbin: $(PTXDIR)/src/blas/ddot.compute_13.ptx $(CUBINDIR)/src/blas/ddot.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/zdotu.fatbin: $(PTXDIR)/src/blas/zdotu.compute_13.ptx $(CUBINDIR)/src/blas/zdotu.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/zdotc.fatbin: $(PTXDIR)/src/blas/zdotc.compute_13.ptx $(CUBINDIR)/src/blas/zdotc.sm_13.cubin | $(FATBINDIR)/src/blas
$(FATBINDIR)/src/blas/zherk.fatbin: $(PTXDIR)/src/blas/zherk.compute_13.ptx $(CUBINDIR)/src/blas/zherk.sm_13.cubin | $(FATBINDIR)/src/blas
$(foreach prec,s c,$(foreach op,gemv gemm syrk trsm trmm,$(eval $(FATBINDIR)/src/blas/$(prec)$(op).fatbin: $(PTXDIR)/src/blas/$(prec)$(op).ptx $(CUBINDIR)/src/blas/$(prec)$(op).sm_11.cubin $(CUBINDIR)/src/blas/$(prec)$(op).sm_13.cubin | $(FATBINDIR)/src/blas)))
$(foreach prec,d z,$(foreach op,gemv gemm syrk trsm trmm,$(eval $(FATBINDIR)/src/blas/$(prec)$(op).fatbin: $(PTXDIR)/src/blas/$(prec)$(op).compute_13.ptx $(CUBINDIR)/src/blas/$(prec)$(op).sm_13.cubin | $(FATBINDIR)/src/blas)))

%.c : %.sh
	$(SHELL) $(.SHELLFLAGS) $(<) > $(@)

$(OBJDIR)/%.o : %.c | $(OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

$(PTXDIR)/%.ptx : %.cu | $(PTXDIR)
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -ptx $(<)

$(CUBINDIR)/%.cubin: $(PTXDIR)/%.ptx | $(CUBINDIR)
	$(PTXAS) $(PTXASFLAGS) -o $(@) $(<)

ptx_template = $(PTXDIR)/%.compute_$(1).ptx: %.cu | $(PTXDIR); $(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -arch compute_$(1) -o $$(@) -ptx $$(<)
cubin_template = $(CUBINDIR)/%.sm_$(1).cubin: $(PTXDIR)/%.compute_$(1).ptx | $(CUBINDIR); $(PTXAS) $(PTXASFLAGS) -arch sm_$(1) -o $$(@) $$(<)
cubin20_template = $(CUBINDIR)/%.sm_$(1).cubin: $(PTXDIR)/%.compute_20.ptx | $(CUBINDIR); $(PTXAS) $(PTXASFLAGS) -arch sm_$(1) -o $$(@) $$(<)

$(foreach code,10 11 12 13 20,$(eval $(call ptx_template,$(code))))
$(foreach arch,10 11 12 13 20,$(eval $(call cubin_template,$(arch))))
$(foreach arch,21 22 23,$(eval $(call cubin20_template,$(arch))))

$(FATBINDIR)/%.fatbin: | $(FATBINDIR)
	$(FATBINARY) $(FATBINARYFLAGS) --create $(@) $(if $(filter %.ptx,$(<)),--ptx $(filter %.ptx,$(<))) $(foreach cubin,$(filter %.cubin,$(^)),--image profile=$(lastword $(subst ., ,$(basename $(cubin)))),file=$(cubin))

$(OBJDIR)/%.o : $(FATBINDIR)/% | $(OBJDIR)
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_start=_$(subst -,_,$(subst .,_,$(notdir $(<)))) --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_size=_$(subst -,_,$(subst .,_,$(notdir $(<))))_size --strip-symbol _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_end $(<) $(@)

%:
	$(MKDIR) $(@)
