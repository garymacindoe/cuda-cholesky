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

ifeq ($(notdir $(CC)), icc)
  CFLAGS = -xHost -O2 -pipe -std=c99 -Wall -openmp
  LDFLAGS += -L$(INTEL_HOME)
  LDLIBS += -liomp5
else
  CFLAGS = -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -ftree-vectorize -fopenmp
  LDLIBS += -lgomp
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
             cumultigpustrsm cumultigpudtrsm cumultigpuctrsm cumultigpuztrsm
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

$(OBJDIR):
	$(MKDIR) $(@)

$(OBJDIR)/src: | $(OBJDIR)
	$(MKDIR) $(@)

$(OBJDIR)/src/error.o: error.h | $(OBJDIR)/src

$(OBJDIR)/src/blas: | $(OBJDIR)/src
	$(MKDIR) $(@)

$(OBJDIR)/src/blas/xerbla.o: blas.h | $(OBJDIR)/src/blas
BLAS_OBJS = $(foreach prec,s d,$(prec)gemm.o $(prec)syrk.o $(prec)trsm.o $(prec)trmm.o) \
            $(foreach prec,c z,$(prec)gemm.o $(prec)herk.o $(prec)trsm.o $(prec)trmm.o)
$(foreach blas,$(BLAS_OBS),$(eval $(OBJDIR)/src/blas/$(blas): blas.h error.h | $(OBJDIR)/src/blas))

$(OBJDIR)/test: | $(OBJDIR)
	$(MKDIR) $(@)
$(foreach prec,s d,$(foreach op,gemm syrk trsm,$(eval $(OBJDIR)/test/$(type)$(prec)$(op).o: test/$(prec)$(op)_ref.c blas.h error.h | $(OBJDIR)/test)))
$(foreach type,cu cumultigpu,$(foreach prec,s d,$(foreach op,gemm syrk trsm,$(eval $(OBJDIR)/test/$(type)$(prec)$(op).o: test/$(prec)$(op)_ref.c blas.h error.h | $(OBJDIR)/test))))
$(foreach prec,c z,$(foreach op,gemm herk trsm,$(eval $(OBJDIR)/test/$(type)$(prec)$(op).o: test/$(prec)$(op)_ref.c blas.h error.h | $(OBJDIR)/test)))
$(foreach type,cu cumultigpu,$(foreach prec,c z,$(foreach op,gemm herk trsm,$(eval $(OBJDIR)/test/$(type)$(prec)$(op).o: test/$(prec)$(op)_ref.c blas.h error.h | $(OBJDIR)/test))))

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
