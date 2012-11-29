CUDA_HOME = /opt/cuda
CUDA_LIBS = -lcuda -lcudart

NVCC = nvcc
PTXAS = ptxas
FATBINARY = fatbinary
NVCFLAGS = -O2 -use_fast_math
PTXASFLAGS = -O2
FATBINARYFLAGS = -cuda

CC = gcc
CFLAGS = -O2 -pipe -std=c99 -Wall
CPPFLAGS = -I../include -I$(CUDA_HOME)/include
NVCPPFLAGS = -I../include
LDFLAGS = -rdynamic -L$(CUDA_HOME)/lib64
LDLIBS = $(CUDA_LIBS) -lm -lrt -ldl

RM = rm -f
RMDIR = rm -rf
MKDIR = mkdir -p
VPATH = . ../include

OBJDIR = obj

TARGETS = bandwidth-test block-test context-test cumemalloc-test event flop-test flop-word gpu-bandwidth kernel-test kernel-test-rt module-test multigpu pitch timer

.PHONY: all debug $(foreach target,$(TARGETS),debug-$(target)) clean distclean
.NOTPARALLEL: $(OBJDIR)

define debug_template =
debug-$(1): CFLAGS += -g
debug-$(1): NVCFLAGS += -G -g
debug-$(1): PTXASFLAGS += -g
debug-$(1): FATBINARYFLAGS += -g
debug-$(1): $(1)
endef

all: $(TARGETS)

debug: $(foreach target,$(TARGETS),debug-$(target))

$(foreach target,$(TARGETS),$(eval $(call debug_template,$(target))))

clean:
	$(RMDIR) $(OBJDIR)

distclean: clean
	$(RM) $(TARGETS) flop-test.fatbin flop-test.compute_10.ptx flop-test.sm_11.cubin flop-test.sm_13.cubin kernel-test.ptx flop-test.fatbin flop-test.cubin flop-test.ptx

bandwidth-test: $(OBJDIR)/bandwidth-test.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

block-test: block-test.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) $(<)

context-test: $(OBJDIR)/context-test.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

cumemalloc-test: $(OBJDIR)/cumemalloc-test.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

event: $(OBJDIR)/event.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

flop-test: $(OBJDIR)/flop-test.o $(OBJDIR)/error.o flop-test.fatbin
	$(CC) $(LDFLAGS) -o $(@) $(filter %.o, $(^)) $(LOADLIBES) $(LDLIBS)

flop-word: $(OBJDIR)/flop-word.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(filter %.o, $(^)) $(LOADLIBES) $(LDLIBS)

gpu-bandwidth: $(OBJDIR)/gpu-bandwidth.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(filter %.o, $(^)) $(LOADLIBES) $(LDLIBS)

kernel-test: $(OBJDIR)/kernel-test.o $(OBJDIR)/error.o kernel-test.ptx
	$(CC) $(LDFLAGS) -o $(@) $(filter %.o, $(^)) $(LOADLIBES) $(LDLIBS)

kernel-test-rt: kernel-test.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) $(<)

module-test: $(OBJDIR)/module-test.o $(OBJDIR)/error.o $(OBJDIR)/flop-test.fatbin.o $(OBJDIR)/flop-test.cubin.o $(OBJDIR)/flop-test.ptx.o flop-test.fatbin flop-test.cubin flop-test.ptx
	$(CC) $(LDFLAGS) -o $(@) $(filter %.o, $(^)) $(LOADLIBES) $(LDLIBS)

multigpu: $(OBJDIR)/multigpu.o $(OBJDIR)/error.o kernel-test.ptx
	$(CC) $(LDFLAGS) -o $(@) $(filter %.o, $(^)) $(LOADLIBES) $(LDLIBS)

pitch: $(OBJDIR)/pitch.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

timer: $(OBJDIR)/timer.o $(OBJDIR)/error.o
	$(CC) $(LDFLAGS) -o $(@) $(^) $(LOADLIBES) $(LDLIBS)

$(OBJDIR)/error.o: ../src/error.c error.h | $(OBJDIR)
$(OBJDIR)/bandwidth-test.o: error.h | $(OBJDIR)
$(OBJDIR)/context-test.o: error.h | $(OBJDIR)
$(OBJDIR)/cumemalloc-test.o: error.h | $(OBJDIR)
$(OBJDIR)/event.o: error.h | $(OBJDIR)
$(OBJDIR)/flop-test.o: error.h | $(OBJDIR)
$(OBJDIR)/flop-word.o: error.h | $(OBJDIR)
$(OBJDIR)/gpu-bandwidth.o: error.h | $(OBJDIR)
$(OBJDIR)/module-test.o: error.h | $(OBJDIR)
$(OBJDIR)/multigpu.o: error.h | $(OBJDIR)
$(OBJDIR)/kernel-test.o: error.h | $(OBJDIR)
$(OBJDIR)/pitch.o: error.h | $(OBJDIR)
$(OBJDIR)/timer.o: CPPFLAGS += -D_POSIX_C_SOURCE=199309L
$(OBJDIR)/timer.o: error.h | $(OBJDIR)

flop-test.fatbin: flop-test.compute_10.ptx flop-test.sm_11.cubin flop-test.sm_13.cubin
module-test.fatbin: module-test.compute_10.ptx module-test.sm_11.cubin module-test.sm_13.cubin

$(OBJDIR)/%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

$(OBJDIR)/%.o : ../src/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

%.ptx : %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -ptx $(<)

%.cubin : %.ptx
	$(PTXAS) $(PTXASFLAGS) -o $(@) $(<)

ptx_template = %.compute_$(1).ptx: %.cu; $(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -arch compute_$(1) -o $$(@) -ptx $$(<)
cubin_template = %.sm_$(1).cubin: %.compute_$(1).ptx; $(PTXAS) $(PTXASFLAGS) -arch sm_$(1) -o $$(@) $$(<)
cubin20_template = %.sm_$(1).cubin: %.compute_20.ptx; $(PTXAS) $(PTXASFLAGS) -arch sm_$(1) -o $$(@) $$(<)

$(foreach code,10 11 12 13 20,$(eval $(call ptx_template,$(code))))
$(foreach arch,10 11 12 13 20,$(eval $(call cubin_template,$(arch))))
$(foreach arch,21 22 23,$(eval $(call cubin20_template,$(arch))))

%.fatbin:
	$(FATBINARY) $(FATBINARYFLAGS) --create $(@) $(if $(filter %.ptx,$(^)),--ptx $(filter %.ptx,$(^))) $(foreach cubin,$(filter %.cubin,$(^)),--image profile=$(lastword $(subst ., ,$(basename $(cubin)))),file=$(cubin))

$(OBJDIR)/%.o : %
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_start=_$(subst -,_,$(subst .,_,$(notdir $(<)))) --redefine-sym _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_size=_$(subst -,_,$(subst .,_,$(notdir $(<))))_size --strip-symbol _binary_$(subst .,_,$(subst /,_,$(subst -,_,$(<))))_end $(<) $(@)

%:
	$(MKDIR) $(@)
