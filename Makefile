.PHONY: all test clean distclean

all: libcumultigpu.a libcumultigpu_seq.a libblas.a liblapack.a

test: libcumultigpu.a libcumultigpu_seq.a libblas.a liblapack.a
	cd test && $(MAKE)

clean:
	cd blas && $(MAKE) clean
	cd lapack && $(MAKE) clean
	cd multigpu && $(MAKE) clean
	cd test && $(MAKE) clean

distclean: clean
	$(RM) libcumultigpu.a libcumultigpu_seq.a libblas.a liblapack.a

libblas.a:
	cd blas && $(MAKE) all

liblapack.a: libblas.a
	cd lapack && $(MAKE) all

libcumultigpu.a libcumultigpu_seq.a:
	cd multigpu && $(MAKE) ../$(@)
