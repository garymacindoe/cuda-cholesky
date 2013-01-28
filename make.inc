NVCC ?= nvcc
BIN2C ?= bin2c
RM ?= rm -f

%.a:
	$(AR) $(ARFLAGS) $(@) $(^)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

%.fatbin: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCFLAGS) -o $(@) -fatbin $(<)

%.fatbin.c: %.fatbin
	$(BIN2C) $(BIN2CFLAGS) $(<) > $(@)

%.d: %.c
	@set -e; $(RM) $@; \
	$(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$
