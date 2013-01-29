NVCC ?= nvcc
BIN2C ?= bin2c
RM ?= rm -f

%.a:
	$(AR) $(ARFLAGS) $(@) $(^)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $(@) -c $(<)

%.ptx : %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -ptx $(<)

%.cubin : %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -cubin $(<)

%.fatbin: %.cu
	$(NVCC) $(NVCPPFLAGS) $(NVCFLAGS) -o $(@) -fatbin $(<)

%.fatbin.c: %.fatbin
	$(BIN2C) $(BIN2CFLAGS) $(<) > $(@)
