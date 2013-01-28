#include "blas.h"

#include <stdio.h>

static void defaultXerbla(const char * function, long info) {
  fprintf(stderr, "On entry to %s parameter %ld had an invalid value\n", function, info);
}

xerbla_t xerbla = &defaultXerbla;
