#include "matrix.h"
#include "align.h"
#include "error.h"
#include "module.h"
#include <string.h>

static inline unsigned int max(unsigned int a, unsigned int b) {
  return (a > b) ? a : b;
}

#define FLOAT_T
#include "templates_on.h"
#include "matrix_source.c"
#include "templates_off.h"
#undef FLOAT_T

#define DOUBLE_T
#include "templates_on.h"
#include "matrix_source.c"
#include "templates_off.h"
#undef DOUBLE_T

#define FLOAT_COMPLEX_T
#include "templates_on.h"
#include "matrix_source.c"
#include "templates_off.h"
#undef FLOAT_COMPLEX_T

#define DOUBLE_COMPLEX_T
#include "templates_on.h"
#include "matrix_source.c"
#include "templates_off.h"
#undef DOUBLE_COMPLEX_T
