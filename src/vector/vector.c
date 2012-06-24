#include "vector.h"
#include "error.h"
#include "multigpu.h"
#include <string.h>

static inline unsigned int max(unsigned int a, unsigned int b) {
  return (a > b) ? a : b;
}

#define FLOAT_T
#include "templates_on.h"
#include "vector_source.c"
#include "templates_off.h"
#undef FLOAT_T

#define DOUBLE_T
#include "templates_on.h"
#include "vector_source.c"
#include "templates_off.h"
#undef DOUBLE_T

#define UINT32_T
#include "templates_on.h"
#include "vector_source.c"
#include "templates_off.h"
#undef UINT32_T

#define UINT64_T
#include "templates_on.h"
#include "vector_source.c"
#include "templates_off.h"
#undef UINT64_T

#define FLOAT_COMPLEX_T
#include "templates_on.h"
#include "vector_source.c"
#include "templates_off.h"
#undef FLOAT_COMPLEX_T

#define DOUBLE_COMPLEX_T
#include "templates_on.h"
#include "vector_source.c"
#include "templates_off.h"
#undef DOUBLE_COMPLEX_T
