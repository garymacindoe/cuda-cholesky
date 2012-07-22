#include <gsl/gsl_gpu_errno.h>
#include <gsl/gsl_gpu_block.h>

#define GSL_GPU_FLOAT_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_FLOAT_T

#define GSL_GPU_DOUBLE_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_DOUBLE_T

#define GSL_GPU_COMPLEX_FLOAT_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_COMPLEX_FLOAT_T

#define GSL_GPU_COMPLEX_DOUBLE_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_COMPLEX_DOUBLE_T

#define GSL_GPU_INT8_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_INT8_T

#define GSL_GPU_INT16_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_INT16_T

#define GSL_GPU_INT32_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_INT32_T

#define GSL_GPU_INT64_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_INT64_T

#define GSL_GPU_UINT8_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_UINT8_T

#define GSL_GPU_UINT16_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_UINT16_T

#define GSL_GPU_UINT32_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_UINT32_T

#define GSL_GPU_UINT64_T
#include "templates_on.h"
#include "block_source.c"
#include "templates_off.h"
#undef GSL_GPU_UINT64_T
