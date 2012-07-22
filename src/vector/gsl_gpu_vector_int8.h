#ifndef __GSL_GPU_VECTOR_INT8_H__
#define __GSL_GPU_VECTOR_INT8_H__

#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t size;
  size_t stride;
  CUdeviceptr data;
  gsl_gpu_block_int8 * block;
  bool owner;
} gsl_gpu_vector_int8;

typedef struct {
  gsl_gpu_vector_int8 vector;
} _gsl_gpu_vector_int8_view;

typedef _gsl_gpu_vector_int8_view gsl_gpu_vector_int8_view;

typedef struct {
  gsl_gpu_vector_int8 vector;
} _gsl_gpu_vector_int8_const_view;

typedef const _gsl_gpu_vector_int8_const_view gsl_gpu_vector_int8_const_view;

#ifdef __cplusplus
}
#endif

#endif
