#ifndef __GSL_GPU_BLOCK_FLOAT_H__
#define __GSL_GPU_BLOCK_FLOAT_H__

#include <stddef.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t width;
  size_t height;
  size_t pitch;
  CUdeviceptr data;
  CUcontext context;
} gsl_gpu_block_float;

gsl_gpu_block_float * gsl_gpu_block_float_alloc(const size_t, const size_t);
gsl_gpu_block_float * gsl_gpu_block_float_calloc(const size_t, const size_t);
void gsl_gpu_block_float_free(gsl_gpu_block_float *);

size_t gsl_gpu_block_float_width(const gsl_gpu_block_float *);
size_t gsl_gpu_block_float_height(const gsl_gpu_block_float *);
size_t gsl_gpu_block_float_pitch(const gsl_gpu_block_float *);
CUdeviceptr gsl_gpu_block_float_data(const gsl_gpu_block_float *);
CUcontext gsl_gpu_block_float_context(const gsl_gpu_block_float *);

#ifdef __cplusplus
}
#endif

#endif
