#ifndef __GSL_GPU_BLOCK_COMPLEX_FLOAT_H__
#define __GSL_GPU_BLOCK_COMPLEX_FLOAT_H__

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
} gsl_gpu_block_complex_float;

gsl_gpu_block_complex_float * gsl_gpu_block_complex_float_alloc(const size_t, const size_t);
gsl_gpu_block_complex_float * gsl_gpu_block_complex_float_calloc(const size_t, const size_t);
void gsl_gpu_block_complex_float_free(gsl_gpu_block_complex_float *);

size_t gsl_gpu_block_complex_float_width(const gsl_gpu_block_complex_float *);
size_t gsl_gpu_block_complex_float_height(const gsl_gpu_block_complex_float *);
size_t gsl_gpu_block_complex_float_pitch(const gsl_gpu_block_complex_float *);
CUdeviceptr gsl_gpu_block_complex_float_data(const gsl_gpu_block_complex_float *);
CUcontext gsl_gpu_block_complex_float_context(const gsl_gpu_block_complex_float *);

#ifdef __cplusplus
}
#endif

#endif
