#ifndef __GSL_GPU_BLOCK_COMPLEX_DOUBLE_H__
#define __GSL_GPU_BLOCK_COMPLEX_DOUBLE_H__

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
} gsl_gpu_block_complex_double;

gsl_gpu_block_complex_double * gsl_gpu_block_complex_double_alloc(const size_t, const size_t);
gsl_gpu_block_complex_double * gsl_gpu_block_complex_double_calloc(const size_t, const size_t);
void gsl_gpu_block_complex_double_free(gsl_gpu_block_complex_double *);

size_t gsl_gpu_block_complex_double_width(const gsl_gpu_block_complex_double *);
size_t gsl_gpu_block_complex_double_height(const gsl_gpu_block_complex_double *);
size_t gsl_gpu_block_complex_double_pitch(const gsl_gpu_block_complex_double *);
CUdeviceptr gsl_gpu_block_complex_double_data(const gsl_gpu_block_complex_double *);
CUcontext gsl_gpu_block_complex_double_context(const gsl_gpu_block_complex_double *);

#ifdef __cplusplus
}
#endif

#endif
