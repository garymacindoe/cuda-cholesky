#ifndef __GSL_GPU_BLOCK_INT64_H__
#define __GSL_GPU_BLOCK_INT64_H__

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
} gsl_gpu_block_int64;

gsl_gpu_block_int64 * gsl_gpu_block_int64_alloc(const size_t, const size_t);
gsl_gpu_block_int64 * gsl_gpu_block_int64_calloc(const size_t, const size_t);
void gsl_gpu_block_int64_free(gsl_gpu_block_int64 *);

size_t gsl_gpu_block_int64_width(const gsl_gpu_block_int64 *);
size_t gsl_gpu_block_int64_height(const gsl_gpu_block_int64 *);
size_t gsl_gpu_block_int64_pitch(const gsl_gpu_block_int64 *);
CUdeviceptr gsl_gpu_block_int64_data(const gsl_gpu_block_int64 *);
CUcontext gsl_gpu_block_int64_context(const gsl_gpu_block_int64 *);

#ifdef __cplusplus
}
#endif

#endif
