#ifndef __GSL_GPU_BLOCK_UINT64_H__
#define __GSL_GPU_BLOCK_UINT64_H__

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
} gsl_gpu_block_uint64;

gsl_gpu_block_uint64 * gsl_gpu_block_uint64_alloc(const size_t, const size_t);
gsl_gpu_block_uint64 * gsl_gpu_block_uint64_calloc(const size_t, const size_t);
void gsl_gpu_block_uint64_free(gsl_gpu_block_uint64 *);

size_t gsl_gpu_block_uint64_width(const gsl_gpu_block_uint64 *);
size_t gsl_gpu_block_uint64_height(const gsl_gpu_block_uint64 *);
size_t gsl_gpu_block_uint64_pitch(const gsl_gpu_block_uint64 *);
CUdeviceptr gsl_gpu_block_uint64_data(const gsl_gpu_block_uint64 *);
CUcontext gsl_gpu_block_uint64_context(const gsl_gpu_block_uint64 *);

#ifdef __cplusplus
}
#endif

#endif
