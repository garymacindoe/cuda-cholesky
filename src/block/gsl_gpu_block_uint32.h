#ifndef __GSL_GPU_BLOCK_UINT32_H__
#define __GSL_GPU_BLOCK_UINT32_H__

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
} gsl_gpu_block_uint32;

gsl_gpu_block_uint32 * gsl_gpu_block_uint32_alloc(const size_t, const size_t);
gsl_gpu_block_uint32 * gsl_gpu_block_uint32_calloc(const size_t, const size_t);
void gsl_gpu_block_uint32_free(gsl_gpu_block_uint32 *);

size_t gsl_gpu_block_uint32_width(const gsl_gpu_block_uint32 *);
size_t gsl_gpu_block_uint32_height(const gsl_gpu_block_uint32 *);
size_t gsl_gpu_block_uint32_pitch(const gsl_gpu_block_uint32 *);
CUdeviceptr gsl_gpu_block_uint32_data(const gsl_gpu_block_uint32 *);
CUcontext gsl_gpu_block_uint32_context(const gsl_gpu_block_uint32 *);

#ifdef __cplusplus
}
#endif

#endif
