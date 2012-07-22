#ifndef __GSL_GPU_BLOCK_UINT16_H__
#define __GSL_GPU_BLOCK_UINT16_H__

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
} gsl_gpu_block_uint16;

gsl_gpu_block_uint16 * gsl_gpu_block_uint16_alloc(const size_t, const size_t);
gsl_gpu_block_uint16 * gsl_gpu_block_uint16_calloc(const size_t, const size_t);
void gsl_gpu_block_uint16_free(gsl_gpu_block_uint16 *);

size_t gsl_gpu_block_uint16_width(const gsl_gpu_block_uint16 *);
size_t gsl_gpu_block_uint16_height(const gsl_gpu_block_uint16 *);
size_t gsl_gpu_block_uint16_pitch(const gsl_gpu_block_uint16 *);
CUdeviceptr gsl_gpu_block_uint16_data(const gsl_gpu_block_uint16 *);
CUcontext gsl_gpu_block_uint16_context(const gsl_gpu_block_uint16 *);

#ifdef __cplusplus
}
#endif

#endif
