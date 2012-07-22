#ifndef __GSL_GPU_BLOCK_UINT8_H__
#define __GSL_GPU_BLOCK_UINT8_H__

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
} gsl_gpu_block_uint8;

gsl_gpu_block_uint8 * gsl_gpu_block_uint8_alloc(const size_t, const size_t);
gsl_gpu_block_uint8 * gsl_gpu_block_uint8_calloc(const size_t, const size_t);
void gsl_gpu_block_uint8_free(gsl_gpu_block_uint8 *);

size_t gsl_gpu_block_uint8_width(const gsl_gpu_block_uint8 *);
size_t gsl_gpu_block_uint8_height(const gsl_gpu_block_uint8 *);
size_t gsl_gpu_block_uint8_pitch(const gsl_gpu_block_uint8 *);
CUdeviceptr gsl_gpu_block_uint8_data(const gsl_gpu_block_uint8 *);
CUcontext gsl_gpu_block_uint8_context(const gsl_gpu_block_uint8 *);

#ifdef __cplusplus
}
#endif

#endif
