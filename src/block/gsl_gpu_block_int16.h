#ifndef __GSL_GPU_BLOCK_INT16_H__
#define __GSL_GPU_BLOCK_INT16_H__

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
} gsl_gpu_block_int16;

gsl_gpu_block_int16 * gsl_gpu_block_int16_alloc(const size_t, const size_t);
gsl_gpu_block_int16 * gsl_gpu_block_int16_calloc(const size_t, const size_t);
void gsl_gpu_block_int16_free(gsl_gpu_block_int16 *);

size_t gsl_gpu_block_int16_width(const gsl_gpu_block_int16 *);
size_t gsl_gpu_block_int16_height(const gsl_gpu_block_int16 *);
size_t gsl_gpu_block_int16_pitch(const gsl_gpu_block_int16 *);
CUdeviceptr gsl_gpu_block_int16_data(const gsl_gpu_block_int16 *);
CUcontext gsl_gpu_block_int16_context(const gsl_gpu_block_int16 *);

#ifdef __cplusplus
}
#endif

#endif
