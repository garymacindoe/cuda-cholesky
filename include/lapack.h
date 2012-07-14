#ifndef LAPACK_H
#define LAPACK_H

#include "blas.h"

// nvcc uses __restrict__ instead of C99's restrict keyword
// CUDA Programming Guide v4.1 Appendix B.2.4
#ifdef __CUDACC__
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** My CPU implementations */
void spotrf(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
void dpotrf(CBlasUplo, size_t, double * restrict, size_t, long * restrict);
void cpotrf(CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
void zpotrf(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

void spotri(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
void dpotri(CBlasUplo, size_t, double * restrict, size_t, long * restrict);
void cpotri(CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
void zpotri(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

/** My GPU/Hybrid implementations */
CUresult cuSpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);
CUresult cuDpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);
CUresult cuCpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);
CUresult cuZpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);

CUresult cuSpotri(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);
CUresult cuDpotri(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);
CUresult cuCpotri(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);
CUresult cuZpotri(CBlasUplo, size_t, CUdeviceptr, size_t, CUdeviceptr);

/** My multi-GPU/Hybrid implementations */
CUresult cuMultiGPUSpotrf(CUcontext *, unsigned int, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
CUresult cuMultiGPUDpotrf(CUcontext *, unsigned int, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
CUresult cuMultiGPUCpotrf(CUcontext *, unsigned int, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
CUresult cuMultiGPUZpotrf(CUcontext *, unsigned int, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

CUresult cuMultiGPUSpotri(CUcontext *, unsigned int, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
CUresult cuMultiGPUDpotri(CUcontext *, unsigned int, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
CUresult cuMultiGPUCpotri(CUcontext *, unsigned int, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
CUresult cuMultiGPUZpotri(CUcontext *, unsigned int, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

#ifdef __CUDACC__
#undef restrict
#endif

#ifdef __cplusplus
}
#endif

#endif
