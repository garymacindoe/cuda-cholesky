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
CUresult cuSpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, long *);
CUresult cuDpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, long *);
CUresult cuCpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, long *);
CUresult cuZpotrf(CBlasUplo, size_t, CUdeviceptr, size_t, long *);

CUresult cuSpotri(CBlasUplo, size_t, CUdeviceptr, size_t, long *);
CUresult cuDpotri(CBlasUplo, size_t, CUdeviceptr, size_t, long *);
CUresult cuCpotri(CBlasUplo, size_t, CUdeviceptr, size_t, long *);
CUresult cuZpotri(CBlasUplo, size_t, CUdeviceptr, size_t, long *);

/** My multi-GPU/Hybrid implementations */
CUresult cuMultiGPUSpotrf(CUcontext *, int, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
CUresult cuMultiGPUDpotrf(CUcontext *, int, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
CUresult cuMultiGPUCpotrf(CUcontext *, int, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
CUresult cuMultiGPUZpotrf(CUcontext *, int, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

CUresult cuMultiGPUSpotri(CUcontext *, int, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
CUresult cuMultiGPUDpotri(CUcontext *, int, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
CUresult cuMultiGPUCpotri(CUcontext *, int, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
CUresult cuMultiGPUZpotri(CUcontext *, int, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

#ifdef __CUDACC__
#undef restrict
#endif

#ifdef __cplusplus
}
#endif

#endif
