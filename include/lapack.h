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
// Single precision Cholesky decomposition
void spotrf(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);

// In-place single precision triangular inverse from Cholesky decomposition
void strtri(CBlasUplo, CBlasDiag, size_t,  float * restrict, size_t, long * restrict);

// Single precision triangular square
void slauum(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);

// Single precision inverse from Cholesky decomposition
void spotri(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);

/** My Hybrid implementations */
// Single precision Cholesky decomposition
CUresult cuSpotrf(CUBLAShandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

// Single precision triangular inverse from Cholesky decomposition
CUresult cuStrtri(CUBLAShandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, long *);

// Single precision triangular square
CUresult cuSlauum(CUBLAShandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

// Single precision inverse from Cholesky decomposition
CUresult cuSpotri(CUBLAShandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

/** My CPU + multiGPU implementations */
// Single precision Cholesky decomposition
CUresult cuMultiGPUSpotrf(CUmultiGPUBLAShandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);

// Single precision triangular inverse from Cholesky decomposition
CUresult cuMultiGPUStrtri(CUmultiGPUBLAShandle, CBlasUplo, CBlasDiag, size_t,  float * restrict, size_t, long * restrict);

// Single precision triangular square
CUresult cuMultiGPUSlauum(CUmultiGPUBLAShandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);

// Single precision inverse from Cholesky decomposition
CUresult cuMultiGPUSpotri(CUmultiGPUBLAShandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);

/** Calculating log determinant - CPU and GPU only */
float slogdet(const float *, size_t, size_t);

CUresult cuSlogdet(CUdeviceptr, size_t, size_t,  float *, CUstream);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#undef restrict
#endif

#endif
