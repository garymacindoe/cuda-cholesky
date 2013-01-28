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
// Double precision Cholesky decomposition
void dpotrf(CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex Cholesky decomposition
void cpotrf(CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex Cholesky decomposition
void zpotrf(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

// In-place single precision triangular inverse from Cholesky decomposition
void strtri(CBlasUplo, CBlasDiag, size_t,  float * restrict, size_t, long * restrict);
// In-place double precision triangular inverse from Cholesky decomposition
void dtrtri(CBlasUplo, CBlasDiag, size_t, double * restrict, size_t, long * restrict);
// In-place single precision complex triangular inverse from Cholesky decomposition
void ctrtri(CBlasUplo, CBlasDiag, size_t,  float complex * restrict, size_t, long * restrict);
// In-place double precision complex triangular inverse from Cholesky decomposition
void ztrtri(CBlasUplo, CBlasDiag, size_t, double complex * restrict, size_t, long * restrict);

// Out of place single precision triangular inverse from Cholesky decomposition
void strtri2(CBlasUplo, CBlasDiag, size_t, const  float * restrict, size_t,  float * restrict, size_t, long * restrict);
// Out of place double precision triangular inverse from Cholesky decomposition
void dtrtri2(CBlasUplo, CBlasDiag, size_t, const double * restrict, size_t, double * restrict, size_t, long * restrict);
// Out of place single precision complex triangular inverse from Cholesky decomposition
void ctrtri2(CBlasUplo, CBlasDiag, size_t, const  float complex * restrict, size_t,  float complex * restrict, size_t, long * restrict);
// Out of place double precision complex triangular inverse from Cholesky decomposition
void ztrtri2(CBlasUplo, CBlasDiag, size_t, const double complex * restrict, size_t, double complex * restrict, size_t, long * restrict);

// Single precision inverse from Cholesky decomposition
void spotri(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision inverse from Cholesky decomposition
void dpotri(CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex inverse from Cholesky decomposition
void cpotri(CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex inverse from Cholesky decomposition
void zpotri(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

/** My Hybrid implementations */
// Single precision Cholesky decomposition
CUresult cuSpotrf(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision Cholesky decomposition
CUresult cuDpotrf(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Single precision complex Cholesky decomposition
CUresult cuCpotrf(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision complex Cholesky decomposition
CUresult cuZpotrf(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

// Single precision inverse from Cholesky decomposition
CUresult cuSpotri(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision inverse from Cholesky decomposition
CUresult cuDpotri(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Single precision complex inverse from Cholesky decomposition
CUresult cuCpotri(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision complex inverse from Cholesky decomposition
CUresult cuZpotri(CUblashandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

/** My CPU + multiGPU implementations */
// Single precision Cholesky decomposition
CUresult cuMultiGPUSpotrf(CUmultiGPUBlasHandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision Cholesky decomposition
CUresult cuMultiGPUDpotrf(CUmultiGPUBlasHandle, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex Cholesky decomposition
CUresult cuMultiGPUCpotrf(CUmultiGPUBlasHandle, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex Cholesky decomposition
CUresult cuMultiGPUZpotrf(CUmultiGPUBlasHandle, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

// Single precision inverse from Cholesky decomposition
CUresult cuMultiGPUSpotri(CUmultiGPUBlasHandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision inverse from Cholesky decomposition
CUresult cuMultiGPUDpotri(CUmultiGPUBlasHandle, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex inverse from Cholesky decomposition
CUresult cuMultiGPUCpotri(CUmultiGPUBlasHandle, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex inverse from Cholesky decomposition
CUresult cuMultiGPUZpotri(CUmultiGPUBlasHandle, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

#ifdef __CUDACC__
#undef restrict
#endif

#ifdef __cplusplus
}
#endif

#endif
