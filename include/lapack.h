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

// Single precision triangular square
void slauum(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision triangular square
void dlauum(CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex triangular square
void clauum(CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex triangular square
void zlauum(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

// Single precision inverse from Cholesky decomposition
void spotri(CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision inverse from Cholesky decomposition
void dpotri(CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex inverse from Cholesky decomposition
void cpotri(CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex inverse from Cholesky decomposition
void zpotri(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

/** My Hybrid implementations */
typedef struct __culapackhandle_st * CULAPACKhandle;
CUresult cuLAPACKCreate(CULAPACKhandle *);
CUresult cuLAPACKDestroy(CULAPACKhandle);

// Single precision Cholesky decomposition
CUresult cuSpotrf(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision Cholesky decomposition
CUresult cuDpotrf(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Single precision complex Cholesky decomposition
CUresult cuCpotrf(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision complex Cholesky decomposition
CUresult cuZpotrf(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

// Single precision triangular inverse from Cholesky decomposition
CUresult cuStrtri(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, long *);
// Double precision triangular inverse from Cholesky decomposition
CUresult cuDtrtri(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, long *);
// Single precision complex triangular inverse from Cholesky decomposition
CUresult cuCtrtri(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, long *);
// Double precision complex triangular inverse from Cholesky decomposition
CUresult cuZtrtri(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, long *);

#define cuStrti2(handle, uplo, diag, n, A, lda, info) cuStrti22(handle, uplo, diag, n, A, lda, A, lda, info)
#define cuDtrti2(handle, uplo, diag, n, A, lda, info) cuDtrti22(handle, uplo, diag, n, A, lda, A, lda, info)
#define cuCtrti2(handle, uplo, diag, n, A, lda, info) cuCtrti22(handle, uplo, diag, n, A, lda, A, lda, info)
#define cuZtrti2(handle, uplo, diag, n, A, lda, info) cuZtrti22(handle, uplo, diag, n, A, lda, A, lda, info)

// Single precision triangular inverse from Cholesky decomposition
CUresult cuStrti22(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, CUdeviceptr, size_t, CUdeviceptr, CUstream);
// Double precision triangular inverse from Cholesky decomposition
CUresult cuDtrti22(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, CUdeviceptr, size_t, CUdeviceptr, CUstream);
// Single precision complex triangular inverse from Cholesky decomposition
CUresult cuCtrti22(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, CUdeviceptr, size_t, CUdeviceptr, CUstream);
// Double precision complex triangular inverse from Cholesky decomposition
CUresult cuZtrti22(CULAPACKhandle, CBlasUplo, CBlasDiag, size_t, CUdeviceptr, size_t, CUdeviceptr, size_t, CUdeviceptr, CUstream);

// Single precision triangular square
CUresult cuSlauum(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision triangular square
CUresult cuDlauum(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Single precision complex triangular square
CUresult cuClauum(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision complex triangular square
CUresult cuZlauum(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

// Single precision inverse from Cholesky decomposition
CUresult cuSpotri(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision inverse from Cholesky decomposition
CUresult cuDpotri(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Single precision complex inverse from Cholesky decomposition
CUresult cuCpotri(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);
// Double precision complex inverse from Cholesky decomposition
CUresult cuZpotri(CULAPACKhandle, CBlasUplo, size_t, CUdeviceptr, size_t, long *);

/** My CPU + multiGPU implementations */
// MultiGPU handle
typedef struct __cumultigpulapackhandle_st * CUmultiGPULAPACKhandle;
CUresult cuMultiGPULAPACKCreate(CUmultiGPULAPACKhandle *, CUmultiGPU);
CUresult cuMultiGPULAPACKDestroy(CUmultiGPULAPACKhandle);
CUresult cuMultiGPULAPACKSynchronize(CUmultiGPULAPACKhandle);

// Single precision Cholesky decomposition
CUresult cuMultiGPUSpotrf(CUmultiGPULAPACKhandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision Cholesky decomposition
CUresult cuMultiGPUDpotrf(CUmultiGPULAPACKhandle, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex Cholesky decomposition
CUresult cuMultiGPUCpotrf(CUmultiGPULAPACKhandle, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex Cholesky decomposition
CUresult cuMultiGPUZpotrf(CUmultiGPULAPACKhandle, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

// Single precision triangular inverse from Cholesky decomposition
CUresult cuMultiGPUStrtri(CUmultiGPULAPACKhandle, CBlasUplo, CBlasDiag, size_t,  float * restrict, size_t, long * restrict);
// Double precision triangular inverse from Cholesky decomposition
CUresult cuMultiGPUDtrtri(CUmultiGPULAPACKhandle, CBlasUplo, CBlasDiag, size_t, double * restrict, size_t, long * restrict);
// Single precision complex triangular inverse from Cholesky decomposition
CUresult cuMultiGPUCtrtri(CUmultiGPULAPACKhandle, CBlasUplo, CBlasDiag, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex triangular inverse from Cholesky decomposition
CUresult cuMultiGPUZtrtri(CUmultiGPULAPACKhandle, CBlasUplo, CBlasDiag, size_t, double complex * restrict, size_t, long * restrict);

// Single precision triangular square
CUresult cuMultiGPUSlauum(CUmultiGPULAPACKhandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision triangular square
CUresult cuMultiGPUDlauum(CUmultiGPULAPACKhandle, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex triangular square
CUresult cuMultiGPUClauum(CUmultiGPULAPACKhandle, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex triangular square
CUresult cuMultiGPUZlauum(CUmultiGPULAPACKhandle, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

// Single precision inverse from Cholesky decomposition
CUresult cuMultiGPUSpotri(CUmultiGPULAPACKhandle, CBlasUplo, size_t,  float * restrict, size_t, long * restrict);
// Double precision inverse from Cholesky decomposition
CUresult cuMultiGPUDpotri(CUmultiGPULAPACKhandle, CBlasUplo, size_t, double * restrict, size_t, long * restrict);
// Single precision complex inverse from Cholesky decomposition
CUresult cuMultiGPUCpotri(CUmultiGPULAPACKhandle, CBlasUplo, size_t,  float complex * restrict, size_t, long * restrict);
// Double precision complex inverse from Cholesky decomposition
CUresult cuMultiGPUZpotri(CUmultiGPULAPACKhandle, CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);

/** Calculating log determinant - CPU and GPU only*/
float slogdet(const float *, size_t, size_t);
double dlogdet(const double *, size_t, size_t);
float clogdet(const float complex *, size_t, size_t);
double zlogdet(const double complex *, size_t, size_t);

CUresult cuSlogdet(CULAPACKhandle, CUdeviceptr, size_t, size_t,  float *, CUstream);
CUresult cuDlogdet(CULAPACKhandle, CUdeviceptr, size_t, size_t, double *, CUstream);
CUresult cuClogdet(CULAPACKhandle, CUdeviceptr, size_t, size_t,  float *, CUstream);
CUresult cuZlogdet(CULAPACKhandle, CUdeviceptr, size_t, size_t, double *, CUstream);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#undef restrict
#endif

#endif
