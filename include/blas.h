#ifndef BLAS_H
#define BLAS_H

#include <stddef.h>
#include <stdbool.h>
#include <complex.h>
#include <cuda.h>

#include "cumultigpu.h"

// nvcc uses __restrict__ instead of C99's restrict keyword
// CUDA Programming Guide v4.1 Appendix B.2.4
#ifdef __CUDACC__
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * BLAS enums - castable to char for compatibility with Fortran BLAS/LAPACK.
 */
typedef enum { CBlasNoTrans = 'N', CBlasTrans = 'T', CBlasConjTrans = 'C' } CBlasTranspose;
typedef enum { CBlasLower = 'L', CBlasUpper = 'U' } CBlasUplo;
typedef enum { CBlasLeft = 'L', CBlasRight = 'R' } CBlasSide;
typedef enum { CBlasNonUnit = 'N', CBlasUnit = 'U' } CBlasDiag;

/**
 * Prefixes:
 *      <none>: Hand coded to run on the CPU using arguments in system memory
 *          cu: Hand coded to run on the GPU using arguments in graphics memory
 *  cuMultiGPU: Hand coded to run on the CPU and multiple GPUs using arguments in system memory
 */

/** Error function */
typedef void (*xerbla_t)(const char *, long);
extern xerbla_t xerbla;
#define XERBLA(info) \
  do { \
    if (xerbla != NULL) \
      xerbla(__func__, info); \
  } while (false)

/** My CPU implementations */
// Single precision rank-K update
void ssyrk(CBlasUplo, CBlasTranspose,
           size_t, size_t,
           float, const  float * restrict, size_t,
           float,  float * restrict, size_t);

// Single precision matrix-multiply
void sgemm(CBlasTranspose, CBlasTranspose,
           size_t, size_t, size_t,
           float, const float * restrict, size_t, const float * restrict, size_t,
           float, float * restrict, size_t);

// In-place single precision triangular matrix multiply
void strmm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
           size_t, size_t,
           float, const float * restrict, size_t,
           float * restrict, size_t);

// Out-of-place single precision triangular matrix multiply
void strmm2(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
            size_t, size_t,
            float, const float * restrict, size_t, const float * restrict, size_t,
            float * restrict, size_t);

// Single precision triangular solve
void strsm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
           size_t, size_t,
           float, const float * restrict, size_t,
           float * restrict, size_t);

/** My GPU implementations */
typedef struct __cublashandle_st * CUBLAShandle;
CUresult cuBLASCreate(CUBLAShandle *);
CUresult cuBLASDestroy(CUBLAShandle);

// Single precision rank-K update
CUresult cuSsyrk(CUBLAShandle, CBlasUplo, CBlasTranspose,
                 size_t, size_t,
                 float, CUdeviceptr, size_t,
                 float, CUdeviceptr, size_t, CUstream);

// In-place single precision matrix multiply
#define cuSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream) \
         cuSgemm2(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, C, ldc, stream)

// Out-of-place single precision matrix multiply
CUresult cuSgemm2(CUBLAShandle, CBlasTranspose, CBlasTranspose,
                  size_t, size_t, size_t,
                  float, CUdeviceptr, size_t, CUdeviceptr, size_t,
                  float, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);

// Single precision triangular matrix multiply
CUresult cuStrmm(CUBLAShandle, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
                 size_t, size_t,
                 float, CUdeviceptr, size_t, CUdeviceptr, size_t,
                 CUstream);

// Single precision triangular matrix multiply
CUresult cuStrmm2(CUBLAShandle, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
                  size_t, size_t,
                  float, CUdeviceptr, size_t, CUdeviceptr, size_t,
                  CUdeviceptr, size_t, CUstream);

// Single precision triangular matrix solve
CUresult cuStrsm(CUBLAShandle, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
                 size_t, size_t,
                 float, CUdeviceptr, size_t,
                 CUdeviceptr, size_t, CUstream);

/** My MultiGPU/Hybrid implementations */
// MultiGPU handle
typedef struct __cumultigpublashandle_st * CUmultiGPUBLAShandle;
CUresult cuMultiGPUBLASCreate(CUmultiGPUBLAShandle *, CUmultiGPU);
CUresult cuMultiGPUBLASDestroy(CUmultiGPUBLAShandle);
CUresult cuMultiGPUBLASSynchronize(CUmultiGPUBLAShandle);

// Single precision rank-K update
CUresult cuMultiGPUSsyrk(CUmultiGPUBLAShandle,
                         CBlasUplo, CBlasTranspose,
                         size_t, size_t,
                         float, const float * restrict, size_t,
                         float, float * restrict, size_t);

// Single precision matrix multiply
CUresult cuMultiGPUSgemm(CUmultiGPUBLAShandle,
                         CBlasTranspose, CBlasTranspose,
                         size_t, size_t, size_t,
                         float, const float * restrict, size_t, const float * restrict, size_t,
                         float, float * restrict, size_t);

// Single precision triangular matrix multiply
CUresult cuMultiGPUStrmm(CUmultiGPUBLAShandle,
                         CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
                         size_t, size_t,
                         float, const float * restrict, size_t,
                         float * restrict, size_t);

// Single precision triangular solve
CUresult cuMultiGPUStrsm(CUmultiGPUBLAShandle,
                         CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag,
                         size_t, size_t,
                         float, const float * restrict, size_t,
                         float * restrict, size_t);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#undef restrict
#endif

#endif
