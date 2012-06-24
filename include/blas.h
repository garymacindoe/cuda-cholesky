#ifndef BLAS_H
#define BLAS_H

#include <stddef.h>
#include <stdbool.h>
#include <complex.h>
#include <cuda.h>
#include <cublas_v2.h>

// nvcc uses __restrict__ instead of C99's restrict keyword
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
 *          cu: Hand coded to run on the GPU (or hybrid CPU/GPU) using arguments in graphics memory
 *  cuMultiGPU: Hand coded to run on multiple GPUs (or hybrid CPU/GPUs) using arguments in system memory
 */

/** CUBLAS Helper macros */
#define cublasUplo(uplo) (((uplo) == CBlasUpper) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER)
#define cublasDiag(diag) (((diag) == CBlasNonUnit) ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT)
#define cublasSide(side) (((side) == CBlasLeft) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT)
#define cublasTrans(trans)(((trans) == CBlasNoTrans) ? CUBLAS_OP_N : ((trans == CBlasTrans) ? CUBLAS_OP_T : CUBLAS_OP_C))

/** Error function */
typedef void (*xerbla_t)(const char *, long);
extern xerbla_t xerbla;
#define XERBLA(info) \
  do { \
    if (xerbla != NULL) \
      xerbla(__func__, info); \
  } while (false)

/** My CPU implementations */
void ssyrk(CBlasUplo, CBlasTranspose, size_t, size_t,  float, const  float * restrict, size_t,  float,  float * restrict, size_t);
void dsyrk(CBlasUplo, CBlasTranspose, size_t, size_t, double, const double * restrict, size_t, double, double * restrict, size_t);

void cherk(CBlasUplo, CBlasTranspose, size_t, size_t,  float, const  float complex * restrict, size_t,  float,  float complex * restrict, size_t);
void zherk(CBlasUplo, CBlasTranspose, size_t, size_t, double, const double complex * restrict, size_t, double, double complex * restrict, size_t);

void sgemm(CBlasTranspose, CBlasTranspose, size_t, size_t, size_t,  float, const  float * restrict, size_t, const  float * restrict, size_t,  float,  float * restrict, size_t);
void dgemm(CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, double, const double * restrict, size_t, const double * restrict, size_t, double, double * restrict, size_t);
void cgemm(CBlasTranspose, CBlasTranspose, size_t, size_t, size_t,  float complex, const  float complex * restrict, size_t, const  float complex * restrict, size_t,  float complex,  float complex * restrict, size_t);
void zgemm(CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, double complex, const double complex * restrict, size_t, const double complex * restrict, size_t, double complex, double complex * restrict, size_t);

void strmm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float, const  float * restrict, size_t,  float * restrict, size_t);
void dtrmm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double, const double * restrict, size_t, double * restrict, size_t);
void ctrmm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float complex, const  float complex * restrict, size_t,  float complex * restrict, size_t);
void ztrmm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double complex, const double complex * restrict, size_t, double complex * restrict, size_t);

void strsm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float, const  float * restrict, size_t,  float * restrict, size_t);
void dtrsm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double, const double * restrict, size_t, double * restrict, size_t);
void ctrsm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float complex, const  float complex * restrict, size_t,  float complex * restrict, size_t);
void ztrsm(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double complex, const double complex * restrict, size_t, double complex * restrict, size_t);

/** My GPU/Hybrid implementations */
CUresult cuSsyrk(CUmodule, CBlasUplo, CBlasTranspose, size_t, size_t,  float, CUdeviceptr, size_t,  float, CUdeviceptr, size_t, CUstream);
CUresult cuDsyrk(CUmodule, CBlasUplo, CBlasTranspose, size_t, size_t, double, CUdeviceptr, size_t, double, CUdeviceptr, size_t, CUstream);

CUresult cuCherk(CUmodule, CBlasUplo, CBlasTranspose, size_t, size_t,  float, CUdeviceptr, size_t,  float, CUdeviceptr, size_t, CUstream);
CUresult cuZherk(CUmodule, CBlasUplo, CBlasTranspose, size_t, size_t, double, CUdeviceptr, size_t, double, CUdeviceptr, size_t, CUstream);

CUresult cuSgemm(CUmodule, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t,  float, CUdeviceptr, size_t, CUdeviceptr, size_t,  float, CUdeviceptr, size_t, CUstream);
CUresult cuDgemm(CUmodule, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, double, CUdeviceptr, size_t, CUdeviceptr, size_t, double, CUdeviceptr, size_t, CUstream);
CUresult cuCgemm(CUmodule, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t,  float complex, CUdeviceptr, size_t, CUdeviceptr, size_t,  float complex, CUdeviceptr, size_t, CUstream);
CUresult cuZgemm(CUmodule, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, double complex, CUdeviceptr, size_t, CUdeviceptr, size_t, double complex, CUdeviceptr, size_t, CUstream);

CUresult cuStrmm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);
CUresult cuDtrmm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);
CUresult cuCtrmm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float complex, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);
CUresult cuZtrmm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double complex, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);

CUresult cuStrsm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);
CUresult cuDtrsm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);
CUresult cuCtrsm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float complex, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);
CUresult cuZtrsm(CUmodule, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double complex, CUdeviceptr, size_t, CUdeviceptr, size_t, CUstream);

/** My MultiGPU/Hybrid implementations */
CUresult cuMultiGPUSsyrk(CUcontext *, unsigned int, CBlasUplo, CBlasTranspose, size_t, size_t,  float, const  float * restrict, size_t,  float,  float * restrict, size_t);
CUresult cuMultiGPUDsyrk(CUcontext *, unsigned int, CBlasUplo, CBlasTranspose, size_t, size_t, double, const double * restrict, size_t, double, double * restrict, size_t);

CUresult cuMultiGPUCherk(CUcontext *, unsigned int, CBlasUplo, CBlasTranspose, size_t, size_t,  float, const  float complex * restrict, size_t,  float,  float complex * restrict, size_t);
CUresult cuMultiGPUZherk(CUcontext *, unsigned int, CBlasUplo, CBlasTranspose, size_t, size_t, double, const double complex * restrict, size_t, double, double complex * restrict, size_t);

CUresult cuMultiGPUSgemm(CUcontext *, unsigned int, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t,  float, const  float * restrict, size_t, const  float * restrict, size_t,  float,  float * restrict, size_t);
CUresult cuMultiGPUDgemm(CUcontext *, unsigned int, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, double, const double * restrict, size_t, const double * restrict, size_t, double, double * restrict, size_t);
CUresult cuMultiGPUCgemm(CUcontext *, unsigned int, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t,  float complex, const  float complex * restrict, size_t, const  float complex * restrict, size_t,  float complex,  float complex * restrict, size_t);
CUresult cuMultiGPUZgemm(CUcontext *, unsigned int, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, double complex, const double complex * restrict, size_t, const double complex * restrict, size_t, double complex, double complex * restrict, size_t);

CUresult cuMultiGPUStrmm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float, const  float * restrict, size_t,  float * restrict, size_t);
CUresult cuMultiGPUDtrmm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double, const double * restrict, size_t, double * restrict, size_t);
CUresult cuMultiGPUCtrmm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float complex, const  float complex * restrict, size_t,  float complex * restrict, size_t);
CUresult cuMultiGPUZtrmm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double complex, const double complex * restrict, size_t, double complex * restrict, size_t);

CUresult cuMultiGPUStrsm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float, const  float * restrict, size_t,  float * restrict, size_t);
CUresult cuMultiGPUDtrsm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double, const double * restrict, size_t, double * restrict, size_t);
CUresult cuMultiGPUCtrsm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t,  float complex, const  float complex * restrict, size_t,  float complex * restrict, size_t);
CUresult cuMultiGPUZtrsm(CUcontext *, unsigned int, CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, double complex, const double complex * restrict, size_t, double complex * restrict, size_t);

#ifdef __CUDACC__
#undef restrict
#endif

#ifdef __cplusplus
}
#endif

#endif
