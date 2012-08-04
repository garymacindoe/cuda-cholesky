#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <complex.h>

static inline size_t min(size_t a, size_t b) { return (a < b) ? a : b; }
static inline size_t max(size_t a, size_t b) { return (a > b) ? a : b; }
static inline unsigned int maxj(unsigned int a, unsigned int b) { return (a > b) ? a : b; }

static inline CUresult cuMemcpyHtoD2DAsync(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                                          const void * B, size_t ldb, size_t bi, size_t bj,
                                          size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_HOST, B, 0, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static inline CUresult cuMemcpyDtoH2DAsync(void * A, size_t lda, size_t ai, size_t aj,
                                          CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                                          size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_HOST, A, 0, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static const double zero = 0.0;
static const double one = 1.0;
static const double complex complex_one = 1.0 + 0.0 * I;

static inline void zpotf2(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < max(1, n))
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0) return;

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < j; i++) {
        double complex temp = A[j * lda + i];
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * conj(A[i * lda + k]);
        A[j * lda + i] = temp / A[i * lda + i];
      }

      double ajj = creal(A[j * lda + j]);
      for (size_t k = 0; k < j; k++)
        ajj -= A[j * lda + k] * conj(A[j * lda + k]);
      if (ajj <= zero || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j;
        return;
      }
      else
        A[j * lda + j] = sqrt(ajj);
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < j; k++) {
        double complex temp = conj(A[k * lda + j]);
        for (size_t i = j; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }

      double ajj = creal(A[j * lda + j]);
      if (ajj <= zero || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j;
        return;
      }
      ajj = sqrt(ajj);
      A[j * lda + j] = ajj;
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] /= ajj;
    }
  }
}

// #ifdef MKL_ILP64
//   extern void zpotrf_(const char *, const long *, double complex *, const long *, long *);
// #else
//   extern void zpotrf_(const char *, const int *, double complex *, const int *, int *);
// #endif
void zpotrf(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
// #ifdef MKL_ILP64
//   zpotrf_((const char *)&uplo, (const long *)&n, A, (const long *)&lda, info);
// #else
//   zpotrf_((const char *)&uplo, (const int *)&n, A, (const int *)&lda, info);
// #endif
//   return;
  if (lda < max(1, n))
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0) return;

  const size_t nb = 64;

  if (n < nb) {
    zpotf2(uplo, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      zherk(CBlasUpper, CBlasConjTrans, jb, j, -complex_one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      zpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        zgemm(CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda);
        ztrsm(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      zherk(CBlasLower, CBlasNoTrans, jb, j, -complex_one, &A[j], lda, one, &A[j * lda + j], lda);
      zpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        zgemm(CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda);
        ztrsm(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

static inline CUresult cuZpotf2(CUmodule module, CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, CUdeviceptr info, CUstream stream) {
  long hInfo = 0;
  CU_ERROR_CHECK(cuMemcpyHtoD(info, &hInfo, sizeof(long)));
  if (lda < max(1, n)) {
    hInfo = -4;
    CU_ERROR_CHECK(cuMemcpyHtoD(info, &hInfo, sizeof(long)));
  }
  if (hInfo != 0) {
    XERBLA(-hInfo);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  const unsigned int bx = (uplo == CBlasUpper) ?  8 : 16;
  const unsigned int by = (uplo == CBlasUpper) ?  8 :  4;

  char name[50];
  snprintf(name, 50, "_Z6zpotf2IL9CBlasUplo%dELj%uELj%uEEviP7double2iPi", uplo, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &A, &lda, &info };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuZpotrf(CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, CUdeviceptr info) {
  long hInfo = 0;
  CU_ERROR_CHECK(cuMemcpyHtoD(info, &hInfo, sizeof(long)));
  if (lda < max(1, n)) {
    hInfo = -4;
    CU_ERROR_CHECK(cuMemcpyHtoD(info, &hInfo, sizeof(long)));
  }
  if (hInfo != 0) {
    XERBLA(-hInfo);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  double complex * B;
  size_t ldb;
  CUstream stream0, stream1;
  CUmodule zpotf2, zherk, zgemm, ztrsm;

  const size_t nb = 1024;

  if (n < nb) {
    CU_ERROR_CHECK(cuModuleLoad(&zpotf2,  "zpotrf.cubin"));
    CU_ERROR_CHECK(cuZpotf2(zpotf2, uplo, n, A, lda, info, NULL));
    CU_ERROR_CHECK(cuModuleUnload(zpotf2));
    return CUDA_SUCCESS;
  }

  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 3u) & ~3u) * nb * sizeof(double complex)));

  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  CU_ERROR_CHECK(cuModuleLoad(&zherk, "zherk.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&zpotf2, "zpotrf.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&zgemm, "zgemm.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&ztrsm, "ztrsm.cubin"));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuZherk(zherk, CBlasUpper, CBlasConjTrans, jb, j, -complex_one, A + j * lda * sizeof(double complex), lda, complex_one, A + (j * lda + j) * sizeof(double complex), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuZgemm(zgemm, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, A + j * lda * sizeof(double complex), lda, A + (j + jb) * lda * sizeof(double complex), lda, complex_one, A + ((j + jb) * lda + j) * sizeof(double complex), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      zpotrf(CBlasUpper, jb, B, ldb, &hInfo);
      if (hInfo != 0) {
        hInfo += (long)j;
        CU_ERROR_CHECK(cuMemcpyHtoD(info, &hInfo, sizeof(long)));
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
//       CU_ERROR_CHECK(cuZtrsm(ctrsm, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, complex_one, A + (j * lda + j) * sizeof(double complex), lda, A + ((j + jb) * lda + j) * sizeof(double complex), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuZherk(zherk, CBlasLower, CBlasNoTrans, jb, j, -complex_one, A + j * sizeof(double complex), lda, complex_one, A + (j * lda + j) * sizeof(double complex), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuZgemm(zgemm, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, A + (j + jb) * sizeof(double complex), lda, A + j * sizeof(double complex), lda, one, A + (j * lda + j + jb) * sizeof(double complex), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      zpotrf(CBlasUpper, jb, B, ldb, &hInfo);
      if (hInfo != 0) {
        hInfo += (long)j;
        CU_ERROR_CHECK(cuMemcpyHtoD(info, &hInfo, sizeof(long)));
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
//       CU_ERROR_CHECK(cuZtrsm(ctrsm, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, complex_one, A + (j * lda + j) * sizeof(double complex), lda, A + (j * lda + j + jb) * sizeof(double complex), lda, stream0));
    }
  }

  CU_ERROR_CHECK(cuModuleUnload(zherk));
  CU_ERROR_CHECK(cuModuleUnload(zpotf2));
  CU_ERROR_CHECK(cuModuleUnload(zgemm));
  CU_ERROR_CHECK(cuModuleUnload(ztrsm));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  CU_ERROR_CHECK(cuMemFreeHost(B));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZpotrf(CUcontext * contexts, int deviceCount, CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < max(1, n))
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  const size_t nb = 1024;

  if (n < nb) {
    zpotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUZherk(contexts, deviceCount, CBlasUpper, CBlasConjTrans, jb, j, -complex_one, &A[j * lda], lda, complex_one, &A[j * lda + j], lda));
      zpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(contexts, deviceCount, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, &A[j * lda], lda, &A[(j + jb) * lda], lda, complex_one, &A[(j + jb) * lda + j], lda));
//         CU_ERROR_CHECK(cuMultiGPUZtrsm(contexts, deviceCount, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, complex_one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUZherk(contexts, deviceCount, CBlasLower, CBlasNoTrans, jb, j, -complex_one, &A[j], lda, one, &A[j * lda + j], lda));
      zpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(contexts, deviceCount, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, &A[j + jb], lda, &A[j], lda, complex_one, &A[j * lda + j + jb], lda));
//         CU_ERROR_CHECK(cuMultiGPUZtrsm(contexts, deviceCount, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, complex_one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}

#if 1
// gcc -I../../include -I/opt/cuda/include -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -c zpotrf.c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

static void zpotrf_ref(CBlasUplo, size_t, double complex * restrict, size_t, long * restrict);
static void * malloc2D(size_t, size_t, size_t *, size_t);
static void rand2D(size_t, size_t, double complex *, size_t);
static void fprintf2D(FILE *, const char *, size_t, size_t, const double complex *, size_t);
#ifdef MKL_ILP64
extern void zgemm_(const char *, const char *, const long *, const long *, const long *, const double complex *, const double complex *, const long *, const double complex *, const long *, const double complex *, double complex *, const long *);
#else
extern void zgemm_(const char *, const char *, const int *, const int *, const int *, const double complex *, const double complex *, const int *, const double complex *, const int *, const double complex *, double complex *, const int *);
#endif
#ifdef GPU
static CUresult cuMemcpyHtoD2D(CUdeviceptr, size_t, size_t, size_t, const void *, size_t, size_t, size_t, size_t, size_t, size_t);
static CUresult cuMemcpyDtoH2D(void *, size_t, size_t, size_t, CUdeviceptr, size_t, size_t, size_t, size_t, size_t, size_t);
#endif

const double complex complex_zero = 0.0 + 0.0 * I;

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  size_t n;
#ifdef GPU
  int d;

  if (argc < 3 || argc > 4) {
    fprintf(stderr, "Usage: %s <uplo> <n> [device]\nwhere:\n  uplo is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n  n                  is the size of the matrix\n  device             is the ordinal of the GPU to use (default 0)\n", argv[0]);
    return 1;
  }
#else

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <uplo> <n>\nwhere:\n  uplo is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n  n                  is the size of the matrix\n", argv[0]);
    return 1;
  }
#endif

  char u;
  if (sscanf(argv[1], "%c", &u) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (u) {
    case 'U': case 'u': uplo = CBlasUpper; break;
    case 'L': case 'l': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 1;
  }

  if (sscanf(argv[2], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[2]);
    return 2;
  }
#ifdef GPU
  if (argc == 4) {
    if (sscanf(argv[3], "%d", &d) != 1) {
      fprintf(stderr, "Unable to parse number from '%s'\n", argv[3]);
      return 3;
    }
  }
  else
    d = 0;
#endif
  srand(0);

  double complex * A, * C, * refA;
  size_t lda, ldc, k = 5 * n;
  long info, rInfo;
#ifdef GPU
  CUdeviceptr dA, dInfo;
  size_t dlda;
#endif

#if defined(GPU) || defined(MULTIGPU)
  CU_ERROR_CHECK(cuInit(0));

#ifdef GPU
  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_BLOCKING_SYNC, device));
#else
  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUcontext contexts[deviceCount];
  for (int i = 0; i < deviceCount; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));
    CU_ERROR_CHECK(cuCtxCreate(&contexts[i], CU_CTX_BLOCKING_SYNC, device));
  }
#endif
#endif

  if ((C = malloc2D(n, k, &ldc, sizeof(double complex))) == NULL) {
    fprintf(stderr, "Unable to allocate C\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  rand2D(n, k, C, ldc);

  if ((A = malloc2D(n, n, &lda, sizeof(double complex))) == NULL) {
    fprintf(stderr, "Unable to allocate A\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  if ((refA = malloc2D(n, n, &lda, sizeof(double complex))) == NULL) {
    fprintf(stderr, "Unable to allocate refA\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

#ifdef MKL_ILP64
  zgemm_("No Transpose", "Conjugate Transpose", (const long *)&n, (const long *)&n, (const long *)&k, &complex_one, C, (const long *)&ldc, C, (const long *)&ldc, &complex_zero, A, (const long *)&lda);
#else
  zgemm_("No Transpose", "Conjugate Transpose", (const int *)&n, (const int *)&n, (const int *)&k, &complex_one, C, (const int *)&ldc, C, (const int *)&ldc, &complex_zero, A, (const int *)&lda);
#endif

  free(C);

#ifdef GPU
  if (n > 0) {
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, n * sizeof(double complex), n, sizeof(double complex)));
    dlda /= sizeof(double complex);
  }
  else {
    dA = 0;
    dlda = 1;
  }
  CU_ERROR_CHECK(cuMemcpyHtoD2D(dA, dlda, 0, 0, A, lda, 0, 0, n, n, sizeof(double complex)));
  CU_ERROR_CHECK(cuMemAlloc(&dInfo, sizeof(long)));
#endif

  if (n <= 8)
    fprintf2D(stdout, "A", n, n, A, lda);

  for (size_t j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(double complex));

  zpotrf_ref(uplo, n, refA, lda, &rInfo);
#ifdef GPU
  CU_ERROR_CHECK(cuZpotrf(uplo, n, dA, dlda, dInfo));

  CU_ERROR_CHECK(cuMemcpyDtoH2D(A, lda, 0, 0, dA, dlda, 0, 0, n, n, sizeof(double complex)));
  CU_ERROR_CHECK(cuMemcpyDtoH(&info, dInfo, sizeof(long)));
#else
#ifdef MULTIGPU
  CU_ERROR_CHECK(cuMultiGPUZpotrf(contexts, deviceCount, uplo, n, A, lda, &info));
#else
  zpotrf(uplo, n, A, lda, &info);
#endif
#endif

  fprintf(stdout, "Reference info: %ld\n", rInfo);
  if (n <= 8)
    fprintf2D(stdout, "Reference ZPOTRF", n, n, refA, lda);
  fprintf(stdout, "Info: %ld\n", info);
  if (n <= 8)
    fprintf2D(stdout, "ZPOTRF", n, n, A, lda);

  bool passed = (info == rInfo);
  double rdiff = zero, idiff = zero;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      double d = fabs(creal(A[j * lda + i]) - creal(refA[j * lda + i]));
      if (d > rdiff)
        rdiff = d;
      d = fabs(cimag(A[j * lda + i]) - cimag(refA[j * lda + i]));
      if (d > idiff)
        idiff = d;
    }
  }

  // Set A to identity so that repeated applications of the cholesky
  // decomposition while benchmarking do not exit early due to
  // non-positive-definite-ness.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? complex_one : complex_zero;
  }

#ifdef GPU
  CU_ERROR_CHECK(cuMemcpyHtoD2D(dA, dlda, 0, 0, A, lda, 0, 0, n, n, sizeof(double complex)));

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuZpotrf(uplo, n, dA, dlda, dInfo));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20000;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));
#else
  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }
  for (size_t i = 0; i < 20; i++)
#ifdef MULTIGPU
    CU_ERROR_CHECK(cuMultiGPUZpotrf(contexts, deviceCount, uplo, n, A, lda, &info));
#else
    zpotrf(uplo, n, A, lda, &info);
#endif
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
#endif

  size_t flops = (((n * n * n) / 6) + ((n * n) / 2) + (n / 3)) * 6 + (((n * n * n) / 6) - (n / 6)) * 2;

  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time, ((double)flops * 1.e-9) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);
#ifdef GPU
  if (dA != 0)
    CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dInfo));

#ifdef MULTIGPU
  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuCtxDestroy(contexts[i]));
#else
  CU_ERROR_CHECK(cuCtxDestroy(context));
#endif
#endif

  return (int)!passed;
}

static inline void zpotrf_ref(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < max(1, n))
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0) return;

  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      double aii = creal(A[i * lda + i]);
      for (size_t k = 0; k < i; k++)
        aii -= A[i * lda + k] * conj(A[i * lda + k]);
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i;
        return;
      }
      aii = sqrt(aii);
      A[i * lda + i] = aii;

      for (size_t j = i + 1; j < n; j++) {
        double complex temp = A[j * lda + i];
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * conj(A[i * lda + k]);
        A[j * lda + i] = temp / aii;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      double ajj = creal(A[j * lda + j]);
      for (size_t k = 0; k < j; k++)
        ajj -= conj(A[k * lda + j]) * A[k * lda + j];
      if (ajj <= zero || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j;
        return;
      }
      ajj = sqrt(ajj);
      A[j * lda + j] = ajj;

      for (size_t i = j + 1; i < n; i++) {
        double complex temp = A[j * lda + i];
        for (size_t k = 0; k < j; k++)
          temp -= conj(A[k * lda + j]) * A[k * lda + i];
        A[j * lda + i] = temp / ajj;
      }
    }
  }
}

static void * malloc2D(size_t m, size_t n, size_t * ld, size_t elemSize) {
  size_t align = (16 / elemSize) - 1;
  *ld = max(1, (m + align) & ~align);
  return malloc(*ld * n * elemSize);
}

static void rand2D(size_t m, size_t n, double complex * A, size_t lda) {
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      A[j * lda + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
  }
}

static void fprintf2D(FILE * stream, const char * label, size_t m, size_t n, const double complex * A, size_t lda) {
  fprintf(stream, "%s =\n", label);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++)
      fprintf(stream, "%15.6f + %9.6fi", creal(A[j * lda + i]), cimag(A[j * lda + i]));
    fputs("\n", stream);
  }
}

#ifdef GPU
static CUresult cuMemcpyHtoD2D(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                               const void * B, size_t ldb, size_t bi, size_t bj,
                               size_t m, size_t n, size_t elemSize) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_HOST, B, 0, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2D(&copy);
}

static CUresult cuMemcpyDtoH2D(void * A, size_t lda, size_t ai, size_t aj,
                               CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                               size_t m, size_t n, size_t elemSize) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_HOST, A, 0, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2D(&copy);
}
#endif
#endif
