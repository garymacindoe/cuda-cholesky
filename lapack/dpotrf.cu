#include "../blas/dgemm.cu"
#include "dtrtri.cu"

#ifndef __UPLO_H
#define __UPLO_H

/*
 * Indexing function for upper triangular packed storage mode.  Only works when
 * i <= j otherwise generates an out-of-bounds access in shared memory and CUDA
 * will segfault.
 */
__device__ int upper(int i, int j) {
  return ((j * (j + 1)) / 2) + i;
}

/*
 * Indexing function for lower triangular packed storage mode.  Only works when
 * i >= j otherwise generates an out-of-bounds access in shared memory and CUDA
 * will segfault.
 */
template <unsigned int bx>
__device__ int lower(int i, int j) {
  return ((2 * bx - j - 1) * j) / 2 + i;
}

#endif

/**
 * SPPTRF:
 *
 * Single precision Positive definite symmetric Packed Triangular Factorisation
 * (unblocked).
 *
 * Performs the Cholesky decomposition on a matrix stored in shared memory using
 * triangular packed storage mode.
 *
 * This can be called with a one or two dimensional thread block.  Two
 * dimensional thread blocks will be unwrapped to one dimension.
 *
 * @param uplo  whether the matrix is upper or lower triangular
 * @param bx    the number of threads per block.  Also controls shared memory
 *                usage.  This function won't work for n > bx.
 * @param n     the size of the matrix
 * @param A     the matrix
 * @param info  if info != 0 on output then the matrix is not positive definite
 */
template <CBlasUplo uplo, unsigned int nb, unsigned int bx>
__device__ void dpptf2(int n, double * __restrict__ A, int * __restrict__ info) {
  // thread 0 is the only thread to write to info
  if (threadIdx.y * bx + threadIdx.x == 0)
    *info = 0;  // initialise info to zero

  if (uplo == CBlasUpper) {
    // Perform the cholesky decomposition in the upper triangle
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also avoids bank
    // conflicts.
    const int j = threadIdx.y * bx + threadIdx.x;
    for (int i = 0; i < n; i++) {
      double temp;
      if (i <= j) {
        // SGEMV/DSYRK
        temp = A[upper(i, j)];
        for (int k = 0; k < i; k++)
          temp -= A[upper(k, i)] * A[upper(k, j)];

        // Thread j calculates the diagonal element
        if (i == j) {
          if (temp <= 0.0 || isnan(temp)) {
            *info = i + 1;
            A[upper(i, i)] = temp;
          }
          else
            A[upper(i, i)] = sqrt(temp);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (*info != 0)
        return;

      // SSCAL
      if (i < j)
        A[upper(i, j)] = temp / A[upper(i, i)];

      __syncthreads();
    }
  }
  else {
    // Perform the cholesky decomposition in the lower triangle
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also avoids bank
    // conflicts.
    const int i = threadIdx.y * bx + threadIdx.x;
    if (i < n) {
      for (int j = 0; j < n; j++) {
        double temp;
        if (i >= j) {
          // SGEMV/DSYRK
          temp = A[lower<nb>(i, j)];
          for (int k = 0; k < j; k++)
            temp -= A[lower<nb>(j, k)] * A[lower<nb>(i, k)];

          // Thread j calculates the diagonal element
          if (i == j) {
            if (temp <= 0.0 || isnan(temp)) {
              *info = j + 1;
              A[lower<nb>(j, j)] = temp;
            }
            else
              A[lower<nb>(j, j)] = sqrt(temp);
          }
        }

        __syncthreads();

        // If info != 0 return (matrix is not positive definite)
        if (*info != 0)
          return;

        // SSCAL
        if (i > j)
          A[lower<nb>(i, j)] = temp / A[lower<nb>(j, j)];

        __syncthreads();
      }
    }
  }
}

/**
 * Global wrapper for spptf2 device function.  Parameters are as for spptf2.
 *
 * This must be called with a one dimensional thread block.
 */
template <CBlasUplo uplo, unsigned int bx>
__global__ void dpotf2(double * __restrict__ A, int * __restrict__ info, int lda, int n) {
  // info parameter cached in shared memory for fast access by all threads in the block
  __shared__ int sino;

  /*
   * For efficient data reuse A needs to be cached in shared memory.  In order
   * to get maximum instruction throughput 64 threads are needed but this would
   * use all 16384 bytes (64 * 64 * sizeof(double)) of shared memory to store A.
   * Triangular packed storage mode is therefore used to store only the
   * triangle of A being updated using 8320 bytes((64 * (64 + 1)) / 2 * sizeof(double))
   * of shared memory.
   * Since this is only ever going to be run using one thread block shared
   * memory and register use can be higher than when trying to fit multiple
   * thread blocks onto each multiprocessor.
   */
  __shared__ double a[(bx * (bx + 1)) / 2];

  const int i = threadIdx.x;

  if (uplo == CBlasUpper) {
    // Read upper triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < bx; j++) {
      if (i <= j)
        a[upper(i, j)] = A[j * lda + i];
    }

    __syncthreads();

    // Perform the cholesky decomposition using the packed device function
    dpptf2<CBlasUpper, bx, bx>(n, a, &sino);

    // Write info back to global memory
    if (i == 0)
      *info = sino;

    // Write the upper triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (i <= j)
        A[j * lda + i] = a[upper(i, j)];
    }
  }
  else {
    // Read lower triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < bx; j++) {
      if (i >= j)
        a[lower<bx>(i, j)] = A[j * lda + i];
    }

    __syncthreads();

    // Perform the cholesky decomposition using the packed device function
    dpptf2<CBlasLower, bx, bx>(n, a, &sino);

    // Write info back to global memory
    if (i == 0)
      *info = sino;

    // Write the lower triangle of A back to global memory
    if (i < n) {
      for (int j = 0; j < n; j++) {
        if (i >= j)
          A[j * lda + i] = a[lower<bx>(i, j)];
      }
    }
  }
}

/**
 * Single precision Positive definite symmetric Factorisation, Inversion and
 * Matrix Multiply (out of place).
 *
 * The unblocked Cholesky decomposition is performed in-place on the diagonal
 * block of A.  The out-of-place STRTI2 places its result in the top/left block
 * of B and the DGEMM places its result in the rest of B.
 *
 * @param A    the current row/column of A
 * @param B    temporary row/column of nb * n - j - jb or n - j - jb * nb for
 *               out-of-place results.
 * @param lda  leading dimension of A.
 * @param ldb  leading dimension of B.
 * @param j    the loop index.
 * @param jb   the block size.
 * @param n    the size of A.
 */
template <CBlasUplo uplo,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dpotfimm2(double * __restrict__ A, double * __restrict__ B,
                          int * __restrict__ info,
                          int lda, int ldb,
                          int j, int jb, int n) {
  if (uplo == CBlasUpper) {
    // If we are the last column of blocks
    if (blockIdx.y == gridDim.y - 1) {
      // and the first of the last column (there should only be one)
      if (blockIdx.x == 0) {
        __shared__ int sino;
        __shared__ double a[(mb * (mb + 1)) / 2];

        const int i = threadIdx.y * bx + threadIdx.x;

        // Read upper triangle of A into shared memory using upper triangular
        // packed storage mode
        #pragma unroll
        for (int k = 0; k < mb; k++) {
          if (i <= k)
            a[upper(i, k)] = A[k * lda + j + i];
        }

        __syncthreads();

        // Perform the cholesky decomposition using the packed device function
        dpptf2<CBlasUpper, mb, bx>(jb, a, &sino);

        // Write the upper triangle of A back to global memory
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            A[k * lda + j + i] = a[upper(i, k)];
        }

        // If cholesky failed don't bother with the inverse
        if (sino != 0) {
          // Write info back to global memory
          if (i == 0)
            *info = sino;
          return;
        }

        // Perform the inverse of A using the cholesky decomposition in shared memory
        dtpti2<CBlasUpper, CBlasNonUnit, mb, bx>(jb, a, &sino);

        // Write info back to global memory
        if (i == 0)
          *info = sino;

        // Write the upper triangle of A back to global memory in the left block of B
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            B[k * ldb + i] = a[upper(i, k)];
        }
      }
    }
    else        // All other blocks perform DGEMM
      dgemm2<CBlasTrans, CBlasNoTrans, mb, nb, kb, bx, by>(jb, n - j - jb, j,
             -1.0, A, lda, &A[jb * lda], lda,
              1.0, &A[jb * lda + j], lda, &B[jb * ldb], ldb);
  }
  else {
    // If we are the last row of blocks
    if (blockIdx.x == gridDim.x - 1) {
      // and the first of the last row (there should only be one)
      if (blockIdx.y == 0) {
        __shared__ int sino;
        __shared__ double a[(nb * (nb + 1)) / 2];

        const int i = threadIdx.y * bx + threadIdx.x;
        A += j * lda;

        // Read lower triangle of A into shared memory using lower triangular
        // packed storage mode
        #pragma unroll
        for (int k = 0; k < nb; k++) {
          if (i >= k)
            a[lower<nb>(i, k)] = A[k * lda + i];
        }

        __syncthreads();

        // Perform the cholesky decomposition using the packed device function
        dpptf2<CBlasLower, nb, bx>(jb, a, &sino);

        // Write the lower triangle of A back to global memory
        if (i < jb) {
          for (int k = 0; k < jb; k++) {
            if (i >= k)
              A[k * lda + i] = a[lower<nb>(i, k)];
          }
        }

        // If cholesky failed don't bother with the inverse
        if (sino != 0) {
          // Write info back to global memory
          if (i == 0)
            *info = sino;
          return;
        }

        // Perform the inverse of A using the cholesky decomposition in shared memory
        dtpti2<CBlasLower, CBlasNonUnit, nb, bx>(jb, a, &sino);

        // Write info back to global memory
        if (i == 0)
          *info = sino;

        // Write the lower triangle of A back to global memory in the top block of B
        if (i < jb) {
          for (int k = 0; k < jb; k++) {
            if (i >= k)
              B[k * ldb + i] = a[lower<nb>(i, k)];
          }
        }
      }
    }
    else        // All other blocks perform DGEMM
      dgemm2<CBlasNoTrans, CBlasTrans, mb, nb, kb, bx, by>(n - j - jb, jb, j,
             -1.0, &A[jb], lda, A, lda,
              1.0, &A[j * lda + jb], lda, &B[jb], ldb);
  }
}

template __global__ void dpotf2<CBlasUpper, 32>(double * __restrict__, int * __restrict__, int, int);
template __global__ void dpotf2<CBlasLower, 32>(double * __restrict__, int * __restrict__, int, int);

template __global__ void dpotfimm2<CBlasUpper, 32, 16, 8, 8, 8>(double * __restrict__, double * __restrict__, int * __restrict__, int, int, int, int, int);
template __global__ void dpotfimm2<CBlasLower, 64,  8, 16, 16, 4>(double * __restrict__, double * __restrict__, int * __restrict__, int, int, int, int, int);

#if 0
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <float.h>
#include <math.h>

#define CUDA_ERROR_CHECK(call) \
  do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA Runtime error in %s (%s:%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(error)); \
      return error; \
    } \
  } while (false)

#define xerbla(info) \
  fprintf(stderr, "On entry to %s parameter %d had an invalid value\n", __func__, (info))

extern "C" void dpotf2_(const char *, const int *, double *, const int *, int *);
static inline void dpotf2(CBlasUplo uplo, int n, double * A, int lda, int * info) {
  if (uplo == CBlasUpper)
    dpotf2<CBlasUpper, 32><<<1,32>>>(A, info, lda, n);
  else
    dpotf2<CBlasLower, 32><<<1,32>>>(A, info, lda, n);
}

static int cond(int, double, double *, size_t);

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  int n;

  if (argc != 3) {
    fprintf(stderr, "Usage %s <uplo> <diag> <n>\n"
                    "where:\n"
                    "  <uplo>  is 'U' or 'u' for CBlasUpper or 'L' or 'l' for CBlasLower\n"
                    "  <n>     is the size of the matrix\n", argv[0]);
    return -1;
  }

  char u;
  if (sscanf(argv[1], "%c", &u) != 1) {
    fprintf(stderr, "Unable to parse character from '%s'\n", argv[1]);
    return 1;
  }
  switch (u) {
    case 'u': case 'U': uplo = CBlasUpper; break;
    case 'l': case 'L': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 1;
  }

  if (sscanf(argv[2], "%d", &n) != 1) {
    fprintf(stderr, "Unable to parse integer from '%s'\n", argv[2]);
    return 2;
  }
  if (n < 1 || n > 32) {
    fputs("n must be between 1 and 32\n", stderr);
    return 2;
  }

  double * A, * dA, * refA;
  size_t lda = (n + 3) & ~3, dlda;
  int * dinfo;
  if ((A = (double *)malloc(lda * n * sizeof(double))) == NULL) {
    fprintf(stderr, "Failed to allocate A\n");
    return -1;
  }
  if ((refA = (double *)malloc(lda * n * sizeof(double))) == NULL) {
    fprintf(stderr, "Failed to allocate refA\n");
    return -2;
  }
  CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, n * sizeof(double), n));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));
  dlda /= sizeof(double);

  cond(n, 2.0, A, lda);

  for (int j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(double));

  CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(double), A, lda * sizeof(double), n * sizeof(double), n, cudaMemcpyHostToDevice));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      fprintf(stderr, "%15.6f", A[j * lda + i]);
    fprintf(stderr, "\n");
  }

  int info = 0, refInfo = 0;
  dpotf2_((const char *)&uplo, &n, refA, (const int *)&lda, &refInfo);
  dpotf2(uplo, n, dA, dlda, dinfo);
  CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(double), dA, dlda * sizeof(double), n * sizeof(double), n, cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));

  fprintf(stderr, "\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      fprintf(stderr, "%15.6f", refA[j * lda + i]);
    fprintf(stderr, "\n");
  }

  fprintf(stderr, "\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      fprintf(stderr, "%15.6f", A[j * lda + i]);
    fprintf(stderr, "\n");
  }

  double error = 0.0;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      double diff = fabs(refA[j * lda + i] - A[j * lda + i]);
      if (diff > error)
        error = diff;
    }
  }

  fprintf(stdout, "Info = %d, refInfo = %d, Error = %6.3e\n", info, refInfo, error);

  free(A);
  free(refA);
  CUDA_ERROR_CHECK(cudaFree(dA));
  CUDA_ERROR_CHECK(cudaFree(dinfo));

  return info;
}

static int cond(int n, double c, double * A, size_t lda) {
  int info = 0;
  if (n < 2)
    info = -1;
  else if (c < 1.0)
    info = -2;
  else if (lda < n)
    info = -4;
  if (info != 0) {
    xerbla(-info);
    return info;
  }

  double * u, * v, * w;
  size_t offset = (n + 1u) & ~1u;

  if ((u = (double *)malloc(3 * offset * sizeof(double))) == NULL)
    return 1;

  v = &u[offset];
  w = &v[offset];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      A[j * lda + i] = 0.0;
  }

  A[0] = 1.0;
  A[lda + 1] = c;
  for (int j = 2; j < n; j++)
    A[j * lda + j] = ((double) rand() / (double)RAND_MAX) * (c - 1.0) + 1.0;

  double t = 0.0, s = 0.0;
  for (int j = 0; j < n; j++) {
    // u is a random vector
    u[j] = (double)rand() / (double)RAND_MAX;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += u[j] * u[j];
    // s = t^2 u'v / 2
    s += u[j] * v[j];
  }
  t = 2.0 / t;
  s = t * t * s / 2.0;

  // w = tv - su
  for (int j = 0; j < n; j++)
    w[j] = t * v[j] - s * u[j];

  // A -= uw' + wu'
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      A[j * lda + i] -= u[i] * w[j] + w[i] * u[j];
  }

  free(u);

  return 0;
}
#endif

#if 0
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <float.h>
#include <math.h>

#define CUDA_ERROR_CHECK(call) \
  do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA Runtime error in %s (%s:%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(error)); \
      return error; \
    } \
  } while (false)

#define xerbla(info) \
  fprintf(stderr, "On entry to %s parameter %d had an invalid value\n", __func__, (info))

extern "C" void dgemm_(const char *, const char *, const int *, const int *, const int *,
                       const double *, const double *, const int *, const double *, const int *,
                       const double *, double *, const int *);
extern "C" void dpotf2_(const char *, const int *, double *, const int *, int *);
extern "C" void dtrti2_(const char *, const char *, const int *, double *, const int *, int *);

static inline void dpotfimm2(CBlasUplo uplo, int j, int jb, int n, double * A, int lda, double * B, int ldb, int * info) {
  if (uplo == CBlasUpper) {
    const unsigned int mb = 32;
    const unsigned int nb = 16;
    const unsigned int kb =  8;
    const unsigned int bx =  8;
    const unsigned int by =  8;

    if (jb > mb) {
      fputs("On entry to dpotfimm2 parameter 3 had an invalid value\n", stderr);
      return;
    }

    const unsigned int gx = (jb + nb - 1) / nb;
    const unsigned int gy = (n - j - jb + nb - 1) / nb;

    dpotfimm2<CBlasUpper, mb, nb, kb, bx, by><<<dim3(max(gx, 1), gy + 1), dim3(bx, by)>>>(A, B, info, lda, ldb, j, jb, n);
  }
  else {
    const unsigned int mb = 64;
    const unsigned int nb =  8;
    const unsigned int kb = 16;
    const unsigned int bx = 16;
    const unsigned int by =  4;

    if (jb > nb) {
      fputs("On entry to dpotfimm2 parameter 3 had an invalid value\n", stderr);
      return;
    }

    const unsigned int gx = (n - j - jb + mb - 1) / mb;
    const unsigned int gy = (jb + nb - 1) / nb;

    dpotfimm2<CBlasLower, mb, nb, kb, bx, by><<<dim3(gx + 1, max(gy, 1)), dim3(bx, by)>>>(A, B, info, lda, ldb, j, jb, n);
  }
}

static void dpotfimm2_(CBlasUplo uplo, int j, int jb, int n, double * A, int lda, double * B, int ldb, int * info) {
  const double mone = -1.0, one = 1.0;
  const int n_j_jb = n - j - jb;
  if (uplo == CBlasUpper) {
    dpotf2_("Upper", &jb, &A[j], &lda, info);
    for (int k = 0; k < n - j; k++)
      memcpy(&B[k * ldb], &A[k * lda + j], jb * sizeof(double));
    dgemm_("Transpose", "No Transpose", &jb, &n_j_jb, &j, &mone, A, &lda, &A[jb * lda], &lda, &one, &B[jb * ldb], &ldb);
    dtrti2_("Upper", "Non-Unit", &jb, B, &ldb, info);
  }
  else {
    dpotf2_("Lower", &jb, &A[j * lda], &lda, info);
    for (int k = 0; k < jb; k++)
      memcpy(&B[k * ldb], &A[(j + k) * lda], (n - j) * sizeof(double));
    dgemm_("No Transpose", "Transpose", &n_j_jb, &jb, &j, &mone, &A[jb], &lda, A, &lda, &one, &B[jb], &ldb);
    dtrti2_("Lower", "Non-Unit", &jb, B, &ldb, info);
  }
}

static int cond(int, double, double *, size_t);

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  int j, nb, n;

  if (argc != 5) {
    fprintf(stderr, "Usage %s <uplo> <diag> <n>\n"
                    "where:\n"
                    "  <uplo>  is 'U' or 'u' for CBlasUpper or 'L' or 'l' for CBlasLower\n"
                    "  <j>     is the current index\n"
                    "  <nb>    is the block size\n"
                    "  <n>     is the size of the matrix\n", argv[0]);
    return -1;
  }

  char u;
  if (sscanf(argv[1], "%c", &u) != 1) {
    fprintf(stderr, "Unable to parse character from '%s'\n", argv[1]);
    return 1;
  }
  switch (u) {
    case 'u': case 'U': uplo = CBlasUpper; break;
    case 'l': case 'L': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 1;
  }

  if (sscanf(argv[2], "%d", &j) != 1) {
    fprintf(stderr, "Unable to parse integer from '%s'\n", argv[2]);
    return 2;
  }

  if (sscanf(argv[3], "%d", &nb) != 1) {
    fprintf(stderr, "Unable to parse integer from '%s'\n", argv[3]);
    return 3;
  }

  if (sscanf(argv[4], "%d", &n) != 1) {
    fprintf(stderr, "Unable to parse integer from '%s'\n", argv[4]);
    return 4;
  }

  if (j >= n || nb > n) {
    fputs("n is too small\n", stderr);
    return 5;
  }

  int jb = min(nb, n - j);

  double * A, * B, * dA, * dB, * refA, * refB;
  size_t lda, ldb, dlda, dldb;
  int * dinfo;
  if (uplo == CBlasUpper) {
    if ((A = (double *)malloc((lda = (j + jb + 1u) & ~1u) * (n - j) * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate A\n");
      return -1;
    }
    if ((B = (double *)calloc((ldb = (jb + 1u) & ~1u), (n - j) * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate B\n");
      return -2;
    }
    if ((refA = (double *)malloc(lda * (n - j) * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate refA\n");
      return -3;
    }
    if ((refB = (double *)calloc(ldb, (n - j) * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate refB\n");
      return -4;
    }
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, (j + jb) * sizeof(double), n - j));
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dB, &dldb, jb * sizeof(double), n - j));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));
    dlda /= sizeof(double);
    dldb /= sizeof(double);

    for (int k = 0; k < n - j; k++) {
      for (int i = 0; i < j + jb; i++)
        A[k * lda + i] = (double)rand() / (double)RAND_MAX;
    }

    cond(jb, 2.0, &A[j], lda);

    for (int k = 0; k < n - j; k++)
      memcpy(&refA[k * lda], &A[k * lda], (j + jb) * sizeof(double));

    CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(double), A, lda * sizeof(double), (j + jb) * sizeof(double), n - j, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy2D(dB, dldb * sizeof(double), B, ldb * sizeof(double), jb * sizeof(double), n - j, cudaMemcpyHostToDevice));

#ifdef PRINT
    for (int i = 0; i < j + jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
#endif
  }
  else {
    if ((A = (double *)malloc((lda = (n - j + 1u) & ~1u) * (j + jb) * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate A\n");
      return -1;
    }
    if ((B = (double *)calloc((ldb = ((n - j) + 1u) & ~1u), jb * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate B\n");
      return -2;
    }
    if ((refA = (double *)malloc(lda * (j + jb) * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate refA\n");
      return -3;
    }
    if ((refB = (double *)calloc(ldb, jb * sizeof(double))) == NULL) {
      fprintf(stderr, "Failed to allocate refB\n");
      return -4;
    }
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, (n - j) * sizeof(double), j + jb));
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dB, &dldb, (n - j) * sizeof(double), jb));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));
    dlda /= sizeof(double);
    dldb /= sizeof(double);

    for (int k = 0; k < j + jb; k++) {
      for (int i = 0; i < n - j; i++)
        A[k * lda + i] = (double)rand() / (double)RAND_MAX;
    }

    cond(jb, 2.0, &A[j * lda], lda);

    for (int k = 0; k < j + jb; k++)
      memcpy(&refA[k * lda], &A[k * lda], (n - j) * sizeof(double));

    CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(double), A, lda * sizeof(double), (n - j) * sizeof(double), j + jb, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy2D(dB, dldb * sizeof(double), B, ldb * sizeof(double), (n - j) * sizeof(double), jb, cudaMemcpyHostToDevice));

#ifdef PRINT
    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < j + jb; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
#endif
  }

  int info = 0, refInfo = 0;
  dpotfimm2(uplo, j, jb, n, dA, dlda, dB, dldb, dinfo);
  dpotfimm2_(uplo, j, jb, n, refA, lda, refB, ldb, &refInfo);
  CUDA_ERROR_CHECK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));

  double error = 0.0;
  if (uplo == CBlasUpper) {
    CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(double), dA, dlda * sizeof(double), (j + jb) * sizeof(double), n - j, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy2D(B, ldb * sizeof(double), dB, dldb * sizeof(double), jb * sizeof(double), n - j, cudaMemcpyDeviceToHost));

#ifdef PRINT
    fputs("\nrefA:\n", stderr);
    for (int i = 0; i < j + jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", refA[k * lda + i]);
      fprintf(stderr, "\n");
    }
    fputs("\nrefB:\n", stderr);
    for (int i = 0; i < jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", refB[k * ldb + i]);
      fprintf(stderr, "\n");
    }

    fputs("\nA:\n", stderr);
    for (int i = 0; i < j + jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
    fputs("\nB:\n", stderr);
    for (int i = 0; i < jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", B[k * ldb + i]);
      fprintf(stderr, "\n");
    }
#endif

    for (int k = 0; k < n - j; k++) {
      for (int i = 0; i < j + jb; i++) {
        double diff = fabs(A[k * lda + i] - refA[k * lda + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = 0; i <= k; i++) {
        double diff = fabs(B[k * ldb + i] - refB[k * ldb + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = jb; k < n - j - jb; k++) {
      for (int i = 0; i < jb; i++) {
        double diff = fabs(B[k * ldb + i] - refB[k * ldb + i]);
        if (diff > error)
          error = diff;
      }
    }
  }
  else {
    CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(double), dA, dlda * sizeof(double), (n - j) * sizeof(double), j + jb, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy2D(B, ldb * sizeof(double), dB, dldb * sizeof(double), (n - j) * sizeof(double), jb, cudaMemcpyDeviceToHost));

#ifdef PRINT
    fputs("\nrefA:\n", stderr);
    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < j + jb; k++)
        fprintf(stderr, "%15.6f", refA[k * lda + i]);
      fprintf(stderr, "\n");
    }
    fputs("\nrefB:\n", stderr);
    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < jb; k++)
        fprintf(stderr, "%15.6f", refB[k * ldb + i]);
      fprintf(stderr, "\n");
    }

    fputs("\nA:\n", stderr);
    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < j + jb; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
    fputs("\nB:\n", stderr);
    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < jb; k++)
        fprintf(stderr, "%15.6f", B[k * ldb + i]);
      fprintf(stderr, "\n");
    }
#endif

    for (int k = 0; k < j + jb; k++) {
      for (int i = 0; i < n - j; i++) {
        double diff = fabs(A[k * lda + i] - refA[k * lda + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = k; i < jb; i++) {
        double diff = fabs(B[k * ldb + i] - refB[k * ldb + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = 0; i < n - j - jb; i++) {
        double diff = fabs(B[k * ldb + jb + i] - refB[k * ldb + jb + i]);
        if (diff > error)
          error = diff;
      }
    }
  }

  fprintf(stdout, "Info = %d, refInfo = %d, Error = %6.3e\n", info, refInfo, error);

  free(A);
  free(B);
  free(refA);
  free(refB);
  CUDA_ERROR_CHECK(cudaFree(dA));
  CUDA_ERROR_CHECK(cudaFree(dB));
  CUDA_ERROR_CHECK(cudaFree(dinfo));

  return info;
}

static int cond(int n, double c, double * A, size_t lda) {
  int info = 0;
  if (n < 2)
    info = -1;
  else if (c < 1.0)
    info = -2;
  else if (lda < n)
    info = -4;
  if (info != 0) {
    xerbla(-info);
    return info;
  }

  double * u, * v, * w;
  size_t offset = (n + 1u) & ~1u;

  if ((u = (double *)malloc(3 * offset * sizeof(double))) == NULL)
    return 1;

  v = &u[offset];
  w = &v[offset];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      A[j * lda + i] = 0.0;
  }

  A[0] = 1.0;
  A[lda + 1] = c;
  for (int j = 2; j < n; j++)
    A[j * lda + j] = ((double) rand() / (double)RAND_MAX) * (c - 1.0) + 1.0;

  double t = 0.0, s = 0.0;
  for (int j = 0; j < n; j++) {
    // u is a random vector
    u[j] = (double)rand() / (double)RAND_MAX;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += u[j] * u[j];
    // s = t^2 u'v / 2
    s += u[j] * v[j];
  }
  t = 2.0 / t;
  s = t * t * s / 2.0;

  // w = tv - su
  for (int j = 0; j < n; j++)
    w[j] = t * v[j] - s * u[j];

  // A -= uw' + wu'
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      A[j * lda + i] -= u[i] * w[j] + w[i] * u[j];
  }

  free(u);

  return 0;
}
#endif
