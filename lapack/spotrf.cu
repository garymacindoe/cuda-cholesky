#define __DEVICE_ONLY
#include "../blas/sgemm.cu"
#include "strtri.cu"
#undef __DEVICE_ONLY

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
__device__ void spptf2(int n, float * __restrict__ A, int * __restrict__ info) {
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
      float temp;
      if (i <= j) {
        // SGEMV/SSYRK
        temp = A[upper(i, j)];
        for (int k = 0; k < i; k++)
          temp -= A[upper(k, i)] * A[upper(k, j)];

        // Thread j calculates the diagonal element
        if (i == j) {
          if (temp <= 0.0f || isnan(temp)) {
            *info = i + 1;
            A[upper(i, i)] = temp;
          }
          else
            A[upper(i, i)] = sqrtf(temp);
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
    for (int j = 0; j < n; j++) {
      float temp;
      if (i >= j) {
        // SGEMV/SSYRK
        temp = A[lower<nb>(i, j)];
        for (int k = 0; k < j; k++)
          temp -= A[lower<nb>(j, k)] * A[lower<nb>(i, k)];

        // Thread j calculates the diagonal element
        if (i == j) {
          if (temp <= 0.0f || isnan(temp)) {
            *info = j + 1;
            A[lower<nb>(j, j)] = temp;
          }
          else
            A[lower<nb>(j, j)] = sqrtf(temp);
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

/**
 * Global wrapper for spptf2 device function.  Parameters are as for spptf2.
 *
 * This must be called with a one dimensional thread block.
 */
template <CBlasUplo uplo, unsigned int bx>
__global__ void spotf2(float * __restrict__ A, int * __restrict__ info, int lda, int n) {
  // info parameter cached in shared memory for fast access by all threads in the block
  __shared__ int sinfo;

  /*
   * For efficient data reuse A needs to be cached in shared memory.  In order
   * to get maximum instruction throughput 64 threads are needed but this would
   * use all 16384 bytes (64 * 64 * sizeof(float)) of shared memory to store A.
   * Triangular packed storage mode is therefore used to store only the
   * triangle of A being updated using 8320 bytes((64 * (64 + 1)) / 2 * sizeof(float))
   * of shared memory.
   * Since this is only ever going to be run using one thread block shared
   * memory and register use can be higher than when trying to fit multiple
   * thread blocks onto each multiprocessor.
   */
  __shared__ float a[(bx * (bx + 1)) / 2];

  if (uplo == CBlasUpper) {
    // Read upper triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < n; j++) {
      if (threadIdx.x <= j)
        a[upper(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    // Perform the cholesky decomposition using the packed device function
    spptf2<CBlasUpper, bx, bx>(n, a, &sinfo);

    // Write info back to global memory
    if (threadIdx.x == 0)
      *info = sinfo;

    // Write the upper triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x <= j)
        A[j * lda + threadIdx.x] = a[upper(threadIdx.x, j)];
    }
  }
  else {
    // Read lower triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        a[lower<bx>(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    // Perform the cholesky decomposition using the packed device function
    spptf2<CBlasLower, bx, bx>(n, a, &sinfo);

    // Write info back to global memory
    if (threadIdx.x == 0)
      *info = sinfo;

    // Write the lower triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        A[j * lda + threadIdx.x] = a[lower<bx>(threadIdx.x, j)];
    }
  }
}

/**
 * Single precision Positive definite symmetric Factorisation, Inversion and
 * Matrix Multiply (out of place).
 *
 * The unblocked Cholesky decomposition is performed in-place on the diagonal
 * block of A.  The out-of-place STRTI2 places its result in the top/left block
 * of B and the SGEMM places its result in the rest of B.
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
__global__ void spotfimm2(float * __restrict__ A, float * __restrict__ B,
                          int * __restrict__ info,
                          int lda, int ldb,
                          int j, int jb, int n) {
  if (uplo == CBlasUpper) {
    if (blockIdx.x == gridDim.x - 1) {
      if (blockIdx.y == 0) {
        __shared__ int sinfo;
        __shared__ float a[(mb * (mb + 1)) / 2];

        const int i = threadIdx.y * bx + threadIdx.x;

        // Read upper triangle of A into shared memory
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            a[upper(i, k)] = A[k * lda + j + i];
        }

        __syncthreads();

        // Perform the cholesky decomposition using the packed device function
        spptf2<CBlasUpper, mb, bx>(jb, a, &sinfo);

        // Write the upper triangle of A back to global memory
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            A[k * lda + j + i] = a[upper(i, k)];
        }

        // If cholesky failed don't bother with the inverse
        if (sinfo != 0) {
          // Write info back to global memory
          if (i == 0)
            *info = sinfo;
          return;
        }

        // Perform the inverse of A using the cholesky decomposition in shared memory
        stpti2<CBlasUpper, CBlasNonUnit, mb, bx>(jb, a, &sinfo);

        // Write info back to global memory
        if (i == 0)
          *info = sinfo;

        // Write the upper triangle of A back to global memory in the left block of B
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            B[k * ldb + i] = a[upper(i, k)];
          else if (i < jb)
            B[k * ldb + i] = A[k * lda + j + i];
        }
      }
    }
    else
      sgemm2<CBlasTrans, CBlasNoTrans, mb, nb, kb, bx, by>(jb, n - j - jb, j,
             -1.0f, A, lda, &A[jb * lda], lda,
              1.0f, &A[jb * lda + j], lda, &B[jb * ldb], ldb);
  }
  else {
    if (blockIdx.y == gridDim.y - 1) {
      if (blockIdx.x == 0) {
        __shared__ int sinfo;
        __shared__ float a[(nb * (nb + 1)) / 2];

        const int i = threadIdx.y * bx + threadIdx.x;

        // Read lower triangle of A into shared memory
        if (i < jb) {
          for (int k = 0; k < jb; k++) {
            if (i >= k)
              a[lower<nb>(i, k)] = A[(j + k) * lda + i];
          }
        }

        __syncthreads();

        if (i < jb) {
          // Perform the cholesky decomposition using the packed device function
          spptf2<CBlasLower, nb, bx>(jb, a, &sinfo);

          // Write the lower triangle of A back to global memory
          for (int k = 0; k < jb; k++) {
            if (i >= k)
              A[(j + k) * lda + i] = a[lower<nb>(i, k)];
          }
        }

        // If cholesky failed don't bother with the inverse
        if (sinfo != 0) {
          // Write info back to global memory
          if (i == 0)
            *info = sinfo;
          return;
        }

        if (i < jb) {
          // Perform the inverse of A using the cholesky decomposition in shared memory
          stpti2<CBlasLower, CBlasNonUnit, nb, bx>(jb, a, &sinfo);

          // Write info back to global memory
          if (i == 0)
            *info = sinfo;

          // Write the lower triangle of A back to global memory in the top block of B
          for (int k = 0; k < jb; k++) {
            if (i < k)
              B[k * ldb + i] = A[(j + k) * lda + i];
            if (i >= k)
              B[k * ldb + i] = a[lower<nb>(i, k)];
          }
        }
      }
    }
    else
      sgemm2<CBlasNoTrans, CBlasTrans, mb, nb, kb, bx, by>(n - j - jb, jb, j,
             -1.0f, &A[jb], lda, A, lda,
              1.0f, &A[j * lda + jb], lda, &B[jb], ldb);
  }
}

#ifndef __DEVICE_ONLY
template __global__ void spotf2<CBlasUpper, 64>(float * __restrict__, int * __restrict__, int, int);
template __global__ void spotf2<CBlasLower, 64>(float * __restrict__, int * __restrict__, int, int);

template __global__ void spotfimm2<CBlasUpper, 32, 32, 8, 8, 8>(float * __restrict__, float * __restrict__, int * __restrict__, int, int, int, int, int);
template __global__ void spotfimm2<CBlasLower, 64, 16, 16, 16, 4>(float * __restrict__, float * __restrict__, int * __restrict__, int, int, int, int, int);
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

extern "C" void spotf2_(const char *, const int *, float *, const int *, int *);
static inline void spotf2(CBlasUplo uplo, int n, float * A, int lda, int * info) {
  if (uplo == CBlasUpper)
    spotf2<CBlasUpper, 64><<<1,64>>>(A, info, lda, n);
  else
    spotf2<CBlasLower, 64><<<1,64>>>(A, info, lda, n);
}

static int cond(int, float, float *, size_t);

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

  float * A, * dA, * refA;
  size_t lda = (n + 3) & ~3, dlda;
  int * dinfo;
  if ((A = (float *)malloc(lda * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Failed to allocate A\n");
    return -1;
  }
  if ((refA = (float *)malloc(lda * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Failed to allocate refA\n");
    return -2;
  }
  CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, n * sizeof(float), n));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));
  dlda /= sizeof(float);

  cond(n, 2.0f, A, lda);

  for (int j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(float));

  CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(float), A, lda * sizeof(float), n * sizeof(float), n, cudaMemcpyHostToDevice));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      fprintf(stderr, "%15.6f", A[j * lda + i]);
    fprintf(stderr, "\n");
  }

  int info = 0, refInfo = 0;
  spotf2_((const char *)&uplo, &n, refA, (const int *)&lda, &refInfo);
  spotf2(uplo, n, dA, dlda, dinfo);
  CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(float), dA, dlda * sizeof(float), n * sizeof(float), n, cudaMemcpyDeviceToHost));
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

  float error = 0.0f;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      float diff = fabsf(refA[j * lda + i] - A[j * lda + i]);
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

static int cond(int n, float c, float * A, size_t lda) {
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

  float * u, * v, * w;
  size_t offset = (n + 3u) & ~3u;

  if ((u = (float *)malloc(3 * offset * sizeof(float))) == NULL)
    return 1;

  v = &u[offset];
  w = &v[offset];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      A[j * lda + i] = 0.0f;
  }

  A[0] = 1.0f;
  A[lda + 1] = c;
  for (int j = 2; j < n; j++)
    A[j * lda + j] = ((float) rand() / (float)RAND_MAX) * (c - 1.0f) + 1.0f;

  float t = 0.0f, s = 0.0f;
  for (int j = 0; j < n; j++) {
    // u is a random vector
    u[j] = (float)rand() / (float)RAND_MAX;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += u[j] * u[j];
    // s = t^2 u'v / 2
    s += u[j] * v[j];
  }
  t = 2.0f / t;
  s = t * t * s / 2.0f;

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

#if 1
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

extern "C" void sgemm_(const char *, const char *, const int *, const int *, const int *,
                       const float *, const float *, const int *, const float *, const int *,
                       const float *, float *, const int *);
extern "C" void spotf2_(const char *, const int *, float *, const int *, int *);
extern "C" void strti2_(const char *, const char *, const int *, float *, const int *, int *);

static inline void spotfimm2(CBlasUplo uplo, int j, int jb, int n, float * A, int lda, float * B, int ldb, int * info) {
  if (uplo == CBlasUpper) {
    spotfimm2<CBlasUpper, 32, 32,  8,  8,  8><<<dim3(((unsigned int)jb + 31) / 32 + 1, (unsigned int)(n - j - jb + 31) / 32), dim3(8, 8)>>>(A, B, info, lda, ldb, j, jb, n);
  }
  else {
    spotfimm2<CBlasLower, 64, 16, 16, 16,  4><<<dim3((unsigned int)(n - j - jb + 63) / 64, ((unsigned int)jb + 15) / 16 + 1), dim3(16, 4)>>>(A, B, info, lda, ldb, j, jb, n);
  }
}

static void spotfimm2_(CBlasUplo uplo, int j, int jb, int n, float * A, int lda, float * B, int ldb, int * info) {
  const float mone = -1.0f, one = 1.0f;
  const int n_j_jb = n - j - jb;
  if (uplo == CBlasUpper) {
    spotf2_("Upper", &jb, &A[j], &lda, info);
    for (int k = 0; k < n - j; k++)
      memcpy(&B[k * ldb], &A[k * lda + j], jb * sizeof(float));
    sgemm_("Transpose", "No Transpose", &jb, &n_j_jb, &j, &mone, A, &lda, &A[jb * lda], &lda, &one, &B[jb * ldb], &ldb);
    strti2_("Upper", "Non-Unit", &jb, B, &ldb, info);
  }
  else {
    spotf2_("Lower", &jb, &A[j * lda], &lda, info);
    for (int k = 0; k < jb; k++)
      memcpy(&B[k * ldb], &A[(j + k) * lda], (n - j) * sizeof(float));
    sgemm_("No Transpose", "Transpose", &n_j_jb, &jb, &j, &mone, &A[jb], &lda, A, &lda, &one, &B[jb], &ldb);
    strti2_("Lower", "Non-Unit", &jb, B, &ldb, info);
  }
}

static int cond(int, float, float *, size_t);

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

  float * A, * B, * dA, * dB, * refA, * refB;
  size_t lda, ldb, dlda, dldb;
  int * dinfo;
  if (uplo == CBlasUpper) {
    if ((A = (float *)malloc((lda = (j + jb + 3u) & ~3u) * (n - j) * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate A\n");
      return -1;
    }
    if ((B = (float *)calloc((ldb = (jb + 3u) & ~3u), (n - j) * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate B\n");
      return -2;
    }
    if ((refA = (float *)malloc(lda * (n - j) * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate refA\n");
      return -3;
    }
    if ((refB = (float *)calloc(ldb, (n - j) * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate refB\n");
      return -4;
    }
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, (j + jb) * sizeof(float), n - j));
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dB, &dldb, jb * sizeof(float), n - j));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));
    dlda /= sizeof(float);
    dldb /= sizeof(float);

    for (int k = 0; k < n - j; k++) {
      for (int i = 0; i < j + jb; i++)
        A[k * lda + i] = (float)rand() / (float)RAND_MAX;
    }

    cond(jb, 2.0f, &A[j], lda);

    for (int k = 0; k < n - j; k++)
      memcpy(&refA[k * lda], &A[k * lda], (j + jb) * sizeof(float));

    CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(float), A, lda * sizeof(float), (j + jb) * sizeof(float), n - j, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy2D(dB, dldb * sizeof(float), B, ldb * sizeof(float), jb * sizeof(float), n - j, cudaMemcpyHostToDevice));

    for (int i = 0; i < j + jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
  }
  else {
    if ((A = (float *)malloc((lda = (n - j + 3u) & ~3u) * (j + jb) * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate A\n");
      return -1;
    }
    if ((B = (float *)calloc((ldb = ((n - j) + 3u) & ~3u), jb * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate B\n");
      return -2;
    }
    if ((refA = (float *)malloc(lda * (j + jb) * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate refA\n");
      return -3;
    }
    if ((refB = (float *)calloc(ldb, jb * sizeof(float))) == NULL) {
      fprintf(stderr, "Failed to allocate refB\n");
      return -4;
    }
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, (n - j) * sizeof(float), j + jb));
    CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dB, &dldb, (n - j) * sizeof(float), jb));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));
    dlda /= sizeof(float);
    dldb /= sizeof(float);

    for (int k = 0; k < j + jb; k++) {
      for (int i = 0; i < n - j; i++)
        A[k * lda + i] = (float)rand() / (float)RAND_MAX;
    }

    cond(jb, 2.0f, &A[j * lda], lda);

    for (int k = 0; k < j + jb; k++)
      memcpy(&refA[k * lda], &A[k * lda], (n - j) * sizeof(float));

    CUDA_ERROR_CHECK(cudaMemcpy2D(dA, dlda * sizeof(float), A, lda * sizeof(float), (n - j) * sizeof(float), j + jb, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy2D(dB, dldb * sizeof(float), B, ldb * sizeof(float), (n - j) * sizeof(float), jb, cudaMemcpyHostToDevice));

    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < j + jb; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
  }

  int info = 0, refInfo = 0;
  spotfimm2_(uplo, j, jb, n, refA, lda, refB, ldb, &refInfo);
  spotfimm2(uplo, j, jb, n, dA, dlda, dB, dldb, dinfo);
  CUDA_ERROR_CHECK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));

  float error = 0.0f;
  if (uplo == CBlasUpper) {
    CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(float), dA, dlda * sizeof(float), (j + jb) * sizeof(float), n - j, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy2D(B, ldb * sizeof(float), dB, dldb * sizeof(float), jb * sizeof(float), n - j, cudaMemcpyDeviceToHost));

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

    for (int k = 0; k < n - j; k++) {
      for (int i = 0; i < j + jb; i++) {
        float diff = fabsf(A[k * lda + i] - refA[k * lda + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < n - j; k++) {
      for (int i = 0; i < jb; i++) {
        float diff = fabsf(B[k * ldb + i] - refB[k * ldb + i]);
        if (diff > error)
          error = diff;
      }
    }
  }
  else {
    CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(float), dA, dlda * sizeof(float), (n - j) * sizeof(float), j + jb, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy2D(B, ldb * sizeof(float), dB, dldb * sizeof(float), (n - j) * sizeof(float), jb, cudaMemcpyDeviceToHost));

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

    for (int k = 0; k < j + jb; k++) {
      for (int i = 0; i < n - j; i++) {
        float diff = fabsf(A[k * lda + i] - refA[k * lda + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = 0; i < n - j; i++) {
        float diff = fabsf(B[k * ldb + i] - refB[k * ldb + i]);
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

static int cond(int n, float c, float * A, size_t lda) {
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

  float * u, * v, * w;
  size_t offset = (n + 3u) & ~3u;

  if ((u = (float *)malloc(3 * offset * sizeof(float))) == NULL)
    return 1;

  v = &u[offset];
  w = &v[offset];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      A[j * lda + i] = 0.0f;
  }

  A[0] = 1.0f;
  A[lda + 1] = c;
  for (int j = 2; j < n; j++)
    A[j * lda + j] = ((float) rand() / (float)RAND_MAX) * (c - 1.0f) + 1.0f;

  float t = 0.0f, s = 0.0f;
  for (int j = 0; j < n; j++) {
    // u is a random vector
    u[j] = (float)rand() / (float)RAND_MAX;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += u[j] * u[j];
    // s = t^2 u'v / 2
    s += u[j] * v[j];
  }
  t = 2.0f / t;
  s = t * t * s / 2.0f;

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
