#include "blas.h"

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

template <CBlasUplo uplo, CBlasDiag diag, unsigned int bx>
__global__ void dtrti2(double * A, int * info, int lda, int n) {
  // info parameter cached in shared memory for fast access by all threads in the block
  __shared__ int sinfo;

  // thread 0 is the only thread to write to info in shared or global memory
  if (threadIdx.x == 0)
    *info = sinfo = 0;  // initialise info to zero and cache

  /*
   * For efficient data reuse A needs to be cached in shared memory.  In order
   * to get maximum instruction throughput 32 threads are needed but this would
   * use 8192 bytes (32 * 32 * sizeof(double)) of shared memory to store A.
   * Triangular packed storage mode is therefore used to store only the
   * triangle of A being updated using 4224 bytes((32 * (32 + 1)) / 2 * sizeof(double))
   * of shared memory.
   * Since this is only ever going to be run using one thread block shared
   * memory and register use can be higher than when trying to fit multiple
   * thread blocks onto each multiprocessor.
   */
  __shared__ double a[(bx * (bx + 1)) / 2];
  __shared__ double ajj;

  if (uplo == CBlasUpper) {
    // Read upper triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < bx; j++) {
      if (threadIdx.x <= j)
        a[upper(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    // Perform the triangular inverse
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also neatly avoids
    // bank conflicts.
    for (int j = 0; j < n; j++) {
      double temp;
      // Read current column into registers
      if (threadIdx.x <= j)
        temp = a[upper(threadIdx.x, j)];

      // Thread j calculates the diagonal element
      if (threadIdx.x == j) {
        if (diag == CBlasNonUnit) {
          if (temp == 0.0) {
            *info = sinfo = j + 1;        // update info in shared and global memory
            break;
          }
          a[upper(threadIdx.x, j)] = 1.0 / temp;
          ajj = -a[upper(threadIdx.x, j)];
        }
        else
          ajj = -1.0;
      }

      __syncthreads();

      // If info != 0 return (matrix is singular)
      if (sinfo != 0)
        return;

      if (threadIdx.x < j) {
        if (diag == CBlasNonUnit)
          temp *= a[upper(threadIdx.x, threadIdx.x)];
        for (int k = threadIdx.x + 1; k < j; k++)
          temp += a[upper(threadIdx.x, k)] * a[upper(k, j)];
      }

      __syncthreads();

      if (threadIdx.x < j)
        a[upper(threadIdx.x, j)] = temp * ajj;

      __syncthreads();
    }

    // Write the upper triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x <= j)
        A[j * lda + threadIdx.x] = a[upper(threadIdx.x, j)];
    }
  }
  else {
    // Read lower triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < bx; j++) {
      if (threadIdx.x >= j)
        a[lower<bx>(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    // Perform the triangular inverse
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also neatly avoids
    // bank conflicts.
    for (int j = n - 1; j >= 0; j--) {
      double temp;
      // Read current column into registers
      if (threadIdx.x >= j)
        temp = a[lower<bx>(threadIdx.x, j)];

      // Thread j calculates the diagonal element
      if (threadIdx.x == j) {
        if (diag == CBlasNonUnit) {
          if (temp == 0.0) {
            *info = sinfo = j + 1;        // update info in shared and global memory
            break;
          }
          a[lower<bx>(threadIdx.x, j)] = 1.0 / temp;
          ajj = -a[lower<bx>(threadIdx.x, j)];
        }
        else
          ajj = -1.0;
      }

      __syncthreads();

      // If info != 0 return (matrix is singular)
      if (sinfo != 0)
        return;

      if (threadIdx.x > j) {
        if (diag == CBlasNonUnit)
          temp *= a[lower<bx>(threadIdx.x, threadIdx.x)];
        for (int k = j + 1; k < threadIdx.x; k++)
          temp += a[lower<bx>(threadIdx.x, k)] * a[lower<bx>(k, j)];
      }

      __syncthreads();

      if (threadIdx.x > j)
        a[lower<bx>(threadIdx.x, j)] = temp * ajj;

      __syncthreads();
    }

    // Write the lower triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        A[j * lda + threadIdx.x] = a[lower<bx>(threadIdx.x, j)];
    }
  }
}

template __global__ void dtrti2<CBlasUpper, CBlasUnit, 32>(double *, int *, int, int);
template __global__ void dtrti2<CBlasUpper, CBlasNonUnit, 32>(double *, int *, int, int);
template __global__ void dtrti2<CBlasLower, CBlasUnit, 32>(double *, int *, int, int);
template __global__ void dtrti2<CBlasLower, CBlasNonUnit, 32>(double *, int *, int, int);

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

extern "C" void dtrti2_(const char *, const char *, const int *, double *, const int *, int *);
static inline void dtrti2(CBlasUplo uplo, CBlasDiag diag, int n, double * A, int lda, int * info) {
  if (uplo == CBlasUpper) {
    if (diag == CBlasNonUnit)
      dtrti2<CBlasUpper, CBlasNonUnit, 32><<<1,32>>>(A, info, lda, n);
    else
      dtrti2<CBlasUpper, CBlasUnit, 32><<<1,32>>>(A, info, lda, n);
  }
  else {
    if (diag == CBlasNonUnit)
      dtrti2<CBlasLower, CBlasNonUnit, 32><<<1,32>>>(A, info, lda, n);
    else
      dtrti2<CBlasLower, CBlasUnit, 32><<<1,32>>>(A, info, lda, n);
  }
}

static int cond(int, double, double *, size_t);

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  CBlasDiag diag;
  int n;

  if (argc != 4) {
    fprintf(stderr, "Usage %s <uplo> <diag> <n>\n"
                    "where:\n"
                    "  <uplo>  is 'U' or 'u' for CBlasUpper or 'L' or 'l' for CBlasLower\n"
                    "  <diag>  is 'U' or 'u' for CBlasUnit or 'N' or 'n' for CBlasNonUnit\n"
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

  char d;
  if (sscanf(argv[2], "%c", &d) != 1) {
    fprintf(stderr, "Unable to parse character from '%s'\n", argv[2]);
    return 2;
  }
  switch (d) {
    case 'u': case 'U': diag = CBlasUnit; break;
    case 'n': case 'N': diag = CBlasNonUnit; break;
    default: fprintf(stderr, "Unknown diag '%c'\n", u); return 1;
  }

  if (sscanf(argv[3], "%d", &n) != 1) {
    fprintf(stderr, "Unable to parse integer from '%s'\n", argv[3]);
    return 3;
  }

  double * A, * dA, * refA;
  size_t lda = (n + 1u) & ~1u, dlda;
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
  dtrti2_((const char *)&uplo, (const char *)&diag, &n, refA, (const int *)&lda, &refInfo);
  dtrti2(uplo, diag, n, dA, dlda, dinfo);
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
