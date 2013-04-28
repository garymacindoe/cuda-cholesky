#include "../blas/sgemm.cu"

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

template <CBlasUplo uplo, unsigned int nb, unsigned int bx>
__device__ void splau2(int n, float * A) {
  if (uplo == CBlasUpper) {
    const int i = threadIdx.y * bx + threadIdx.x;
    // Perform the upper triangular multiply
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also avoids bank
    // conflicts.
    for (int j = 0; j < n; j++) {
      if (i <= j) {
        float temp = A[upper(i, j)] * A[upper(j, j)];
        for (int k = j + 1; k < n; k++)
          temp += A[upper(i, k)] * A[upper(j, k)];
        A[upper(i, j)] = temp;
      }
      __syncthreads();
    }
  }
  else {
    const int j = threadIdx.y * bx + threadIdx.x;
    // Perform the lower triangular multiply
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also avoids bank
    // conflicts.
    for (int i = 0; i < n; i++) {
      if (i >= j) {
        float temp = A[lower<bx>(i, j)] * A[lower<bx>(i, i)];
        for (int k = i + 1; k < n; k++)
          temp += A[lower<bx>(k, i)] * A[lower<bx>(k, j)];
        A[lower<bx>(i, j)] = temp;
      }
      __syncthreads();
    }
  }
}

template <CBlasUplo uplo, unsigned int bx>
__global__ void slauu2(float * A, int lda, int n) {
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
    for (int j = 0; j < n; j++) {
      if (threadIdx.x <= j)
        a[upper(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    splau2<CBlasUpper, bx, bx>(n, a);

    // Write upper triangle of A into global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x <= j)
        A[j * lda + threadIdx.x] = a[upper(threadIdx.x, j)];
    }

  }
  else {
    // Read lower triangle of A into shared memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        a[lower<bx>(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    splau2<CBlasLower, bx, bx>(n, a);

    // Write the lower triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        A[j * lda + threadIdx.x] = a[lower<bx>(threadIdx.x, j)];
    }
  }
}

template <CBlasUplo uplo,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void slaumm2(float * __restrict__ A, float * __restrict__ B,
                        int lda, int ldb, int j, int jb, int n) {
  if (uplo == CBlasUpper) {
    if (blockIdx.x == gridDim.x - 1) {
      if (blockIdx.y == 0) {
        __shared__ float a[(nb * (nb + 1)) / 2];

        const int i = threadIdx.y * bx + threadIdx.x;

        // Read upper triangle of A into shared memory
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            a[upper(i, k)] = A[k * lda + j + i];
        }

        __syncthreads();

        // Perform the matrix square using the packed device function
        splau2<CBlasUpper, nb, bx>(jb, a);

        // Write the upper triangle of A back to global memory
        for (int k = 0; k < jb; k++) {
          if (i <= k)
            A[k * lda + j + i] = a[upper(i, k)];
        }
      }
    }
    else
      sgemm2<CBlasNoTrans, CBlasTrans, mb, nb, kb, bx, by>(j, jb, n - j - jb,
              1.0f, &A[jb * lda], lda, &A[jb * lda + j], lda,
              1.0f, B, ldb, A, lda);
  }
  else {
    if (blockIdx.y == gridDim.y - 1) {
      if (blockIdx.x == 0) {
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
          // Perform the triangular multiply using the packed device function
          splau2<CBlasLower, nb, bx>(jb, a);

          // Write the lower triangle of A back to global memory
          for (int k = 0; k < jb; k++) {
            if (i >= k)
              A[(j + k) * lda + i] = a[lower<nb>(i, k)];
          }
        }
      }
    }
    else
      sgemm2<CBlasTrans, CBlasNoTrans, mb, nb, kb, bx, by>(jb, j, n - j - jb,
              1.0f, &A[j * lda + jb], lda, &A[jb], lda,
              1.0f, B, ldb, A, lda);
  }
}

template __global__ void slauu2<CBlasUpper, 64>(float *, int, int);
template __global__ void slauu2<CBlasLower, 64>(float *, int, int);

template __global__ void slaumm2<CBlasUpper, 64, 16, 16, 16, 4>(float * __restrict__, float * __restrict__, int, int, int, int, int);
template __global__ void slaumm2<CBlasLower, 32, 32,  8,  8, 8>(float * __restrict__, float * __restrict__, int, int, int, int, int);

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

extern "C" void slauu2_(const char *, const int *, float *, const int *, int *);
static inline void slauu2(CBlasUplo uplo, int n, float * A, int lda) {
  if (uplo == CBlasUpper)
    slauu2<CBlasUpper, 64><<<1,64>>>(A, lda, n);
  else
    slauu2<CBlasLower, 64><<<1,64>>>(A, lda, n);
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
  size_t lda = (n + 3u) & ~3u, dlda;
  if ((A = (float *)malloc(lda * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Failed to allocate A\n");
    return -1;
  }
  if ((refA = (float *)malloc(lda * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Failed to allocate refA\n");
    return -2;
  }
  CUDA_ERROR_CHECK(cudaMallocPitch((void **)&dA, &dlda, n * sizeof(float), n));
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

  int refInfo;
  slauu2_((const char *)&uplo, &n, refA, (const int *)&lda, &refInfo);
  slauu2(uplo, n, dA, dlda);
  CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(float), dA, dlda * sizeof(float), n * sizeof(float), n, cudaMemcpyDeviceToHost));

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

  fprintf(stdout, "refInfo = %d, Error = %6.3e\n", refInfo, error);

  free(A);
  free(refA);
  CUDA_ERROR_CHECK(cudaFree(dA));

  return refInfo;
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

extern "C" void sgemm_(const char *, const char *, const int *, const int *, const int *,
                       const float *, const float *, const int *, const float *, const int *,
                       const float *, float *, const int *);
extern "C" void slauu2_(const char *, const int *, float *, const int *);

static inline void slaumm2(CBlasUplo uplo, int j, int jb, int n, float * A, int lda, float * B, int ldb) {
  if (uplo == CBlasUpper) {
    const unsigned int mb = 32;
    const unsigned int nb = 32;
    const unsigned int kb =  8;
    const unsigned int bx =  8;
    const unsigned int by =  8;

    if (jb > mb) {
      fputs("On entry to slaumm2 parameter 3 had an invalid value\n", stderr);
      return;
    }

    const unsigned int gx = (jb + nb - 1) / nb;
    const unsigned int gy = (n - j - jb + nb - 1) / nb;

    slaumm2<CBlasUpper, mb, nb, kb, bx, by><<<dim3(max(gx, 1), gy + 1), dim3(bx, by)>>>(A, B, lda, ldb, j, jb, n);
  }
  else {
    const unsigned int mb = 64;
    const unsigned int nb = 16;
    const unsigned int kb = 16;
    const unsigned int bx = 16;
    const unsigned int by =  4;

    if (jb > nb) {
      fputs("On entry to slaumm2 parameter 3 had an invalid value\n", stderr);
      return;
    }

    const unsigned int gx = (n - j - jb + mb - 1) / mb;
    const unsigned int gy = (jb + nb - 1) / nb;

    slaumm2<CBlasLower, mb, nb, kb, bx, by><<<dim3(gx + 1, max(gy, 1)), dim3(bx, by)>>>(A, B, lda, ldb, j, jb, n);
  }
}

static void slaumm2_(CBlasUplo uplo, int j, int jb, int n, float * A, int lda, float * B, int ldb) {
  const float mone = -1.0f, one = 1.0f;
  const int n_j_jb = n - j - jb;
  if (uplo == CBlasUpper) {
    slauu2_("Upper", &jb, &A[j], &lda);
    sgemm_("No Transpose", "Transpose", &j, &jb, &n_j_jb, &one, &A[jb * lda + j], &lda, &A[jb * lda], &lda, &one, A, &lda);
  }
  else {
    slauu2_("Lower", &jb, &A[j * lda], &lda);
    sgemm_("Transpose", "No Transpose", &jb, &j, &n_j_jb, &one, &A[j * lda + jb], &lda, &A[jb], &lda, &one, A, &lda);
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

#ifdef PRINT
    for (int i = 0; i < j + jb; i++) {
      for (int k = 0; k < n - j; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
#endif
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

#ifdef PRINT
    for (int i = 0; i < n - j; i++) {
      for (int k = 0; k < j + jb; k++)
        fprintf(stderr, "%15.6f", A[k * lda + i]);
      fprintf(stderr, "\n");
    }
#endif
  }

  int info = 0, refInfo = 0;
  spotfimm2(uplo, j, jb, n, dA, dlda, dB, dldb, dinfo);
  spotfimm2_(uplo, j, jb, n, refA, lda, refB, ldb, &refInfo);
  CUDA_ERROR_CHECK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));

  float error = 0.0f;
  if (uplo == CBlasUpper) {
    CUDA_ERROR_CHECK(cudaMemcpy2D(A, lda * sizeof(float), dA, dlda * sizeof(float), (j + jb) * sizeof(float), n - j, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy2D(B, ldb * sizeof(float), dB, dldb * sizeof(float), jb * sizeof(float), n - j, cudaMemcpyDeviceToHost));

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
        float diff = fabsf(A[k * lda + i] - refA[k * lda + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = 0; i <= k; i++) {
        float diff = fabsf(B[k * ldb + i] - refB[k * ldb + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = jb; k < n - j - jb; k++) {
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
        float diff = fabsf(A[k * lda + i] - refA[k * lda + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = k; i < jb; i++) {
        float diff = fabsf(B[k * ldb + i] - refB[k * ldb + i]);
        if (diff > error)
          error = diff;
      }
    }

    for (int k = 0; k < jb; k++) {
      for (int i = 0; i < n - j - jb; i++) {
        float diff = fabsf(B[k * ldb + jb + i] - refB[k * ldb + jb + i]);
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
