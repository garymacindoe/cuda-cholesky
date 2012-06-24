// nvcc -I../../include -O2 -arch=compute_13 -code=sm_13 -use_fast_math -Xptxas=-v -maxrregcount=32 -cubin spotrf.cu
#include "blas.h"

template <unsigned int bs>
__device__ float sdot(int ti, int n, const float * x, const float * y) {
  __shared__ float temp[bs];

  float res = 0.0f;

  for (int i = ti; i < n; i += bs * 2) {
    res += x[i] * y[i];
    if (i + bs < n)
      res += x[i + bs] * y[i + bs];
  }

  temp[ti] = res;
  __syncthreads();

  if (bs >= 512) { if (ti < 256) { temp[ti] = res = res + temp[ti + 256]; } __syncthreads(); }
  if (bs >= 256) { if (ti < 128) { temp[ti] = res = res + temp[ti + 128]; } __syncthreads(); }
  if (bs >= 128) { if (ti <  64) { temp[ti] = res = res + temp[ti +  64]; } __syncthreads(); }

  if (ti < 32) {
    volatile float * vtemp = temp;
    if (bs >= 64) { vtemp[ti] = res = res + vtemp[ti + 32]; }
    if (bs >= 32) { vtemp[ti] = res = res + vtemp[ti + 16]; }
    if (bs >= 16) { vtemp[ti] = res = res + vtemp[ti +  8]; }
    if (bs >=  8) { vtemp[ti] = res = res + vtemp[ti +  4]; }
    if (bs >=  4) { vtemp[ti] = res = res + vtemp[ti +  2]; }
    if (bs >=  2) { vtemp[ti] = res = res + vtemp[ti +  1]; }
  }

  return res;
}

template <CBlasUplo uplo, unsigned int bx, unsigned int by>
__global__ void spotf2(int n, float * A, int lda, int * info) {
  const int ti = threadIdx.y * bx + threadIdx.x;

  __shared__ int s_info;
  if (ti == 0)
    s_info = 0;

  if (uplo == CBlasUpper) {
    for (int i = 0; i < n; i++) {
      float temp = sdot<bx * by>(ti, i, &A[i * lda], &A[i * lda]);

      float aii;
      if (ti == 0) {
        temp = A[i * lda + i] - temp;
        if (temp <= 0.0f || isnan(temp)) {
          A[i * lda + i] = temp;
          *info = s_info = i;
        }
        else
          A[i * lda + i] = aii = sqrtf(temp);
      }

      __syncthreads();

      if (s_info != 0)
        return;

      for (int j = i + 1; j < n; j++) {
        temp = sdot<bx * by>(ti, i, &A[i * lda], &A[j * lda]);
        if (ti == 0)
          A[j * lda + i] = (A[j * lda + i] - temp) / aii;
      }

      __syncthreads();
    }
  }
  else {
    __shared__ float ajj;
    for (int j = 0; j < n; j++) {
      if (j + ti < n) {
        float temp = A[j * lda + j + ti];
        for (int k = 0; k < j; k++)
          temp -= A[k * lda + j] * A[k * lda + j + ti];

        if (ti == 0) {
          if (temp <= 0.0f || isnan(temp)) {
            A[j * lda + j] = temp;
            *info = s_info = j;
          }
          else
            A[j * lda + j] = ajj = sqrtf(temp);
        }

        __syncthreads();

        if (s_info != 0)
          return;

        if (ti > 0)
          A[j * lda + j + ti] = temp / ajj;
      }

      for (int i = j + bx * by + ti; i < n; i += bx * by) {
        float temp = A[j * lda + i];
        for (int k = 0; k < j; k++)
          temp -= A[k * lda + j] * A[k * lda + i];
        A[j * lda + i] = temp / ajj;
      }

      __syncthreads();
    }
  }
}

template void spotf2<CBlasUpper,  8, 8>(int, float *, int, int *);
template void spotf2<CBlasLower, 16, 4>(int, float *, int, int *);
