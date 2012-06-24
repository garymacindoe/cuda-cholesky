// nvcc -I../../include -O2 -arch=compute_13 -code=sm_13 -use_fast_math -Xptxas=-v -maxrregcount=32 -cubin cpotrf.cu
#include "blas.h"
#include <cuComplex.h>

template <unsigned int bs>
__device__ cuComplex cdotc(int ti, int n, const cuComplex * x, const cuComplex * y) {
  __shared__ float temp_real[bs], temp_imag[bs];

  cuComplex res = make_cuComplex(0.0f, 0.0f);

  for (int i = ti; i < n; i += bs * 2) {
    res = cuCfmaf(cuConjf(x[i]), y[i], res);
    if (i + bs < n)
      res = cuCfmaf(cuConjf(x[i + bs]), y[i + bs], res);
  }

  temp_real[ti] = cuCrealf(res);
  temp_imag[ti] = cuCimagf(res);
  __syncthreads();

  if (bs >= 512) { if (ti < 256) res = make_cuComplex(temp_real[ti] = cuCrealf(res) + temp_real[ti + 256], temp_imag[ti] = cuCimagf(res) + temp_imag[ti + 256]); __syncthreads(); }
  if (bs >= 256) { if (ti < 128) res = make_cuComplex(temp_real[ti] = cuCrealf(res) + temp_real[ti + 128], temp_imag[ti] = cuCimagf(res) + temp_imag[ti + 128]); __syncthreads(); }
  if (bs >= 128) { if (ti <  64) res = make_cuComplex(temp_real[ti] = cuCrealf(res) + temp_real[ti +  64], temp_imag[ti] = cuCimagf(res) + temp_imag[ti +  64]); __syncthreads(); }

  if (ti < 32) {
    volatile float * vtemp_real = temp_real;
    volatile float * vtemp_imag = temp_imag;
    if (bs >= 64) { res = make_cuComplex(vtemp_real[ti] = cuCrealf(res) + vtemp_real[ti + 32], vtemp_imag[ti] = cuCimagf(res) + vtemp_imag[ti + 32]); }
    if (bs >= 32) { res = make_cuComplex(vtemp_real[ti] = cuCrealf(res) + vtemp_real[ti + 16], vtemp_imag[ti] = cuCimagf(res) + vtemp_imag[ti + 16]); }
    if (bs >= 16) { res = make_cuComplex(vtemp_real[ti] = cuCrealf(res) + vtemp_real[ti +  8], vtemp_imag[ti] = cuCimagf(res) + vtemp_imag[ti +  8]); }
    if (bs >=  8) { res = make_cuComplex(vtemp_real[ti] = cuCrealf(res) + vtemp_real[ti +  4], vtemp_imag[ti] = cuCimagf(res) + vtemp_imag[ti +  4]); }
    if (bs >=  4) { res = make_cuComplex(vtemp_real[ti] = cuCrealf(res) + vtemp_real[ti +  2], vtemp_imag[ti] = cuCimagf(res) + vtemp_imag[ti +  2]); }
    if (bs >=  2) { res = make_cuComplex(vtemp_real[ti] = cuCrealf(res) + vtemp_real[ti +  1], vtemp_imag[ti] = cuCimagf(res) + vtemp_imag[ti +  1]); }
  }

  return res;
}

template <CBlasUplo uplo, unsigned int bx, unsigned int by>
__global__ void cpotf2(int n, cuComplex * A, int lda, int * info) {
  const int ti = threadIdx.y * bx + threadIdx.x;

  __shared__ int s_info;
  if (ti == 0)
    s_info = 0;

  if (uplo == CBlasUpper) {
    for (int i = 0; i < n; i++) {
      cuComplex temp = cdotc<bx * by>(ti, i, &A[i * lda], &A[i * lda]);

      float aii;
      if (ti == 0) {
        aii = cuCrealf(A[i * lda + i]) - cuCrealf(temp);
        if (aii <= 0.0f || isnan(aii)) {
          A[i * lda + i] = temp;
          *info = s_info = i;
        }
        else
          A[i * lda + i] = make_cuComplex(aii = sqrtf(aii), 0.0f);
      }

      __syncthreads();

      if (s_info != 0)
        return;

      for (int j = i + 1; j < n; j++) {
        temp = cdotc<bx * by>(ti, i, &A[i * lda], &A[j * lda]);
        if (ti == 0)
          A[j * lda + i] = make_cuComplex((cuCrealf(A[j * lda + i]) - cuCrealf(temp)) / aii, (cuCimagf(A[j * lda + i]) - cuCimagf(temp)) / aii);
      }

      __syncthreads();
    }
  }
  else {
    __shared__ float ajj;
    for (int j = 0; j < n; j++) {
      if (j + ti < n) {
        cuComplex temp = A[j * lda + j + ti];
        for (int k = 0; k < j; k++)
          temp = cuCsubf(temp, cuCmulf(cuConjf(A[k * lda + j]), A[k * lda + j + ti]));

        if (ti == 0) {
          if (cuCrealf(temp) <= 0.0f || isnan(cuCrealf(temp))) {
            A[j * lda + j] = temp;
            *info = s_info = j;
          }
          else
            A[j * lda + j] = make_cuComplex(ajj = sqrtf(cuCrealf(temp)), 0.0f);
        }

        __syncthreads();

        if (s_info != 0)
          return;

        if (ti > 0)
          A[j * lda + j + ti] = make_cuComplex(cuCrealf(temp) / ajj, cuCimagf(temp) / ajj);
      }

      for (int i = j + bx * by + ti; i < n; i += bx * by) {
        cuComplex temp = A[j * lda + i];
        for (int k = 0; k < j; k++)
          temp = cuCsubf(temp, cuCmulf(cuConjf(A[k * lda + j]), A[k * lda + i]));
        A[j * lda + i] = make_cuComplex(cuCrealf(temp) / ajj, cuCimagf(temp) / ajj);
      }

      __syncthreads();
    }
  }
}

template void cpotf2<CBlasUpper,  8, 8>(int, cuComplex *, int, int *);
template void cpotf2<CBlasLower, 16, 4>(int, cuComplex *, int, int *);
