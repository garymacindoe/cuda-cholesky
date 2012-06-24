// nvcc -I../../include -O2 -arch=compute_13 -code=sm_13 -use_fast_math -Xptxas=-v -maxrregcount=32 -cubin zpotrf.cu
#include "blas.h"
#include <cuComplex.h>

template <unsigned int bs>
__device__ cuDoubleComplex zdotc(int ti, int n, const cuDoubleComplex * x, const cuDoubleComplex * y) {
  __shared__ double temp_real[bs], temp_imag[bs];

  cuDoubleComplex res = make_cuDoubleComplex(0.0, 0.0);

  for (int i = ti; i < n; i += bs * 2) {
    res = cuCfma(cuConj(x[i]), y[i], res);
    if (i + bs < n)
      res = cuCfma(cuConj(x[i + bs]), y[i + bs], res);
  }

  temp_real[ti] = cuCreal(res);
  temp_imag[ti] = cuCimag(res);
  __syncthreads();

  if (bs >= 512) { if (ti < 256) res = make_cuDoubleComplex(temp_real[ti] = cuCreal(res) + temp_real[ti + 256], temp_imag[ti] = cuCimag(res) + temp_imag[ti + 256]); __syncthreads(); }
  if (bs >= 256) { if (ti < 128) res = make_cuDoubleComplex(temp_real[ti] = cuCreal(res) + temp_real[ti + 128], temp_imag[ti] = cuCimag(res) + temp_imag[ti + 128]); __syncthreads(); }
  if (bs >= 128) { if (ti <  64) res = make_cuDoubleComplex(temp_real[ti] = cuCreal(res) + temp_real[ti +  64], temp_imag[ti] = cuCimag(res) + temp_imag[ti +  64]); __syncthreads(); }

  if (ti < 32) {
    volatile double * vtemp_real = temp_real;
    volatile double * vtemp_imag = temp_imag;
    if (bs >= 64) { res = make_cuDoubleComplex(vtemp_real[ti] = cuCreal(res) + vtemp_real[ti + 32], vtemp_imag[ti] = cuCimag(res) + vtemp_imag[ti + 32]); }
    if (bs >= 32) { res = make_cuDoubleComplex(vtemp_real[ti] = cuCreal(res) + vtemp_real[ti + 16], vtemp_imag[ti] = cuCimag(res) + vtemp_imag[ti + 16]); }
    if (bs >= 16) { res = make_cuDoubleComplex(vtemp_real[ti] = cuCreal(res) + vtemp_real[ti +  8], vtemp_imag[ti] = cuCimag(res) + vtemp_imag[ti +  8]); }
    if (bs >=  8) { res = make_cuDoubleComplex(vtemp_real[ti] = cuCreal(res) + vtemp_real[ti +  4], vtemp_imag[ti] = cuCimag(res) + vtemp_imag[ti +  4]); }
    if (bs >=  4) { res = make_cuDoubleComplex(vtemp_real[ti] = cuCreal(res) + vtemp_real[ti +  2], vtemp_imag[ti] = cuCimag(res) + vtemp_imag[ti +  2]); }
    if (bs >=  2) { res = make_cuDoubleComplex(vtemp_real[ti] = cuCreal(res) + vtemp_real[ti +  1], vtemp_imag[ti] = cuCimag(res) + vtemp_imag[ti +  1]); }
  }

  return res;
}

template <CBlasUplo uplo, unsigned int bx, unsigned int by>
__global__ void zpotf2(int n, cuDoubleComplex * A, int lda, int * info) {
  const int ti = threadIdx.y * bx + threadIdx.x;

  __shared__ int s_info;
  if (ti == 0)
    s_info = 0;

  if (uplo == CBlasUpper) {
    for (int i = 0; i < n; i++) {
      cuDoubleComplex temp = zdotc<bx * by>(ti, i, &A[i * lda], &A[i * lda]);

      double aii;
      if (ti == 0) {
        aii = cuCreal(A[i * lda + i]) - cuCreal(temp);
        if (aii <= 0.0 || isnan(aii)) {
          A[i * lda + i] = temp;
          *info = s_info = i;
        }
        else
          A[i * lda + i] = make_cuDoubleComplex(aii = sqrt(aii), 0.0);
      }

      __syncthreads();

      if (s_info != 0)
        return;

      for (int j = i + 1; j < n; j++) {
        temp = zdotc<bx * by>(ti, i, &A[i * lda], &A[j * lda]);
        if (ti == 0)
          A[j * lda + i] = make_cuDoubleComplex((cuCreal(A[j * lda + i]) - cuCreal(temp)) / aii, (cuCimag(A[j * lda + i]) - cuCimag(temp)) / aii);
      }

      __syncthreads();
    }
  }
  else {
    __shared__ double ajj;
    for (int j = 0; j < n; j++) {
      if (j + ti < n) {
        cuDoubleComplex temp = A[j * lda + j + ti];
        for (int k = 0; k < j; k++)
          temp = cuCsub(temp, cuCmul(cuConj(A[k * lda + j]), A[k * lda + j + ti]));

        if (ti == 0) {
          if (cuCreal(temp) <= 0.0 || isnan(cuCreal(temp))) {
            A[j * lda + j] = temp;
            *info = s_info = j;
          }
          else
            A[j * lda + j] = make_cuDoubleComplex(ajj = sqrt(cuCreal(temp)), 0.0);
        }

        __syncthreads();

        if (s_info != 0)
          return;

        if (ti > 0)
          A[j * lda + j + ti] = make_cuDoubleComplex(cuCreal(temp) / ajj, cuCimag(temp) / ajj);
      }

      for (int i = j + bx * by + ti; i < n; i += bx * by) {
        cuDoubleComplex temp = A[j * lda + i];
        for (int k = 0; k < j; k++)
          temp = cuCsub(temp, cuCmul(cuConj(A[k * lda + j]), A[k * lda + i]));
        A[j * lda + i] = make_cuDoubleComplex(cuCreal(temp) / ajj, cuCimag(temp) / ajj);
      }

      __syncthreads();
    }
  }
}

template void zpotf2<CBlasUpper,  8, 8>(int, cuDoubleComplex *, int, int *);
template void zpotf2<CBlasLower, 16, 4>(int, cuDoubleComplex *, int, int *);
