#include "matrix.cu"

template void      all<float>(float *, size_t, float, size_t, size_t);
template void identity<float>(float *, size_t, size_t, size_t);
template void           add<float>(float *, size_t, float *, size_t, size_t, size_t);
template void      addConst<float>(float *, size_t, float, size_t, size_t);
template void      subtract<float>(float *, size_t, float *, size_t, size_t, size_t);
template void subtractConst<float>(float *, size_t, float, size_t, size_t);
template void      multiply<float>(float *, size_t, float *, size_t, size_t, size_t);
template void multiplyConst<float>(float *, size_t, float, size_t, size_t);

// '/' on floats is not IEEE compliant, __fdiv_rn is.
template <> void      divide<float>(float * A, size_t lda, float * B, size_t ldb, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] = __fdiv_rn(A[j * lda + i], B[j * ldb + i]);
}

template <> void divideConst<float>(float * A, size_t lda, float a, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] = __fdiv_rn(A[j * lda + i], a);
}
