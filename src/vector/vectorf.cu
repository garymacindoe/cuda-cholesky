#include "vector.cu"

template void   all<float>(float *, size_t, float, size_t);
template void basis<float>(float *, size_t, size_t, size_t);
template void           add<float>(float *, size_t, float *, size_t, size_t);
template void      addConst<float>(float *, size_t, float, size_t);
template void      subtract<float>(float *, size_t, float *, size_t, size_t);
template void subtractConst<float>(float *, size_t, float, size_t);
template void      multiply<float>(float *, size_t, float *, size_t, size_t);
template void multiplyConst<float>(float *, size_t, float, size_t);

// '/' on floats is not IEEE compliant, __fdiv_rn is.
template <> void      divide<float>(float * x, size_t xinc, float * y, size_t yinc, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] = __fdiv_rn(x[i * xinc], y[i * yinc]);
}

template <> void divideConst<float>(float * x, size_t xinc, float a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i * xinc] = __fdiv_rn(x[i * xinc], a);
}