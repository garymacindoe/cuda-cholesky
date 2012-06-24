#include "matrix.cu"
#include "../cuComplex.cuh"

template void      all<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float>, size_t, size_t);
template void identity<cuComplex<float> >(cuComplex<float> *, size_t, size_t, size_t);
template void           add<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float> *, size_t, size_t, size_t);
template void      addConst<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float>, size_t, size_t);
template void      subtract<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float> *, size_t, size_t, size_t);
template void subtractConst<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float>, size_t, size_t);
template void      multiply<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float> *, size_t, size_t, size_t);
template void multiplyConst<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float>, size_t, size_t);
template void        divide<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float> *, size_t, size_t, size_t);
template void   divideConst<cuComplex<float> >(cuComplex<float> *, size_t, cuComplex<float>, size_t, size_t);
