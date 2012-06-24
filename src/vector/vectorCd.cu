#include "../cuComplex.cuh"
#include "vector.cu"

template void           all<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double>, size_t);
template void         basis<cuComplex<double> >(cuComplex<double> *, size_t, size_t, size_t);
template void           add<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double> *, size_t, size_t);
template void      addConst<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double>, size_t);
template void      subtract<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double> *, size_t, size_t);
template void subtractConst<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double>, size_t);
template void      multiply<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double> *, size_t, size_t);
template void multiplyConst<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double>, size_t);
template void        divide<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double> *, size_t, size_t);
template void   divideConst<cuComplex<double> >(cuComplex<double> *, size_t, cuComplex<double>, size_t);
