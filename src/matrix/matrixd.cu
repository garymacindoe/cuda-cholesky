#include "matrix.cu"

template void      all<double>(double *, size_t, double, size_t, size_t);
template void identity<double>(double *, size_t, size_t, size_t);
template void           add<double>(double *, size_t, double *, size_t, size_t, size_t);
template void      addConst<double>(double *, size_t, double, size_t, size_t);
template void      subtract<double>(double *, size_t, double *, size_t, size_t, size_t);
template void subtractConst<double>(double *, size_t, double, size_t, size_t);
template void      multiply<double>(double *, size_t, double *, size_t, size_t, size_t);
template void multiplyConst<double>(double *, size_t, double, size_t, size_t);
template void        divide<double>(double *, size_t, double *, size_t, size_t, size_t);
template void   divideConst<double>(double *, size_t, double, size_t, size_t);
