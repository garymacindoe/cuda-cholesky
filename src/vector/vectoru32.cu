#include "vector.cu"
#include <stdint.h>

template void   all<uint32_t>(uint32_t *, size_t, uint32_t, size_t);
template void basis<uint32_t>(uint32_t *, size_t, size_t, size_t);
template void           add<uint32_t>(uint32_t *, size_t, uint32_t *, size_t, size_t);
template void      addConst<uint32_t>(uint32_t *, size_t, uint32_t, size_t);
template void      subtract<uint32_t>(uint32_t *, size_t, uint32_t *, size_t, size_t);
template void subtractConst<uint32_t>(uint32_t *, size_t, uint32_t, size_t);
template void      multiply<uint32_t>(uint32_t *, size_t, uint32_t *, size_t, size_t);
template void multiplyConst<uint32_t>(uint32_t *, size_t, uint32_t, size_t);
template void        divide<uint32_t>(uint32_t *, size_t, uint32_t *, size_t, size_t);
template void   divideConst<uint32_t>(uint32_t *, size_t, uint32_t, size_t);
