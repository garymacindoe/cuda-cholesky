#include "vector.cu"
#include <stdint.h>

template void   all<uint64_t>(uint64_t *, size_t, uint64_t, size_t);
template void basis<uint64_t>(uint64_t *, size_t, size_t, size_t);
template void           add<uint64_t>(uint64_t *, size_t, uint64_t *, size_t, size_t);
template void      addConst<uint64_t>(uint64_t *, size_t, uint64_t, size_t);
template void      subtract<uint64_t>(uint64_t *, size_t, uint64_t *, size_t, size_t);
template void subtractConst<uint64_t>(uint64_t *, size_t, uint64_t, size_t);
template void      multiply<uint64_t>(uint64_t *, size_t, uint64_t *, size_t, size_t);
template void multiplyConst<uint64_t>(uint64_t *, size_t, uint64_t, size_t);
template void        divide<uint64_t>(uint64_t *, size_t, uint64_t *, size_t, size_t);
template void   divideConst<uint64_t>(uint64_t *, size_t, uint64_t, size_t);
