#include <stdint.h>

__device__ __constant__ uint32_t pos_tbl[200];
__device__ __constant__ uint32_t sh1_tbl[200];
__device__ __constant__ uint32_t sh2_tbl[200];

#include "mtgp64.cu"

template void sample<176, 128, uint64_t, convert>            (mt_state<176> *, uint64_t *, size_t, size_t);
template void sample<176, 128,   double, convert_open_open  >(mt_state<176> *,   double *, size_t, size_t);
template void sample<176, 128,   double, convert_open_close >(mt_state<176> *,   double *, size_t, size_t);
template void sample<176, 128,   double, convert_close_open >(mt_state<176> *,   double *, size_t, size_t);
template void sample<176, 128,   double, convert_close_close>(mt_state<176> *,   double *, size_t, size_t);
