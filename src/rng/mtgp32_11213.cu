#include <stdint.h>

__device__ __constant__ uint32_t pos_tbl[200];
__device__ __constant__ uint32_t sh1_tbl[200];
__device__ __constant__ uint32_t sh2_tbl[200];

#include "mtgp32.cu"

template void sample<351, 256, uint32_t, convert>            (mt_state<351> *, uint32_t *, size_t, size_t);
template void sample<351, 256,    float, convert_open_open  >(mt_state<351> *,    float *, size_t, size_t);
template void sample<351, 256,    float, convert_open_close >(mt_state<351> *,    float *, size_t, size_t);
template void sample<351, 256,    float, convert_close_open >(mt_state<351> *,    float *, size_t, size_t);
template void sample<351, 256,    float, convert_close_close>(mt_state<351> *,    float *, size_t, size_t);
