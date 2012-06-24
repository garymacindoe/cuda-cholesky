#define MEXP 11213

#define POS1	37
#define SL1	19
#define MSK1	UINT64_C(0x000ffffffdf7fffd)
#define MSK2	UINT64_C(0x000dfffffff6bfff)
#define MSK32_1	0x000fffff
#define MSK32_2	0xfdf7fffd
#define MSK32_3	0x000dffff
#define MSK32_4	0xfff6bfff
#define FIX1	UINT64_C(0xd0ef7b7c75b06793)
#define FIX2	UINT64_C(0x9c50ff4caae0a641)
#define PCV1	UINT64_C(0x8234c51207c80000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^11213"
#define RNG_T dsfmt_11213_t

#include "dsfmt.c"
