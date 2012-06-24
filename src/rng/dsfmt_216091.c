#define MEXP 216091

#define POS1	1890
#define SL1	23
#define MSK1	UINT64_C(0x000bf7df7fefcfff)
#define MSK2	UINT64_C(0x000e7ffffef737ff)
#define MSK32_1	0x000bf7df
#define MSK32_2	0x7fefcfff
#define MSK32_3	0x000e7fff
#define MSK32_4	0xfef737ff
#define FIX1	UINT64_C(0xd7f95a04764c27d7)
#define FIX2	UINT64_C(0x6a483861810bebc2)
#define PCV1	UINT64_C(0x3af0a8f3d5600000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^216091"
#define RNG_T dsfmt_216091_t

#include "dsfmt.c"
