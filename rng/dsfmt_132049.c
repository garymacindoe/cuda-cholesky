#define MEXP 132049

#define POS1	371
#define SL1	23
#define MSK1	UINT64_C(0x000fb9f4eff4bf77)
#define MSK2	UINT64_C(0x000fffffbfefff37)
#define MSK32_1	0x000fb9f4
#define MSK32_2	0xeff4bf77
#define MSK32_3	0x000fffff
#define MSK32_4	0xbfefff37
#define FIX1	UINT64_C(0x4ce24c0e4e234f3b)
#define FIX2	UINT64_C(0x62612409b5665c2d)
#define PCV1	UINT64_C(0x181232889145d000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^132049"
#define RNG_T dsfmt_132049_t

#include "dsfmt.c"
