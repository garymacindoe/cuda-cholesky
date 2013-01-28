#define MEXP 86243

#define POS1	231
#define SL1	13
#define MSK1	UINT64_C(0x000ffedff6ffffdf)
#define MSK2	UINT64_C(0x000ffff7fdffff7e)
#define MSK32_1	0x000ffedf
#define MSK32_2	0xf6ffffdf
#define MSK32_3	0x000ffff7
#define MSK32_4	0xfdffff7e
#define FIX1	UINT64_C(0x1d553e776b975e68)
#define FIX2	UINT64_C(0x648faadf1416bf91)
#define PCV1	UINT64_C(0x5f2cd03e2758a373)
#define PCV2	UINT64_C(0xc0b7eb8410000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^86243"
#define RNG_T dsfmt_86243_t

#include "dsfmt.c"
