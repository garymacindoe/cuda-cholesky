#define MEXP 4253

#define POS1	19
#define SL1	19
#define MSK1	UINT64_C(0x0007b7fffef5feff)
#define MSK2	UINT64_C(0x000ffdffeffefbfc)
#define MSK32_1	0x0007b7ff
#define MSK32_2	0xfef5feff
#define MSK32_3	0x000ffdff
#define MSK32_4	0xeffefbfc
#define FIX1	UINT64_C(0x80901b5fd7a11c65)
#define FIX2	UINT64_C(0x5a63ff0e7cb0ba74)
#define PCV1	UINT64_C(0x1ad277be12000000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^4253"
#define RNG_T dsfmt_4253_t

#include "dsfmt.c"
