#define MEXP 2203

#define POS1	7
#define SL1	19
#define MSK1	UINT64_C(0x000fdffff5edbfff)
#define MSK2	UINT64_C(0x000f77fffffffbfe)
#define MSK32_1	0x000fdfff
#define MSK32_2	0xf5edbfff
#define MSK32_3	0x000f77ff
#define MSK32_4	0xfffffbfe
#define FIX1	UINT64_C(0xb14e907a39338485)
#define FIX2	UINT64_C(0xf98f0735c637ef90)
#define PCV1	UINT64_C(0x8000000000000000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^2203"
#define RNG_T dsfmt_2203_t

#include "dsfmt.c"
