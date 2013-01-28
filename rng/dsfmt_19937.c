#define MEXP 19937

#define POS1	117
#define SL1	19
#define MSK1	UINT64_C(0x000ffafffffffb3f)
#define MSK2	UINT64_C(0x000ffdfffc90fffd)
#define MSK32_1	0x000ffaff
#define MSK32_2	0xfffffb3f
#define MSK32_3	0x000ffdff
#define MSK32_4	0xfc90fffd
#define FIX1	UINT64_C(0x90014964b32f4329)
#define FIX2	UINT64_C(0x3b8d12ac548a7c7a)
#define PCV1	UINT64_C(0x3d84e1ac0dc82880)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^19937"
#define RNG_T dsfmt_19937_t

#include "dsfmt.c"
