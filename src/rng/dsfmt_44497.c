#define MEXP 44497

#define POS1	304
#define SL1	19
#define MSK1	UINT64_C(0x000ff6dfffffffef)
#define MSK2	UINT64_C(0x0007ffdddeefff6f)
#define MSK32_1	0x000ff6df
#define MSK32_2	0xffffffef
#define MSK32_3	0x0007ffdd
#define MSK32_4	0xdeefff6f
#define FIX1	UINT64_C(0x75d910f235f6e10e)
#define FIX2	UINT64_C(0x7b32158aedc8e969)
#define PCV1	UINT64_C(0x4c3356b2a0000000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^44497"
#define RNG_T dsfmt_44497_t

#include "dsfmt.c"
