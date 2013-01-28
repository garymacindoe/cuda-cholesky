#define MEXP 521

#define POS1	3
#define SL1	25
#define MSK1	UINT64_C(0x000fbfefff77efff)
#define MSK2	UINT64_C(0x000ffeebfbdfbfdf)
#define MSK32_1	0x000fbfef
#define MSK32_2	0xff77efff
#define MSK32_3	0x000ffeeb
#define MSK32_4	0xfbdfbfdf
#define FIX1	UINT64_C(0xcfb393d661638469)
#define FIX2	UINT64_C(0xc166867883ae2adb)
#define PCV1	UINT64_C(0xccaa588000000000)
#define PCV2	UINT64_C(0x0000000000000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^521"
#define RNG_T dsfmt_521_t

#include "dsfmt.c"
