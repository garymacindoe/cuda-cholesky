#define MEXP 1279

#define POS1	9
#define SL1	19
#define MSK1	UINT64_C(0x000efff7ffddffee)
#define MSK2	UINT64_C(0x000fbffffff77fff)
#define MSK32_1	0x000efff7
#define MSK32_2	0xffddffee
#define MSK32_3	0x000fbfff
#define MSK32_4	0xfff77fff
#define FIX1	UINT64_C(0xb66627623d1a31be)
#define FIX2	UINT64_C(0x04b6c51147b6109b)
#define PCV1	UINT64_C(0x7049f2da382a6aeb)
#define PCV2	UINT64_C(0xde4ca84a40000001)

#define NAME "Double precision SIMD-oriented Fast Mersenne Twister 2^1279"
#define RNG_T dsfmt_1279_t

#include "dsfmt.c"
