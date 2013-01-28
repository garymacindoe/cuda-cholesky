#define MEXP 132049

#define POS1	110
#define SL1	19
#define SL2	1
#define SR1	21
#define SR2	1
#define MSK1	0xffffbb5f
#define MSK2	0xfb6ebf95
#define MSK3	0xfffefffa
#define MSK4	0xcff77fff
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0xcb520000
#define PARITY4	0xc7e91c7d

#define NAME "SIMD-oriented Fast Mersenne Twister 2^132049"
#define RNG_T sfmt_132049_t

#include "sfmt.c"
