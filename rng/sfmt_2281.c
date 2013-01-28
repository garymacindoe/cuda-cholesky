#define MEXP 2281

#define POS1	12
#define SL1	19
#define SL2	1
#define SR1	5
#define SR2	1
#define MSK1	0xbff7ffbf
#define MSK2	0xfdfffffe
#define MSK3	0xf7ffef7f
#define MSK4	0xf2f7cbbf
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0x00000000
#define PARITY4	0x41dfa600

#define NAME "SIMD-oriented Fast Mersenne Twister 2^2281"
#define RNG_T sfmt_2281_t

#include "sfmt.c"
