#define MEXP 607

#define POS1	2
#define SL1	15
#define SL2	3
#define SR1	13
#define SR2	3
#define MSK1	0xfdff37ff
#define MSK2	0xef7f3f7d
#define MSK3	0xff777b7d
#define MSK4	0x7ff7fb2f
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0x00000000
#define PARITY4	0x5986f054

#define NAME "SIMD-oriented Fast Mersenne Twister 2^607"
#define RNG_T sfmt_607_t

#include "sfmt.c"
