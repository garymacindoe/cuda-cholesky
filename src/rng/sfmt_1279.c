#define MEXP 1279

#define POS1	7
#define SL1	14
#define SL2	3
#define SR1	5
#define SR2	1
#define MSK1	0xf7fefffd
#define MSK2	0x7fefcfff
#define MSK3	0xaff3ef3f
#define MSK4	0xb5ffff7f
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0x00000000
#define PARITY4	0x20000000

#define NAME "SIMD-oriented Fast Mersenne Twister 2^1279"
#define RNG_T sfmt_1279_t

#include "sfmt.c"
