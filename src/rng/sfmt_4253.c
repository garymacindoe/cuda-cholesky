#define MEXP 4253

#define POS1	17
#define SL1	20
#define SL2	1
#define SR1	7
#define SR2	1
#define MSK1	0x9f7bffff
#define MSK2	0x9fffff5f
#define MSK3	0x3efffffb
#define MSK4	0xfffff7bb
#define PARITY1	0xa8000001
#define PARITY2	0xaf5390a3
#define PARITY3	0xb740b3f8
#define PARITY4	0x6c11486d

#define NAME "SIMD-oriented Fast Mersenne Twister 2^4253"
#define RNG_T sfmt_4253_t

#include "sfmt.c"
