#define MEXP 19937

#define POS1	122
#define SL1	18
#define SL2	1
#define SR1	11
#define SR2	1
#define MSK1	0xdfffffef
#define MSK2	0xddfecb7f
#define MSK3	0xbffaffff
#define MSK4	0xbffffff6
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0x00000000
#define PARITY4	0x13c9e684

#define NAME "SIMD-oriented Fast Mersenne Twister 2^19937"
#define RNG_T sfmt_19937_t

#include "sfmt.c"
