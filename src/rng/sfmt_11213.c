#define MEXP 11213

#define POS1	68
#define SL1	14
#define SL2	3
#define SR1	7
#define SR2	3
#define MSK1	0xeffff7fb
#define MSK2	0xffffffef
#define MSK3	0xdfdfbfff
#define MSK4	0x7fffdbfd
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0xe8148000
#define PARITY4	0xd0c7afa3

#define NAME "SIMD-oriented Fast Mersenne Twister 2^11213"
#define RNG_T sfmt_11213_t

#include "sfmt.c"
