#define MEXP 44497

#define POS1	330
#define SL1	5
#define SL2	3
#define SR1	9
#define SR2	3
#define MSK1	0xeffffffb
#define MSK2	0xdfbebfff
#define MSK3	0xbfbf7bef
#define MSK4	0x9ffd7bff
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0xa3ac4000
#define PARITY4	0xecc1327a

#define NAME "SIMD-oriented Fast Mersenne Twister 2^44497"
#define RNG_T sfmt_44497_t

#include "sfmt.c"
