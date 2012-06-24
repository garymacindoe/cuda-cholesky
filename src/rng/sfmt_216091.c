#define MEXP 216091

#define POS1	627
#define SL1	11
#define SL2	3
#define SR1	10
#define SR2	1
#define MSK1	0xbff7bff7
#define MSK2	0xbfffffff
#define MSK3	0xbffffa7f
#define MSK4	0xffddfbfb
#define PARITY1	0xf8000001
#define PARITY2	0x89e80709
#define PARITY3	0x3bd2b64b
#define PARITY4	0x0c64b1e4

#define NAME "SIMD-oriented Fast Mersenne Twister 2^216091"
#define RNG_T sfmt_216091_t

#include "sfmt.c"
