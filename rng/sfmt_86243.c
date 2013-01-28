#define MEXP 86243

#define POS1	366
#define SL1	6
#define SL2	7
#define SR1	19
#define SR2	1
#define MSK1	0xfdbffbff
#define MSK2	0xbff7ff3f
#define MSK3	0xfd77efff
#define MSK4	0xbf9ff3ff
#define PARITY1	0x00000001
#define PARITY2	0x00000000
#define PARITY3	0x00000000
#define PARITY4	0xe9528d85

#define NAME "SIMD-oriented Fast Mersenne Twister 2^86243"
#define RNG_T sfmt_86243_t

#include "sfmt.c"
