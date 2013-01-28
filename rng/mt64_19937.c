#define TYPES_ONLY
#include "rng.h"
#undef TYPES_ONLY
#include "vector_double.h"
#include "vector_uint64.h"

#define N 312
#define M 156
#define MATRIX_A UINT64_C(0xB5026F5AA96619E9)
#define UPPER_MASK UINT64_C(0xFFFFFFFF80000000)
#define LOWER_MASK UINT64_C(0x7FFFFFFF)

typedef struct {
  uint64_t state[N];
  size_t index;
} mt_state;

static void set(uint64_t seed, void * state) {
  mt_state * mt = (mt_state *)state;

  mt->state[0] = seed;
  for (size_t i = 1; i < N; i++)
    mt->state[i] = (UINT64_C(6364136223846793005) * (mt->state[i - 1] ^ (mt->state[i - 1] >> 62)) + i);
  mt->index = N;
}

static inline uint64_t generate(mt_state * mt) {
  static uint64_t magic[2] = { UINT64_C(0), MATRIX_A };

  if (mt->index >= N) {
    size_t i;
    for (i = 0; i < N - M; i++) {
      uint64_t x = (mt->state[i] & UPPER_MASK) | (mt->state[i + 1] & LOWER_MASK);
      mt->state[i] = mt->state[i + M] ^ (x >> 1) ^ magic[x & UINT64_C(1)];
    }

    for (; i < N - 1; i++) {
      uint64_t x = (mt->state[i] & UPPER_MASK) | (mt->state[i + 1] & LOWER_MASK);
      mt->state[i] = mt->state[i + M - N] ^ (x >> 1) ^ magic[x & UINT64_C(1)];
    }

    uint64_t x = (mt->state[N - 1] & UPPER_MASK) | (mt->state[0] & LOWER_MASK);
    mt->state[N - 1] = mt->state[M - 1] ^ (x >> 1) ^ magic[x & UINT64_C(1)];

    mt->index = 0;
  }

  uint64_t x = mt->state[mt->index++];

  x ^= (x >> 29) & UINT64_C(0x5555555555555555);
  x ^= (x << 17) & UINT64_C(0x71D67FFFEDA60000);
  x ^= (x << 37) & UINT64_C(0xFFF7EEE000000000);
  x ^= (x >> 43);

  return x;
}

static void get(vectoru64 * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectoru64Set(res, i, generate(mt));
}

static void getOpenOpen(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectordSet(res, i,       ((double)(generate(mt) >> 12) + 0.5) * (1.0 / 4503599627370496.0));
}

static void getOpenClose(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectordSet(res, i, 1.0 - ((double)(generate(mt) >> 11       ) * (1.0 / 9007199254740992.0)));
}

static void getCloseOpen(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectordSet(res, i,        (double)(generate(mt) >> 11       ) * (1.0 / 9007199254740992.0));
}

static void getCloseClose(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectordSet(res, i,        (double)(generate(mt) >> 11       ) * (1.0 / 9007199254740991.0));
}

static rng64_t type = { "Mersenne Twister (64 bit) 2^19937", sizeof(mt_state), UINT64_C(0), UINT64_C(0xffffffffffffffff), set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const rng64_t * mt64_19937_t = &type;
