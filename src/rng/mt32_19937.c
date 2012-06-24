#define TYPES_ONLY
#include "rng.h"
#undef TYPES_ONLY
#include "vector_float.h"
#include "vector_uint32.h"

#define N 624
#define M 397
#define MATRIX_A UINT32_C(0x9908b0df)
#define UPPER_MASK UINT32_C(0x80000000)
#define LOWER_MASK UINT32_C(0x7fffffff)

typedef struct {
  uint32_t state[N];
  size_t index;
} mt_state;

static void set(uint32_t seed, void * state) {
  mt_state * mt = (mt_state *)state;

  mt->state[0] = seed & UINT32_C(0xffffffff);
  for (size_t i = 1; i < N; i++)
    mt->state[i] = ((UINT32_C(1812433253) * (mt->state[i - 1] ^ (mt->state[i - 1] >> 30)) + i)) & UINT32_C(0xffffffff);
  mt->index = N;
}

static inline uint32_t generate(mt_state * mt) {
  static uint32_t magic[2] = { UINT32_C(0), MATRIX_A };

  if (mt->index >= N) {
    size_t k;

    for (k = 0; k < N - M; k++) {
      uint32_t y = (mt->state[k] & UPPER_MASK) | (mt->state[k + 1] & LOWER_MASK);
      mt->state[k] = mt->state[k + M] ^ (y >> 1) ^ magic[y & UINT32_C(1)];
    }

    for (; k < N - 1; k++) {
      uint32_t y = (mt->state[k] & UPPER_MASK) | (mt->state[k + 1] & LOWER_MASK);
      mt->state[k] = mt->state[k + M - N] ^ (y >> 1) ^ magic[y & UINT32_C(1)];
    }

    uint32_t y = (mt->state[N - 1] & UPPER_MASK) | (mt->state[0] & LOWER_MASK);
    mt->state[N - 1] = mt->state[M - 1] ^ (y >> 1) ^ magic[y & UINT32_C(1)];

    mt->index = 0;
  }

  uint32_t x = mt->state[mt->index++];

  x ^= (x >> 11);
  x ^= (x << 7 ) & UINT32_C(0x9d2c5680);
  x ^= (x << 15) & UINT32_C(0xefc60000);
  x ^= (x >> 18);

  return x;
}

static void get(vectoru32 * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectoru32Set(res, i, generate(mt));
}

static void getOpenOpen(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, ((float)(generate(mt) >> 9) + 0.5f) * (1.0f / 8388608.0f));
}

static void getOpenClose(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, 1.0f - ((float)(generate(mt) >> 8) * (1.0f / 16777216.0f)));
}

static void getCloseOpen(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, (float)(generate(mt) >> 8) * (1.0f / 16777216.0f));
}

static void getCloseClose(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, (float)(generate(mt) >> 8) * (1.0f / 16777215.0f));
}

static rng32_t type = { "Mersenne Twister 2^19937", sizeof(mt_state), UINT32_C(0), UINT32_C(0xffffffff), set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const rng32_t * mt32_19937_t = &type;
