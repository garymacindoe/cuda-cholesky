#include <stdlib.h>
#define TYPES_ONLY
#include "rng.h"
#undef TYPES_ONLY
#include "vector_float.h"
#include "vector_uint32.h"

static void set(uint32_t seed, void * state) {
  (void)state;
  srand(seed);
}

static void get(vectoru32 * res, void * state) {
  (void)state;
  for (size_t i = 0; i < res->n; i++)
    vectoru32Set(res, i, (uint32_t)rand());
}

static void getOpenOpen(vectorf * res, void * state) {
  (void)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, ((float)(rand() >> 9) + 0.5f) * (1.0f / 4194304.0f));
}

static void getOpenClose(vectorf * res, void * state) {
  (void)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, 1.0f - ((float)(rand() >> 8) * (1.0f / 8388608.0f)));
}

static void getCloseOpen(vectorf * res, void * state) {
  (void)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, (float)(rand() >> 8) * (1.0f / 8388608.0f));
}

static void getCloseClose(vectorf * res, void * state) {
  (void)state;
  for (size_t i = 0; i < res->n; i++)
    vectorfSet(res, i, (float)(rand() >> 8) * (1.0f / 8388607.0f));
}

static rng32_t type = { "stdlib.h rand()", 0ul, UINT32_C(0), (uint32_t)RAND_MAX, set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const rng32_t * std_rand_t = &type;
