#define TYPES_ONLY
#include "rng.h"
#undef TYPES_ONLY
#include "vector_float.h"
#include "vector_uint32.h"

#include <emmintrin.h>

#define N (MEXP / 128 + 1)

typedef union {
  uint32_t u[4];
  __m128i si;
} w128_t;

typedef union {
  float f[4];
  __m128 sf;
} w128f_t;

typedef struct {
  w128_t state[N];
  size_t index; // In the original code this is an index into the state as an array of uint32_t (over [0, 4N)).  Here it is a direct index into the state as an array of w128_t (over [0, N)).
} mt_state;

static uint32_t parity[4] = {PARITY1, PARITY2, PARITY3, PARITY4};

// Constants as SSE types
static const w128f_t half = {{               0.5f,               0.5f,               0.5f,               0.5f }}, // Half
                      one = {{               1.0f,               1.0f,               1.0f,               1.0f }}, // One
                   rtwo23 = {{ 1.0f /  8388608.0f, 1.0f /  8388608.0f, 1.0f /  8388608.0f, 1.0f /  8388608.0f }}, // 1 /  (max >> 10)
                   rtwo24 = {{ 1.0f / 16777216.0f, 1.0f / 16777216.0f, 1.0f / 16777216.0f, 1.0f / 16777216.0f }}, // 1 /   max >> 9
                 rtwo24m1 = {{ 1.0f / 16777215.0f, 1.0f / 16777215.0f, 1.0f / 16777215.0f, 1.0f / 16777215.0f }}, // 1 / ((max >> 9) - 1)
                 half_max = {{      2147483648.0f,      2147483648.0f,      2147483648.0f,      2147483648.0f }}; // 2^31 (used to cast unsigned int -> type-punned int -> float)

// Converts an __m128i containing unsigned ints as signed ints into an __m128
static inline __m128 _mm_cvtepu32_ps(__m128i i) { return _mm_add_ps(_mm_cvtepi32_ps(i), half_max.sf); }

static void set(uint32_t seed, void * state) {
  mt_state * mt = (mt_state *)state;

  // This section is from init_gen_rand
  uint32_t * ptr = &mt->state[0].u[0];

  ptr[0] = seed;
  for (size_t i = 1; i < (4 * N); i++)
    ptr[i] = UINT32_C(1812433253) * (ptr[i - 1] ^ (ptr[i - 1] >> 30)) + (uint32_t)i;
  mt->index = N;

  // This section is inlined from period_certification
  uint32_t inner = 0;
  for (size_t i = 0; i < 4; i++)
    inner ^= ptr[i] & parity[i];
  for (size_t i = 16; i > 0; i >>= 1)
    inner ^= inner >> i;
  inner &= 1;

  if (inner == 1)
    return;

  for (size_t i = 0; i < 4; i++) {
    uint32_t work = 1;
    for (size_t j = 0; j < 32; j++) {
      if ((work & parity[i]) != 0) {
        ptr[i] ^= work;
        return;
      }
      work <<= 1;
    }
  }

}

static inline w128_t generate(mt_state * mt) {
  // This is converted from gen_rand32 to return a w128_t
  if (mt->index >= N) {
    // This is inlined from the SSE2 version of gen_rand_all
    __m128i mask = _mm_set_epi32((int)MSK4, (int)MSK3, (int)MSK2, (int)MSK1);

    __m128i r1 = _mm_load_si128(&mt->state[N - 2].si);
    __m128i r2 = _mm_load_si128(&mt->state[N - 1].si);

    size_t i;
    for (i = 0; i < N - POS1; i++) {
      // Inlined call to SSE2 version of do_recursion (mm_recursion)
      __m128i x = _mm_load_si128(&mt->state[i].si);
      __m128i y = _mm_srli_epi32( mt->state[i + POS1].si, SR1);
      __m128i z = _mm_srli_si128(r1, SR2);
      __m128i v = _mm_slli_epi32(r2, SL1);
      z = _mm_xor_si128(z, x);
      z = _mm_xor_si128(z, v);
      x = _mm_slli_si128(x, SL2);
      y = _mm_and_si128(y, mask);
      z = _mm_xor_si128(z, x);
      z = _mm_xor_si128(z, y);

      _mm_store_si128(&mt->state[i].si, z);
      r1 = r2;
      r2 = z;
    }
    for (; i < N; i++) {
      // Inlined call to SSE2 version of do_recursion (mm_recursion)
      __m128i x = _mm_load_si128(&mt->state[i].si);
      __m128i y = _mm_srli_epi32( mt->state[i + POS1 - N].si, SR1);
      __m128i z = _mm_srli_si128(r1, SR2);
      __m128i v = _mm_slli_epi32(r2, SL1);
      z = _mm_xor_si128(z, x);
      z = _mm_xor_si128(z, v);
      x = _mm_slli_si128(x, SL2);
      y = _mm_and_si128(y, mask);
      z = _mm_xor_si128(z, x);
      z = _mm_xor_si128(z, y);

      _mm_store_si128(&mt->state[i].si, z);
      r1 = r2;
      r2 = z;
    }
    mt->index = 0;
  }
  return mt->state[mt->index++];
}

static void get(vectoru32 * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 4;       // Size of vector data as an array of w128_t
  const size_t m = res->n - (n * 4); // Remainder of data to be processed sequentially

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128_t * ptr = (w128_t *)res->data; // Pointer into vector data as an array of w128_t
    // Generate vector data in blocks of 4
    for (size_t i = 0; i < n; i++)
      ptr[i] = generate(mt);
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 4) {
      w128_t r = generate(mt);
      vectoru32Set(res, i    , r.u[0]);
      vectoru32Set(res, i + 1, r.u[1]);
      vectoru32Set(res, i + 2, r.u[2]);
      vectoru32Set(res, i + 3, r.u[3]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one or more individual elements so generate 4
  w128_t r = generate(mt);

  // Put them into the vector
  switch (m) {
    case 3:
      vectoru32Set(res, res->n - 1, r.u[2]);
    case 2:
      vectoru32Set(res, res->n - 2, r.u[1]);
    case 1:
      vectoru32Set(res, res->n - 3, r.u[0]);
  }
}

static void getOpenOpen(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 4;       // Size of vector data as an array of w128f_t
  const size_t m = res->n - (n * 4); // Remainder of data to be processed sequentially

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128f_t * ptr = (w128f_t *)res->data; // Pointer into vector data as an array of w128f_t
    // Generate vector data in blocks of 4
    for (size_t i = 0; i < n; i++) {
      w128_t s = generate(mt);
      ptr[i].sf = _mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 9)), half.sf), rtwo23.sf);  // ((float)(x >> 9) + 0.5f) * (1.0f / 8388608.0f)
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 4) {
      w128_t s = generate(mt);
      w128f_t r;
      r.sf = _mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 9)), half.sf), rtwo23.sf);  // ((float)(x >> 9) + 0.5f) * (1.0f / 8388608.0f)
      vectorfSet(res, i    , r.f[0]);
      vectorfSet(res, i + 1, r.f[1]);
      vectorfSet(res, i + 2, r.f[2]);
      vectorfSet(res, i + 3, r.f[3]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one or more individual elements so generate 4
  w128_t s = generate(mt);
  w128f_t r;
  r.sf = _mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 9)), half.sf), rtwo23.sf);  // ((float)(x >> 9) + 0.5f) * (1.0f / 8388608.0f)

  // Put them into the vector
  switch (m) {
    case 3:
      vectorfSet(res, res->n - 1, r.f[2]);
    case 2:
      vectorfSet(res, res->n - 2, r.f[1]);
    case 1:
      vectorfSet(res, res->n - 3, r.f[0]);
  }
}

static void getOpenClose(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 4;       // Size of vector data as an array of w128f_t
  const size_t m = res->n - (n * 4); // Remainder of data to be processed sequentially

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128f_t * ptr = (w128f_t *)res->data; // Pointer into vector data as an array of w128f_t
    // Generate vector data in blocks of 4
    for (size_t i = 0; i < n; i++) {
      w128_t s = generate(mt);
      ptr[i].sf = _mm_sub_ps(one.sf, _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24.sf)); // 1.0f - (float)(x >> 8) * (1.0f / 16777216.0f)
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 4) {
      w128_t s = generate(mt);
      w128f_t r;
      r.sf = _mm_sub_ps(one.sf, _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24.sf)); // 1.0f - (float)(x >> 8) * (1.0f / 16777216.0f)
      vectorfSet(res, i    , r.f[0]);
      vectorfSet(res, i + 1, r.f[1]);
      vectorfSet(res, i + 2, r.f[2]);
      vectorfSet(res, i + 3, r.f[3]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one or more individual elements so generate 4
  w128_t s = generate(mt);
  w128f_t r;
  r.sf = _mm_sub_ps(one.sf, _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24.sf)); // 1.0f - (float)(x >> 8) * (1.0f / 16777216.0f)

  // Put them into the vector
  switch (m) {
    case 3:
      vectorfSet(res, res->n - 1, r.f[2]);
    case 2:
      vectorfSet(res, res->n - 2, r.f[1]);
    case 1:
      vectorfSet(res, res->n - 3, r.f[0]);
  }
}

static void getCloseOpen(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 4;       // Size of vector data as an array of w128f_t
  const size_t m = res->n - (n * 4); // Remainder of data to be processed sequentially

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128f_t * ptr = (w128f_t *)res->data; // Pointer into vector data as an array of w128f_t
    // Generate vector data in blocks of 4
    for (size_t i = 0; i < n; i++) {
      w128_t s = generate(mt);
      ptr[i].sf = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24.sf); // (float)(x >> 8) * (1.0f / 16777216.0f)
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 4) {
      w128_t s = generate(mt);
      w128f_t r;
      r.sf = _mm_mul_ps(_mm_cvtepu32_ps(_mm_srli_epi32(s.si, 8)), rtwo24.sf); // (float)(x >> 8) * (1.0f / 16777216.0f)
      vectorfSet(res, i    , r.f[0]);
      vectorfSet(res, i + 1, r.f[1]);
      vectorfSet(res, i + 2, r.f[2]);
      vectorfSet(res, i + 3, r.f[3]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one or more individual elements so generate 4
  w128_t s = generate(mt);
  w128f_t r;
  r.sf = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24.sf); // (float)(x >> 8) * (1.0f / 16777216.0f)

  // Put them into the vector
  switch (m) {
    case 3:
      vectorfSet(res, res->n - 1, r.f[2]);
    case 2:
      vectorfSet(res, res->n - 2, r.f[1]);
    case 1:
      vectorfSet(res, res->n - 3, r.f[0]);
  }
}

static void getCloseClose(vectorf * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 4;       // Size of vector data as an array of w128f_t
  const size_t m = res->n - (n * 4); // Remainder of data to be processed sequentially

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128f_t * ptr = (w128f_t *)res->data; // Pointer into vector data as an array of w128f_t
    // Generate vector data in blocks of 4
    for (size_t i = 0; i < n; i++) {
      w128_t s = generate(mt);
      ptr[i].sf = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24m1.sf); // (float)(x >> 8) * (1.0f / 16777215.0f)
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 4) {
      w128_t s = generate(mt);
      w128f_t r;
      r.sf = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24m1.sf); // (float)(x >> 8) * (1.0f / 16777215.0f)
      vectorfSet(res, i    , r.f[0]);
      vectorfSet(res, i + 1, r.f[1]);
      vectorfSet(res, i + 2, r.f[2]);
      vectorfSet(res, i + 3, r.f[3]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one or more individual elements so generate 4
  w128_t s = generate(mt);
  w128f_t r;
  r.sf = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(s.si, 8)), rtwo24m1.sf); // (float)(x >> 8) * (1.0f / 16777215.0f)

  // Put them into the vector
  switch (m) {
    case 3:
      vectorfSet(res, res->n - 1, r.f[2]);
    case 2:
      vectorfSet(res, res->n - 2, r.f[1]);
    case 1:
      vectorfSet(res, res->n - 3, r.f[0]);
  }
}

static rng32_t type = { NAME, sizeof(mt_state), UINT32_C(0), UINT32_C(0xffffffff), set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const rng32_t * RNG_T = &type;
