#include <stdbool.h>
#include <emmintrin.h>

#define N ((MEXP - 128) / 104 + 1)
#define LOW_MASK   UINT64_C(0x000FFFFFFFFFFFFF)
#define HIGH_CONST UINT64_C(0x3FF0000000000000)
#define SR 12
#define SHUFF 0x1b

typedef union {
  __m128i si;
  __m128d sd;
  uint64_t u64[2];
  uint32_t u32[4];
  double d[2];
} w128_t;

typedef struct {
  w128_t state[N + 1];
  size_t index; // In the original code this is an index into the state as an array of uint64_t (over [0, 2N)).  Here it is a direct index into the state as an array of w128_t (over [0, N)).
} mt_state;

// Constants as SSE types
static __m128i param_mask;
static __m128i onei;
static __m128d one;
static __m128d two;
static __m128d rtwo53m1;  // 1 / ((max >> 12) - 1)

static inline __m128d _mm_cvtepi64_pd(__m128i si) {
  w128_t v, w = { si };
  v.d[0] = (double)w.u64[0];
  v.d[1] = (double)w.u64[1];
  return v.sd;
}

static void set(uint64_t seed, void * state) {
  mt_state * mt = (mt_state *)state;
  // Converted from dsfmt_chk_gen_init_rand

  // Inlined from setup_const
  static bool init = false;
  if (!init) {
    param_mask = _mm_set_epi32((int)MSK32_3, (int)MSK32_4, (int)MSK32_1, (int)MSK32_2);
    onei = _mm_set_epi32(0, 1, 0, 1);
    one  = _mm_set1_pd(1.0);
    two  = _mm_set1_pd(2.0);
    rtwo53m1 = _mm_set1_pd(1.0 / 9007199254740991.0);  // 1 / ((max >> 12) - 1)
    init = true;
  }

  uint32_t * u32 = &mt->state[0].u32[0];
  u32[0] = (uint32_t)seed;
  for (size_t i = 1; i < (N + 1) * 4; i++)
    u32[i] = UINT32_C(1812433253) * (u32[i - 1] ^ (u32[i - 1] >> 30)) + (uint32_t)i;

  mt->index = N;

  // Inlined from initial_mask
  uint64_t * u64 = &mt->state[0].u64[0];
  for (size_t i = 0; i < N * 2; i++)
    u64[i] = (u64[i] & LOW_MASK) | HIGH_CONST;

  // Inlined from period_certification
  uint64_t pcv[2] = { PCV1, PCV2 };
  uint64_t tmp[2] = { mt->state[N].u64[0] ^ FIX1, mt->state[N].u64[1] ^ FIX2 };

  uint64_t inner = (tmp[0] & pcv[0]) ^(tmp[1] & pcv[1]);
  for (size_t i = 32; i > 0; i >>= 1)
    inner ^= inner >> i;
  inner &= 1;

  if (inner == 1)
    return;

#if (PCV2 & 1) == 1
  mt->state[N].u64[1] ^= 1;
#else
  for (size_t i = 1; i >= 0; i--) {
    uint64_t work = 1;
    for (size_t j = 0; j < 64; j++) {
      if ((work & pcv[i]) != 0) {
        mt->state[N].u64[i] ^= work;
        return;
      }
      work = work << 1;
    }
  }
#endif
}

static inline w128_t generate(mt_state * mt) {
  // Converted from dsfmt_genrand_uint32 to return a w128_t
  if (mt->index >= N) {
    // Inlined from dsfmt_gen_rand_all
    w128_t lung = mt->state[N];

    // Inlined from do_recursion
    __m128i x = _mm_xor_si128(_mm_slli_epi64(mt->state[0].si, SL1), mt->state[POS1].si);
    lung.si = _mm_xor_si128(_mm_shuffle_epi32(lung.si, SHUFF), x);
    __m128i y = _mm_xor_si128(_mm_srli_epi64(lung.si, SR), mt->state[0].si);
    mt->state[0].si = _mm_xor_si128(y, _mm_and_si128(lung.si, param_mask));

    for (size_t i = 1; i < N - POS1; i++) {
      // Inlined from do_recursion
      x = _mm_xor_si128(_mm_slli_epi64(mt->state[i].si, SL1), mt->state[POS1 + i].si);
      lung.si = _mm_xor_si128(_mm_shuffle_epi32(lung.si, SHUFF), x);
      y = _mm_xor_si128(_mm_srli_epi64(lung.si, SR), mt->state[i].si);
      mt->state[i].si = _mm_xor_si128(y, _mm_and_si128(lung.si, param_mask));
    }

    for (size_t i = N - POS1; i < N; i++) {
      // Inlined from do_recursion
      x = _mm_xor_si128(_mm_slli_epi64(mt->state[i].si, SL1), mt->state[POS1 + i - N].si);
      lung.si = _mm_xor_si128(_mm_shuffle_epi32(lung.si, SHUFF), x);
      y = _mm_xor_si128(_mm_srli_epi64(lung.si, SR), mt->state[i].si);
      mt->state[i].si = _mm_xor_si128(y, _mm_and_si128(lung.si, param_mask));
    }

    mt->state[N] = lung;

    mt->index = 0;
  }
  return mt->state[mt->index++];
}

static void get(uint64_t * res, size_t n, void * state) {
  mt_state * mt = (mt_state *)state;

  w128_t * ptr = (w128_t *)res; // Pointer into vector data as an array of w128_t
  // Generate vector data in blocks of 2
  size_t i = 0;
  while (n > 2) {
    ptr[i++] = generate(mt);
    n -= 2;
  }

  // If there are no elements left over in the vector return now
  if (n == 0) return;

  // We need one more individual element so generate 2
  w128_t r = generate(mt);
  res[i] = r.u64[0];
}

static void getOpenOpen(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 2;       // Size of vector data as an array of w128_t
  const size_t m = res->n - (n * 2); // Remainder of data to be processed sequentially [0, 1]

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128_t * ptr = (w128_t *)res->data; // Pointer into vector data as an array of w128_t
    // Generate vector data in blocks of 2
    for (size_t i = 0; i < n; i++) {
      w128_t r = generate(mt);
      r.si = _mm_or_si128(r.si, onei);
      ptr[i].sd = _mm_sub_pd(r.sd, one);
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 2) {
      w128_t r = generate(mt);
      r.si = _mm_or_si128(r.si, onei);
      r.sd = _mm_sub_pd(r.sd, one);
      vectordSet(res, i    , r.d[0]);
      vectordSet(res, i + 1, r.d[1]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one more individual element so generate 2
  w128_t r = generate(mt);
  r.si = _mm_or_si128(r.si, onei);
  r.sd = _mm_sub_pd(r.sd, one);

  // Put the last one into the vector
  vectordSet(res, res->n - 1, r.d[0]);
}

static void getOpenClose(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 2;       // Size of vector data as an array of w128_t
  const size_t m = res->n - (n * 2); // Remainder of data to be processed sequentially [0, 1]

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128_t * ptr = (w128_t *)res->data; // Pointer into vector data as an array of w128_t
    // Generate vector data in blocks of 2
    for (size_t i = 0; i < n; i++) {
      w128_t r = generate(mt);
      ptr[i].sd = _mm_sub_pd(two, r.sd);
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 2) {
      w128_t r = generate(mt);
      r.sd = _mm_sub_pd(two, r.sd);
      vectordSet(res, i    , r.d[0]);
      vectordSet(res, i + 1, r.d[1]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one more individual element so generate 2
  w128_t r = generate(mt);
  r.sd = _mm_sub_pd(two, r.sd);

  // Put the last one into the vector
  vectordSet(res, res->n - 1, r.d[0]);
}

static void getCloseOpen(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 2;       // Size of vector data as an array of w128_t
  const size_t m = res->n - (n * 2); // Remainder of data to be processed sequentially [0, 1]

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128_t * ptr = (w128_t *)res->data; // Pointer into vector data as an array of w128_t
    // Generate vector data in blocks of 2
    for (size_t i = 0; i < n; i++) {
      w128_t r = generate(mt);
      ptr[i].sd = _mm_sub_pd(r.sd, one);
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 2) {
      w128_t r = generate(mt);
      r.sd = _mm_sub_pd(r.sd, one);
      vectordSet(res, i    , r.d[0]);
      vectordSet(res, i + 1, r.d[1]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one more individual element so generate 2
  w128_t r = generate(mt);
  r.sd = _mm_sub_pd(r.sd, one);

  // Put the last one into the vector
  vectordSet(res, res->n - 1, r.d[0]);
}

static void getCloseClose(vectord * res, void * state) {
  mt_state * mt = (mt_state *)state;

  const size_t n = res->n / 2;       // Size of vector data as an array of w128_t
  const size_t m = res->n - (n * 2); // Remainder of data to be processed sequentially [0, 1]

  // If the vector data is contiguous
  if (res->inc == 1) {
    w128_t * ptr = (w128_t *)res->data; // Pointer into vector data as an array of w128_t
    // Generate vector data in blocks of 2
    for (size_t i = 0; i < n; i++) {
      w128_t r = generate(mt);
      ptr[i].sd = _mm_mul_pd(_mm_cvtepi64_pd(_mm_srli_epi64(r.si, 10)), rtwo53m1);  // (double)(x >> 12) * (1.0 / 9007199254740991.0)
    }
  }
  else {  // Non-contiguous vector
    for (size_t i = 0; i < res->n; i += 2) {
      w128_t r = generate(mt);
      r.sd = _mm_mul_pd(_mm_cvtepi64_pd(_mm_srli_epi64(r.si, 10)), rtwo53m1);  // (double)(x >> 12) * (1.0 / 9007199254740991.0)
      vectordSet(res, i    , r.d[0]);
      vectordSet(res, i + 1, r.d[1]);
    }
  }

  // If there are no elements left over in the vector return now
  if (m == 0) return;

  // We need one more individual element so generate 2
  w128_t r = generate(mt);
  r.sd = _mm_mul_pd(_mm_cvtepi64_pd(_mm_srli_epi64(r.si, 10)), rtwo53m1);  // (double)(x >> 12) * (1.0 / 9007199254740991.0)

  // Put the last one into the vector
  vectordSet(res, res->n - 1, r.d[0]);
}

static rng64_t type = { NAME, sizeof(mt_state), UINT64_C(0), UINT64_C(0xffffffffffffffff), set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const rng64_t * RNG_T = &type;
