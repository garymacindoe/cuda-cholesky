template <size_t N>
struct mt_state {
  uint64_t state[N];
};

texture<uint32_t, 1, cudaReadModeElementType> tex_param_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_temper_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_double_ref;

__device__ __constant__ uint32_t high_mask;
__device__ __constant__ uint32_t low_mask;

__device__ void recursion(uint32_t * rh, uint32_t * rl, uint32_t x1h, uint32_t x1l, uint32_t x2h, uint32_t x2l, uint32_t yh, uint32_t yl, size_t bid) {
  uint32_t xh = (x1h & high_mask) ^ x2h;
  uint32_t xl = (x1l & low_mask) ^ x2l;

  xh ^= xh << sh1_tbl[bid];
  xl ^= xl << sh1_tbl[bid];
  yh = xl ^ (yh >> sh2_tbl[bid]);
  yl = xh ^ (yl >> sh2_tbl[bid]);

  uint32_t mat = tex1Dfetch(tex_param_ref, bid * 16 + (yl & 0x0f));
  *rh = yh ^ mat;
  *rl = yl;
}

__device__ uint32_t temper(texture<uint32_t, 1, cudaReadModeElementType> tex, uint32_t t, size_t bid) {
  t ^= t >> 16;
  t ^= t >>  8;
  return tex1Dfetch(tex, bid * 16 + (t & 0x0f));
}

struct convert {
  __device__ uint64_t operator()(uint32_t vh, uint32_t vl, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_temper_ref, t, bid);
    return ((uint64_t)(vh ^ mat) << 32) | vl;
  }
};

struct convert_close_open {
  __device__ double operator()(uint32_t vh, uint32_t vl, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_double_ref, t, bid);
    convert c;
    return __longlong_as_double((c(vh, vl, t, bid) >> 12) ^ ((uint64_t)mat << 32)) - 1.0;
  }
};

struct convert_open_close {
  __device__ double operator()(uint32_t vh, uint32_t vl, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_double_ref, t, bid);
    convert c;
    return 2.0 - __longlong_as_double((c(vh, vl, t, bid) >> 12) ^ ((uint64_t)mat << 32));
  }
};

struct convert_open_open {
  __device__ double operator()(uint32_t vh, uint32_t vl, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_double_ref, t, bid);
    convert c;
    return __longlong_as_double((c(vh, vl, t, bid) >> 12) ^ ((uint64_t)mat << 32) | 1) - 1.0;
  }
};

struct convert_close_close {
  __device__ double operator()(uint32_t vh, uint32_t vl, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_double_ref, t, bid);
    convert c;
    return (double)c(vh, vl, t, bid) * (1.0 / (double)0xffffffffffffffff);
  }
};

template <size_t N, size_t bs, class T, class C>
__global__ void sample(mt_state<N> * mt, T * data, size_t inc, size_t n) {
  extern __shared__ uint32_t shared[];
  C c;
  const size_t bid = blockIdx.x;
  const size_t tid = threadIdx.x;

  // copy status data from global memory to shared memory.
  uint64_t x = mt[bid].state[tid];
  shared[3 * bs - N + tid] = x >> 32;
  shared[6 * bs - N + tid] = x & 0xffffffff;
  if (tid < N - bs) {
    x = mt[bid].state[bs + tid];
    shared[3 * bs - N + bs + tid] = x >> 32;
    shared[6 * bs - N + bs + tid] = x & 0xffffffff;
  }
  __syncthreads();

  // main loop
  const uint32_t pos = pos_tbl[bid];
  uint32_t yh, yl;
  for (size_t i = 0; i < n; i += 3 * bs) {
    recursion(&yh, &yl, shared[3 * bs - N + tid], shared[6 * bs - N + tid], shared[3 * bs - N + tid + 1], shared[6 * bs - N + tid + 1], shared[3 * bs - N + tid + pos], shared[6 * bs - N + tid + pos], bid);
    shared[tid] = yh;
    shared[3 * bs + tid] = yl;
    data[(n * bid + i + tid) * inc] = c(yh, yl, shared[6 * bs - N + tid + pos - 1], bid);
    __syncthreads();

    recursion(&yh, &yl, shared[(4 * bs - N + tid) % (3 * bs)], shared[(7 * bs - N + tid) % (3 * bs)], shared[(4 * bs - N + tid + 1) % (3 * bs)], shared[(7 * bs - N + tid + 1) % (3 * bs)], shared[(4 * bs - N + tid + pos) % (3 * bs)], shared[(7 * bs - N + tid + pos) % (3 * bs)], bid);
    shared[bs + tid] = yh;
    shared[4 * bs + tid] = yl;
    data[(n * bid + bs + i + tid) * inc] = c(yh, yl, shared[(7 * bs - N + tid + pos - 1) % (3 * bs)], bid);
    __syncthreads();

    recursion(&yh, &yl, shared[2 * bs - N + tid], shared[5 * bs - N + tid], shared[2 * bs - N + tid + 1], shared[5 * bs - N + tid + 1], shared[2 * bs - N + tid + pos], shared[5 * bs - N + tid + pos], bid);
    shared[2 * bs + tid] = yh;
    shared[5 * bs + tid] = yl;
    data[(n * bid + 2 * bs + i + tid) * inc] = c(yh, yl, shared[5 * bs + tid + pos - N - 1], bid);
    __syncthreads();
  }

  // write back status for next call
  x = (uint64_t)shared[3 * bs - N + tid] << 32;
  x = x | shared[6 * bs - N + tid];
  mt[bid].state[tid] = x;
  if (tid < N - bs) {
    x = (uint64_t)shared[4 * bs - N + tid] << 32;
    x = x | shared[7 * bs - N + tid];
    mt[bid].state[bs + tid] = x;
  }
  __syncthreads();
}
