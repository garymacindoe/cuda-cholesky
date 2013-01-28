template <size_t N>
struct mt_state {
  uint32_t state[N];
};

texture<uint32_t, 1, cudaReadModeElementType> tex_param_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_temper_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_single_ref;

__device__ __constant__ uint32_t mask;

__device__ uint32_t recursion(uint32_t x1, uint32_t x2, uint32_t y, size_t bid) {
  uint32_t x = (x1 & mask) ^ x2;
  x ^= x << sh1_tbl[bid];
  y = x ^ (y >> sh2_tbl[bid]);
  uint32_t mat = tex1Dfetch(tex_param_ref, bid * 16 + (y & 0x0f));
  return y ^ mat;
}

__device__ uint32_t temper(texture<uint32_t, 1, cudaReadModeElementType> tex, uint32_t t, size_t bid) {
  t ^= t >> 16;
  t ^= t >>  8;
  return tex1Dfetch(tex, bid * 16 + (t & 0x0f));
}

struct convert {
  __device__ uint32_t operator()(uint32_t v, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_temper_ref, t, bid);
    return v ^ mat;
  }
};

struct convert_close_open {
  __device__ float operator()(uint32_t v, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_single_ref, t, bid);
    return __int_as_float((v >> 9) ^ mat) - 1.0f;
  }
};

struct convert_open_close {
  __device__ float operator()(uint32_t v, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_single_ref, t, bid);
    return 2.0f - __int_as_float((v >> 9) ^ mat);
  }
};

struct convert_open_open {
  __device__ float operator()(uint32_t v, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_single_ref, t, bid);
    return __int_as_float((v >> 9) ^ mat | 1) - 1.0f;
  }
};

struct convert_close_close {
  __device__ float operator()(uint32_t v, uint32_t t, size_t bid) {
    uint32_t mat = temper(tex_single_ref, t, bid);
    return (float)(v ^ mat) * (1.0f / (float)0xffffffff);
  }
};

template <size_t N, size_t bs, class T, class C>  // N = 176, bs = 128
__global__ void sample(mt_state<N> * mt, T * data, size_t inc, size_t n) {
  extern __shared__ uint32_t shared[];  // shared[384]
  C c;
  const size_t bid = blockIdx.x;  // [0...119]
  const size_t tid = threadIdx.x; // [0...127]

  // copy status data from global memory to shared memory. - Reads mt[bid].state[0...175] into shared[208...383]
  shared[(bs * 3) - N + tid] = mt[bid].state[tid];  // shared[208...335] = mt[bid].state[0...127]
  if (tid < N - bs)                                 // 0...127 < 48
    shared[(bs * 3) - N + bs + tid] = mt[bid].state[bs + tid];  // shared[336...383] = mt[bid].state[128...175]
  __syncthreads();

  // main loop
  const uint32_t pos = pos_tbl[bid];  // Must be [0...48] to avoid out-of-bounds (between 0 and N - bs)
  uint32_t r;
  for (size_t i = 0; i < n; i += (bs * 3)) {  // i = 0,384,768...n (which is data->n / gridDim.x)
    r = recursion(shared[(bs * 3) - N + tid], shared[(bs * 3) - N + tid + 1], shared[(bs * 3) - N + tid + pos], bid); // recursion(shared[208...335], shared[209...336], shared[208...335 + pos], bid)
    shared[tid] = r;  // shared[0...127]
    data[(n * bid + i + tid) * inc] = c(r, shared[(bs * 3) - N + tid + pos - 1], bid);  // data[] = shared[207...334 + pos]
    __syncthreads();

    r = recursion(shared[(4 * bs - N + tid) % (bs * 3)], shared[(4 * bs - N + tid + 1) % (bs * 3)], shared[(4 * bs - N + tid + pos) % (bs * 3)], bid);  // recursion(shared[336...383,0...79], shared[337...383,0...80], shared[336...383,0...79 + pos], bid)
    shared[tid + bs] = r; // shared[128...255]
    data[(n * bid + bs + i + tid) * inc] = c(r, shared[(4 * bs - N + tid + pos - 1) % (bs * 3)], bid);  // data[] = shared[335...383,0...78 + pos]
    __syncthreads();

    r = recursion(shared[2 * bs - N + tid], shared[2 * bs - N + tid + 1], shared[2 * bs - N + tid + pos], bid); // recursion(shared[80...207], shared[81...208], shared[80...207 + pos], bid)
    shared[tid + 2 * bs] = r; // shared[256...383]
    data[(n * bid + 2 * bs + i + tid) * inc] = c(r, shared[tid + pos - 1 + 2 * bs - N], bid); // data[] = shared[79...206 + pos]
    __syncthreads();
  }

  // write back status for next call - Writes shared[208...383] into mt[bid].state[0...175]
  mt[bid].state[tid] = shared[(bs * 3) - N + tid];  // mt[bid].state[0...127] = shared[208...383]
  if (tid < N - bs)                                 // 0...127 < 48
    mt[bid].state[bs + tid] = shared[4 * bs - N + tid]; // mt[bid].state[128...175] = shared[336...383]
  __syncthreads();
}
