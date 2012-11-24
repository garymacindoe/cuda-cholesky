#define FMAD16(a, b) \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a

#define FMAD256(a, b) \
 FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); \
 FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); \
 FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); \
 FMAD16(a, b); FMAD16(a, b); FMAD16(a, b); FMAD16(a, b)

/**
 * This kernel is used to measure single precision floating point throughput.
 *
 * It contains 4096 32-bit FMAD instructions executed in a loop 16 times.  It
 * has 131072 single precision FLOPs in total.
 */
extern "C" __global__ void fmad(float * data) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  float a = index, b = data[index];

  for (int i = 0; i < 16; i++) {
    FMAD256(a, b); FMAD256(a, b); FMAD256(a, b); FMAD256(a, b);
    FMAD256(a, b); FMAD256(a, b); FMAD256(a, b); FMAD256(a, b);
    FMAD256(a, b); FMAD256(a, b); FMAD256(a, b); FMAD256(a, b);
    FMAD256(a, b); FMAD256(a, b); FMAD256(a, b); FMAD256(a, b);
  }

  data[index] = a + b;
}

#define FMADMUL16(a, b, c, d) \
 a = b * a + b; c = d * c; b = a * b + a; d = c * d; a = b * a + b; c = d * c; b = a * b + a; d = c * d; \
 a = b * a + b; c = d * c; b = a * b + a; d = c * d; a = b * a + b; c = d * c; b = a * b + a; d = c * d; \
 a = b * a + b; c = d * c; b = a * b + a; d = c * d; a = b * a + b; c = d * c; b = a * b + a; d = c * d; \
 a = b * a + b; c = d * c; b = a * b + a; d = c * d; a = b * a + b; c = d * c; b = a * b + a; d = c * d

#define FMADMUL256(a, b, c, d) \
 FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); \
 FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); \
 FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); \
 FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d); FMADMUL16(a, b, c, d)

/**
 * This kernel attempts to force the warp scheduler to dual-issue instructions
 * to the scalar processors and special function units in parallel.
 *
 * It interleaves each fmad instruction to be executed on the scalar processors
 * with an independent fmul instruction that can be executed by the special
 * function unit at the same time.
 *
 * It contains 196608 single precision FLOPs in total.
 */
extern "C" __global__ void fmadmul(float * data) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  float a = index, b = data[index], c = -index, d = -data[index];

  for (int i = 0; i < 16; i++) {
    FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d);
    FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d);
    FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d);
    FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d); FMADMUL256(a, b, c, d);
  }

  data[index] = a + b;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 120
#define DMAD16(a, b) \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a; \
 a = b * a + b; b = a * b + a; a = b * a + b; b = a * b + a

#define DMAD256(a, b) \
 DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); \
 DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); \
 DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); \
 DMAD16(a, b); DMAD16(a, b); DMAD16(a, b); DMAD16(a, b)

/**
 * This kernel is used to measure double precision floating point throughput.
 *
 * It contains 4096 32-bit DMAD instructions executed in a loop 16 times.  It
 * has 131072 double-precision FLOPs in total.
 */
extern "C" __global__ void dmad(double * data) {
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
  double a = index, b = data[index];

  for (int i = 0; i < 16; i++) {
    DMAD256(a, b); DMAD256(a, b); DMAD256(a, b); DMAD256(a, b);
    DMAD256(a, b); DMAD256(a, b); DMAD256(a, b); DMAD256(a, b);
    DMAD256(a, b); DMAD256(a, b); DMAD256(a, b); DMAD256(a, b);
    DMAD256(a, b); DMAD256(a, b); DMAD256(a, b); DMAD256(a, b);
  }

  data[index] = a + b;
}
#endif
