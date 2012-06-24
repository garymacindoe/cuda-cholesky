#include <stdio.h>

#define ITERATIONS 1E6

#define CUDA_ERROR_CHECK(call) \
 do { \
   cudaError_t __error__; \
   if ((__error__ = (call)) != cudaSuccess) { \
     fprintf(stderr, "CUDA error: %s\n\t at %s(%s:%d)\n", cudaGetErrorString(__error__), __func__, __FILE__, __LINE__); \
     return (int)__error__; \
   } \
 } while (false)

/**
 * This is a minimal kernel used to measure kernel launch overhead on the GPU.
 *
 * It copies one single precision 32-bit floating point value from one area of
 * global memory to another.
 */
extern "C" __global__ void kernel(const float * in, float * out) {
  *out = *in;
}

/**
 * This measures the overhead in launching a kernel function on each GPU in the
 * system.
 *
 * It does this by executing a small kernel (copying 1 value in global memory) a
 * very large number of times and taking the average execution time.  This
 * program uses the CUDA runtime API.
 */
int main() {
  int count;
  CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));

  float x = 5.0f;
  for (int d = 0; d < count; d++) {
    CUDA_ERROR_CHECK(cudaSetDevice(d));

    float * in, * out;
    CUDA_ERROR_CHECK(cudaMalloc((void **)&in, sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&out, sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(in, &x, sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    CUDA_ERROR_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++)
      kernel<<<1, 1>>>(in, out);
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));

    float time;
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&time, start, stop));

    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));

    CUDA_ERROR_CHECK(cudaFree(in));
    CUDA_ERROR_CHECK(cudaFree(out));

    fprintf(stdout, "Device %d: %fms\n", d, (time / (double)ITERATIONS));

    CUDA_ERROR_CHECK(cudaThreadExit());
  }

  return 0;
}
