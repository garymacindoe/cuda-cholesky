#include <stdio.h>

#define ITERATIONS 1E6
#define MULTIPLE 1

#define CUDA_ERROR_CHECK(call) \
 do { \
   cudaError_t __error__; \
   if ((__error__ = (call)) != cudaSuccess) { \
     fprintf(stderr, "CUDA error: %s\n\t at %s(%s:%d)\n", cudaGetErrorString(__error__), __func__, __FILE__, __LINE__); \
     return (int)__error__; \
   } \
 } while (false)

/**
 * This is an empty kernel used to measure the overhead of launching "empty"
 * blocks on the GPU.
 */
__global__ void kernel() {
  return;
}

/**
 * This measures the overhead of launching extra thread blocks that exit
 * immediately on multiprocessors in the GPU.
 *
 * It does this by executing an empty kernel a very large number of times with
 * an increasing number of blocks and taking the average execution time.  It
 * then performs an Ordinary Least Squares regression to measure the increase
 * in time as the number of empty blocks increases.  If there is no overhead
 * the gradient should be zero.  The intersect will be the overhead of
 * launching the kernel which should be the same as calculated by the
 * kernel-test benchmark. This program uses the CUDA runtime API.
 */
int main() {
  int count;
  CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));

  fprintf(stdout, "Cost of running n empty blocks on the GPU:\n");

  for (int d = 0; d < count; d++) {
    CUDA_ERROR_CHECK(cudaSetDevice(d));

    struct cudaDeviceProp prop;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, d));

    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));

    const size_t n = MULTIPLE * prop.multiProcessorCount;

    float sumX = 0.0f, sumXX = 0.0f, sumY = 0.0f, sumXY = 0.0f;
    for (int j = 1; j <= n; j++) {
      CUDA_ERROR_CHECK(cudaEventRecord(start));
      for (int i = 0; i < ITERATIONS; i++)
        kernel<<<j, prop.maxThreadsPerBlock>>>();
      CUDA_ERROR_CHECK(cudaEventRecord(stop));
      CUDA_ERROR_CHECK(cudaEventSynchronize(stop));

      float time;
      CUDA_ERROR_CHECK(cudaEventElapsedTime(&time, start, stop));

      sumX += (float)j;
      sumXX += (float)(j * j);
      sumY += time / (float)ITERATIONS;
      sumXY += (time / (float)ITERATIONS) * (float)j;
    }

    float sxx = sumXX - (sumX * sumX) / (float)n;
    float sxy = sumXY - (sumX * sumY) / (float)n;
    float xbar = sumX / (float)n;
    float ybar = sumY / (float)n;
    float m = sxy / sxx;

    fprintf(stdout, "\tDevice %d: %fms * n + %fms\n", d, m, ybar - m * xbar);

    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));

    CUDA_ERROR_CHECK(cudaThreadExit());
  }

  return 0;
}
