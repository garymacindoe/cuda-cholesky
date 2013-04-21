#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include "error.h"
#include "blas.h"

static inline unsigned int min(unsigned int a, unsigned int b) { return (a < b) ? a : b; }

static CUresult cuDeviceGetMaxGFLOPs(int *);
static CUresult cuFuncMaxBlocksPerMP(CUfunction, CUdevice, unsigned int, unsigned int *);
static void getMaxReduction(unsigned int, unsigned int, unsigned int, unsigned int *, unsigned int *);
static CUresult cuSgemmBenchmark(CUfunction, CBlasTranspose, CBlasTranspose, size_t, size_t, size_t, float *);
static CUresult bandwidth_test(double *, double *, double *, double *);

int main() {
  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGetMaxGFLOPs(&device));

  int major, minor;
  CU_ERROR_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CU_ERROR_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  int multiProcessorCount;
  CU_ERROR_CHECK(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));

  unsigned int maxBlocks, mb, nb, kb;
  float prevGFlops;

  CUmodule sgemm;
  CU_ERROR_CHECK(cuModuleLoad(&sgemm, "sgemm.fatbin"));

  CUfunction sgemmN, sgemmT;
  CU_ERROR_CHECK(cuModuleGetFunction(&sgemmN, sgemm,
  "_Z6sgemm2IL14CBlasTranspose78ELS0_84ELj64ELj16ELj16ELj16ELj4EEvPKfS2_S2_Pfffiiiiiii"));
  CU_ERROR_CHECK(cuModuleGetFunction(&sgemmT, sgemm,
  "_Z6sgemm2IL14CBlasTranspose84ELS0_78ELj32ELj32ELj8ELj8ELj8EEvPKfS2_S2_Pfffiiiiiii"));

  CU_ERROR_CHECK(cuFuncMaxBlocksPerMP(sgemmN, device, 64, &maxBlocks));
  getMaxReduction((unsigned int)multiProcessorCount * maxBlocks, 64, 16, &mb, &nb);

  prevGFlops = 0.0f;
  kb = 16;
  while (kb < 1024) {
    float time;
    CU_ERROR_CHECK(cuSgemmBenchmark(sgemmN, CBlasNoTrans, CBlasTrans, mb, nb, kb, &time));
    const size_t flops = 2 * (size_t)mb * (size_t)nb * ((size_t)kb + 1);
    float gflops = ((float)flops * 1.e-9f) / time;
    if (prevGFlops * 1.01f > gflops)
      break;
    prevGFlops = gflops;
    kb += 16;
  }

  fprintf(stdout, "#define SGEMM_N_MB %d\n", mb);
  fprintf(stdout, "#define SGEMM_N_NB %d\n", nb);
  fprintf(stdout, "#define SGEMM_N_KB %d\n\n", kb);

  CU_ERROR_CHECK(cuFuncMaxBlocksPerMP(sgemmT, device, 64, &maxBlocks));
  getMaxReduction((unsigned int)multiProcessorCount * maxBlocks, 32, 32, &mb, &nb);

  prevGFlops = 0.0f;
  kb = 8;
  while (kb < 1024) {
    float time;
    CU_ERROR_CHECK(cuSgemmBenchmark(sgemmT, CBlasTrans, CBlasNoTrans, mb, nb, kb, &time));
    const size_t flops = 2 * (size_t)mb * (size_t)nb * ((size_t)kb + 1);
    float gflops = ((float)flops * 1.e-9f) / time;
    if (prevGFlops * 1.01f > gflops)
      break;
    prevGFlops = gflops;
    kb += 8;
  }

  fprintf(stdout, "#define SGEMM_T_MB %d\n", mb);
  fprintf(stdout, "#define SGEMM_T_NB %d\n", nb);
  fprintf(stdout, "#define SGEMM_T_KB %d\n\n", kb);

  CU_ERROR_CHECK(cuModuleUnload(sgemm));

  double bandwidth_htod, overhead_htod, bandwidth_dtoh, overhead_dtoh;
  CU_ERROR_CHECK(bandwidth_test(&bandwidth_htod, &overhead_htod,
                                &bandwidth_dtoh, &overhead_dtoh));

  fprintf(stdout, "#define BANDWIDTH_HTOD %.10e\n", bandwidth_htod);
  fprintf(stdout, "#define OVERHEAD_HTOD %.10e\n", overhead_htod);
  fprintf(stdout, "#define BANDWIDTH_DTOH %.10e\n", bandwidth_dtoh);
  fprintf(stdout, "#define OVERHEAD_DTOH %.10e\n", overhead_dtoh);

  CU_ERROR_CHECK(cuCtxDestroy(context));

  return 0;
}

static CUresult cuDeviceGetMaxGFLOPs(CUdevice * res) {
  *res = 0;

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  int maxFlops = 0;
  for (int d = 0; d < count; d++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, d));

    int multiProcessorCount, clockRate;
    CU_ERROR_CHECK(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

    int flops = multiProcessorCount * clockRate;
    if (flops >= maxFlops)
      *res = device;
  }

  return CUDA_SUCCESS;
}

static CUresult cuFuncMaxBlocksPerMP(CUfunction function, CUdevice device, unsigned int threads, unsigned int * maxBlocks) {
  // Get the resources available per multiprocessor
  int maxThreadsPerBlock, maxRegistersPerBlock, maxSharedMemoryPerBlock;
  CU_ERROR_CHECK(cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  CU_ERROR_CHECK(cuDeviceGetAttribute(&maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
  CU_ERROR_CHECK(cuDeviceGetAttribute(&maxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));

  // Get the resource usage of the function
  int numRegs, sharedSizeBytes;
  CU_ERROR_CHECK(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
  CU_ERROR_CHECK(cuFuncGetAttribute(&sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
  numRegs *= (int)threads;
  numRegs = (numRegs + 511) & ~511;
  sharedSizeBytes = (sharedSizeBytes + 511) & ~511;

  // Work out the maximum number of thread blocks per multiprocessor
  unsigned int maxBlocksByThread = (unsigned int)maxThreadsPerBlock / threads;
  unsigned int maxBlocksByRegs = (unsigned int)(maxRegistersPerBlock / numRegs);
  unsigned int maxBlocksByShared = (unsigned int)(maxSharedMemoryPerBlock / sharedSizeBytes);

  *maxBlocks = min(maxBlocksByThread, min(maxBlocksByRegs, maxBlocksByShared));

  return CUDA_SUCCESS;
}

static void getMaxReduction(unsigned int blocks, unsigned int mb, unsigned int nb, unsigned int * m, unsigned int * n) {
  // Find all factors of the number of blocks required
  double reduction = 0.0;
  unsigned int limit = (unsigned int)floor(sqrt((double)blocks));
  for (unsigned int x = 1; x <= limit; x++) {
    if (blocks % x == 0) {
      // Work out the cofactor
      unsigned int y = blocks / x;

      // Work out the bandwidth reduction
      unsigned int bx = x * mb;
      unsigned int by = y * nb;
      double r = 2.0 / ((1.0 / (double)bx) + (1.0 / (double)by));
      // Find the maximum bandwidth reduction
      if (r > reduction) {
        reduction = r;
        *m = bx;
        *n = by;
      }

      // Try y * x as well as x * y
      bx = y * mb;
      by = x * nb;
      r = 2.0 / ((1.0 / (double)bx) + (1.0 / (double)by));
      // Find the maximum bandwidth reduction
      if (r > reduction) {
        reduction = r;
        *m = bx;
        *n = by;
      }
    }
  }
}

static CUresult cuSgemmBenchmark(CUfunction function, CBlasTranspose transA, CBlasTranspose transB, size_t m, size_t n, size_t k, float * time) {
  float alpha, beta, * A, * B, * C;
  CUdeviceptr dA, dB, dC, dD;
  size_t lda, ldb, ldc, dlda, dldb, dldc, dldd;

  alpha = (float)rand() / (float)RAND_MAX;
  beta = (float)rand() / (float)RAND_MAX;

  if (transA == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocHost((void **)&A, (lda = (m + 3u) & ~3u) * k * sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, m * sizeof(float), k, sizeof(float)));
    dlda /= sizeof(float);

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, 0, lda * sizeof(float),
                            0, 0, CU_MEMORYTYPE_DEVICE, 0, dA, 0, dlda * sizeof(float),
                            m * sizeof(float), k };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocHost((void **)&A, (lda = (k + 3u) & ~3u) * m * sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, k * sizeof(float), m, sizeof(float)));
    dlda /= sizeof(float);

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, 0, lda * sizeof(float),
                            0, 0, CU_MEMORYTYPE_DEVICE, 0, dA, 0, dlda * sizeof(float),
                            k * sizeof(float), m };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  if (transB == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (k + 3u) & ~3u) * n * sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, k * sizeof(float), n, sizeof(float)));
    dldb /= sizeof(float);

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, 0, ldb * sizeof(float),
                            0, 0, CU_MEMORYTYPE_DEVICE, 0, dB, 0, dldb * sizeof(float),
                            k * sizeof(float), n };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (n + 3u) & ~3u) * k * sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, n * sizeof(float), k, sizeof(float)));
    dldb /= sizeof(float);

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, 0, ldb * sizeof(float),
                            0, 0, CU_MEMORYTYPE_DEVICE, 0, dB, 0, dldb * sizeof(float),
                            n * sizeof(float), k };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  CU_ERROR_CHECK(cuMemAllocHost((void **)&C, (ldc = (m + 3u) & ~3u) * n * sizeof(float)));
  CU_ERROR_CHECK(cuMemAllocPitch(&dC, &dldc, m * sizeof(float), n, sizeof(float)));
  dldc /= sizeof(float);
  CU_ERROR_CHECK(cuMemAllocPitch(&dD, &dldd, m * sizeof(float), n, sizeof(float)));
  dldd /= sizeof(float);

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      C[j * ldc + i] = (float)rand() / (float)RAND_MAX;
  }

  CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, C, 0, 0, ldc * sizeof(float),
                          0, 0, CU_MEMORYTYPE_DEVICE, 0, dC, 0, dldc * sizeof(float),
                          m * sizeof(float), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  void * params[] = { &dA, &dB, &dC, &dD, &alpha, &beta, &dlda, &dldb, &dldc, &dldd, &m, &n, &k };

  CU_ERROR_CHECK(cuEventRecord(start, 0));
  for (size_t i = 0; i < 2000; i++) {
    if (transA == CBlasNoTrans)
      CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + 63) / 64, (unsigned int)(n + 15) / 16, 1, 16, 4, 1, 0, NULL, params, NULL));
    else
      CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + 31) / 32, (unsigned int)(n + 31) / 32, 1, 8, 8, 1, 0, NULL, params, NULL));
  }
  CU_ERROR_CHECK(cuEventRecord(stop, 0));
  CU_ERROR_CHECK(cuEventSynchronize(stop));
  CU_ERROR_CHECK(cuEventElapsedTime(time, start, stop));
  *time /= 2000.0f;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));

  CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dB));
  CU_ERROR_CHECK(cuMemFree(dC));
  CU_ERROR_CHECK(cuMemFree(dD));
  CU_ERROR_CHECK(cuMemFreeHost(A));
  CU_ERROR_CHECK(cuMemFreeHost(B));
  CU_ERROR_CHECK(cuMemFreeHost(C));

  return CUDA_SUCCESS;
}

static CUresult bandwidth_test(double * bandwidth_htod, double * overhead_htod,
                               double * bandwidth_dtoh, double * overhead_dtoh) {
  CUstream stream;
  CU_ERROR_CHECK(cuStreamCreate(&stream, 0));

  CUdeviceptr dPointer;
  CU_ERROR_CHECK(cuMemAlloc(&dPointer, 16777216));

  void * hPointer;
  CU_ERROR_CHECK(cuMemAllocHost(&hPointer, 134217728));

  double sumX = 0.0, sumXX = 0.0, sumY = 0.0, sumXY = 0.0;
  const size_t n = 16;
  for (size_t j = 1; j <= n; j++) {
    size_t size = j * 1048576;
    struct timeval start, stop;

    int error;
    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t k = 0; k < 20; k++)
      CU_ERROR_CHECK(cuMemcpyHtoDAsync(dPointer, hPointer, size, stream));
    CU_ERROR_CHECK(cuStreamSynchronize(stream));
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }
    double time = ((double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6)) / 20.0;

    sumX += (double)size;
    sumXX += (double)size * (double)size;
    sumY += time;
    sumXY += time * (double)size;
  }
  double sxx = sumXX - (sumX * sumX) / (double)n;
  double sxy = sumXY - (sumX * sumY) / (double)n;
  double xbar = sumX / (double)n;
  double ybar = sumY / (double)n;

  *bandwidth_htod = sxy / sxx;
  *overhead_htod = ybar - *bandwidth_htod * xbar;

  sumY = 0.0, sumXY = 0.0;
  for (size_t j = 1; j <= n; j++) {
    size_t size = j * 1048576;
    struct timeval start, stop;

    int error;
    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t k = 0; k < 20; k++)
      CU_ERROR_CHECK(cuMemcpyDtoHAsync(hPointer, dPointer, size, stream));
    CU_ERROR_CHECK(cuStreamSynchronize(stream));
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }
    double time = ((double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6)) / 20.0;

    sumY += time;
    sumXY += time * (double)size;
  }
  sxx = sumXX - (sumX * sumX) / (double)n;
  sxy = sumXY - (sumX * sumY) / (double)n;
  xbar = sumX / (double)n;
  ybar = sumY / (double)n;

  *bandwidth_dtoh = sxy / sxx;
  *overhead_dtoh = ybar - *bandwidth_dtoh * xbar;

  CU_ERROR_CHECK(cuMemFreeHost(hPointer));

  CU_ERROR_CHECK(cuMemFree(dPointer));

  CU_ERROR_CHECK(cuStreamDestroy(stream));

  return CUDA_SUCCESS;
}
