#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "dgemm_ref.c"

/**
 * Test program for multiGPU DGEMM.
 *
 * Block Sizes:
 *
 * From the CUDA Programming Guide each GPU multiprocessor requires at least 192
 * threads/6 warps to hide global memory latency.  The number of thread blocks
 * should also be as large as possible so that there are enough to distribute
 * among future GPUs with more multiprocessors.
 *
 * These calculations give the minimum block sizes to use for maximum
 * performance when executing GPU DGEMM on one or more GTX 280s with arguments
 * in host memory.
 *
 * When transA == CBlasNoTrans each GPU multiprocessor processes 64x8 blocks of
 * C using 64 threads.  A minimum of 3 blocks is required to mask global memory
 * latency.  Due to register and shared memory requirements a maximum of 8
 * blocks can fit concurrently on each multiprocessor.  This is enough to hide
 * global memory latency and is the minimum number of blocks needed to fully
 * utilise the GPU and give maximum performance.
 *
 * Since there are 30 multiprocessors on a GTX 280 GPU 240 blocks need to be
 * scheduled to give each the 8 blocks required to get maximum performance.
 * Valid block sizes are listed in the table below along with the bandwidth
 * reduction provided by the block size.  The performance across all block sizes
 * is constant for a given k.
 *
 * -------------------------------------------
 * | Factors |   Overall  | Bandwidth |  k   |
 * | of 240  | Block Size | Reduction |      |
 * -------------------------------------------
 * |   1x240 |    64x1920 |   123.87  |  32* |
 * |   2x120 |   128x 960 |   225.88  |  32* |
 * |   3x 80 |   192x 640 |   295.38  |  32* |
 * |   4x 60 |   256x 480 |   333.91  |  32* |
 * |   5x 48 |   320x 384 |   349.09  |  32* |
 * |   6x 40 |   384x 320 |   349.09  |  32* |
 * |   8x 30 |   448x 240 |   312.56  |  32* |
 * |  10x 24 |   640x 192 |   295.38  |  32* |
 * |  12x 20 |   768x 160 |   264.83  |  32* |
 * |  15x 16 |   960x 128 |   225.88  |  32* |
 * |  16x 15 |  1024x 120 |   227.56  |  32* |
 * |  20x 12 |  1280x  96 |   178.60  |  32* |
 * |  24x 10 |  1536x  80 |   152.08  |  32* |
 * |  30x  8 |  1920x  64 |   123.87  |  32* |
 * |  40x  6 |  2560x  48 |    94.23  |  32  |
 * |  48x  5 |  3072x  40 |    78.97  |  16  |
 * |  60x  4 |  3840x  32 |    63.47  |   8  |
 * |  80x  3 |  5120x  24 |    47.78  |   8  |
 * | 120x  2 |  7680x  16 |    31.93  |   8  |
 * | 240x  1 | 15360x   8 |    15.99  |   8  |
 * -------------------------------------------
 * (*minimum value to be compute bound - throughput cannot outperform bandwidth)
 *
 * The GPU is connected to main memory by a PCI Express 2.0 x16 bus.  Using the
 * bandwidth-test benchmark in the minibench directory it is found that this
 * will transfer data at a minimum of 5.5 GB/s with a maximum of 0.06 ms latency
 * (depending on whether it is host-device, device-host and if there is a
 * display attached to the GPU).  Since the internal bandwidth of the GPU is far
 * in excess of the PCI bandwidth and the latency of a memory copy is greater
 * than the latency of a kernel launch it is not possible to choose a kb > 0
 * such that the time taken to transfer a block of A and B matches the time
 * taken to process them.  A single tuning run using a block size of 384x320 was
 * used to measure performance for all block sizes when kb varies from 8-512
 * in steps of 8 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 *
 * When transA != CBlasNoTrans each GPU multiprocessor processes 32x16 blocks of
 * C using 64 threads.  A minimum of 3 blocks is required to mask global memory
 * latency.  Due to register and shared memory requirements a maximum of 4
 * blocks can fit concurrently on each multiprocessor.  This is enough to hide
 * global memory latency and is the minimum number of blocks needed to give
 * maximum performance.
 *
 * Since there are 30 multiprocessors on a GTX 280 GPU 120 blocks need to be
 * scheduled to give each the 4 blocks required to get maximum performance.
 * Valid block sizes are listed in the table below along with the bandwidth
 * reduction provided by the block size.  The performance across all block sizes
 * is constant for a given k.
 *
 * -------------------------------------------
 * | Factors |   Overall  | Bandwidth |  k   |
 * | of 180  | Block Size | Reduction |      |
 * -------------------------------------------
 * |   1x120 |    32x3840 |   63.47   |   8  |
 * |   2x60  |    64x1920 |  123.87   |  24* |
 * |   3x40  |    96x1280 |  178.60   |  24* |
 * |   4x30  |   128x960  |  225.88   |  24* |
 * |   5x24  |   160x768  |  264.83   |  24* |
 * |   6x20  |   192x640  |  295.38   |  24* |
 * |   8x15  |   256x480  |  333.91   |  24* |
 * |  10x12  |   320x384  |  349.09   |  24* |
 * |  12x10  |   384x320  |  349.09   |  24* |
 * |  15x8   |   480x256  |  333.91   |  24* |
 * |  20x6   |   640x192  |  295.38   |  24* |
 * |  24x5   |   768x160  |  264.83   |  24* |
 * |  30x4   |   960x128  |  225.88   |  24* |
 * |  40x3   |  1280x96   |  178.60   |  24* |
 * |  60x2   |  1920x64   |  123.87   |  24* |
 * | 120x1   |  3840x32   |   63.47   |   8  |
 * -------------------------------------------
 * (*minimum value to be compute bound - throughput cannot outperform bandwidth)
 *
 * The GPU is connected to main memory by a PCI Express 2.0 x16 bus.  Using the
 * bandwidth-test benchmark in the minibench directory it is found that this
 * will transfer data at a minimum of 5.5 GB/s with a maximum of 0.06 ms latency
 * (depending on whether it is host-device, device-host and if there is a
 * display attached to the GPU).  Since the internal bandwidth of the GPU is far
 * in excess of the PCI bandwidth and the latency of a memory copy is greater
 * than the latency of a kernel launch it is not possible to choose a kb > 0
 * such that the time taken to transfer a block of A and B matches the time
 * taken to process them.  A single tuning run using a block size of 384x320 was
 * used to measure performance for all block sizes when kb varies from 8-512
 * in steps of 8 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 */

int main(int argc, char * argv[]) {
  CBlasTranspose transA, transB;
  size_t m, n, k;

  if (argc != 6) {
    fprintf(stderr, "Usage: %s <transA> <transB> <m> <n> <k>\n"
                    "where:\n"
                    "  transA and transB  are 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
                    "  m, n and k         are the sizes of the matrices\n", argv[0]);
    return 1;
  }

  char t;
  if (sscanf(argv[1], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (t) {
    case 'N': case 'n': transA = CBlasNoTrans; break;
    case 'T': case 't': transA = CBlasTrans; break;
    case 'C': case 'c': transA = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[2], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (t) {
    case 'N': case 'n': transB = CBlasNoTrans; break;
    case 'T': case 't': transB = CBlasTrans; break;
    case 'C': case 'c': transB = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[3], "%zu", &m) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[3]);
    return 3;
  }

  if (sscanf(argv[4], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[4]);
    return 4;
  }

  if (sscanf(argv[5], "%zu", &k) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 5;
  }

  srand(0);
  double alpha, beta, * A, * B, * C, * refC;
  size_t lda, ldb, ldc;

  CU_ERROR_CHECK(cuInit(0));

  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUdevice devices[deviceCount];
  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], i));

  CUmultiGPU mGPU;
  CU_ERROR_CHECK(cuMultiGPUCreate(&mGPU, devices, deviceCount));

  alpha = (double)rand() / (double)RAND_MAX;
  beta = (double)rand() / (double)RAND_MAX;

  if (transA == CBlasNoTrans) {
    lda = (m + 1u) & ~1u;
    if ((A = malloc(lda * k * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }
  }
  else {
    lda = (k + 1u) & ~1u;
    if ((A = malloc(lda * m * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }
  }

  if (transB == CBlasNoTrans) {
    ldb = (k + 1u) & ~1u;
    if ((B = malloc(ldb * n * sizeof(double))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = (double)rand() / (double)RAND_MAX;
    }
  }
  else {
    ldb = (n + 1u) & ~1u;
    if ((B = malloc(ldb * k * sizeof(double))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = (double)rand() / (double)RAND_MAX;
    }
  }

  ldc = (m + 1u) & ~1u;
  if ((C = malloc(ldc * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -3;
  }
  if ((refC = malloc(ldc * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate refC\n", stderr);
    return -4;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refC[j * ldc + i] = C[j * ldc + i] = (double)rand() / (double)RAND_MAX;
  }

  CUmultiGPUDBlasConfig config;
  CU_ERROR_CHECK(cuMultiGPUDBlasConfigCreate(&config, mGPU, transA, transB, 384, 320,
                                             (transA == CBlasNoTrans) ?  32 : 24));

  dgemm_ref(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, refC, ldc);
  CU_ERROR_CHECK(cuMultiGPUDgemm(config, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));

  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(C[j * ldc + i] - refC[j * ldc + i]);
      if (d > diff)
        diff = d;
    }
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUDgemm(config, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
  if (gettimeofday(&stop, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;

  size_t flops = 2 * k - 1;
  if (alpha != 1.0)
    flops += 1;
  if (beta != 0.0)
    flops += 2;
  double error = (double)flops * 2.0 * DBL_EPSILON;
  flops *= m * n;

  bool passed = (diff <= error);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refC);

  CU_ERROR_CHECK(cuMultiGPUDBlasConfigDestroy(config));
  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));

  return (int)!passed;
}
