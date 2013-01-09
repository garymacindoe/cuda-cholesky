#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "zgemm_ref.c"

/**
 * Test program for multiGPU ZGEMM.
 *
 * Block Sizes:
 *
 * From the CUDA Programming Guide each GPU multiprocessor requires at least 192
 * threads/6 warps to hide global memory latency.  The number of thread blocks
 * should also be as large as possible so that there are enough to distribute
 * among future GPUs with more multiprocessors.
 *
 * These calculations give the minimum block sizes to use for maximum
 * performance when executing GPU SGEMM on one or more GTX 280s with arguments
 * in host memory.
 *
 * When transA == CBlasNoTrans each GPU multiprocessor processes 64x4 blocks of
 * C using 64 threads.  A minimum of 3 blocks is required to mask global memory
 * latency.  Due to register and shared memory requirements a maximum of 8
 * blocks can fit concurrently on each multiprocessor.  This is enough to hide
 * global memory latency and is the minimum number of blocks needed to give
 * maximum performance.
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
 * |   1x240 |    64x960  |   120.00  |  16  |
 * |   2x120 |   128x480  |   202.11  |  16  |
 * |   3x80  |   192x320  |   240.00  |  32* |
 * |   4x60  |   256x240  |   247.74  |  32* |
 * |   5x48  |   320x192  |   240.00  |  32* |
 * |   6x40  |   384x160  |   225.88  |  32* |
 * |   8x30  |   512x120  |   194.43  |  16  |
 * |  10x24  |   640x 96  |   166.96  |  16  |
 * |  12x20  |   768x 80  |   144.91  |  16  |
 * |  15x16  |   960x 64  |   120.00  |  16  |
 * |  16x15  |  1024x 60  |   113.36  |  16  |
 * |  20x12  |  1280x 48  |    92.53  |  16  |
 * |  24x10  |  1536x 40  |    77.97  |  16  |
 * |  30x8   |  1920x 32  |    62.95  |  16  |
 * |  40x6   |  2560x 24  |    47.55  |  16  |
 * |  48x5   |  3072x 20  |    39.74  |  16  |
 * |  60x4   |  3840x 16  |    31.87  |  16  |
 * |  80x3   |  5120x 12  |    23.94  |  16  |
 * | 120x2   |  7680x  8  |    15.98  |  16  |
 * | 240x1   | 15360x  4  |     8.00  |  16  |
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
 * taken to process them.  A single tuning run using a block size of 256x240 was
 * used to measure performance for all block sizes when kb varies from 16-512
 * in steps of 16 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 *
 * When transA != CBlasNoTrans and transB == CBlasNoTrans each GPU
 * multiprocessor processes 8x8 blocks of C using 32 threads.  A minimum of 3
 * blocks is required to mask global memory latency.  Due to register and shared
 * memory requirements a maximum of 8 blocks can fit concurrently on each
 * multiprocessor.  This is enough to hide global memory latency and is the
 * minimum number of blocks needed to give maximum performance.
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
 * |   1x240 |    8x1920  |    15.93  |   4  |
 * |   2x120 |   16x 960  |    31.48  |   4  |
 * |   3x80  |   24x 640  |    46.27  |   4  |
 * |   4x60  |   32x 480  |    60.00  |   4  |
 * |   5x48  |   40x 384  |    72.45  |   4  |
 * |   6x40  |   48x 320  |    83.48  |   4  |
 * |   8x30  |   64x 240  |   101.05  |   8  |
 * |  10x24  |   80x 192  |   112.94  |   8  |
 * |  12x20  |   96x 160  |   120.00  |  12  |
 * |  15x16  |  120x 128  |   123.87  |  16  |
 * |  16x15  |  128x 120  |   123.87  |  16  |
 * |  20x12  |  160x  96  |   120.00  |  12  |
 * |  24x10  |  192x  80  |   112.94  |   8  |
 * |  30x8   |  240x  64  |   101.05  |   8  |
 * |  40x6   |  320x  48  |    83.48  |   4  |
 * |  48x5   |  384x  40  |    72.45  |   4  |
 * |  60x4   |  480x  32  |    60.00  |   4  |
 * |  80x3   |  640x  24  |    46.27  |   4  |
 * | 120x2   |  960x  16  |    31.48  |   4  |
 * | 240x1   | 1920x   8  |    15.93  |   4  |
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
 * taken to process them.  A single tuning run using a block size of 640x384 was
 * used to measure performance for all block sizes when kb varies from 4-512
 * in steps of 4 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 *
 * When transA != CBlasNoTrans and transB != CBlasNoTrans each GPU
 * multiprocessor processes 8x16 blocks of C using 64 threads.  A minimum of 3
 * blocks is required to mask global memory latency.  Due to register and shared
 * memory requirements a maximum of 4 blocks can fit concurrently on each
 * multiprocessor.  This is enough to hide global memory latency and is the
 * minimum number of blocks needed to give maximum performance.
 *
 * Since there are 30 multiprocessors on a GTX 280 GPU 120 blocks need to be
 * scheduled to give each the 4 blocks required to get maximum performance.
 * Valid block sizes are listed in the table below along with the bandwidth
 * reduction provided by the block size.  The performance across all block sizes
 * is constant for a given k.
 *
 * -------------------------------------------
 * | Factors |   Overall  | Bandwidth |  k   |
 * | of 240  | Block Size | Reduction |      |
 * -------------------------------------------
 * |   1x120 |    8x1920  |    15.93  |   8  |
 * |   2x60  |   16x960   |    31.48  |   8  |
 * |   3x40  |   24x640   |    46.27  |   8  |
 * |   4x30  |   32x480   |    60.00  |   8  |
 * |   5x24  |   40x384   |    72.45  |   8  |
 * |   6x20  |   48x320   |    83.48  |   8  |
 * |   8x15  |   64x240   |   101.05  |   8  |
 * |  10x12  |   80x192   |   112.94  |  16  |
 * |  12x10  |   96x160   |   120.00  |  16  |
 * |  15x8   |  120x128   |   123.87  |  16  |
 * |  20x6   |  160x96    |   120.00  |  16  |
 * |  24x5   |  192x80    |   112.94  |  16  |
 * |  30x4   |  240x64    |   101.05  |   8  |
 * |  40x3   |  320x48    |    83.48  |   8  |
 * |  60x2   |  480x32    |    60.00  |   8  |
 * | 120x1   |  960x16    |    31.48  |   8  |
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
 * taken to process them.  A single tuning run using a block size of 120x128 was
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
  double complex alpha, beta, * A, * B, * C, * refC;
  size_t lda, ldb, ldc;

  CU_ERROR_CHECK(cuInit(0));

  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUdevice devices[deviceCount];
  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], i));

  CUmultiGPU mGPU;
  CU_ERROR_CHECK(cuMultiGPUCreate(&mGPU, devices, deviceCount));

  alpha = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
  beta = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;

  if (transA == CBlasNoTrans) {
    lda = m;
    if ((A = malloc(lda * k * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }
  }
  else {
    lda = k;
    if ((A = malloc(lda * m * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }
  }

  if (transB == CBlasNoTrans) {
    ldb = k;
    if ((B = malloc(ldb * n * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }
  }
  else {
    ldb = n;
    if ((B = malloc(ldb * k * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }
  }

  ldc = m;
  if ((C = malloc(ldc * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -3;
  }
  if ((refC = malloc(ldc * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate refC\n", stderr);
    return -4;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refC[j * ldc + i] = C[j * ldc + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
  }

  CUmultiGPUZBlasConfig config;
  CU_ERROR_CHECK(cuMultiGPUZBlasConfigCreate(&config, mGPU, transA, transB,
                                             (transA == CBlasNoTrans) ? 256 : ((transB == CBlasNoTrans) ? 128 : 120),
                                             (transA == CBlasNoTrans) ? 240 : ((transB == CBlasNoTrans) ? 120 : 128),
                                             (transA == CBlasNoTrans) ?  32 : 16));

  zgemm_ref(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, refC, ldc);
  CU_ERROR_CHECK(cuMultiGPUZgemm(config, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));

  double rdiff = 0.0, idiff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(creal(C[j * ldc + i]) - creal(refC[j * ldc + i]));
      if (d > rdiff)
        rdiff = d;
      d = fabs(cimag(C[j * ldc + i]) - cimag(refC[j * ldc + i]));
      if (d > idiff)
        idiff = d;
    }
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUZgemm(config, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
  if (gettimeofday(&stop, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;

  size_t flops = 8 * k - 2;
  if (alpha != 1.0 + 0.0 * I)
    flops += 6;
  if (beta != 0.0 + 0.0 * I)
    flops += 8;
  double error = (double)flops * DBL_EPSILON;
  flops *= m * n;

  bool passed = (rdiff <= error) && (idiff <= error);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refC);

  CU_ERROR_CHECK(cuMultiGPUZBlasConfigDestroy(config));
  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));

  return (int)!passed;
}
