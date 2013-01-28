#ifndef CONFIG_H
#define CONFIG_H

/**
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
 * When transA == CBlasNoTrans each GPU multiprocessor processes 64x16 blocks of
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
 * |   1x240 |    64x3840 |   125.90  |  16  | 185.8
 * |   2x120 |   128x1920 |   240.00  | 112  | 354.3
 * |   3x80  |   192x1280 |   333.91  | 192* | 492.9
 * |   4x60  |   256x960  |   404.21  | 192* | 596.7
 * |   5x48  |   320x768  |   451.76  | 192* | 666.9
 * |   6x40  |   384x640  |   480.00  | 192* | 708.6
 * |   8x30  |   512x480  |   495.48  | 192* | 731.5
 * |  10x24  |   640x384  |   480.00  | 192* | 708.6
 * |  12x20  |   768x320  |   451.76  | 192* | 666.9
 * |  15x16  |   960x256  |   404.21  | 192* | 596.7
 * |  16x15  |  1024x240  |   388.86  | 192* | 574.1
 * |  20x12  |  1280x192  |   333.91  | 192* | 492.9
 * |  24x10  |  1536x160  |   289.81  | 192* | 427.8
 * |  30x8   |  1920x128  |   240.00  | 112  | 354.3
 * |  40x6   |  2560x96   |   185.06  |  32  | 273.2
 * |  48x5   |  3072x80   |   155.94  |  16  | 230.2
 * |  60x4   |  3840x64   |   125.90  |  16  | 185.8
 * |  80x3   |  5120x48   |    95.11  |  16  | 140.4
 * | 120x2   |  7680x32   |    63.73  |  16  |  94.0
 * | 240x1   | 15360x16   |    31.97  |  16  |  47.1
 * -------------------------------------------
 * (*throughput cannot outperform bandwidth so algorithm is bandwidth bound)
 *
 * The GPU is connected to main memory by a PCI Express 2.0 x16 bus.  Using the
 * bandwidth-test benchmark in the minibench directory it is found that this
 * will transfer data at a minimum of 5.5 GB/s with a maximum of 0.06 ms latency
 * (depending on whether it is host-device, device-host and if there is a
 * display attached to the GPU).  Since the internal bandwidth of the GPU is far
 * in excess of the PCI bandwidth and the latency of a memory copy is greater
 * than the latency of a kernel launch it is not possible to choose a kb > 0
 * such that the time taken to transfer a block of A and B matches the time
 * taken to process them.  A single tuning run using a block size of 512x480 was
 * used to measure performance for all block sizes when kb varies from 16-2048
 * in steps of 16 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 *
 * When transA != CBlasNoTrans each GPU multiprocessor processes 32x32 blocks of
 * C using 64 threads.  A minimum of 3 blocks is required to mask global memory
 * latency.  Due to register and shared memory requirements a maximum of 6
 * blocks can fit concurrently on each multiprocessor.  This is enough to hide
 * global memory latency and is the minimum number of blocks needed to give
 * maximum performance.
 *
 * Since there are 30 multiprocessors on a GTX 280 GPU 180 blocks need to be
 * scheduled to give each the 6 blocks required to get maximum performance.
 * Valid block sizes are listed in the table below along with the bandwidth
 * reduction provided by the block size.  The performance across all block sizes
 * is constant for a given k.
 *
 * -------------------------------------------
 * | Factors |   Overall  | Bandwidth |  k   |
 * | of 180  | Block Size | Reduction |      |
 * -------------------------------------------
 * |   1x180 |    32x5760 |    63.65  |   8  |  93.9
 * |   2x90  |    64x2880 |   125.22  |  24  | 184.8
 * |   3x60  |    96x1920 |   182.86  |  80  | 269.9
 * |   4x45  |   128x1440 |   235.10  | 136* | 347.1
 * |   5x36  |   160x1152 |   280.98  | 136* | 414.8
 * |   6x30  |   192x960  |   320.00  | 136* | 472.4
 * |   9x20  |   288x640  |   397.24  | 136* | 586.4
 * |  10x18  |   320x576  |   411.43  | 136* | 607.4
 * |  12x15  |   384x480  |   426.67  | 136* | 629.9
 * |  15x12  |   480x384  |   426.67  | 136* | 629.9
 * |  18x10  |   576x320  |   411.43  | 136* | 607.4
 * |  20x9   |   640x288  |   397.24  | 136* | 586.4
 * |  30x6   |   960x192  |   320.00  | 136* | 472.4
 * |  36x5   |  1152x160  |   280.98  | 136* | 414.8
 * |  45x4   |  1440x128  |   235.10  | 136* | 347.1
 * |  60x3   |  1920x96   |   182.86  |  80  | 269.9
 * |  90x2   |  2880x64   |   125.22  |  24  | 184.8
 * | 180x1   |  5760x32   |    63.65  |   8  |  93.9
 * -------------------------------------------
 * (*throughput cannot outperform bandwidth so algorithm is bandwidth bound)
 *
 * The GPU is connected to main memory by a PCI Express 2.0 x16 bus.  Using the
 * bandwidth-test benchmark in the minibench directory it is found that this
 * will transfer data at a minimum of 5.5 GB/s with a maximum of 0.06 ms latency
 * (depending on whether it is host-device, device-host and if there is a
 * display attached to the GPU).  Since the internal bandwidth of the GPU is far
 * in excess of the PCI bandwidth and the latency of a memory copy is greater
 * than the latency of a kernel launch it is not possible to choose a kb > 0
 * such that the time taken to transfer a block of A and B matches the time
 * taken to process them.  A single tuning run using a block size of 480x384 was
 * used to measure performance for all block sizes when kb varies from 8-2048
 * in steps of 8 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 */
#define SGEMM_N_MB 512
#define SGEMM_N_NB 480
#define SGEMM_N_KB 192
#define SGEMM_T_MB 480
#define SGEMM_T_NB 384
#define SGEMM_T_KB 136

/**
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
 * |   1x240 |    64x1920 |   123.87  | 128* |  91.4
 * |   2x120 |   128x960  |   225.88  | 128* | 166.7
 * |   3x80  |   192x640  |   295.38  | 128* | 218.0
 * |   4x60  |   256x480  |   333.91  | 128* | 246.4
 * |   5x48  |   320x384  |   349.09  | 128* | 257.6
 * |   6x40  |   384x320  |   349.09  | 128* | 257.6
 * |   8x30  |   512x240  |   326.81  | 128* | 241.2
 * |  10x24  |   640x192  |   295.38  | 128* | 218.0
 * |  12x20  |   768x160  |   264.83  | 128* | 195.4
 * |  15x16  |   960x128  |   225.88  | 128* | 166.7
 * |  16x15  |  1024x120  |   214.83  | 128* | 158.5
 * |  20x12  |  1280x96   |   178.60  | 128* | 131.8
 * |  24x10  |  1536x80   |   152.08  | 128* | 112.2
 * |  30x8   |  1920x64   |   123.87  | 128* |  91.4
 * |  40x6   |  2560x48   |    94.23  |  32  |  69.5
 * |  48x5   |  3072x40   |    78.97  |  16  |  58.2
 * |  60x4   |  3840x32   |    63.47  |  16  |  46.8
 * |  80x3   |  5120x24   |    47.78  |  16  |  35.2
 * | 120x2   |  7680x16   |    31.93  |  16  |  23.5
 * | 240x1   | 15360x8    |    15.99  |  16  |  11.8
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
 * used to measure performance for all block sizes when kb varies from 16-512
 * in steps of 16 (the amount of unrolling applied to the inner loop of the
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
 * | of 120  | Block Size | Reduction |      |
 * -------------------------------------------
 * |   1x120 |    32x1920 |    62.95  |  16  |  46.4
 * |   2x60  |    64x960  |   120.00  |  32* |  88.5
 * |   3x40  |    96x640  |   166.96  |  32* | 123.2
 * |   4x30  |   128x480  |   202.11  |  32* | 149.1
 * |   5x24  |   160x384  |   225.88  |  32* | 166.7
 * |   6x20  |   192x320  |   240.00  |  32* | 177.1
 * |   8x15  |   256x240  |   247.74  |  32* | 182.8
 * |  10x12  |   320x192  |   240.00  |  32* | 177.1
 * |  12x10  |   384x160  |   225.88  |  32* | 166.7
 * |  15x8   |   480x128  |   202.11  |  32* | 149.1
 * |  20x6   |   640x96   |   166.96  |  32* | 123.2
 * |  24x5   |   768x80   |   144.91  |  32* | 106.9
 * |  30x4   |   960x64   |   120.00  |  32* |  88.5
 * |  40x3   |  1280x48   |    92.53  | 144  |  68.3
 * |  60x2   |  1920x32   |    62.95  |  16  |  46.4
 * | 120x1   |  3840x16   |    31.87  |   8  |  23.5
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
 * used to measure performance for all block sizes when kb varies from 8-512
 * in steps of 8 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 */
#define DGEMM_N_MB 384
#define DGEMM_N_NB 320
#define DGEMM_N_KB 128
#define DGEMM_T_MB 256
#define DGEMM_T_NB 240
#define DGEMM_T_KB  32

/**
 * Block Sizes:
 *
 * From the CUDA Programming Guide each GPU multiprocessor requires at least 192
 * threads/6 warps to hide global memory latency.  The number of thread blocks
 * should also be as large as possible so that there are enough to distribute
 * among future GPUs with more multiprocessors.
 *
 * These calculations give the minimum block sizes to use for maximum
 * performance when executing GPU CGEMM on one or more GTX 280s with arguments
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
 * |   1x240 |    64x1920 |   123.87  |  16  |  91.4
 * |   2x120 |   128x960  |   225.88  |  16  | 166.7
 * |   3x80  |   192x640  |   295.38  |  16  | 218.0
 * |   4x60  |   256x480  |   333.91  |  16  | 246.4
 * |   5x48  |   320x384  |   349.09  |  16  | 257.6
 * |   6x40  |   384x320  |   349.09  |  16  | 257.6
 * |   8x30  |   512x240  |   326.81  |  16  | 241.2
 * |  10x24  |   640x192  |   295.38  |  16  | 218.0
 * |  12x20  |   768x160  |   264.83  |  16  | 195.4
 * |  15x16  |   960x128  |   225.88  |  16  | 166.7
 * |  16x15  |  1024x120  |   214.83  |  16  | 158.5
 * |  20x12  |  1280x96   |   178.60  |  16  | 131.8
 * |  24x10  |  1536x80   |   152.08  |  16  | 112.2
 * |  30x8   |  1920x64   |   123.87  |  16  |  91.4
 * |  40x6   |  2560x48   |    94.23  |  16  |  69.5
 * |  48x5   |  3072x40   |    78.97  |  16  |  58.2
 * |  60x4   |  3840x32   |    63.47  |  16  |  46.8
 * |  80x3   |  5120x24   |    47.78  |  16  |  35.2
 * | 120x2   |  7680x16   |    31.93  |  16  |  23.5
 * | 240x1   | 15360x8    |    15.99  |  16  |  11.8
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
 * used to measure performance for all block sizes when kb varies from 16-512
 * in steps of 16 (the amount of unrolling applied to the inner loop of the
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
 * |   1x120 |    32x1920 |    62.95  |   8  |  46.4
 * |   2x60  |    64x960  |   120.00  |   8  |  88.5
 * |   3x40  |    96x640  |   166.96  |   8  | 123.2
 * |   4x30  |   128x480  |   202.11  |   8  | 149.1
 * |   5x24  |   160x384  |   225.88  |   8  | 166.7
 * |   6x20  |   192x320  |   240.00  |  16  | 177.1
 * |   8x15  |   256x240  |   247.74  |  16  | 182.8
 * |  10x12  |   320x192  |   240.00  |  16  | 177.1
 * |  12x10  |   384x160  |   225.88  |   8  | 166.7
 * |  15x8   |   480x128  |   202.11  |   8  | 149.1
 * |  20x6   |   640x96   |   166.96  |   8  | 123.2
 * |  24x5   |   768x80   |   144.91  |   8  | 106.9
 * |  30x4   |   960x64   |   120.00  |   8  |  88.5
 * |  40x3   |  1280x48   |    92.53  |   8  |  68.3
 * |  60x2   |  1920x32   |    62.95  |   8  |  46.4
 * | 120x1   |  3840x16   |    31.87  |   8  |  23.5
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
 * used to measure performance for all block sizes when kb varies from 8-512
 * in steps of 8 (the amount of unrolling applied to the inner loop of the
 * kernel). As performance increases with k (up to a point), kb is chosen to be
 * the maximum value such that the algorithm remains compute bound (unless
 * performance levels off, then it is taken to be the minimum value that gives
 * maximum performance in order to minimise the difference in time taken for
 * transfers).
 *
 */
#define CGEMM_N_MB 384
#define CGEMM_N_NB 320
#define CGEMM_N_KB  16
#define CGEMM_C_MB 256
#define CGEMM_C_NB 240
#define CGEMM_C_KB  16

/**
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
 * |   1x240 |    64x960  |   120.00  |  16  | 44.2
 * |   2x120 |   128x480  |   202.11  |  32  | 74.5
 * |   3x80  |   192x320  |   240.00  |  64* | 88.5
 * |   4x60  |   256x240  |   247.74  |  64* | 91.4
 * |   5x48  |   320x192  |   240.00  |  64* | 88.5
 * |   6x40  |   384x160  |   225.88  |  64* | 83.3
 * |   8x30  |   512x120  |   194.43  |  32  | 71.7
 * |  10x24  |   640x96   |   166.96  |  16  | 61.6
 * |  12x20  |   768x80   |   144.91  |  16  | 53.4
 * |  15x16  |   960x64   |   120.00  |  16  | 44.2
 * |  16x15  |  1024x60   |   113.36  |  16  | 41.8
 * |  20x12  |  1280x48   |    92.53  |  16  | 34.1
 * |  24x10  |  1536x40   |    77.97  |  16  | 28.7
 * |  30x8   |  1920x32   |    62.95  |  16  | 23.2
 * |  40x6   |  2560x24   |    47.55  |  16  | 17.5
 * |  48x5   |  3072x20   |    39.74  |  16  | 14.6
 * |  60x4   |  3840x16   |    31.87  |  16  | 11.7
 * |  80x3   |  5120x12   |    23.94  |  16  |  8.8
 * | 120x2   |  7680x8    |    15.98  |  16  |  5.8
 * | 240x1   | 15360x4    |     8.00  |  16  |  2.9
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
 * |   1x240 |     8x1920 |    15.93  |   4  |  5.8
 * |   2x120 |    16x960  |    31.48  |   4  | 11.6
 * |   3x80  |    24x640  |    46.27  |   4  | 17.0
 * |   4x60  |    32x480  |    60.00  |   4  | 22.1
 * |   5x48  |    40x384  |    72.45  |   4  | 26.7
 * |   6x40  |    48x320  |    83.48  |   4  | 30.8
 * |   8x30  |    64x240  |   101.05  |   4  | 37.2
 * |  10x24  |    80x192  |   112.94  |   8  | 41.6
 * |  12x20  |    96x160  |   120.00  |  16  | 44.2
 * |  15x16  |   120x128  |   123.87  |  44* | 45.7
 * |  16x15  |   128x120  |   123.87  |  44* | 45.7
 * |  20x12  |   160x96   |   120.00  |  16  | 44.2
 * |  24x10  |   192x80   |   112.94  |   8  | 41.6
 * |  30x8   |   240x64   |   101.05  |   4  | 37.2
 * |  40x6   |   320x48   |    83.48  |   4  | 30.8
 * |  48x5   |   384x40   |    72.45  |   4  | 26.7
 * |  60x4   |   480x32   |    60.00  |   4  | 22.1
 * |  80x3   |   640x24   |    46.27  |   4  | 17.0
 * | 120x2   |   960x16   |    31.48  |   4  | 11.6
 * | 240x1   |  1920x8    |    15.93  |   4  |  5.8
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
 * taken to process them.  A single tuning run using a block size of 128x120 was
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
 * | of 120  | Block Size | Reduction |      |
 * -------------------------------------------
 * |   1x120 |     8x1920 |    15.93  |   8  |   5.8
 * |   2x60  |    16x960  |    31.48  |   8  |  11.6
 * |   3x40  |    24x640  |    46.27  |   8  |  17.0
 * |   4x30  |    32x480  |    60.00  |   8  |  22.1
 * |   5x24  |    40x384  |    72.45  |   8  |  26.7
 * |   6x20  |    48x320  |    83.48  |   8  |  30.8
 * |   8x15  |    64x240  |   101.05  |   8  |  37.2
 * |  10x12  |    80x192  |   112.94  |  16  |  41.6
 * |  12x10  |    96x160  |   120.00  |  16  |  44.2
 * |  15x8   |   120x128  |   123.87  |  16  |  45.7
 * |  20x6   |   160x96   |   120.00  |  16  |  44.2
 * |  24x5   |   192x80   |   112.94  |  16  |  41.6
 * |  30x4   |   240x64   |   101.05  |   8  |  37.2
 * |  40x3   |   320x48   |    83.48  |   8  |  30.8
 * |  60x2   |   480x32   |    60.00  |   8  |  22.1
 * | 120x1   |   960x16   |    31.48  |   8  |  11.6
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
#define ZGEMM_N_MB  256
#define ZGEMM_N_NB  240
#define ZGEMM_N_KB   64
#define ZGEMM_CN_MB 128
#define ZGEMM_CN_NB 120
#define ZGEMM_CN_KB  44
#define ZGEMM_CC_MB 120
#define ZGEMM_CC_NB 128
#define ZGEMM_CC_KB  16

#endif
