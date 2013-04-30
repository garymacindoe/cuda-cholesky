/**
 * Copied from CUDA SDK version 5.0.35 with the following changes:
 *  - reduceBlocks is inlined to reduceSinglePass
 *  - reduceMultipass is removed
 *  - reduceSinglePass is renamed reduce
 *
 * To calculate the log of the determinant (log(prod(x)^2) = 2 * sum(log(x))):
 *  - incx is added to sum elements that are not contiguous (i.e. along the
 *    diagonal of A)
 *  - elements of x are calculated as log(x) when reading from global memory
 *  - the final sum is written as 2 * sum(log(x)) to compute the determinant
 */

template <unsigned int bs>
__device__ void reduceBlock(volatile double * sdata, double sum, const unsigned int i) {
  sdata[i] = sum;
  __syncthreads();

  // do reduction in shared mem
  if (bs >= 512) { if (i < 256) { sdata[i] = sum = sum + sdata[i + 256]; } __syncthreads(); }
  if (bs >= 256) { if (i < 128) { sdata[i] = sum = sum + sdata[i + 128]; } __syncthreads(); }
  if (bs >= 128) { if (i <  64) { sdata[i] = sum = sum + sdata[i +  64]; } __syncthreads(); }

  if (i < 32) {
    if (bs >=  64) { sdata[i] = sum = sum + sdata[i + 32]; }
    if (bs >=  32) { sdata[i] = sum = sum + sdata[i + 16]; }
    if (bs >=  16) { sdata[i] = sum = sum + sdata[i +  8]; }
    if (bs >=   8) { sdata[i] = sum = sum + sdata[i +  4]; }
    if (bs >=   4) { sdata[i] = sum = sum + sdata[i +  2]; }
    if (bs >=   2) { sdata[i] = sum = sum + sdata[i +  1]; }
  }
}

// Global variable used by reduce to count how many blocks have finished
__device__ unsigned int retirementCount = 0;

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.
template <unsigned int bs, bool nIsPow2>
__global__ void reduce(const double *g_idata, double *g_odata, int inc, int n) {
  __shared__ double sdata[bs];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const unsigned int tid = threadIdx.x;
  const unsigned int gs = bs * 2 * gridDim.x;
  double sum = 0.0f;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gs and therefore fewer elements per thread
  for (int i = blockIdx.x * (bs * 2) + threadIdx.x; i < n; i += gs) {
    sum += log(g_idata[i * inc]);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + bs < n)
      sum += log(g_idata[(i + bs) * inc]);
  }

  // do reduction in shared mem
  reduceBlock<bs>(sdata, sum, tid);

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = 2.0 * sdata[0];

  if (gridDim.x > 1) {
    __shared__ bool amLast;

    // wait until all outstanding memory instructions in this thread are finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
      // If the ticket ID is equal to the number of blocks, we are the last block!
      amLast = (ticket == gridDim.x-1);
    }

    __syncthreads();

    // The last block sums the results of all other blocks
    if (amLast) {
      double sum = 0.0f;

      for (int i = tid; i < gridDim.x; i += bs)
        sum += g_odata[i];

      reduceBlock<bs>(sdata, sum, tid);

      if (tid == 0) {
        g_odata[0] = sdata[0];

        // reset retirement count so that next run succeeds
        retirementCount = 0;
      }
    }
  }
}

template __global__ void reduce<512,  true>(const double *, double *, int, int);
template __global__ void reduce<256,  true>(const double *, double *, int, int);
template __global__ void reduce<128,  true>(const double *, double *, int, int);
template __global__ void reduce< 64,  true>(const double *, double *, int, int);
template __global__ void reduce< 32,  true>(const double *, double *, int, int);
template __global__ void reduce< 16,  true>(const double *, double *, int, int);
template __global__ void reduce<  8,  true>(const double *, double *, int, int);
template __global__ void reduce<  4,  true>(const double *, double *, int, int);
template __global__ void reduce<  2,  true>(const double *, double *, int, int);
template __global__ void reduce<  1,  true>(const double *, double *, int, int);

template __global__ void reduce<512, false>(const double *, double *, int, int);
template __global__ void reduce<256, false>(const double *, double *, int, int);
template __global__ void reduce<128, false>(const double *, double *, int, int);
template __global__ void reduce< 64, false>(const double *, double *, int, int);
template __global__ void reduce< 32, false>(const double *, double *, int, int);
template __global__ void reduce< 16, false>(const double *, double *, int, int);
template __global__ void reduce<  8, false>(const double *, double *, int, int);
template __global__ void reduce<  4, false>(const double *, double *, int, int);
template __global__ void reduce<  2, false>(const double *, double *, int, int);
template __global__ void reduce<  1, false>(const double *, double *, int, int);
