#ifndef CUMULTIGPU_H
#define CUMULTIGPU_H

#include <cuda.h>

/**
 * MultiGPU context.  May be single threaded or multi-threaded.
 */
typedef struct __cumultigpu_st * CUmultiGPU;

/**
 * Creates a multiGPU context with a single CUDA context created on each of the
 * devices given.
 *
 * @param mGPU     the newly created context is returned through this pointer.
 * @param devices  the CUDA devices to use.
 * @param n        the number of CUDA devices to use.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUCreate(CUmultiGPU *, CUdevice *, int);

/**
 * Destroys a multiGPU context.
 *
 * @param mGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUDestroy(CUmultiGPU);

/**
 * Gets the number of contexts available in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return the number of contexts in the multiGPU context.
 */
int cuMultiGPUGetContextCount(CUmultiGPU);

/**
 * Synchronises all contexts in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return any errors.
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU);

#endif
