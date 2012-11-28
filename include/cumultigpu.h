#ifndef CU_MULTIGPU_H
#define CU_MULTIGPU_H

#include <cuda.h>

typedef struct __cumultigpu_st * CUmultiGPU;

/**
 * Creates a multiGPU context.
 *
 * @param multiGPU  the handle to the created context is returned through this
 *                  pointer.
 * @param flags     flags for the context created on each device.
 * @param devices   the devices to create the contexts on.
 * @param n         the number of devices.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
 *         CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OPERATING_SYSTEM,
 *         CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
 */
CUresult cuMultiGPUCreate(CUmultiGPU *, unsigned int, CUdevice *, int);

/**
 * Destroys a multiGPU context.
 *
 * @param multiGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUDestroy(CUmultiGPU);

/**
 * Blocks until all outstanding tasks are completed.  Any error encountered
 * completing the remaining tasks is returned.
 *
 * @param multiGPU  the context to synchronise.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU);

/**
 * Returns the number of GPU contexts running on background threads.
 *
 * @param multiGPU  the context.
 * @return the number of background threads.
 */
int cuMultiGPUGetNumberOfContexts(CUmultiGPU);

#endif
