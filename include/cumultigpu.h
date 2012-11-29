#ifndef CUMULTIGPU_H
#define CUMULTIGPU_H

#include <cuda.h>

typedef struct __cumultigpu_st * CUmultiGPU;

/**
 * Creates a multiGPU context with a number of background threads each with a
 * context for a device and with a shared queue of tasks to execute.
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
 * Destroys a multiGPU context.  Any tasks currently scheduled will not be run
 * and any currently running will be cancelled.
 *
 * @param multiGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUDestroy(CUmultiGPU);

#endif
