#ifndef CUMULTIGPU_H
#define CUMULTIGPU_H

#include <cuda.h>

typedef struct __cumultigpu_st * CUmultiGPU;

/**
 * Creates a multiGPU context with a single GPU context on each device given.
 * Each context is owned by a background thread and tasks are sent to them
 * asynchronously via a shared queue.
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
