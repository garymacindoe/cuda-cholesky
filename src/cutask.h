#ifndef CUTASK_H
#define CUTASK_H

#include "cumultigpu.h"

typedef struct __cutask_st * CUtask;

/**
 * Schedules a task to run on a GPU.
 *
 * @param task      a handle to the background task is returned through this
 *                  pointer (may be NULL if the result is not needed).
 * @param multiGPU  the multiGPU context to use.
 * @param i         the device to use.
 * @param function  the function to run.
 * @param args      the parameters for the function (may be NULL).
 * @param n         the size of the parameter object (must be zero if args is NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuTaskSchedule(CUtask *, CUmultiGPU, int, CUresult (*)(const void *),
                        const void *, size_t);

/**
 * Blocks until the background task has completed and returns the result.
 *
 * @param task  the background task.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM or any error returned by the
 *         background task.
 */
CUresult cuTaskGetResult(CUtask);

/**
 * Destroys the handle to the background task.  If the task has not yet completed
 * this will block until it has.
 *
 * @param task  the background task to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM or any error returned by the
 *         background task.
 */
CUresult cuTaskDestroy(CUtask);

#endif
