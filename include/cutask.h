#ifndef CUTASK_H
#define CUTASK_H

#include <stddef.h>
#include <cuda.h>

typedef struct __cutask_st * CUtask;

/**
 * Creates a task.
 *
 * @param task      the newly created task is returned through this pointer.
 * @param function  the function to execute.
 * @param args      arguments for the function.
 * @param size      the size of the arguments.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_INVALID_VALUE if <b>function</b> is NULL or <b>args</b> is
 *         NULL and <b>size</b> is greater than 0,
 *         CUDA_ERROR_OUT_OF_MEMORY if there is not enough memory to create
 *         another task.
 */
CUresult cuTaskCreate(CUtask *, CUresult (*)(const void *), const void *, size_t);

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskDestroy(CUtask, CUresult *);

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskExecute(CUtask);

#endif
