#ifndef TASKQUEUE_H
#define TASKQUEUE_H

#include <stddef.h>
#include <stdbool.h>
#include <cuda.h>
#include "task.h"

/**
 * An implementation of a queue of tasks backed by a resizable array.
 */
typedef struct __cutaskqueue_st * CUtaskqueue;

/**
 * Creates a new queue with at least the specified capacity.
 *
 * @param queue     the newly created queue is returned through this pointer.
 * @param capacity  the initial capacity of the queue.
 * @return CUDA_SUCCESS if the queue was created successfully,
 *         CUDA_ERROR_OUT_OF_MEMORY if there isn't enough memory to allocate the
 *         queue.
 */
CUresult cuTaskQueueCreate(CUtaskqueue *, size_t);

/**
 * Destroys the queue.
 *
 * @param queue    the queue to destroy.
 */
void cuTaskQueueDestroy(CUtaskqueue);

/**
 * Gets the object currently at the front of the queue.
 *
 * @param queue  the queue.
 * @param obj    the object at the front of the queue is returned through this
 *               pointer.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_INVALID_VALUE if the queue is empty.
 */
CUresult cuTaskQueuePeek(const CUtaskqueue, CUtask *);

/**
 * Places an object at the back of the queue, expanding the queue if necessary.
 *
 * @param queue  the queue.
 * @param obj    the object to place in the queue.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OUT_OF_MEMORY if the queue needed to be expanded and there
 *         isn't enough memory.
 */
CUresult cuTaskQueuePush(CUtaskqueue, CUtask);

/**
 * Removes a task from the front of the queue.
 *
 * @param queue  the queue.
 * @param task   the task at the front of the queue is returned through this
 *               pointer.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_INVALID_VALUE if the queue is currently empty.
 */
CUresult cuTaskQueuePop(CUtaskqueue, CUtask *);

/**
 * Calculates the size of the queue.
 *
 * @param queue  the queue.
 * @return the size of the queue.
 */
size_t cuTaskQueueSize(const CUtaskqueue);

/**
 * Checks whether the queue is empty.
 *
 * @param queue  the queue.
 * @return <b>true</b> if the queue is empty, <b>false</b> otherwise.
 */
bool cuTaskQueueIsEmpty(const CUtaskqueue);

/**
 * Removes all items in the queue.
 *
 * @param queue  the queue.
 */
void cuTaskQueueClear(CUtaskqueue);

#endif
