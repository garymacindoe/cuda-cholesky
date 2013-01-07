#include "taskqueue.h"
#include <stdlib.h>
#include <string.h>
#include "error.h"

/**
 * An implementation of a queue of CUtasks backed by a resizable array.
 */
struct __cutaskqueue_st {
  CUtask * data;                /** Array of pointers to tasks in the queue.  */
  size_t head, tail, capacity;  /** Indices to head and tail of queue and queue
                                    capacity. */
};

/**
 * Creates a new queue with at least the specified capacity.
 *
 * @param queue     the newly created queue is returned through this pointer.
 * @param capacity  the initial capacity of the queue.
 * @return CUDA_SUCCESS if the queue was created successfully,
 *         CUDA_ERROR_OUT_OF_MEMORY if there isn't enough memory.
 */
CUresult cuTaskQueueCreate(CUtaskqueue * queue, size_t capacity) {
  // Allocate a new queue object
  if ((*queue = malloc(sizeof(struct __cutaskqueue_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Initialise head, tail and capacity
  (*queue)->head = 0;
  (*queue)->tail = 0;
  (*queue)->capacity = capacity;

  // Allocate space for objects
  if (((*queue)->data = malloc((*queue)->capacity * sizeof(CUtask))) == NULL) {
    free(*queue);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys the queue.
 *
 * @param queue  the queue to destroy.
 */
void cuTaskQueueDestroy(CUtaskqueue queue) {
  // Free the data and queue
  free(queue->data);
  free(queue);
}

/**
 * Gets the task currently at the front of the queue.
 *
 * @param queue  the queue.
 * @param task   the task at the front of the queue is returned through this
 *               pointer.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_INVALID_VALUE if the queue is empty.
 */
CUresult cuTaskQueuePeek(const CUtaskqueue queue, CUtask * task) {
  if (cuTaskQueueIsEmpty(queue))
    return CUDA_ERROR_INVALID_VALUE;

  *task = queue->data[queue->head];

  return CUDA_SUCCESS;
}

/**
 * Places a task at the back of the queue, expanding the queue if necessary.
 *
 * @param queue  the queue.
 * @param task   the task to place in the queue.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OUT_OF_MEMORY if the queue needs to be expanded and there
 *         isn't enough memory.
 */
CUresult cuTaskQueuePush(CUtaskqueue queue, CUtask task) {
  // Place the object at the back of the queue
  queue->data[queue->tail++] = task;

  // Wrap the tail around
  if (queue->tail == queue->capacity)
    queue->tail = 0;

  // Expand the array if needed
  if (queue->tail == queue->head) {
    size_t capacity = queue->capacity + 10;

    // Allocate new data array
    CUtask * ptr = malloc(capacity * sizeof(CUtask));
    if (ptr == NULL)
      return CUDA_ERROR_OUT_OF_MEMORY;

    // Copy objects into new array
    if (queue->head < queue->tail)
      memcpy(ptr, &queue->data[queue->head],
             (queue->tail - queue->head) * sizeof(CUtask));
    else {
      memcpy(ptr, &queue->data[queue->head],
             (queue->capacity - queue->head) * sizeof(CUtask));
      memcpy(&ptr[queue->capacity - queue->head],
             queue->data, queue->tail * sizeof(CUtask));
    }

    // Free old data
    free(queue->data);

    // Update queue
    queue->data = ptr;
    queue->head = 0;
    queue->tail = queue->capacity;      // New tail is old capacity
    queue->capacity = capacity;
  }

  return CUDA_SUCCESS;
}

/**
 * Removes a task from the front of the queue.
 *
 * @param queue  the queue.
 * @param task   the task at the front of the queue is returned through this
 *               pointer.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_INVALID_VALUE if the queue is currently empty.
 */
CUresult cuTaskQueuePop(CUtaskqueue queue, CUtask * task) {
  if (cuTaskQueueIsEmpty(queue))
    return CUDA_ERROR_INVALID_VALUE;

  *task = queue->data[queue->head++];

  // Wrap the head around
  if (queue->head == queue->capacity)
    queue->head = 0;

  return CUDA_SUCCESS;
}

/**
 * Calculates the size of the queue.
 *
 * @param queue  the queue.
 * @return the size of the queue.
 */
size_t cuTaskQueueSize(const CUtaskqueue queue) {
  return (queue->head > queue->tail)    // Head can be greater than tail
       ? (queue->capacity - queue->head) + queue->tail
       : queue->tail - queue->head;
}

/**
 * Checks whether the queue is empty.
 *
 * @param queue  the queue.
 * @return <b>true</b> if the queue is empty, <b>false</b> otherwise.
 */
bool cuTaskQueueIsEmpty(const CUtaskqueue queue) {
  return (queue->head == queue->tail);
}

/**
 * Removes all items in the queue.
 *
 * @param queue  the queue.
 */
void cuTaskQueueClear(CUtaskqueue queue) {
  queue->head = queue->tail;
}
