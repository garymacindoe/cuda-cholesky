#include "util/arrayqueue.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/**
 * An implementation of a queue backed by a resizable array.
 */
struct __arrayqueue_st {
  void ** data;                 /** Array of pointers to objects in the
                                    queue. */
  size_t head, tail, capacity;  /** Indices to head and tail of queue and queue
                                    capacity. */
};

/**
 * Rounds up to the next nearest power of two.
 *
 * @param n  the number to round up.
 * @return the number rounded up.
 */
static inline size_t nextPow2(size_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  return n;
}

/**
 * Creates a new queue with at least the specified capacity.
 *
 * @param queue     the newly created queue is returned through this pointer.
 * @param capacity  the initial capacity of the queue.
 * @return zero if the queue was created successfully, or ENOMEM if there isn't
 *        enough memory.
 */
int arrayqueue_create(arrayqueue_t * queue, size_t capacity) {
  // Allocate a new queue object
  if ((*queue = malloc(sizeof(struct __arrayqueue_st))) == NULL)
    return ENOMEM;

  // Initialise head, tail and capacity
  (*queue)->head = 0;
  (*queue)->tail = 0;
  (*queue)->capacity = nextPow2(capacity);

  // Allocate space for objects
  if (((*queue)->data = malloc((*queue)->capacity * sizeof(void *))) == NULL) {
    free(*queue);
    return ENOMEM;
  }

  return 0;
}

/**
 * Destroys the queue.
 *
 * @param queue  the queue to destroy.
 */
void arrayqueue_destroy(arrayqueue_t queue) {
  // Free the data and queue
  free(queue->data);
  free(queue);
}

/**
 * Gets the object currently at the front of the queue.
 *
 * @param queue  the queue.
 * @param obj    the object at the front of the queue is returned through this
 *               pointer.
 * @return zero on success, EINVAL if the queue is empty.
 */
int arrayqueue_peek(const arrayqueue_t queue, void ** obj) {
  if (arrayqueue_isempty(queue))
    return EINVAL;

  *obj = queue->data[queue->head];

  return 0;
}

/**
 * Places an object at the back of the queue, expanding the queue if necessary.
 *
 * @param queue  the queue.
 * @param obj    the object to place in the queue.
 * @return zero on success, ENOMEM if the queue needed to be expanded and there
 *         isn't enough memory.
 */
int arrayqueue_push(arrayqueue_t queue, void * obj) {
  // Place the object at the back of the queue
  queue->data[queue->tail++] = obj;

  // Wrap the tail around
  if (queue->tail == queue->capacity)
    queue->tail = 0;

  // Expand the array if needed
  if (queue->tail == queue->head) {
    size_t capacity = queue->capacity * 2; // Double the capacity

    // Allocate new data array
    void ** ptr = malloc(capacity * sizeof(void *));
    if (ptr == NULL)
      return ENOMEM;

    // Copy objects into new array
    if (queue->head < queue->tail)
      memcpy(ptr, &queue->data[queue->head],
             (queue->tail - queue->head) * sizeof(void *));
    else {
      memcpy(ptr, &queue->data[queue->head],
             (queue->capacity - queue->head) * sizeof(void *));
      memcpy(&ptr[queue->capacity - queue->head],
             queue->data, queue->tail * sizeof(void *));
    }

    // Free old data
    free(queue->data);

    // Update queue
    queue->data = ptr;
    queue->head = 0;
    queue->tail = queue->capacity;      // New tail is old capacity
    queue->capacity = capacity;
  }

  return 0;
}

/**
 * Removes the object from the front of the queue.
 *
 * @param queue  the queue.
 * @param obj    the object at the front of the queue is returned through this
 *               pointer.
 * @return zero on success, EINVAL if the queue is currently empty.
 */
int arrayqueue_pop(arrayqueue_t queue, void ** obj) {
  if (arrayqueue_isempty(queue))
    return EINVAL;

  *obj = queue->data[queue->head++];

  // Wrap the head around
  if (queue->head == queue->capacity)
    queue->head = 0;

  return 0;
}

/**
 * Calculates the size of the queue.
 *
 * @param queue  the queue.
 * @return the size of the queue.
 */
size_t arrayqueue_size(const arrayqueue_t queue) {
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
bool arrayqueue_isempty(const arrayqueue_t queue) {
  return (queue->head == queue->tail);
}

/**
 * Removes all items in the queue.
 *
 * @param queue  the queue.
 */
void arrayqueue_clear(arrayqueue_t queue) {
  queue->head = queue->tail;
}
