#ifndef ARRAYQUEUE_H
#define ARRAYQUEUE_H

#include <stddef.h>
#include <stdbool.h>

/**
 * An implementation of a queue backed by a resizable array.
 */
typedef struct __arrayqueue_st * arrayqueue_t;

/**
 * Creates a new queue with at least the specified capacity.
 *
 * @param queue     the newly created queue is returned through this pointer.
 * @param capacity  the initial capacity of the queue.
 * @return zero if the queue was created successfully, or ENOMEM if there isn't
 *        enough memory.
 */
int arrayqueue_create(arrayqueue_t *, size_t);

/**
 * Destroys the queue using the given destructor function to destroy any objects
 * still in the queue.
 *
 * @param queue    the queue to destroy.
 * @param destroy  the destructor function for objects in the queue.
 */
void arrayqueue_destroy(arrayqueue_t, void (*)(void *));

/**
 * Gets the object currently at the front of the queue.
 *
 * @param queue  the queue.
 * @param obj    the object at the front of the queue is returned through this
 *               pointer.
 * @return zero on success, EINVAL if the queue is empty.
 */
int arrayqueue_peek(const arrayqueue_t, void **);

/**
 * Places an object at the back of the queue, expanding the queue if necessary.
 *
 * @param queue  the queue.
 * @param obj    the object to place in the queue.
 * @return zero on success, ENOMEM if the queue needed to be expanded and there
 *         isn't enough memory.
 */
int arrayqueue_push(arrayqueue_t, void *);

/**
 * Removes the object from the front of the queue.
 *
 * @param queue  the queue.
 * @param obj    the object at the front of the queue is returned through this
 *               pointer.
 * @return zero on success, EINVAL if the queue is currently empty.
 */
int arrayqueue_pop(arrayqueue_t, void **);

/**
 * Calculates the size of the queue.
 *
 * @param queue  the queue.
 * @return the size of the queue.
 */
size_t arrayqueue_size(const arrayqueue_t);

/**
 * Checks whether the queue is empty.
 *
 * @param queue  the queue.
 * @return <b>true</b> if the queue is empty, <b>false</b> otherwise.
 */
bool arrayqueue_isempty(const arrayqueue_t);

#endif
