#include "util/thread.h"
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>

#include "util/arrayqueue.h"

/**
 * Background thread type.
 */
struct __thread_st {
  arrayqueue_t queue;           /** Queue of tasks                            */
  pthread_t thread;             /** Background thread                         */
  pthread_mutex_t mutex;        /** Mutex to protect queue                    */
  pthread_cond_t nonEmpty;      /** Condition to wait on non-empty queue      */
  int error;                    /** Thread error status                       */
};

/**
 * Background thread main function.  Waits for tasks to appear in the queue and
 * executes them.
 *
 * @param args  thread object.
 * @return This function does not return.
 */
static void * thread_main(void * args) {
  thread_t this = (thread_t)args;

  // Enter main loop
  while (true) {
    // Lock the mutex for the task queue
    if ((this->error = pthread_mutex_lock(&this->mutex)) != 0)
      return this;

    // Wait until there are tasks in the queue
    while (arrayqueue_isempty(this->queue))
    if ((this->error = pthread_cond_wait(&this->nonEmpty, &this->mutex)) != 0)
      return this;

    // Remove a task from the head of the queue
    task_t task;
    if ((this->error = arrayqueue_pop(this->queue, (void **)&task)) != 0)
      return this;

    // Unlock the mutex for the task queue
    if ((this->error = pthread_mutex_unlock(&this->mutex)) != 0)
      return this;

    // A NULL task is the signal to exit
    if (task == NULL)
      break;

    // Execute task
    if ((this->error = task_execute(task)) != 0)
      return this;
  }

  return this;
}

/**
 * Starts a background thread.
 *
 * @param thread  a handle to the background thread is returned through this
 *                pointer.
 * @return zero on success,
 *         ENOMEM if there is not enough memory.
 */
int thread_create(thread_t * thread) {
  // Allocate space on the heap for the thread object
  if ((*thread = malloc(sizeof(struct __thread_st))) == NULL)
    return ENOMEM;

  // Create the task queue for the thread
  if (arrayqueue_create(&(*thread)->queue, 1) != 0) {
    free(*thread);
    return ENOMEM;
  }

  // Set up the mutex and condition variable
  (*thread)->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  (*thread)->nonEmpty = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // Initialise thread error status to zero
  (*thread)->error = 0;

  // Start the background thread
  int error;
  if ((error = pthread_create(&(*thread)->thread, NULL, thread_main, *thread)) != 0)
    return error;

  return 0;
}

/**
 * Destroys the background thread.  This function will block until all tasks are
 * completed.
 *
 * @param thread  the thread to destroy.
 * @return zero on success.
 */
int thread_destroy(thread_t thread) {
  int error;

  // Lock the queue
  if ((error = pthread_mutex_lock(&thread->mutex)) != 0)
    return error;

  // Place a NULL task on the queue to signal the thread to exit the main loop
  if ((error = arrayqueue_push(thread->queue, NULL)) != 0)
    return error;

  // Unlock the queue
  if ((error = pthread_mutex_unlock(&thread->mutex)) != 0)
    return error;

  // Signal to the thread that there are tasks waiting
  if ((error = pthread_cond_signal(&thread->nonEmpty)) != 0)
    return error;

  // Wait for the thread to exit
  if ((error = pthread_join(thread->thread, (void **)&thread)) != 0)
    return error;

  // Destroy the queue
  arrayqueue_destroy(thread->queue);

  // Copy the thread error status
  error = thread->error;

  // Free the thread object
  free(thread);

  // Return the thread error status
  return error;
}

/**
 * Schedules a task to run on a background thread.
 *
 * @param thread  the background thread to run the task on.
 * @param task    the task to run.
 * @return zero on success,
 *         ENOMEM if there is not enough memory.
 */
int thread_run_task(thread_t thread, task_t task) {
  int error;

  // Lock the mutex for the queue
  if ((error = pthread_mutex_lock(&thread->mutex)) != 0)
    return error;

  // Place the task in the queue
  if ((error = arrayqueue_push(thread->queue, task)) != 0)
    return error;

  // Unlock the queue mutex
  if ((error = pthread_mutex_unlock(&thread->mutex)) != 0)
    return error;

  // Signal to background thread that there are now tasks available
  if ((error = pthread_cond_signal(&thread->nonEmpty)) != 0)
    return error;

  return 0;
}
