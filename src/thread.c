#include "thread.h"
#include <pthread.h>
#include "error.h"
#include "taskqueue.h"

/**
 * Background thread type.
 */
struct __cuthread_st {
  CUtaskqueue queue;            /** Queue of tasks                            */
  pthread_t thread;             /** Background thread                         */
  pthread_mutex_t mutex;        /** Mutex to protect queue                    */
  pthread_cond_t nonEmpty;      /** Condition to wait on non-empty queue      */
  CUresult error;               /** Thread error status                       */
};

/**
 * Background thread error handling macros.
 */
#define CU_THREAD_ERROR_CHECK(call) \
  do { \
    if ((this->error = (call)) != CUDA_SUCCESS) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, this->error, \
                     (const char * (*)(int))cuGetErrorString); \
      return this; \
    } \
  } while (false)

#define THREAD_ERROR_CHECK(call) \
  do { \
    int __error__; \
    if ((__error__ = (call)) != CUDA_SUCCESS) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))strerror); \
      this->error = CUDA_ERROR_OPERATING_SYSTEM; \
      return this; \
    } \
  } while (false)

/**
 * Background thread main function.  Waits for tasks to appear in the queue and
 * executes them.
 *
 * @param args  thread object.
 * @return thread object.
 */
static void * cu_thread_main(void * args) {
  CUthread this = (CUthread)args;

  // Enter main loop
  while (true) {
    // Lock the mutex for the task queue
    THREAD_ERROR_CHECK(pthread_mutex_lock(&this->mutex));

    // Wait until there are tasks in the queue
    while (cuTaskQueueIsEmpty(this->queue))
      THREAD_ERROR_CHECK(pthread_cond_wait(&this->nonEmpty, &this->mutex));

    // Remove a task from the head of the queue
    CUtask task;
    CU_THREAD_ERROR_CHECK(cuTaskQueuePop(this->queue, &task));

    // Unlock the mutex for the task queue
    THREAD_ERROR_CHECK(pthread_mutex_unlock(&this->mutex));

    // A NULL task is the signal to exit
    if (task == NULL)
      break;

    // Execute task
    CU_THREAD_ERROR_CHECK(cuTaskExecute(task));
  }

  return this;
}

/**
 * Starts a background thread.
 *
 * @param thread  a handle to the background thread is returned through this
 *                pointer.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OUT_OF_MEMORY if there is not enough memory,
 *         CUDA_ERROR_OPERATING_SYSTEM if the thread could not be started.
 */
CUresult cuThreadCreate(CUthread * thread) {
  // Allocate space on the heap for the thread object
  if ((*thread = malloc(sizeof(struct __cuthread_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Create the task queue for the thread
  if (cuTaskQueueCreate(&(*thread)->queue, 1) != CUDA_SUCCESS) {
    free(*thread);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Set up the mutex and condition variable
  (*thread)->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  (*thread)->nonEmpty = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // Initialise thread error status to zero
  (*thread)->error = CUDA_SUCCESS;

  // Start the background thread
  ERROR_CHECK(pthread_create(&(*thread)->thread, NULL, cu_thread_main, *thread));

  return CUDA_SUCCESS;
}

/**
 * Destroys the background thread.  This function will block until all tasks are
 * completed.
 *
 * @param thread  the thread to destroy.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OPERATING_SYSTEM if the thread could not be stopped.
 */
CUresult cuThreadDestroy(CUthread thread) {
  // Lock the queue
  ERROR_CHECK(pthread_mutex_lock(&thread->mutex));

  // Place a NULL task on the queue to signal the thread to exit the main loop
  CU_ERROR_CHECK(cuTaskQueuePush(thread->queue, NULL));

  // Unlock the queue
  ERROR_CHECK(pthread_mutex_unlock(&thread->mutex));

  // Signal to the thread that there are tasks waiting
  ERROR_CHECK(pthread_cond_signal(&thread->nonEmpty));

  // Wait for the thread to exit
  ERROR_CHECK(pthread_join(thread->thread, (void **)&thread));

  // Destroy the queue
  cuTaskQueueDestroy(thread->queue);

  // Copy the thread error status
  CUresult error = thread->error;

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
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OUT_OF_MEMORY if there is not enough memory,
 *         CUDA_ERROR_OPERATING_SYSTEM if there was a problem communicating the
 *         task to the thread.
 */
CUresult cuThreadRunTask(CUthread thread, CUtask task) {
  // Lock the mutex for the queue
  ERROR_CHECK(pthread_mutex_lock(&thread->mutex));

  // Place the task in the queue
  CU_ERROR_CHECK(cuTaskQueuePush(thread->queue, task));

  // Unlock the queue mutex
  ERROR_CHECK(pthread_mutex_unlock(&thread->mutex));

  // Signal to background thread that there are now tasks available
  ERROR_CHECK(pthread_cond_signal(&thread->nonEmpty));

  return CUDA_SUCCESS;
}
