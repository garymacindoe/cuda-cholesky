#include "multigpu.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>

#include "error.h"
#include "util/arrayqueue.h"

/**
 * Background thread arguments/type.
 */
typedef struct __cu_thread_st {
  arrayqueue_t queue;           /** Queue of tasks                            */
  pthread_t thread;             /** Background thread                         */
  pthread_mutex_t mutex;        /** Mutex to protect queue                    */
  pthread_cond_t nonEmpty;      /** Condition to wait on non-empty queue      */
} * CUthread;

/**
 * Background thread function.
 *
 * @param args  thread arguments.
 * @return thread error status.
 */
static void * cu_thread_main(void * args) {
  CUthread this = (CUthread)args;

  // Enter main loop
  while (true) {
    // Lock the mutex for the task queue
    ERROR_CHECK(pthread_mutex_lock(&this->mutex));

    // Wait until there are tasks in the queue
    while (arrayqueue_isempty(this->queue))
      ERROR_CHECK(pthread_cond_wait(&this->nonEmpty, &this->mutex));

    // Remove a task from the head of the queue
    CUtask task;
    ERROR_CHECK(arrayqueue_pop(this->queue, (void **)&task));

    // Unlock the mutex for the task queue
    ERROR_CHECK(pthread_mutex_unlock(&this->mutex));

    // Execute task
    cuTaskExecute(task);
  }
}

CUresult cuThreadCreate(CUthread * thread) {
  // Allocate space on the heap for the thread object
  if ((*thread = malloc(sizeof(struct __cu_thread_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Create the task queue for the thread
  if (arrayqueue_create(&(*thread)->queue, 1) != 0) {
    free(*thread);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Set up the mutex and condition variable
  (*thread)->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  (*thread)->nonEmpty = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // Start the background thread
  ERROR_CHECK(pthread_create(&(*thread)->threads[mGPU->n], NULL, cu_thread_main, *thread));

  return CUDA_SUCCESS;
}

/**
 * MultiGPU context structure.
 */
struct __cumultigpu_st {
  CUthread * threads;           /** Background threads                        */
  int n;                        /** Number of background threads              */
};

static CUresult cu_thread_exit(const void * args) {
  (void)args;
  CUcontext context;
  CU_ERROR_CHECK(cuCtxGetCurrent(&context));
  CU_ERROR_CHECK(cuCtxDestroy(context));
  pthread_exit(NULL);
  return CUDA_SUCCESS;
}

static void destroy_task(void * args) {
  CUtask task = (CUtask)args;
  CUresult result;
  cuTaskDestroy(task, &result);
}

/**
 * Creates a multiGPU context with a single GPU context on each device given.
 * Each context is owned by a background thread and tasks are sent to them
 * asynchronously via a shared queue.
 *
 * @param multiGPU  the handle to the created context is returned through this
 *                  pointer.
 * @param devices   devices to create contexts on.
 * @param n         the number of devices.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
 *         CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OPERATING_SYSTEM,
 *         CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
 */
CUresult cuMultiGPUCreate(CUmultiGPU * multiGPU, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  // Allocate space for a new thread pool on the heap
  CUmultiGPU mGPU;
  if ((mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Set up the queue
  if (arrayqueue_create(&mGPU->tasks, (size_t)n) != 0) {
    free(mGPU);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Allocate space for the threads
  if ((mGPU->threads = malloc((size_t)n * sizeof(pthread_t))) == NULL) {
    arrayqueue_destroy(mGPU->tasks, destroy_task);
    free(mGPU);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Set up the mutex and condition variable
  mGPU->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  mGPU->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // From this point on the pool is functional even if no threads start so
  // assign the result so it can be destroyed later
  *multiGPU = mGPU;

  // Start the background threads
  for (mGPU->n = 0; mGPU->n < n; mGPU->n++) {
    struct thread_args * args;
    if ((args = malloc(sizeof(struct thread_args))) == NULL)
      return CUDA_ERROR_OUT_OF_MEMORY;

    args->multiGPU = mGPU;
    args->device = devices[mGPU->n];

    PTHREAD_ERROR_CHECK(pthread_create(&mGPU->threads[mGPU->n], NULL,
                                       cu_thread_main, args));
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys the multiGPU context.  This will block until all currently executing
 * tasks have completed.  Queued tasks will be destroyed.
 *
 * @param multiGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUDestroy(CUmultiGPU multiGPU) {
  // Lock the queue
  PTHREAD_ERROR_CHECK(pthread_mutex_lock(&multiGPU->mutex));

  // Destroy all tasks currently on the queue
  CUtask task;
  CUresult result;
  while (arrayqueue_pop(multiGPU->tasks, (void **)&task) == 0)
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));

  // Place one exit task for each thread on the queue
  CU_ERROR_CHECK(cuTaskCreate(&task, cu_thread_exit, NULL, 0));
  for (int i = 0; i < multiGPU->n; i++)
    arrayqueue_push(multiGPU->tasks, task);

  // Unlock the queue
  PTHREAD_ERROR_CHECK(pthread_mutex_unlock(&multiGPU->mutex));

  // Signal to the threads that there are tasks waiting
  PTHREAD_ERROR_CHECK(pthread_cond_broadcast(&multiGPU->cond));

  // Wait for each thread to exit
  for (int i = 0; i < multiGPU->n; i++)
    PTHREAD_ERROR_CHECK(pthread_join(multiGPU->threads[i], NULL));

  // Destroy the queue
  arrayqueue_destroy(multiGPU->tasks, destroy_task);

  free(multiGPU);

  return CUDA_SUCCESS;
}

/**
 * Schedules a task to run on a GPU managed by a background thread.
 *
 * @param multiGPU  the multiGPU context to use.
 * @param task      the task to schedule.
 * @return CUDA_SUCCESS,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUSchedule(CUmultiGPU multiGPU, CUtask task) {
  // Lock the mutex for the queue
  ERROR_CHECK(pthread_mutex_lock(&multiGPU->mutex));

  ERROR_CHECK(arrayqueue_push(multiGPU->tasks, task));

  // Unlock the queue mutex
  ERROR_CHECK(pthread_mutex_unlock(&multiGPU->mutex));

  // Signal to background threads that there are now tasks available
  ERROR_CHECK(pthread_cond_broadcast(&multiGPU->cond));

  return CUDA_SUCCESS;
}
