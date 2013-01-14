#include "cumultigpu.h"
#include "error.h"

#ifdef MGPU_SEQ
// Single threaded versions of CUtask and CUmultiGPU.

/**
 * Task structure.
 */
struct __cutask_st {
  CUresult (*function)(const void *);  /** The function to run                */
  void * args;                         /** Arguments for the function         */
  CUresult result;                     /** Result of the function             */
};

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
CUresult cuTaskCreate(CUtask * task, CUresult (*function)(const void *),
                      const void * args, size_t size) {
  if (function == NULL || (args == NULL && size > 0))
    return CUDA_ERROR_INVALID_VALUE;

  if (((*task) = malloc(sizeof(struct __cutask_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*task)->function = function;

  // Allocate space on heap for arguments so that they can be accessed by other
  // threads
  if (((*task)->args = malloc(size)) == NULL) {
    free(*task);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Copy arguments onto heap
  (*task)->args = memcpy((*task)->args, args, size);

  return CUDA_SUCCESS;
}

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskDestroy(CUtask task, CUresult * result) {
  // Copy the result
  if (result != NULL)
    *result = task->result;

  // Free the task and arguments
  free(task->args);
  free(task);

  return CUDA_SUCCESS;
}

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskExecute(CUtask task) {
  // Run the task function using the arguments and assign the result
  task->result = task->function(task->args);
  return CUDA_SUCCESS;
}

/**
 * MultiGPU context.  Single threaded version is simply an array of CUDA
 * contexts.
 */
struct __cumultigpu_st {
  CUcontext * contexts;
  int n;
};

/**
 * Creates a multiGPU context with a single CUDA context created on each of the
 * devices given.
 *
 * @param mGPU     the newly created context is returned through this pointer.
 * @param devices  the CUDA devices to use.
 * @param n        the number of CUDA devices to use.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUCreate(CUmultiGPU * mGPU, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  if ((*mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  if (((*mGPU)->contexts = malloc((size_t)n * sizeof(CUcontext))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*mGPU)->n = n;
  for (int i = 0; i < n; i++) {
    CU_ERROR_CHECK(cuCtxCreate(&(*mGPU)->contexts[i], CU_CTX_SCHED_AUTO, devices[i]));
    CU_ERROR_CHECK(cuCtxPopCurrent(&(*mGPU)->contexts[i]));
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys a multiGPU context.
 *
 * @param mGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUDestroy(CUmultiGPU mGPU) {
  for (int i = 0; i < mGPU->n; i++)
    CU_ERROR_CHECK(cuCtxDestroy(mGPU->contexts[i]));
  free(mGPU->contexts);
  free(mGPU);
  return CUDA_SUCCESS;
}

/**
 * Runs a task using a particular CUDA context.
 *
 * @param mGPU  the multiGPU context.
 * @param i     the index of the CUDA context to use.
 * @param task  the task to run.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPURunTask(CUmultiGPU mGPU, int i, CUtask task) {
  if (i < 0 || i >= mGPU->n)
    return CUDA_ERROR_INVALID_VALUE;

  CU_ERROR_CHECK(cuCtxPushCurrent(mGPU->contexts[i]));
  CU_ERROR_CHECK(cuTaskExecute(task));
  CU_ERROR_CHECK(cuCtxPopCurrent(&mGPU->contexts[i]));

  return CUDA_SUCCESS;
}

/**
 * Synchronises all contexts in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return any errors.
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU mGPU) {
  for (int i = 0; i < mGPU->n; i++) {
    CU_ERROR_CHECK(cuCtxPushCurrent(mGPU->contexts[i]));
    CU_ERROR_CHECK(cuCtxSynchronize());
    CU_ERROR_CHECK(cuCtxPopCurrent(&mGPU->contexts[i]));
  }
  return CUDA_SUCCESS;
}

#else
// Multi-threaded versions of CUtask and CUmultiGPU.
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>

/**
 * An implementation of a queue of CUtasks backed by a resizable array.
 */
typedef struct {
  CUtask * data;                /** Array of pointers to tasks in the queue.  */
  size_t head, tail, capacity;  /** Indices to head and tail of queue and queue
                                    capacity. */
} CUtaskqueue;

/**
 * Creates a new queue with at least the specified capacity.
 *
 * @param queue     the newly created queue is returned through this pointer.
 * @param capacity  the initial capacity of the queue.
 * @return CUDA_SUCCESS if the queue was created successfully,
 *         CUDA_ERROR_OUT_OF_MEMORY if there isn't enough memory.
 */
static inline CUresult cuTaskQueueCreate(CUtaskqueue * queue, size_t capacity) {
  // Initialise head, tail and capacity
  queue->head = 0;
  queue->tail = 0;
  queue->capacity = capacity;

  // Allocate space for objects
  if ((queue->data = malloc(queue->capacity * sizeof(CUtask))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  return CUDA_SUCCESS;
}

/**
 * Destroys the queue.
 *
 * @param queue  the queue to destroy.
 */
static inline void cuTaskQueueDestroy(CUtaskqueue * queue) {
  // Free the data
  free(queue->data);
}

/**
 * Checks whether the queue is empty.
 *
 * @param queue  the queue.
 * @return <b>true</b> if the queue is empty, <b>false</b> otherwise.
 */
static inline bool cuTaskQueueIsEmpty(const CUtaskqueue * queue) {
  return (queue->head == queue->tail);
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
static inline CUresult cuTaskQueuePush(CUtaskqueue * queue, CUtask task) {
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
 */
static inline void cuTaskQueuePop(CUtaskqueue * queue, CUtask * task) {
  *task = queue->data[queue->head++];

  // Wrap the head around
  if (queue->head == queue->capacity)
    queue->head = 0;
}

/**
 * Background thread type.
 */
typedef struct __cuthread_st {
  CUtaskqueue queue;            /** Queue of tasks                            */
  pthread_t thread;             /** Background thread                         */
  pthread_mutex_t mutex;        /** Mutex to protect queue                    */
  pthread_cond_t nonEmpty;      /** Condition to wait on non-empty queue      */
  CUresult error;               /** Thread error status                       */
} * CUthread;

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
    while (cuTaskQueueIsEmpty(&this->queue))
      THREAD_ERROR_CHECK(pthread_cond_wait(&this->nonEmpty, &this->mutex));

    // Remove a task from the head of the queue
    CUtask task;
    cuTaskQueuePop(&this->queue, &task);

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
static inline CUresult cuThreadCreate(CUthread * thread) {
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
static inline CUresult cuThreadDestroy(CUthread thread) {
  // Lock the queue
  ERROR_CHECK(pthread_mutex_lock(&thread->mutex));

  // Place a NULL task on the queue to signal the thread to exit the main loop
  CU_ERROR_CHECK(cuTaskQueuePush(&thread->queue, NULL));

  // Unlock the queue
  ERROR_CHECK(pthread_mutex_unlock(&thread->mutex));

  // Signal to the thread that there are tasks waiting
  ERROR_CHECK(pthread_cond_signal(&thread->nonEmpty));

  // Wait for the thread to exit
  ERROR_CHECK(pthread_join(thread->thread, (void **)&thread));

  // Destroy the queue
  cuTaskQueueDestroy(&thread->queue);

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
static inline CUresult cuThreadRunTask(CUthread thread, CUtask task) {
  // Lock the mutex for the queue
  ERROR_CHECK(pthread_mutex_lock(&thread->mutex));

  // Place the task in the queue
  CU_ERROR_CHECK(cuTaskQueuePush(&thread->queue, task));

  // Unlock the queue mutex
  ERROR_CHECK(pthread_mutex_unlock(&thread->mutex));

  // Signal to background thread that there are now tasks available
  ERROR_CHECK(pthread_cond_signal(&thread->nonEmpty));

  return CUDA_SUCCESS;
}

/**
 * Task structure.
 */
struct __cutask_st {
  CUresult (*function)(const void *);  /** The function to run                */
  void * args;                         /** Arguments for the function         */
  CUresult result;                     /** Result of the function             */
  bool complete;                       /** Flag set when function is finished */
  pthread_mutex_t mutex;               /** Mutex to protect access to result and
                                           flag                               */
  pthread_cond_t cond;                 /** Condition to wait on function
                                           completion                         */
};

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
CUresult cuTaskCreate(CUtask * task, CUresult (*function)(const void *),
                      const void * args, size_t size) {
  if (function == NULL || (args == NULL && size > 0))
    return CUDA_ERROR_INVALID_VALUE;

  if (((*task) = malloc(sizeof(struct __cutask_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*task)->function = function;
  (*task)->complete = false;
  (*task)->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  (*task)->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // Allocate space on heap for arguments so that they can be accessed by other
  // threads
  if (((*task)->args = malloc(size)) == NULL) {
    free(*task);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Copy arguments onto heap
  (*task)->args = memcpy((*task)->args, args, size);

  return CUDA_SUCCESS;
}

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskDestroy(CUtask task, CUresult * result) {
  // Lock the mutex for the task
  ERROR_CHECK(pthread_mutex_lock(&task->mutex));

  // Wait until the task has been completed
  while (!task->complete)
    ERROR_CHECK(pthread_cond_wait(&task->cond, &task->mutex));

  // Copy the result
  if (result != NULL)
    *result = task->result;

  // Unlock the task mutex
  ERROR_CHECK(pthread_mutex_unlock(&task->mutex));

  // Free the task and arguments
  free(task->args);
  free(task);

  return CUDA_SUCCESS;
}

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskExecute(CUtask task) {
  // Lock the mutex for the task
  ERROR_CHECK(pthread_mutex_lock(&task->mutex));

  // Run the task function using the arguments and assign the result
  task->result = task->function(task->args);

  // Set the task as completed
  task->complete = true;

  // Unlock the task mutex
  ERROR_CHECK(pthread_mutex_unlock(&task->mutex));

  // Signal to waiting threads that the task has now completed
  ERROR_CHECK(pthread_cond_signal(&task->cond));

  return CUDA_SUCCESS;
}

/**
 * MultiGPU context.  Multithreaded version is an array of CUthreads.
 */
struct __cumultigpu_st {
  CUthread * threads;
  int n;
};

static CUresult createContext(const void * args) {
  CUdevice * device = (CUdevice *)args;
  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_YIELD, *device));
  return CUDA_SUCCESS;
}

static CUresult destroyContext(const void * args) {
  (void)args;

  CUcontext context;
  CU_ERROR_CHECK(cuCtxGetCurrent(&context));
  CU_ERROR_CHECK(cuCtxDestroy(context));

  return CUDA_SUCCESS;
}

/**
 * Creates a multiGPU context with a single CUDA context created on each of the
 * devices given.
 *
 * @param mGPU     the newly created context is returned through this pointer.
 * @param devices  the CUDA devices to use.
 * @param n        the number of CUDA devices to use.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUCreate(CUmultiGPU * mGPU, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  if ((*mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  if (((*mGPU)->threads = malloc((size_t)n * sizeof(CUthread))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*mGPU)->n = n;
  for (int i = 0; i < n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, createContext, &devices[i], sizeof(CUdevice)));

    CU_ERROR_CHECK(cuThreadCreate(&(*mGPU)->threads[i]));
    CU_ERROR_CHECK(cuThreadRunTask((*mGPU)->threads[i], task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return (int)result;
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys a multiGPU context.
 *
 * @param mGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUDestroy(CUmultiGPU mGPU) {
  for (int i = 0; i < mGPU->n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, destroyContext, NULL, 0));
    CU_ERROR_CHECK(cuThreadRunTask(mGPU->threads[i], task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;

    CU_ERROR_CHECK(cuThreadDestroy(mGPU->threads[i]));
  }
  free(mGPU->threads);
  free(mGPU);
  return CUDA_SUCCESS;
}

/**
 * Runs a task using a particular CUDA context.
 *
 * @param mGPU  the multiGPU context.
 * @param i     the index of the CUDA context to use.
 * @param task  the task to run.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPURunTask(CUmultiGPU mGPU, int i, CUtask task) {
  if (i < 0 || i >= mGPU->n)
    return CUDA_ERROR_INVALID_VALUE;
  CU_ERROR_CHECK(cuThreadRunTask(mGPU->threads[i], task));
  return CUDA_SUCCESS;
}

static CUresult synchronize() {
  CU_ERROR_CHECK(cuCtxSynchronize());
  return CUDA_SUCCESS;
}

/**
 * Synchronises all contexts in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return any errors.
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU mGPU) {
  CUtask task;
  CUresult result;

  for (int i = 0; i < mGPU->n; i++) {
    CU_ERROR_CHECK(cuTaskCreate(&task, synchronize, NULL, 0));
    CU_ERROR_CHECK(cuMultiGPURunTask(mGPU, i, task));
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }

  return CUDA_SUCCESS;
}

#endif

/**
 * Gets the number of contexts available in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return the number of contexts in the multiGPU context.
 */
int cuMultiGPUGetContextCount(CUmultiGPU mGPU) {
  return mGPU->n;
}
