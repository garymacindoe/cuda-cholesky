#include "cumultigpu.h"
#include "cutask.h"
#include "error.h"
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <string.h>

struct __cutask_st {
  CUresult (*function)(const void *);
  void * args;
  CUresult result;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool owner, complete;
};

typedef struct __cutaskqueue_st {
  CUtask * data;
  size_t head, tail, n;
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t empty, nonEmpty;
  CUcontext context;
} * CUtaskqueue;

static void * thread_main(void * q) {
  CUtaskqueue queue = (CUtaskqueue)q;

  CU_ERROR_CHECK_VAL(cuCtxSetCurrent(q->context), NULL);

  while (true) {
    ERROR_CHECK_VAL(pthread_mutex_lock(&queue->mutex), strerror, NULL);

    while (queue->head == queue->tail)
      ERROR_CHECK_VAL(pthread_cond_wait(&queue->nonEmpty, &queue->mutex), strerror, NULL);

    CUtask task = queue->data[queue->head++];
    if (queue->head == queue->n)
      queue->head = 0;

    ERROR_CHECK_VAL(pthread_mutex_unlock(&queue->mutex), strerror, NULL);

    if (task->function == NULL)
      break;

    ERROR_CHECK_VAL(pthread_mutex_lock(&task->mutex), strerror, NULL);

    task->result = task->function(task->args);
    task->complete = true;

    ERROR_CHECK_VAL(pthread_mutex_unlock(&task->mutex), strerror, NULL);
    pthread_cond_signal(&task->cond);

    if (task->owner)
      free(task->args);
  }

  return NULL;
}

/**
 * Creates a new task queue and starts a background thread to execute tasks
 * submitted to the queue.
 *
 * @param queue   the task queue to create.
 * @param device  the device to execute tasks on.
 * @param flags   context creation flags.
 * @return ENOMEM if the queue could not be allocated, EAGAIN if there are
 *         insufficient resources to create another thread.
 */
static int cuTaskqueueCreate(CUtaskqueue * queue, CUcontext context) {
  queue->head = 0;
  queue->tail = 0;
  queue->n = 16;
  queue->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  queue->nonEmpty = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
  queue->device = device;
  queue->flags = flags;

  if ((queue->data = malloc(queue->n * sizeof(task_t))) == NULL)
    return ENOMEM;

  int error = pthread_create(&queue->thread, NULL, thread_main, queue);
  if (error != 0) {
    free(queue->data);
    return error;
  }

  return 0;
}

static int taskqueue_destroy(taskqueue_t queue) {
  pthread_mutex_lock(&queue->mutex);
  queue->data[queue->tail++] = (CUtask){ NULL, NULL, CUDA_SUCCESS,
                                         PTHREAD_MUTEX_INITIALIZER,
                                         PTHREAD_COND_INITIALIZER, false, false };
  pthread_mutex_unlock(&queue->mutex);
  pthread_cond_signal(&queue->nonEmpty);
  pthread_join(queue->thread, NULL);

  free(queue->data);

  return 0;
}

static int taskqueue_push(taskqueue_t queue, CUresult (*function)(const void *), const void * args, size_t size) {
  CUtask task;
  task.function = function;
  if ((task.args = malloc(size)) == NULL)
    return ENOMEM;
  task.args = memcpy(task.args, args, size);
  task.mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  task.cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
  task.owner = true;
  task.complete = false;

  pthread_mutex_lock(&queue->mutex);

  queue->data[queue->tail++] = task;

  if (queue->tail == queue->n)
    queue->tail = 0;

  if (queue->tail == queue->head) {
    size_t n = queue->n * 2;
    task_t * ptr = malloc(n * sizeof(task_t));
    if (ptr == NULL) {
      free(task.args);
      return ENOMEM;
    }

    if (queue->head < queue->tail)
      memcpy(ptr, &queue->data[queue->head], (queue->tail - queue->head) * sizeof(task_t));
    else {
      memcpy(ptr, &queue->data[queue->head], (queue->n - queue->head) * sizeof(task_t));
      memcpy(&ptr[queue->n - queue->head], queue->data, queue->tail * sizeof(task_t));
    }
    queue->data = ptr;
    queue->head = 0;
    queue->tail = queue->n;
    queue->n = n;
  }

  pthread_mutex_unlock(&queue->mutex);
  pthread_cond_signal(&queue->nonEmpty);

  return CUDA_SUCCESS;
}

static void synchronise(const void * args) {
  pthread_barrier_t ** barrier = (pthread_barrier_t **)args;
  CU_ERROR_CHECK_NORETURN(cuCtxSynchronize());
  PTHREAD_ERROR_CHECK_EXIT(pthread_barrier_wait(*barrier));
}

static inline CUresult taskqueue_synchronise(taskqueue_t * queue) {
  pthread_barrier_t * barrier;
  if ((barrier = malloc(sizeof(pthread_barrier_t))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  barrier->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  barrier->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
  barrier->count = 2;

  CU_ERROR_CHECK(taskqueue_push(queue, synchronise, &barrier, sizeof(pthread_barrier_t *)));

  PTHREAD_ERROR_CHECK(pthread_barrier_wait(barrier));

  free(barrier);

  return CUDA_SUCCESS;
}

struct __cumultigpu_st {
  taskqueue_t * queues;
  int n;
};

/* Internal API */
CUresult cuMultiGPUSchedule(CUmultiGPU multiGPU, int i, void (* function)(const void *),
                            const void * args, size_t size) {
  if (function == NULL || (args == NULL && size > 0) || i >= multiGPU->n)
    return CUDA_ERROR_INVALID_VALUE;

  return taskqueue_push(&multiGPU->queues[i], function, args, size);
}

int cuMultiGPUGetNumberOfContexts(CUmultiGPU multiGPU) {
  return multiGPU->n;
}

/* Public API */

CUresult cuMultiGPUCreate(CUmultiGPU * multiGPU, unsigned int flags, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  if ((*multiGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  if (((*multiGPU)->queues = malloc((size_t)n * sizeof(taskqueue_t))) == NULL) {
    free(*multiGPU);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  (*multiGPU)->n = n;

  for (int i = 0; i < n; i++)
    CU_ERROR_CHECK(taskqueue_create(&(*multiGPU)->queues[i], devices[i], flags));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDestroy(CUmultiGPU multiGPU) {
  for (int i = 0; i < multiGPU->n; i++)
    CU_ERROR_CHECK(taskqueue_destroy(&multiGPU->queues[i]));
  free(multiGPU);
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSynchronize(CUmultiGPU multiGPU) {
  for (int i = 0; i < multiGPU->n; i++)
    CU_ERROR_CHECK(taskqueue_synchronise(&multiGPU->queues[i]));
  return CUDA_SUCCESS;
}
