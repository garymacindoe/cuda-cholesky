#include "../src/thread.h"
#include <stddef.h>
#include <assert.h>

struct args_st {
  const int x, y;
};

CUresult add(const void * args) {
  struct args_st * a = (struct args_st *)args;
  return a->x + a->y;
}

int main() {
  struct args_st args = { 2, 5 };

  CUtask task;
  assert(cuTaskCreate(&task, add, &args, sizeof(struct args_st)) == CUDA_SUCCESS);

  CUthread thread;
  assert(cuThreadCreate(&thread) == CUDA_SUCCESS);

  assert(cuThreadRunTask(thread, task) == CUDA_SUCCESS);

  CUresult result;
  assert(cuTaskDestroy(task, &result) == CUDA_SUCCESS);
  assert(result == 7);

  assert(cuThreadDestroy(thread) == CUDA_SUCCESS);

  return 0;
}
