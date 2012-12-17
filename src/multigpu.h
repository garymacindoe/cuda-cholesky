#ifndef MULTIGPU_H
#define MULTIGPU_H

#include "cumultigpu.h"
#include "task.h"
#include <stddef.h>

/**
 * Schedules a task to run on a GPU managed by a background thread.
 *
 * @param multiGPU  the multiGPU context to use.
 * @param task      the task.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OPERATING_SYSTEM on failure.
 */
CUresult cuMultiGPUSchedule(CUmultiGPU, CUtask);

#endif
