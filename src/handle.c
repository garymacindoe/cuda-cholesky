#include "handle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 8 blas + lapack routines in 4 precisions
#define MODULE_COUNT 8
static const char * module_precisions[] = { "s", "d", "c", "z" };
static const char * module_names[] = { "gemm", "syrk", "herk", "trsm", "trmm",
                                       "potrf", "potri", "trtri" };
struct __CUhandle {
  CUcontext context;            /* The context */
  CUmodule modules[4][8];       /* The modules currently loaded into the context */
  CUstream * streams;           /* The streams currently active in the context */
  int nstreams, maxstreams;     /* Counters for arraylist of streams */
  CUdeviceptr linear;           /* Linear GPU memory allocated in the context */
  size_t size;                  /* The size of the current linear GPU memory allocation */
  CUdeviceptr pitched;          /* Pitched GPU memory allocated in the context */
  size_t widthInBytes, height, pitch;   /* The width, height and pitch of the current 2D GPU memory allocation */
  unsigned int elemSize;        /* The size of element the current 2D GPU memory allocation is aligned for */
  void * host_linear;           /* Pinned host memory allocated in the context */
  size_t host_size;             /* The size of the current linear host memory allocation */
  void * host_pitched;          /* Pinned pitched host memory allocated in the context */
  size_t host_widthInBytes, host_height, host_pitch;   /* The width, height and pitch of the current 2D host memory allocation */
  unsigned int host_elemSize;   /* The size of element the current 2D host memory allocation is aligned for */
};

/* Public API */
CUresult cuHandleCreate(CUhandle * handle, unsigned int flags, CUdevice device) {
  CUhandle res;
  CUresult error;
  if ((res = malloc(sizeof(struct __CUhandle))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Initialise context
  if ((error = cuCtxCreate(&res->context, flags, device)) != CUDA_SUCCESS)
    return error;

  // Initialise modules
  memset(&res->modules[CU_HANDLE_SINGLE], 0, MODULE_COUNT * sizeof(CUmodule));
  memset(&res->modules[CU_HANDLE_DOUBLE], 0, MODULE_COUNT * sizeof(CUmodule));
  memset(&res->modules[CU_HANDLE_COMPLEX], 0, MODULE_COUNT * sizeof(CUmodule));
  memset(&res->modules[CU_HANDLE_DOUBLE_COMPLEX], 0, MODULE_COUNT * sizeof(CUmodule));

  // Initialise array-backed list of streams
  res->nstreams = 0;
  res->maxstreams = 2;
  if ((res->streams = malloc((size_t)res->maxstreams * sizeof(CUstream))) == NULL) {
    cuHandleDestroy(res);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Initialise memory pointers
  res->linear = 0;
  res->pitched = 0;
  res->host_linear = NULL;
  res->host_pitched = NULL;

  // Assign the result
  *handle = res;

  return CUDA_SUCCESS;
}

CUresult cuHandleDestroy(CUhandle handle) {
  CUresult error;

  /* Unload any modules from the context */
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < MODULE_COUNT; j++) {
      if (handle->modules[i][j] != 0) {
        if ((error = cuModuleUnload(handle->modules[i][j])) != CUDA_SUCCESS)
          return error;
        handle->modules[i][j] = 0;
      }
    }
  }

  /* Destroy any streams */
  for (int i = 0; i < handle->nstreams; i++) {
    if ((error = cuStreamDestroy(handle->streams[i])) != CUDA_SUCCESS)
      return error;
  }

  /* Free any device memory allocated */
  if (handle->linear != 0) {
    if ((error = cuMemFree(handle->linear)) != CUDA_SUCCESS)
      return error;
    handle->linear = 0;
  }

  if (handle->pitched != 0) {
    if ((error = cuMemFree(handle->pitched)) != CUDA_SUCCESS)
      return error;
    handle->pitched = 0;
  }

  /* Free any host memory allocated */
  if (handle->host_linear != NULL) {
    if ((error = cuMemFreeHost(handle->host_linear)) != CUDA_SUCCESS)
      return error;
    handle->host_linear = NULL;
  }

  if (handle->host_pitched != NULL) {
    if ((error = cuMemFreeHost(handle->host_pitched)) != CUDA_SUCCESS)
      return error;
    handle->host_pitched = NULL;
  }

  /* Destroy the context */
  if ((error = cuCtxDestroy(handle->context)) != CUDA_SUCCESS)
    return error;

  free(handle);

  return CUDA_SUCCESS;
}

/* Internal API */
CUcontext cuHandleGetContext(CUhandle handle) {
  return handle->context;
}

CUresult cuHandleGetModule(CUhandle handle, CUmodule * module, int precision, int function) {
  // Check arguments
  if (precision < 0 || precision >= 4 ||
      function < 0 || function >= MODULE_COUNT)
    return CUDA_ERROR_INVALID_VALUE;

  // If the module is not in the cache
  if (handle->modules[precision][function] == 0) {
    // Format the name of the file containing the module from the module_precisions
    // and module_names lookup tables
    char file[14];
    snprintf(file, 14, "%s%s.fatbin", module_precisions[precision], module_names[function]);

    // Load the module into the cache
    CUresult error;
    if ((error = cuModuleLoad(&handle->modules[precision][function], file)) != CUDA_SUCCESS)
      return error;
  }

  // Set the result to point to the cached module
  *module = handle->modules[precision][function];

  return CUDA_SUCCESS;
}

/* Pre-allocated linear and 2D memory */
CUresult cuHandleMemAlloc(CUhandle handle, CUdeviceptr * ptr, size_t size) {
  CUresult error;

  // If no memory has been allocated already
  if (handle->linear == 0) {
    // Allocate new GPU memory and record the size
    if ((error = cuMemAlloc(&handle->linear, size)) != CUDA_SUCCESS)
      return error;
    handle->size = size;
  }
  // Else if more memory is required than already allocated
  else if (size > handle->size) {
    // Free the existing memory
    if ((error = cuMemFree(handle->linear)) != CUDA_SUCCESS)
      return error;
    handle->linear = 0;    // Set the pointer to zero in case the new allocation fails
    // Allocate the new memory
    if ((error = cuMemAlloc(&handle->linear, size)) != CUDA_SUCCESS)
      return error;
    handle->size = size;        // Update the size
  }

  *ptr = handle->linear;

  return CUDA_SUCCESS;
}

CUresult cuHandleMemAllocHost(CUhandle handle, void ** ptr, size_t size) {
  CUresult error;

  // If no memory has been allocated already
  if (handle->host_linear == 0) {
    // Allocate new GPU memory and record the size
    if ((error = cuMemAllocHost(&handle->host_linear, size)) != CUDA_SUCCESS)
      return error;
    handle->host_size = size;
  }
  // Else if more memory is required than already allocated
  else if (size > handle->host_size) {
    // Free the existing memory
    if ((error = cuMemFreeHost(handle->host_linear)) != CUDA_SUCCESS)
      return error;
    handle->host_linear = NULL;    // Set the pointer to NULL in case the new allocation fails
    // Allocate the new memory
    if ((error = cuMemAllocHost(&handle->host_linear, size)) != CUDA_SUCCESS)
      return error;
    handle->host_size = size;        // Update the size
  }

  *ptr = handle->host_linear;

  return CUDA_SUCCESS;
}

CUresult cuHandleMemAllocPitch(CUhandle handle, CUdeviceptr * ptr, size_t * pitch,
                                   size_t widthInBytes, size_t height, unsigned int elemSize) {
  CUresult error;

  // If no memory has been allocated already
  if (handle->pitched == 0) {
    // Allocate new GPU memory and record the size
    if ((error = cuMemAllocPitch(&handle->pitched, &handle->pitch, widthInBytes,
                                 height, elemSize)) != CUDA_SUCCESS)
      return error;
    handle->widthInBytes = widthInBytes;
    handle->height = height;
    handle->elemSize = elemSize;
  }
  // Else if more memory is required than already allocated
  else if (widthInBytes > handle->pitch || height > handle->height ||
           elemSize > handle->elemSize) {
    // Free the existing memory
    if ((error = cuMemFree(handle->pitched)) != CUDA_SUCCESS)
      return error;
    handle->pitched = 0;    // Set the pointer to zero in case the new allocation fails
    // Allocate the new memory
    if ((error = cuMemAllocPitch(&handle->pitched, &handle->pitch, widthInBytes,
                                 height, elemSize)) != CUDA_SUCCESS)
      return error;
    handle->widthInBytes = widthInBytes;        // Update the size
    handle->height = height;
    handle->elemSize = elemSize;
  }

  *ptr = handle->pitched;
  *pitch = handle->pitch;

  return CUDA_SUCCESS;
}

CUresult cuHandleMemAllocPitchHost(CUhandle handle, void ** ptr, size_t * pitch,
                                 size_t widthInBytes, size_t height, unsigned int elemSize) {
  CUresult error;

  // If no memory has been allocated already
  if (handle->host_pitched == 0) {
    // Allocate new GPU memory and record the size
    const size_t align = 16u / elemSize;
    if ((error = cuMemAllocHost(&handle->host_pitched,
                                (handle->host_pitch = (widthInBytes + align - 1) & ~(align - 1))
                                * height)) != CUDA_SUCCESS)
      return error;
    handle->host_widthInBytes = widthInBytes;
    handle->host_height = height;
    handle->host_elemSize = elemSize;
  }
  // Else if more memory is required than already allocated
  else if (widthInBytes > handle->host_pitch || height > handle->host_height ||
           elemSize > handle->host_elemSize) {
    // Free the existing memory
    if ((error = cuMemFreeHost(handle->host_pitched)) != CUDA_SUCCESS)
      return error;
    handle->host_pitched = 0;    // Set the pointer to zero in case the new allocation fails
    // Allocate the new memory
    const size_t align = 16u / elemSize;
    if ((error = cuMemAllocHost(&handle->host_pitched,
                                (handle->host_pitch = (widthInBytes + align - 1) & ~(align - 1))
                                * height)) != CUDA_SUCCESS)
      return error;
    handle->host_widthInBytes = widthInBytes;   // Update the size
    handle->host_height = height;
    handle->host_elemSize = elemSize;
  }

  *ptr = handle->host_pitched;
  *pitch = handle->host_pitch;

  return CUDA_SUCCESS;
}

/* Lazy allocated streams per context */
CUresult cuHandleGetStream(CUhandle handle, CUstream * stream, int i) {
  if (i < 0 || i > handle->nstreams)
    return CUDA_ERROR_INVALID_VALUE;

  // If i is equal to the number of streams this is the signal to create a new stream
  if (i == handle->nstreams) {
    // Enlarge the array if the maximum number of streams has been reached
    if (handle->nstreams + 1 >= handle->maxstreams) {
      CUstream * streams;
      if ((streams = realloc(handle->streams, (size_t)handle->maxstreams + 2)) == NULL)
        return CUDA_ERROR_OUT_OF_MEMORY;
      handle->streams = streams;
      handle->maxstreams += 2;
    }

    // Create a new stream in the list
    CUresult error;
    if ((error = cuStreamCreate(&handle->streams[i], 0)) != CUDA_SUCCESS)
      return error;

    // Increment the number of streams
    handle->nstreams++;
  }

  // Set the result to point to the stream in the list
  *stream = handle->streams[i];

  return CUDA_SUCCESS;
}
