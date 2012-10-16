#include "error.h"

#include <stdio.h>

static void defaultErrorHandler(const char * call, const char * function, const char * file, unsigned int line, int error, strerror_t strerror) {
#ifdef HAS_CULA
  if (strerror == culaGetErrorString) {
    if (error == culaArgumentError)
      fprintf(stderr, "Error:\n\t%s returned %d (Argument %d has an illegal value) in %s (%s:%u)\n", call, error, culaGetErrorInfo(), function, file, line);
    else if (error = culaDataError)
      fprintf(stderr, "Error:\n\t%s returned %d (CULA Data Error with code %d) in %s (%s:%u)\n", call, error, culaGetErrorInfo(), function, file, line);
    else
      fprintf(stderr, "Error:\n\t%s returned %d in %s (%s:%u)\n", call, error, strerror(error), function, file, line);
  }
  else
#endif
  if (strerror != NULL)
    fprintf(stderr, "Error:\n\t%s returned %d (%s) in %s (%s:%u)\n", call, error, strerror(error), function, file, line);
  else
    fprintf(stderr, "Error:\n\t%s returned %d in %s (%s:%u)\n", call, error, function, file, line);
}

errorHandler_t errorHandler = &defaultErrorHandler;

const char * cuGetErrorString(CUresult result) {
  switch (result) {
    case CUDA_SUCCESS:                             return "No errors";
    case CUDA_ERROR_INVALID_VALUE:                 return "Invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY:                 return "Out of memory";
    case CUDA_ERROR_NOT_INITIALIZED:               return "Driver not initialized";
    case CUDA_ERROR_DEINITIALIZED:                 return "Driver deinitialized";
    case CUDA_ERROR_PROFILER_DISABLED:             return "Profiler disabled";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:      return "Profiler not initialized";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:      return "Profiler already started";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:      return "Profiler already stopped";
    case CUDA_ERROR_NO_DEVICE:                     return "No CUDA-capable device available";
    case CUDA_ERROR_INVALID_DEVICE:                return "Invalid device";
    case CUDA_ERROR_INVALID_IMAGE:                 return "Invalid kernel image";
    case CUDA_ERROR_INVALID_CONTEXT:               return "Invalid context";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:       return "Context already current";
    case CUDA_ERROR_MAP_FAILED:                    return "Map failed";
    case CUDA_ERROR_UNMAP_FAILED:                  return "Unmap failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED:               return "Array is mapped";
    case CUDA_ERROR_ALREADY_MAPPED:                return "Already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:             return "No binary for GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:              return "Already acquired";
    case CUDA_ERROR_NOT_MAPPED:                    return "Not mapped";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:           return "Not mapped as array";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:         return "Not mapped as pointer";
    case CUDA_ERROR_ECC_UNCORRECTABLE:             return "Uncorrectable ECC error";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:             return "Unsupported CUlimit";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:        return "Context already in use";
    case CUDA_ERROR_INVALID_SOURCE:                return "Invalid source";
    case CUDA_ERROR_FILE_NOT_FOUND:                return "File not found";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Shared object symbol not found";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:     return "Shared object initialization failed";
    case CUDA_ERROR_OPERATING_SYSTEM:              return "Operating System call failed";
    case CUDA_ERROR_INVALID_HANDLE:                return "Invalid handle";
    case CUDA_ERROR_NOT_FOUND:                     return "Not found";
    case CUDA_ERROR_NOT_READY:                     return "CUDA not ready";
    case CUDA_ERROR_LAUNCH_FAILED:                 return "Launch failed";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:       return "Launch exceeded resources";
    case CUDA_ERROR_LAUNCH_TIMEOUT:                return "Launch exceeded timeout";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:   return "Peer access already enabled";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:       return "Peer access not enabled";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:        return "Primary context active";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:          return "Context is destroyed";
    case CUDA_ERROR_ASSERT:                        return "Device assert failed";
    case CUDA_ERROR_TOO_MANY_PEERS:                return "Too many peers";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:return "Host memory already registered";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:    return "Host memory not registered";
    case CUDA_ERROR_UNKNOWN:                       return "Unknown error";
    default:                                       return "Unknown error code";
  }
}

const char * cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:          return "operation completed successfully";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS library not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "resource allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:    return "unsupported numerical value was passed to function";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "function requires an architectural feature absent from the architecture of the device";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "access to GPU memory space failed";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "GPU program failed to execute";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "an internal CUBLAS operation failed";
    default:                             return "unknown error code";
  }
}
