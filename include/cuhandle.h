#ifndef CUHANDLE_H
#define CUHANDLE_H

#include <cuda.h>

typedef struct __CUhandle * CUhandle;

/**
 * Creates a handle to pass to CUDA functions that contains a CUDA context along
 * with caches for modules, streams and temporary memory on the host and device.
 *
 * @param handle  the newly created handle is returned through this pointer.
 * @param flags   context creation flags.
 * @param device  the device to create the context on.
 * @return CUDA_ERROR_OUT_OF_MEMORY or any other errors associated with context
 *         creation.
 */
CUresult cuHandleCreate(CUhandle *, unsigned int, CUdevice);

/**
 * Destroys a handle unloading any modules in the cache, destroying any streams
 * and freeing temporary memory before destroying the context itself.
 *
 * @param handle  the handle to destroy.
 * @return any error associated with freeing memory, destroying streams,
 *         unloading modules or destroying contexts.
 */
CUresult cuHandleDestroy(CUhandle);

#endif
