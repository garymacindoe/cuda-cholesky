#ifndef CUHANDLE_H
#define CUHANDLE_H

#include <cuda.h>

typedef struct __cuhandle_st * CUhandle;

CUresult cuHandleCreate(CUhandle *, unsigned int, CUdevice);

CUresult cuHandleDestroy(CUhandle);

#endif
