#ifndef HANDLE_H
#define HANDLE_H

struct __cublashandle_st {
  CUcontext context;
  CUmodule sgemm, ssyrk, strsm, strmm2;
  bool contextOwner;
};

struct __cumultigpublashandle_st {
  CUmultiGPU mGPU;
  struct __cublashandle_st * handles;
};

#endif
