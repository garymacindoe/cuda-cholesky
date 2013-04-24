#ifndef HANDLE_H
#define HANDLE_H

struct __cublashandle_st {
  CUcontext context;
  CUmodule sgemm, ssyrk, strsm, strmm;
  CUmodule cgemm, cherk, ctrsm, ctrmm;
  CUmodule dgemm, dsyrk, dtrsm, dtrmm;
  CUmodule zgemm, zherk, ztrsm, ztrmm;
  bool contextOwner;
};

struct __cumultigpublashandle_st {
  CUmultiGPU mGPU;
  struct __cublashandle_st * handles;
};

#endif
