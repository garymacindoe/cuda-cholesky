static CUresult set(uint32_t seed, CUdeviceptr state, CUstream stream) {
  (void)stream;

  uint32_t * h_state;
  CU_ERROR_CHECK(cuMemAllocHost((void **)&h_state, PARAMS_NUM * N * sizeof(uint32_t)));

  for (size_t i = 0; i < PARAMS_NUM; i++) {
    uint32_t * array = &h_state[i * N];
    uint32_t hidden_seed = mtgp32dc_params[i].tbl[4] ^ (mtgp32dc_params[i].tbl[8] << 16);
    uint32_t tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(uint32_t) * N);
    array[0] = seed;
    array[1] = hidden_seed;
    for (size_t j = 1; j < N; j++)
      array[j] ^= UINT32_C(1812433253) * (array[j - 1] ^ (array[j - 1] >> 30)) + (uint32_t)j;
  }

  CU_ERROR_CHECK(cuMemcpyHtoD(state, h_state, PARAMS_NUM * N * sizeof(uint32_t)));

  CU_ERROR_CHECK(cuMemFreeHost(h_state));

  return CUDA_SUCCESS;
}

static inline CUresult sample(const char * name, CUdeviceptr state, CUdeviceptr data, size_t inc, size_t n, CUstream stream) {
  CUdevice device;
  int major, minor, smem, mp;
  CU_ERROR_CHECK(cuCtxGetDevice(&device));
  CU_ERROR_CHECK(cuDeviceComputeCapability(&major, &minor, device));
  CU_ERROR_CHECK(cuDeviceGetAttribute(&smem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
  CU_ERROR_CHECK(cuDeviceGetAttribute(  &mp, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,        device));

  const int num_blocks = MIN(smem / (int)((FLOOR_2P * 3) * sizeof(uint32_t)), ((major * 100 + minor <= 101) ? 768 : 1024) / FLOOR_2P) * mp;
  size_t samples_per_block = n / (size_t)num_blocks;

  static CUmodule module = NULL;

  if (module == NULL) {
    CU_ERROR_CHECK(cuModuleLoadResource(&module, "mtgp32_" STRING(MEXP)));

    // Set up parameters in constant device memory
    CUdeviceptr dPosTbl,  dSh1Tbl,  dSh2Tbl, dMask;
    size_t posSize, sh1Size, sh2Size, maskSize;
    CU_ERROR_CHECK(cuModuleGetGlobal(&dPosTbl,  &posSize, module, "pos_tbl"));
    CU_ERROR_CHECK(cuModuleGetGlobal(&dSh1Tbl,  &sh1Size, module, "sh1_tbl"));
    CU_ERROR_CHECK(cuModuleGetGlobal(&dSh2Tbl,  &sh2Size, module, "sh2_tbl"));
    CU_ERROR_CHECK(cuModuleGetGlobal(  &dMask, &maskSize, module, "mask"));

    uint32_t  *  posTbl, * sh1Tbl, * sh2Tbl;
    CU_ERROR_CHECK(cuMemAllocHost((void **)&posTbl, posSize));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&sh1Tbl, sh1Size));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&sh2Tbl, sh2Size));

    for (size_t i = 0; i < PARAMS_NUM; i++) {
      posTbl[i] = mtgp32dc_params[i].pos;
      sh1Tbl[i] = mtgp32dc_params[i].sh1;
      sh2Tbl[i] = mtgp32dc_params[i].sh2;
    }

    CU_ERROR_CHECK(cuMemcpyHtoD(dPosTbl, posTbl, posSize));
    CU_ERROR_CHECK(cuMemcpyHtoD(dSh1Tbl, sh1Tbl, sh1Size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dSh2Tbl, sh2Tbl, sh2Size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dMask, &mtgp32dc_params[0].mask, maskSize));

    CU_ERROR_CHECK(cuMemFreeHost(posTbl));
    CU_ERROR_CHECK(cuMemFreeHost(sh1Tbl));
    CU_ERROR_CHECK(cuMemFreeHost(sh2Tbl));

    // Set up parameters in texture memory
    const size_t size = sizeof(uint32_t) * PARAMS_NUM * 16;

    CUtexref tParam, tTemper, tSingle;
    CU_ERROR_CHECK(cuModuleGetTexRef( &tParam, module, "tex_param_ref"));
    CU_ERROR_CHECK(cuModuleGetTexRef(&tTemper, module, "tex_temper_ref"));
    CU_ERROR_CHECK(cuModuleGetTexRef(&tSingle, module, "tex_single_ref"));

    CUdeviceptr dParam, dTemper, dSingle;
    CU_ERROR_CHECK(cuMemAlloc( &dParam, size));
    CU_ERROR_CHECK(cuMemAlloc(&dTemper, size));
    CU_ERROR_CHECK(cuMemAlloc(&dSingle, size));

    uint32_t * hParam, * hTemper, * hSingle;
    CU_ERROR_CHECK(cuMemAllocHost((void **) &hParam, size));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&hTemper, size));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&hSingle, size));

    for (size_t i = 0; i < PARAMS_NUM; i++) {
      for (size_t j = 0; j < 16; j++) {
        hParam [i * 16 + j] = mtgp32dc_params[i].tbl[j];
        hTemper[i * 16 + j] = mtgp32dc_params[i].tmp_tbl[j];
        hSingle[i * 16 + j] = mtgp32dc_params[i].flt_tmp_tbl[j];
      }
    }

    CU_ERROR_CHECK(cuMemcpyHtoD( dParam,  hParam, size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dTemper, hTemper, size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dSingle, hSingle, size));

    CU_ERROR_CHECK(cuTexRefSetFilterMode( tParam, CU_TR_FILTER_MODE_POINT));
    CU_ERROR_CHECK(cuTexRefSetFilterMode(tTemper, CU_TR_FILTER_MODE_POINT));
    CU_ERROR_CHECK(cuTexRefSetFilterMode(tSingle, CU_TR_FILTER_MODE_POINT));

    CU_ERROR_CHECK(cuTexRefSetAddress(NULL,  tParam,  dParam, size));
    CU_ERROR_CHECK(cuTexRefSetAddress(NULL, tTemper, dTemper, size));
    CU_ERROR_CHECK(cuTexRefSetAddress(NULL, tSingle, dSingle, size));

    CU_ERROR_CHECK(cuMemFreeHost( hParam));
    CU_ERROR_CHECK(cuMemFreeHost(hTemper));
    CU_ERROR_CHECK(cuMemFreeHost(hSingle));
  }

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &state, &data, &inc, &samples_per_block };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)num_blocks, 1, 1, FLOOR_2P, 1, 1, (unsigned int)((FLOOR_2P * 3) * sizeof(uint64_t)), stream, params, NULL));

  return CUDA_SUCCESS;
}

static CUresult get(CUvectoru32 * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ej7convertEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getOpenOpen(CUvectorf * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ef17convert_open_openEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getOpenClose(CUvectorf * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ef18convert_open_closeEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getCloseOpen(CUvectorf * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ef18convert_close_openEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getCloseClose(CUvectorf * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ef19convert_close_closeEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUrng32_t type = { NAME, PARAMS_NUM * N * sizeof(uint32_t), UINT32_C(0), UINT32_C(0xffffffff), set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const CUrng32_t * RNG_T = &type;
