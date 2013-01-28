static CUresult set(uint64_t seed, CUdeviceptr state, CUstream stream) {
  (void)stream;

  uint64_t * h_state;
  CU_ERROR_CHECK(cuMemAllocHost((void **)&h_state, PARAMS_NUM * N * sizeof(uint64_t)));

  for (size_t i = 0; i < PARAMS_NUM; i++) {
    uint64_t * array = &h_state[i * N];
    uint64_t hidden_seed = mtgp64dc_params[i].tbl[4] ^ (mtgp64dc_params[i].tbl[8] << 16);
    uint64_t tmp = hidden_seed >> 32;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(uint64_t) * N);
    array[0] = seed;
    array[1] = hidden_seed;
    for (size_t j = 1; j < N; j++)
      array[j] ^= UINT64_C(6364136223846793005) * (array[j - 1] ^ (array[j - 1] >> 62)) + (uint64_t)j;
  }

  CU_ERROR_CHECK(cuMemcpyHtoD(state, h_state, PARAMS_NUM * N * sizeof(uint64_t)));

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

  const int num_blocks = MIN(smem / (int)((FLOOR_2P * 3) * sizeof(uint64_t)), ((major * 100 + minor <= 101) ? 768 : 1024) / FLOOR_2P) * mp;
  size_t samples_per_block = n / (size_t)num_blocks;

  static CUmodule module = NULL;

  if (module == NULL) {
    CU_ERROR_CHECK(cuModuleLoadResource(&module, "mtgp64_" STRING(MEXP)));

    // Set up parameters in constant device memory
    CUdeviceptr dPosTbl,  dSh1Tbl,  dSh2Tbl, dHighMask, dLowMask;
    size_t posSize, sh1Size, sh2Size;
    CU_ERROR_CHECK(cuModuleGetGlobal(  &dPosTbl, &posSize, module, "pos_tbl"));
    CU_ERROR_CHECK(cuModuleGetGlobal(  &dSh1Tbl, &sh1Size, module, "sh1_tbl"));
    CU_ERROR_CHECK(cuModuleGetGlobal(  &dSh2Tbl, &sh2Size, module, "sh2_tbl"));
    CU_ERROR_CHECK(cuModuleGetGlobal(&dHighMask,     NULL, module, "high_mask"));
    CU_ERROR_CHECK(cuModuleGetGlobal( &dLowMask,     NULL, module, "low_mask"));

    uint32_t  *  posTbl, * sh1Tbl, * sh2Tbl;
    CU_ERROR_CHECK(cuMemAllocHost((void **)&posTbl, posSize));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&sh1Tbl, sh1Size));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&sh2Tbl, sh2Size));

    uint32_t highMask = (uint32_t)(mtgp64dc_params[0].mask >> 32);
    uint32_t  lowMask = mtgp64dc_params[0].mask & 0xffffffff;

    for (size_t i = 0; i < PARAMS_NUM; i++) {
      posTbl[i] = mtgp64dc_params[i].pos;
      sh1Tbl[i] = mtgp64dc_params[i].sh1;
      sh2Tbl[i] = mtgp64dc_params[i].sh2;
    }

    CU_ERROR_CHECK(cuMemcpyHtoD(  dPosTbl,    posTbl, posSize));
    CU_ERROR_CHECK(cuMemcpyHtoD(  dSh1Tbl,    sh1Tbl, sh1Size));
    CU_ERROR_CHECK(cuMemcpyHtoD(  dSh2Tbl,    sh2Tbl, sh2Size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dHighMask, &highMask, sizeof(uint32_t)));
    CU_ERROR_CHECK(cuMemcpyHtoD( dLowMask,  &lowMask, sizeof(uint32_t)));

    CU_ERROR_CHECK(cuMemFreeHost(posTbl));
    CU_ERROR_CHECK(cuMemFreeHost(sh1Tbl));
    CU_ERROR_CHECK(cuMemFreeHost(sh2Tbl));

    // Set up parameters in texture memory
    const size_t size = sizeof(uint32_t) * PARAMS_NUM * 16;

    CUtexref tParam, tTemper, tDouble;
    CU_ERROR_CHECK(cuModuleGetTexRef( &tParam, module, "tex_param_ref"));
    CU_ERROR_CHECK(cuModuleGetTexRef(&tTemper, module, "tex_temper_ref"));
    CU_ERROR_CHECK(cuModuleGetTexRef(&tDouble, module, "tex_double_ref"));

    CUdeviceptr dParam, dTemper, dDouble;
    CU_ERROR_CHECK(cuMemAlloc( &dParam, size));
    CU_ERROR_CHECK(cuMemAlloc(&dTemper, size));
    CU_ERROR_CHECK(cuMemAlloc(&dDouble, size));

    uint32_t * hParam, * hTemper, * hDouble;
    CU_ERROR_CHECK(cuMemAllocHost((void **) &hParam, size));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&hTemper, size));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&hDouble, size));

    for (size_t i = 0; i < PARAMS_NUM; i++) {
      for (size_t j = 0; j < 16; j++) {
        hParam [i * 16 + j] = (uint32_t)(mtgp64dc_params[i].tbl[j] >> 32);
        hTemper[i * 16 + j] = (uint32_t)(mtgp64dc_params[i].tmp_tbl[j] >> 32);
        hDouble[i * 16 + j] = (uint32_t)(mtgp64dc_params[i].dbl_tmp_tbl[j] >> 32);
      }
    }

    CU_ERROR_CHECK(cuMemcpyHtoD( dParam,  hParam, size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dTemper, hTemper, size));
    CU_ERROR_CHECK(cuMemcpyHtoD(dDouble, hDouble, size));

    CU_ERROR_CHECK(cuTexRefSetFilterMode( tParam, CU_TR_FILTER_MODE_POINT));
    CU_ERROR_CHECK(cuTexRefSetFilterMode(tTemper, CU_TR_FILTER_MODE_POINT));
    CU_ERROR_CHECK(cuTexRefSetFilterMode(tDouble, CU_TR_FILTER_MODE_POINT));

    CU_ERROR_CHECK(cuTexRefSetAddress(NULL,  tParam,  dParam, size));
    CU_ERROR_CHECK(cuTexRefSetAddress(NULL, tTemper, dTemper, size));
    CU_ERROR_CHECK(cuTexRefSetAddress(NULL, tDouble, dDouble, size));

    CU_ERROR_CHECK(cuMemFreeHost( hParam));
    CU_ERROR_CHECK(cuMemFreeHost(hTemper));
    CU_ERROR_CHECK(cuMemFreeHost(hDouble));
  }

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &state, &data, &inc, &samples_per_block };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)num_blocks, 1, 1, FLOOR_2P, 1, 1, (unsigned int)((FLOOR_2P * 3) * sizeof(uint64_t)), stream, params, NULL));

  return CUDA_SUCCESS;
}

static CUresult get(CUvectoru64 * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Em7convertEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getOpenOpen(CUvectord * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ed17convert_open_openEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getOpenClose(CUvectord * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ed18convert_open_closeEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getCloseOpen(CUvectord * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ed18convert_close_openEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUresult getCloseClose(CUvectord * res, CUdeviceptr state, CUstream stream) {
  CU_ERROR_CHECK(sample("_Z6sampleILm" STRING(N) "ELm" STRING(FLOOR_2P) "Ed19convert_close_closeEvP8mt_stateIXT_EEPT1_mm", state, res->data, res->inc, res->n, stream));
  return CUDA_SUCCESS;
}

static CUrng64_t type = { NAME, PARAMS_NUM * N * sizeof(uint64_t), UINT64_C(0), UINT64_C(0xffffffffffffffff), set, get, getOpenOpen, getOpenClose, getCloseOpen, getCloseClose };

const CUrng64_t * RNG_T = &type;
