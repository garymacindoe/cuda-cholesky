#include <pthread.h>

static pthread_key_t key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

static void make_key() {
  ERROR_CHECK_VOID(pthread_key_create(&key, NULL));
}

/**
 * Thread-specific data.
 */
struct cgemm_data {
  CUmodule module;
  CUstream compute, copy;
  CUdeviceptr A0, A1, B0, B1, C;
  size_t lda, ldb, ldc;
};

/**
 * Arguments for background function.
 */
struct cgemm_args {
  CBlasTranspose transA, transB;
  size_t m, n, k, lda, ldb, ldc;
  const float complex * A, * B;
  float complex * C;
  float complex alpha, beta;
};

/**
 * Background function.
 */
static CUresult background_cgemm(const void * a) {
  struct cgemm_args * args = (struct cgemm_args *)a;

  // Unpack the arguments
  CBlasTranspose transA = args->transA;
  CBlasTranspose transB = args->transB;
  size_t m = args->m;
  size_t n = args->n;
  size_t k = args->k;
  float complex alpha = args->alpha;
  const float complex * A = args->A;
  size_t lda = args->lda;
  const float complex * B = args->B;
  size_t ldb = args->ldb;
  float complex beta = args->beta;
  float complex * C = args->C;
  size_t ldc = args->ldc;

  const size_t mb = MB;
  const size_t nb = NB;
  const size_t kb = KB;

  // Get thread-specific module, streams and memory pointers
  struct cgemm_data * data;
  ERROR_CHECK(pthread_once(&key_once, make_key));
  if ((data = pthread_getspecific(key)) == NULL) {
    if ((data = malloc(sizeof(struct cgemm_data))) == NULL)
      return CUDA_ERROR_OUT_OF_MEMORY;

    // Load the cgemm module
    CU_ERROR_CHECK(cuModuleLoad(&data->module, "cgemm.fatbin"));

    // Create separate streams for concurrent copy and execute
    CU_ERROR_CHECK(cuStreamCreate(&data->compute, 0));
    CU_ERROR_CHECK(cuStreamCreate(&data->copy, 0));

    // Allocate C (always mb * nb)
    CU_ERROR_CHECK(cuMemAllocPitch(&data->C, &data->ldc, mb * sizeof(float complex), nb,
                                   sizeof(float complex)));
    data->ldc /= sizeof(float complex);

    if (transA == CBlasNoTrans) {
      // A is mb * kb
      CU_ERROR_CHECK(cuMemAllocPitch(&data->A0, &data->lda, mb * sizeof(float complex), kb,
                                     sizeof(float complex)));
      CU_ERROR_CHECK(cuMemAllocPitch(&data->A1, &data->lda, mb * sizeof(float complex), kb,
                                     sizeof(float complex)));
      data->lda /= sizeof(float complex);
    }
    else {
      // A is kb * mb
      CU_ERROR_CHECK(cuMemAllocPitch(&data->A0, &data->lda, kb * sizeof(float complex), mb,
                                     sizeof(float complex)));
      CU_ERROR_CHECK(cuMemAllocPitch(&data->A1, &data->lda, kb * sizeof(float complex), mb,
                                     sizeof(float complex)));
      data->lda /= sizeof(float complex);
    }

    if (transB == CBlasNoTrans) {
      // B is kb * nb
      CU_ERROR_CHECK(cuMemAllocPitch(&data->B0, &data->ldb, kb * sizeof(float complex), nb,
                                     sizeof(float complex)));
      CU_ERROR_CHECK(cuMemAllocPitch(&data->B1, &data->ldb, kb * sizeof(float complex), nb,
                                     sizeof(float complex)));
      data->ldb /= sizeof(float complex);
    }
    else {
      // B is nb * kb
      CU_ERROR_CHECK(cuMemAllocPitch(&data->B0, &data->ldb, nb * sizeof(float complex), kb,
                                    sizeof(float complex)));
      CU_ERROR_CHECK(cuMemAllocPitch(&data->B1, &data->ldb, nb * sizeof(float complex), kb,
                                    sizeof(float complex)));
      data->ldb /= sizeof(float complex);
    }
  }

  // Copy C onto the device using the compute stream
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->C, data->ldc, 0, 0, C, ldc, 0, 0,
                                     m, n, sizeof(float complex), data->compute));

  // Perform C *= beta on the compute stream to ensure C has finished copying
  CU_ERROR_CHECK(cuCgemm(data->module, CBlasNoTrans, CBlasNoTrans, m, n, 0,
                         zero, 0, ldc, 0, 0, beta, data->C, data->ldc, data->compute));

  // Can exit early if alpha * op(A) * op(B) will evaluate to zero
  if (alpha != zero && k > 0) {

    // Perform C += alpha * op(A) * op(B)
    if (transB == CBlasNoTrans) {
      if (transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A0, data->lda, 0, 0, A, lda, 0, 0,
                                           m, lb, sizeof(float complex), data->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B0, data->ldb, 0, 0, B, ldb, 0, 0,
                                           lb, n, sizeof(float complex), data->compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(data->module, transA, transB, m, n, min(k - l, kb),
                                 alpha, data->A0, data->lda, data->B0, data->ldb,
                                 one, data->C, data->ldc, data->compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A1, data->lda, 0, 0, A, lda, 0, l + kb,
                                               m, lb, sizeof(float complex), data->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B1, data->ldb, 0, 0, B, ldb, l + kb, 0,
                                               lb, n, sizeof(float complex), data->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = data->compute; data->compute = data->copy; data->copy = stream;
            CUdeviceptr ptr = data->A0; data->A0 = data->A1; data->A1 = ptr;
            ptr = data->B0; data->B0 = data->B1; data->B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A0, data->lda, 0, 0, A, lda, 0, 0,
                                           lb, m, sizeof(float complex), data->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B0, data->ldb, 0, 0, B, ldb, 0, 0,
                                           lb, n, sizeof(float complex), data->compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(data->module, transA, transB, m, n, min(k - l, kb),
                                 alpha, data->A0, data->lda, data->B0, data->ldb,
                                 one, data->C, data->ldc, data->compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A1, data->lda, 0, 0, A, lda, l + kb, 0,
                                               lb, m, sizeof(float complex), data->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B1, data->ldb, 0, 0, B, ldb, l + kb, 0,
                                               lb, n, sizeof(float complex), data->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = data->compute; data->compute = data->copy; data->copy = stream;
            CUdeviceptr ptr = data->A0; data->A0 = data->A1; data->A1 = ptr;
            ptr = data->B0; data->B0 = data->B1; data->B1 = ptr;
          }
        }
      }
    }
    else {
      if (transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A0, data->lda, 0, 0, A, lda, 0, 0,
                                           m, lb, sizeof(float complex), data->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B0, data->ldb, 0, 0, B, ldb, 0, 0,
                                           n, lb, sizeof(float complex), data->compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(data->module, transA, transB, m, n, min(k - l, kb),
                                 alpha, data->A0, data->lda, data->B0, data->ldb,
                                 one, data->C, data->ldc, data->compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A1, data->lda, 0, 0, A, lda, 0, l + kb,
                                               m, lb, sizeof(float complex), data->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B1, data->ldb, 0, 0, B, ldb, 0, l + kb,
                                               n, lb, sizeof(float complex), data->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = data->compute; data->compute = data->copy; data->copy = stream;
            CUdeviceptr ptr = data->A0; data->A0 = data->A1; data->A1 = ptr;
            ptr = data->B0; data->B0 = data->B1; data->B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A0, data->lda, 0, 0, A, lda, 0, 0,
                                           lb, m, sizeof(float complex), data->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B0, data->ldb, 0, 0, B, ldb, 0, 0,
                                           n, lb, sizeof(float complex), data->compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(data->module, transA, transB, m, n, min(k - l, kb),
                                 alpha, data->A0, data->lda, data->B0, data->ldb,
                                 one, data->C, data->ldc, data->compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->A1, data->lda, 0, 0, A, lda, l + kb, 0,
                                               lb, m, sizeof(float complex), data->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(data->B1, data->ldb, 0, 0, B, ldb, 0, l + kb,
                                               n, lb, sizeof(float complex), data->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = data->compute; data->compute = data->copy; data->copy = stream;
            CUdeviceptr ptr = data->A0; data->A0 = data->A1; data->A1 = ptr;
            ptr = data->B0; data->B0 = data->B1; data->B1 = ptr;
          }
        }
      }
    }
  }

  // Copy C back onto the host on the compute stream
  CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(C, ldc, 0, 0, data->C, data->ldc, 0, 0,
                                     m, n, sizeof(float complex), data->compute));

  return CUDA_SUCCESS;
}