template <typename T>
__global__ void all(T * A, size_t lda, T a, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] = a;
}

template <typename T>
__global__ void identity(T * A, size_t lda, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] = (i == j) ? T(1) : T(0);
}

template <typename T>
__global__ void add(T * A, size_t lda, T * B, size_t ldb, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] += B[j * ldb + i];
}

template <typename T>
__global__ void addConst(T * A, size_t lda, T a, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] += a;
}

template <typename T>
__global__ void subtract(T * A, size_t lda, T * B, size_t ldb, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] -= B[j * ldb + i];
}

template <typename T>
__global__ void subtractConst(T * A, size_t lda, T a, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] -= a;
}

template <typename T>
__global__ void multiply(T * A, size_t lda, T * B, size_t ldb, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] *= B[j * ldb + i];
}

template <typename T>
__global__ void multiplyConst(T * A, size_t lda, T a, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] *= a;
}

template <typename T>
__global__ void divide(T * A, size_t lda, T * B, size_t ldb, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] /= B[j * ldb + i];
}

template <typename T>
__global__ void divideConst(T * A, size_t lda, T a, size_t m, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n)
    A[j * lda + i] /= a;
}
